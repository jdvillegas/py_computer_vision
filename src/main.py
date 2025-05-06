# main.py

import cv2
import threading
import tkinter as tk
from ultralytics import YOLO
from deepface import DeepFace
from modules.webcam_stream import WebcamStream
import math
import json
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Flag to control the application loop
exit_flag = False

# Dictionary to store tracked persons and their information
tracked_persons = {}
next_person_id = 1  # Counter for unique IDs

# Dictionary to store greetings
greetings = {}

def load_greetings():
    """
    Load greetings from saludos.json.
    """
    global greetings
    with open("d:\\python\\py_computer_vision\\src\\saludos.json", "r", encoding="utf-8") as file:
        greetings = json.load(file)

def draw_text_with_pillow(frame, text, x, y, font_path="arial.ttf", font_size=16, text_color=(255, 255, 255), bg_color=(0, 0, 0, 77)):
    """
    Dibuja texto con fondo translúcido en un frame de OpenCV usando Pillow.

    Parámetros:
    - frame: imagen de OpenCV (BGR)
    - text: texto a mostrar
    - x, y: coordenadas del texto
    - font_path: ruta a la fuente TTF
    - font_size: tamaño del texto
    - text_color: color del texto en RGB
    - bg_color: color del fondo como RGBA (opacidad 0–255)

    Devuelve:
    - Frame modificado con el texto y fondo dibujados
    """
    # Convertir de BGR a RGB y a imagen Pillow
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Fuente
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"⚠️ Fuente no encontrada en '{font_path}', usando fuente por defecto.")
        font = ImageFont.load_default()

    # Medidas del texto
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    padding = 6
    box = [x - padding, y - padding, x + text_width + padding, y + text_height + padding]

    # Dibujar fondo translúcido
    draw.rectangle(box, fill=bg_color)

    # Dibujar texto
    draw.text((x, y), text, font=font, fill=text_color)

    # Combinar imagen original y overlay con transparencia
    combined = Image.alpha_composite(pil_img, overlay)

    # Convertir de nuevo a BGR para OpenCV
    return cv2.cvtColor(np.array(combined.convert("RGB")), cv2.COLOR_RGB2BGR)

def close_application():
    global exit_flag
    exit_flag = True

def create_control_window():
    # Create a Tkinter window with a close button
    root = tk.Tk()
    root.title("Control Panel")
    root.geometry("200x100")
    close_button = tk.Button(root, text="Close Application", command=close_application)
    close_button.pack(expand=True)
    root.mainloop()

def calculate_distance(box1, box2):
    """
    Calculate the Euclidean distance between the centers of two bounding boxes.
    """
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2
    return math.sqrt((x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2)

def assign_id_to_person(box, threshold=50):
    """
    Assign a unique ID to a detected person based on proximity to existing tracked persons.
    """
    global tracked_persons, next_person_id

    for person_id, tracked_box in tracked_persons.items():
        distance = calculate_distance(box, tracked_box['box'])
        if distance < threshold:  # If the box is close to an existing person, assign the same ID
            tracked_persons[person_id]['box'] = box  # Update the tracked box
            return person_id

    # If no match is found, assign a new ID
    tracked_persons[next_person_id] = {'box': box, 'gender': None, 'greeting': None}  # Initialize with no gender or greeting
    next_person_id += 1
    return next_person_id - 1

def wrap_text(text, max_width, font, font_scale, thickness):
    """
    Wrap text into multiple lines if it exceeds the max width.
    """
    words = text.split(' ')
    lines = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        text_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
        if text_size[0] > max_width:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line

    if current_line:
        lines.append(current_line)

    return lines

def analyze_gender(face):
    """
    Analyze the gender of a detected face using DeepFace.
    """
    try:
        # Perform gender analysis
        analysis = DeepFace.analyze(face, actions=['gender'], enforce_detection=False)
        
        # Handle the case where DeepFace returns a list
        if isinstance(analysis, list):
            analysis = analysis[0]  # Take the first result if it's a list
        
        # Extract gender probabilities
        gender_probs = analysis.get('gender', {})
        if isinstance(gender_probs, dict):
            # Get the gender with the highest probability
            return max(gender_probs, key=gender_probs.get)
        else:
            return None
    except Exception as e:
        print(f"Error analyzing gender: {e}")
        return None

def assign_greeting(person_id, gender):
    """
    Assign a greeting to a person based on their gender.
    """
    global tracked_persons, greetings

    if tracked_persons[person_id]['greeting'] is None:
        if gender == "Man":
            tracked_persons[person_id]['greeting'] = random.choice(greetings["saludos_hombre"])
        elif gender == "Woman":
            tracked_persons[person_id]['greeting'] = random.choice(greetings["saludos_mujer"])
        else:
            tracked_persons[person_id]['greeting'] = random.choice(greetings["saludos_neutros"])

    # Ensure proper encoding for special characters
    return tracked_persons[person_id]['greeting'].encode('utf-8').decode('utf-8')


def main():
    global exit_flag

    # Load greetings from saludos.json
    load_greetings()

    # Start the control window in a separate thread
    control_thread = threading.Thread(target=create_control_window)
    control_thread.daemon = True
    control_thread.start()

    # Initialize webcam stream
    webcam = WebcamStream()
    webcam.start()

    # Load YOLO model
    model = YOLO('yolov8n.pt')  # Use YOLOv8 nano model (you can replace with 'yolov8s.pt' or others)

    # Create a resizable window
    cv2.namedWindow("Webcam Stream", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Webcam Stream", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while not exit_flag:
        # Capture frame from webcam
        frame = webcam.get_frame()
        if frame is None:
            break

        # Reduce resolution for better performance
        frame = cv2.resize(frame, (640, 480))

        # Run YOLO detection
        results = model(frame, stream=True, conf=0.5, iou=0.4)  # Adjusted confidence and IOU thresholds

        # Variables to track the closest person
        max_area = 0
        closest_box = None
        closest_person_id = None

        # Identify the closest person
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                cls = int(box.cls[0])  # Class ID
                conf = box.conf[0]  # Confidence score

                # Only consider "person" class (class ID 0 in COCO dataset) with confidence >= 85%
                if cls == 0 and conf >= 0.85:
                    # Calculate the area of the bounding box
                    area = (x2 - x1) * (y2 - y1)

                    # Check if this is the largest area so far
                    if area > max_area:
                        max_area = area
                        closest_box = (x1, y1, x2, y2)
                        closest_person_id = assign_id_to_person((x1, y1, x2, y2))

        # If a closest person is identified, process their information
        if closest_box and closest_person_id is not None:
            x1, y1, x2, y2 = closest_box

            # Extract the face region
            face = frame[y1:y2, x1:x2]

            # Check if gender is already known
            if tracked_persons[closest_person_id]['gender'] is None or tracked_persons[closest_person_id]['gender'] == 'Unknown':
                gender = analyze_gender(face)
                tracked_persons[closest_person_id]['gender'] = gender if gender else 'Unknown'
            else:
                gender = tracked_persons[closest_person_id]['gender']

            # Assign a greeting based on gender
            greeting = assign_greeting(closest_person_id, gender)

            # Wrap the greeting text to fit inside the bounding box
            max_width = x2 - x1
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            wrapped_text = wrap_text(greeting, max_width, font, font_scale, thickness)

            # Draw a rectangle and label for the closest person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green rectangle for the closest person
            label = f"ID: {closest_person_id}, {greeting}"
            frame = draw_text_with_pillow(frame, label, x1, y1 - 20, font_size=16, text_color=(0, 255, 0))

            # Draw the wrapped greeting text below the bounding box
            y_offset = y2 + 15
            for line in wrapped_text:
                frame = draw_text_with_pillow(frame, line, x1, y_offset, font_size=16, text_color=(255, 255, 255))
                y_offset += 20

        # Display the frame
        cv2.imshow("Webcam Stream", frame)

        # Check if the window is closed or 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or exit_flag:  # Exit on 'q' or button press
            break
        if cv2.getWindowProperty("Webcam Stream", cv2.WND_PROP_VISIBLE) < 1:  # Window closed
            break

    # Release resources
    webcam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()