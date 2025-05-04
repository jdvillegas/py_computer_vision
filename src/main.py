# main.py

import cv2
import threading
import tkinter as tk
from ultralytics import YOLO
from deepface import DeepFace
from modules.webcam_stream import WebcamStream

# Flag to control the application loop
exit_flag = False

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

def main():
    global exit_flag

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

    frame_count = 0  # Counter to limit the frequency of gender analysis

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

        # Draw bounding boxes for detected people
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
                        closest_box = (x1, y1, x2, y2, conf)

                    # Draw a red rectangle for all detected people
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"Person {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Perform gender analysis only every 10 frames and for the closest person
        if frame_count % 10 == 0 and closest_box:
            x1, y1, x2, y2, conf = closest_box
            face = frame[y1:y2, x1:x2]

            # Analyze gender
            gender = analyze_gender(face)
            if gender:
                cv2.putText(frame, f"Closest: {gender}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Highlight the closest person
        if closest_box:
            x1, y1, x2, y2, conf = closest_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green rectangle for the closest person

        # Display the frame
        cv2.imshow("Webcam Stream", frame)

        # Increment frame count
        frame_count += 1

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