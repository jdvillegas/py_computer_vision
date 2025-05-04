# main.py

import cv2
import threading
import tkinter as tk
from ultralytics import YOLO
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

    while not exit_flag:
        # Capture frame from webcam
        frame = webcam.get_frame()
        if frame is None:
            break

        # Run YOLO detection
        results = model(frame, stream=True)  # Stream=True for real-time processing

        # Draw bounding boxes for detected people
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                cls = int(box.cls[0])  # Class ID
                conf = box.conf[0]  # Confidence score

                # Only draw boxes for "person" class (class ID 0 in COCO dataset) with confidence >= 85%
                if cls == 0 and conf >= 0.75:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red rectangle
                    cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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