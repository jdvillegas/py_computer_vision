import cv2

class WebcamStream:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")

    def get_frame(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

def display_video_stream():
    """Displays the video stream from the webcam."""
    webcam_stream = WebcamStream()
    webcam_stream.start()
    
    while True:
        frame = webcam_stream.get_frame()
        if frame is None:
            break
        cv2.imshow('Webcam Stream', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    webcam_stream.stop()
    cv2.destroyAllWindows()