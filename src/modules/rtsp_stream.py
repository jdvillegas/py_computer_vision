import cv2

class RTSPStream:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            raise Exception(f"Could not open RTSP stream: {self.rtsp_url}")

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

def display_stream(rtsp_url):
    """
    Connects to the RTSP stream and displays the video.
    
    Parameters:
        rtsp_url (str): The URL of the RTSP stream.
    """
    stream = RTSPStream(rtsp_url)
    stream.start()
    
    while True:
        frame = stream.get_frame()
        if frame is None:
            break
        cv2.imshow('RTSP Stream', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    stream.stop()
    cv2.destroyAllWindows()