# Video Stream Application

This project is a Python application that streams video from a webcam or an RTSP source. It is designed to be modular, allowing for easy maintenance and expansion.

## Project Structure

```
video-stream-app
├── src
│   ├── main.py                # Entry point of the application
│   ├── modules
│   │   ├── webcam_stream.py    # Module for webcam streaming
│   │   └── rtsp_stream.py      # Module for RTSP streaming
├── requirements.txt           # Dependencies for the project
└── README.md                  # Documentation for the project
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd video-stream-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command:

```
python src/main.py
```

You can choose to stream from either the webcam or an RTSP source by modifying the code in `main.py`.

## Modules

### Webcam Stream

- **File:** `src/modules/webcam_stream.py`
- **Description:** This module handles the functionality for streaming video from the webcam. It includes functions to initialize the webcam, capture frames, and display the video stream.

### RTSP Stream

- **File:** `src/modules/rtsp_stream.py`
- **Description:** This module is designed to handle video streaming from an RTSP source. It includes functions to connect to the RTSP stream, capture frames, and display the video.

## Contributing

Feel free to submit issues or pull requests to improve the application. 

## License

This project is licensed under the MIT License.