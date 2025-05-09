import cv2
import numpy as np
import json
import random
import os
import time

# Dictionary to store greetings
greetings = {}

def load_greetings():
    """
    Load greetings from saludos.json.
    """
    global greetings
    current_dir = os.path.dirname(os.path.abspath(__file__))
    greetings_path = os.path.join(current_dir, "saludos.json")

    if os.path.exists(greetings_path):
        with open(greetings_path, "r", encoding="utf-8-sig") as file:
            greetings = json.load(file)
    else:
        print("Error: saludos.json no encontrado.")
        exit()

# URL del stream RTSP
rtsp_url = 'rtsp://admin:Andi0783@192.169.10.10:554/cam/realmonitor?channel=1&subtype=0&unicast=true'

# Cargar YOLO
net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Definir las clases (solo 'person')
classes = ['person']

# Captura del video
cap = cv2.VideoCapture(rtsp_url)

# Verificación de la conexión
if not cap.isOpened():
    print("Error al conectar al stream RTSP")
    exit()

# Configuración de ventana en modo redimensionable
cv2.namedWindow("Stream RTSP", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Stream RTSP", 800, 600)  # Tamaño inicial

# Variables para suavizar el cuadro
previous_box = None
stabilization_factor = 0.85
text_stabilization_factor = 0.9

stable_center_x, stable_center_y = 0, 0
show_message = False
current_message = ""

load_greetings()

# Control de FPS
fps_limit = 15  # 15 FPS para mejor rendimiento
last_time = time.time()

while True:
    # Control de FPS
    current_time = time.time()
    if (current_time - last_time) < (1 / fps_limit):
        continue
    last_time = current_time

    ret, frame = cap.read()

    if not ret:
        print("Error al recibir frame del stream")
        break

    # Proceso de detección con YOLO
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (256, 256), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Dividir la pantalla en tres partes
    section_width = width // 3
    section_start = section_width
    section_end = section_width * 2

    show_message = False

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Detectar personas solamente
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype("int")

                if section_start < center_x < section_end:
                    show_message = True

    # Mostrar el mensaje solo si hay persona
    if show_message:
        if not current_message:  # Solo cambiar mensaje si está vacío
            current_message = random.choice(greetings.get("saludos_neutros", ["¡Bienvenido!"]))

        message_bg_x = 0
        message_bg_width = width
        message_bg_y = 0
        message_bg_height = 60
        cv2.rectangle(frame, (message_bg_x, message_bg_y), (message_bg_x + message_bg_width, message_bg_y + message_bg_height), (50, 50, 50), -1)

        # Texto del mensaje (utf-8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(current_message, font, 1.5, 2)[0]
        text_x = message_bg_x + (message_bg_width - text_size[0]) // 2
        text_y = message_bg_y + (message_bg_height + text_size[1]) // 2

        cv2.putText(frame, current_message.encode('utf-8').decode('utf-8'), (text_x, text_y), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

    else:
        current_message = ""  # Limpiar mensaje si no hay persona

    # Mostrar el frame
    cv2.imshow("Stream RTSP", frame)

    # Verificar cierre de ventana
    if cv2.getWindowProperty("Stream RTSP", cv2.WND_PROP_AUTOSIZE) == -1:
        break

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
