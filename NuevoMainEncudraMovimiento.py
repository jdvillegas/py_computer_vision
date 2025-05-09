import cv2
import numpy as np

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

# Configuración de ventana en pantalla completa
cv2.namedWindow("Stream RTSP", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Stream RTSP", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Variables para suavizar el cuadro
previous_box = None
stabilization_factor = 0.85
text_stabilization_factor = 0.9

stable_center_x, stable_center_y = 0, 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error al recibir frame del stream")
        break

    # Proceso de detección con YOLO
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Detectar personas solamente
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype("int")

                # Suavizar la posición del cuadro
                if previous_box is not None:
                    center_x = int(stabilization_factor * previous_box[0] + (1 - stabilization_factor) * center_x)
                    center_y = int(stabilization_factor * previous_box[1] + (1 - stabilization_factor) * center_y)
                    w = int(stabilization_factor * previous_box[2] + (1 - stabilization_factor) * w)
                    h = int(stabilization_factor * previous_box[3] + (1 - stabilization_factor) * h)

                previous_box = (center_x, center_y, w, h)

                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))

                # Estabilizar el texto
                stable_center_x = int(text_stabilization_factor * stable_center_x + (1 - text_stabilization_factor) * center_x)
                stable_center_y = int(text_stabilization_factor * stable_center_y + (1 - text_stabilization_factor) * center_y)

                # Dibujar el rectángulo solo si es 'person'
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Mensaje de saludo en el centro del cuadro (estabilizado)
                text = "¡Hola, bienvenido!"
                text_x = int(stable_center_x - len(text) * 4)
                text_y = int(stable_center_y)
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Mostrar el frame
    cv2.imshow("Stream RTSP", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()