import cv2

# URL del stream RTSP
rtsp_url = 'rtsp://admin:Andi0783@192.169.10.10:554/cam/realmonitor?channel=1&subtype=0&unicast=true'

# Captura del video
cap = cv2.VideoCapture(rtsp_url)

# Verificación de la conexión
if not cap.isOpened():
    print("Error al conectar al stream RTSP")
    exit()

# Configuración de ventana en pantalla completa
cv2.namedWindow("Stream RTSP", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Stream RTSP", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error al recibir frame del stream")
        break

    # Mostrar el frame
    cv2.imshow("Stream RTSP", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
