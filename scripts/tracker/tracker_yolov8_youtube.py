import yt_dlp
import cv2
from ultralytics import YOLO

# URL del video de YouTube
url = "https://www.youtube.com/watch?v=EIal0QdVq7c"

# Configuración de yt-dlp para obtener la mejor URL de video
ydl_opts = {
    'format': 'best',
    'quiet': True
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(url, download=False)
    video_url = info_dict.get('url', None)

if video_url is None:
    raise Exception("Failed to retrieve video URL")

# Cargar el modelo de YOLO
model = YOLO('/home/herex/Escritorio/MASTER/Vision_por_computador/Trabajo_final/Trabajo_final_VC_udit/scripts/default_petrained_yolov8n/run/weights/best.pt')
# Inicia la captura de video desde la URL del mejor stream
cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    raise ConnectionError(f"Failed to read images from {url}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Realiza la inferencia con YOLO
    results = model.track(frame, conf=0.5, iou=0.5, persist = True)
    # Renderizar y mostrar los resultados
    for result in results:
        frame = result.plot()

    cv2.imshow('YOLOv8 Tracking', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
