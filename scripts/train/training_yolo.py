from ultralytics import YOLO
import torch.backends.cudnn as cudnn
cudnn.benchmark = False
cudnn.deterministic = True

# Determinar capas a congelar: desde la capa 0 hasta la capa 5 (antes de `ultralytics.nn.modules.conv.Conv` con out_channels=384)
# Cargar un modelo YOLO preentrenado (recomendado para entrenamiento)
model = YOLO('/home/herex/Escritorio/MASTER/Vision_por_computador/Trabajo_final/Trabajo_final_VC_udit/scripts/train/freezed_petrained_yolov8n/run/weights/best.pt')

# # Ruta al archivo YAML
# yaml_path = '/home/herex/Escritorio/MASTER/Vision_por_computador/Trabajo_final/SDC_dataset/dataset.yaml'

# # Capas a congelar
# freeze_layers = [0, 1, 2, 3, 4]

# # Entrenar el modelo con 2 GPUs
# results = model.train(
#     data=yaml_path,
#     epochs=600,
#     imgsz=640,
#     patience=20,
#     batch=26,
#     workers=18,
#     project='freezed_v2_petrained_yolov8m_pruebaaa',
#     name='run',
#     exist_ok=True,
#     cache=False,
#     pretrained=True,
#     optimizer='auto',  # Options include SGD, Adam, AdamW, NAdam, RAdam, RMSProp etc.
#     verbose=False,
#     deterministic=True,  # Ensures reproducibility but may affect performance.
#     amp=True,  # Automatic mixed precision training (fp16/fp32) - faster and more memory efficient.
#     seed=0,
#     freeze=freeze_layers,  # Freeze specified layers
#     plots=True,  # Generates and saves plots of training and validation metrics.
#     device=[0]  # Especificar ambas GPUs.
# )

# Evaluar el rendimiento del modelo en el conjunto de validación
results = model.val()

# # Asumiendo que 'results' puede ser convertido a un diccionario para la serialización JSON
# metrics_dict = results.to_dict()

# import json
# with open('metrics.json', 'w') as f:
#     json.dump(metrics_dict, f)

"""@software{yolov8_ultralytics,
  author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title = {Ultralytics YOLOv8},
  version = {8.0.0},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}"""

