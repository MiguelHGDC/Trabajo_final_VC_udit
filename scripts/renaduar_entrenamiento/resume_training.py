from ultralytics import YOLO
import torch.backends.cudnn as cudnn
cudnn.benchmark = False
cudnn.deterministic = True

model = YOLO('/home/herex/Escritorio/MASTER/Vision_por_computador/Trabajo_final/Trabajo_final_VC_udit/scripts/train/freezed_petrained_yolov8n/run/weights/last.pt')

# Ruta al archivo YAML
yaml_path = '/home/herex/Escritorio/MASTER/Vision_por_computador/Trabajo_final/SDC_dataset/dataset.yaml'

# Entrenar el modelo con 2 GPUs
results = model.train(
    data=yaml_path,
    epochs=104,
    resume=True,
    device=[0]  # Especificar ambas GPUs.
)
results = model.val()