from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("/home/herex/Escritorio/MASTER/Vision_por_computador/Trabajo_final/Trabajo_final_VC_udit/scripts/yolov8m.pt")  # build a new model from scratch
    # Use the model, cfg = es la ruta a un YAML con los hiperparametros
    model.val(data = "/home/herex/Escritorio/MASTER/Vision_por_computador/Trabajo_final/SDC_dataset/dataset.yaml",cfg=r"E:\UDIT\UDIT_SDC\cfg\default.yaml")