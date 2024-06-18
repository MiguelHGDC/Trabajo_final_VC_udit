from ultralytics import YOLO

def load_model(model_path):
    """
    Carga el modelo YOLO desde la ruta especificada.

    :param model_path: Ruta al archivo del modelo YOLO.
    :return: Modelo YOLO cargado.
    """
    return YOLO(model_path)

def process_video(video_path,
                  model, 
                  conf=0.5, 
                  iou=0.7, 
                  imgsz=640, 
                  half=False, 
                  device='cuda', 
                  max_det=300, 
                  vid_stride=1, 
                  stream_buffer=False, 
                  visualize=False, 
                  augment=False, 
                  agnostic_nms=False, 
                  classes=None, 
                  retina_masks=False, 
                  show=False, 
                  save=True, 
                  save_frames=False, 
                  save_txt=False, 
                  save_conf=False, 
                  save_crop=False, 
                  show_labels=True, 
                  show_conf=True, 
                  show_boxes=True, 
                  line_width=None):
    """
    Procesa el video realizando predicciones frame por frame y guarda el resultado.

    :param video_path: Ruta del video de entrada.
    :param model: Modelo YOLO cargado.
    :param conf: Umbral de confianza mínimo para las detecciones.
    :param iou: Umbral de IoU para la supresión no máxima.
    :param imgsz: Tamaño de la imagen para la inferencia.
    :param half: Habilita la inferencia en media precisión (FP16).
    :param device: Dispositivo para la inferencia (cpu, cuda:0, etc.).
    :param max_det: Número máximo de detecciones permitidas por imagen.
    :param vid_stride: Intervalo de frames para entradas de video.
    :param stream_buffer: Determina si se deben almacenar todos los frames en el buffer.
    :param visualize: Activa la visualización de las características del modelo durante la inferencia.
    :param augment: Habilita la augmentación en tiempo de prueba (TTA).
    :param agnostic_nms: Habilita la supresión no máxima agnóstica a la clase.
    :param classes: Filtra las predicciones a un conjunto de IDs de clase.
    :param retina_masks: Utiliza máscaras de segmentación de alta resolución.
    :param show: Muestra las imágenes o videos anotados en una ventana.
    :param save: Habilita la opción de guardar las imágenes o videos anotados en un archivo.
    :param save_frames: Guarda frames individuales como imágenes al procesar videos.
    :param save_txt: Guarda los resultados de detección en un archivo de texto.
    :param save_conf: Incluye las puntuaciones de confianza en los archivos de texto guardados.
    :param save_crop: Guarda imágenes recortadas de las detecciones.
    :param show_labels: Muestra las etiquetas de cada detección en la salida visual.
    :param show_conf: Muestra la puntuación de confianza para cada detección junto a la etiqueta.
    :param show_boxes: Dibuja cuadros delimitadores alrededor de los objetos detectados.
    :param line_width: Especifica el ancho de línea de los cuadros delimitadores.
    """
    
    model = model.to(device)
    model.predict(video_path, 
                            imgsz=imgsz, 
                            conf=conf, 
                            iou=iou, 
                            half=half, 
                            device=device, 
                            max_det=max_det, 
                            vid_stride=vid_stride, 
                            stream_buffer=stream_buffer, 
                            visualize=visualize, 
                            augment=augment, 
                            agnostic_nms=agnostic_nms, 
                            classes=classes, 
                            retina_masks=retina_masks,
                            show=show,
                            save=save,
                            save_frames=save_frames,
                            save_txt=save_txt,
                            save_conf=save_conf,
                            save_crop=save_crop,
                            show_labels=show_labels,
                            show_conf=show_conf,
                            show_boxes=show_boxes,
                            line_width=line_width)


def main():
    # Ruta al archivo del modelo YOLO entrenado
    model_path = '/home/herex/Escritorio/MASTER/Vision_por_computador/Trabajo_final/Trabajo_final_VC_udit/scripts/freezed_petrained_yolov8m/run/weights/best.pt'
    # Ruta del video de entrada
    video_path = '/home/herex/Escritorio/MASTER/Vision_por_computador/Trabajo_final/Entrega_tracker/Traffic_top_view.mp4'

    # Cargar el modelo
    model = load_model(model_path)
    
    # Procesar el video
    process_video(
                    video_path,
                    model,
                    conf=0.1,
                    iou=0.5,
                    imgsz=640,
                    half=False,
                    device='cuda',
                    max_det=300,
                    vid_stride=1,
                    stream_buffer=False,
                    visualize=False,
                    augment=False,
                    agnostic_nms=True,
                    classes=None,
                    retina_masks=True,
                    show=True,  # Mostrar el video anotado en una ventana
                    save=True,  # Guardar el video anotado
                    save_frames=False,
                    save_txt=False,
                    save_conf=False,
                    save_crop=False,
                    show_labels=True,
                    show_conf=True,
                    show_boxes=True,
                    line_width=None
                  )

if __name__ == "__main__":
    main()