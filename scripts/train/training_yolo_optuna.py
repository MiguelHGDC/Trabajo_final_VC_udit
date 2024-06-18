import optuna
from ultralytics import YOLO

PATH = './'

# Definir el espacio de búsqueda para los hiperparámetros
def objective(trial):
    # Definir los hiperparámetros a optimizar
    
    """
        lr0 (Initial Learning Rate):
        Descripción: La tasa de aprendizaje inicial para el optimizador. 
        Influencia: Controla la magnitud de las actualizaciones de los parámetros del modelo. 
        Un valor demasiado alto puede causar inestabilidad y no converger, 
        mientras que uno muy bajo puede resultar en una convergencia lenta y subóptima.
        
        lrf (Final Learning Rate):
        Descripción: La tasa de aprendizaje final como una fracción de la tasa de aprendizaje inicial.
        Influencia: Ayuda a ajustar la tasa de aprendizaje de manera gradual durante el entrenamiento. 
        Un lrf más bajo puede ayudar a afinar el modelo hacia el final del entrenamiento.
        
        momentum:
        Descripción: El factor de momento para optimizadores como SGD o beta1 para Adam.
        Influencia: Ayuda a acelerar el entrenamiento en la dirección de los gradientes persistentes, 
        suavizando las actualizaciones. Un valor adecuado puede mejorar la convergencia, 
        mientras que uno inapropiado puede causar oscilaciones.
        
        weight_decay:
        Descripción: La tasa de decaimiento del peso, utilizada para regularización L2.
        Influencia: Penaliza los pesos grandes para evitar el sobreajuste. 
        Un valor alto puede conducir a un subajuste, mientras que un valor demasiado bajo puede no ser efectivo para prevenir el sobreajuste.
        
        warmup_epochs:
        Descripción: El número de épocas para la fase de calentamiento de la tasa de aprendizaje.
        Influencia: Gradualmente incrementa la tasa de aprendizaje desde un valor bajo hasta el valor inicial definido (lr0). 
        Esto puede estabilizar el entrenamiento al comienzo y prevenir grandes actualizaciones abruptas.
        
        warmup_momentum:
        Descripción: El momento inicial para la fase de calentamiento.
        Influencia: Ajusta el momento durante el calentamiento, comenzando desde un valor más bajo y aumentando gradualmente. 
        Ayuda a estabilizar las actualizaciones iniciales del modelo.
        
        warmup_bias_lr:
        Descripción: La tasa de aprendizaje para los parámetros de sesgo durante el calentamiento.
        Influencia: Un ajuste específico para los sesgos, 
        que pueden necesitar una tasa de aprendizaje diferente para estabilizarse correctamente durante las primeras épocas.
        
        box:
        Descripción: Peso del componente de pérdida de la caja en la función de pérdida.
        Influencia: Influencia la precisión con la que se predicen las coordenadas de los cuadros delimitadores. 
        Un valor adecuado balancea entre la precisión de la ubicación y otras pérdidas.
        
        cls:
        Descripción: Peso de la pérdida de clasificación en la función de pérdida total.
        Influencia: Afecta la importancia de la predicción correcta de la clase en comparación con otros componentes de la pérdida. 
        Un valor adecuado equilibra la precisión de la clasificación con la localización y otras tareas.
        
        dfl:
        Descripción: Peso de la pérdida focal de distribución.
        Influencia: Utilizado en versiones específicas de YOLO para clasificación detallada. 
        Un valor adecuado puede mejorar la precisión al focalizarse en los ejemplos difíciles.
    """
    
    lr0 = trial.suggest_float('lr0', 1e-5, 1e-2, log=True)
    lrf = trial.suggest_uniform('lrf', 0.01, 1.0)
    momentum = trial.suggest_uniform('momentum', 0.6, 0.98)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    warmup_epochs = trial.suggest_uniform('warmup_epochs', 0, 10)
    warmup_momentum = trial.suggest_uniform('warmup_momentum', 0.0, 0.95)
    warmup_bias_lr = trial.suggest_float('warmup_bias_lr', 1e-6, 1e-2, log=True)
    box = trial.suggest_uniform('box', 0.02, 0.2)
    cls = trial.suggest_uniform('cls', 0.2, 4.0)
    dfl = trial.suggest_uniform('dfl', 0.2, 4.0)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.4)

    # Ruta al archivo YAML
    yaml_path = '/home/herex/Escritorio/MASTER/Vision_por_computador/Trabajo_final/SDC_dataset/dataset.yaml'

    # Cargar un modelo YOLO preentrenado
    model = YOLO('yolov8n.pt')

    # Entrenar el modelo con los hiperparámetros sugeridos
    results = model.train(data=yaml_path,
                          epochs=100,
                          imgsz=640,
                          patiente= 15,
                          batch_size=16,
                          workers=18,
                          project='optuna_optimization_yolo',
                          name = 'trial_' + str(trial.number),
                          exist_ok=True,
                          cache_images=True,
                          pretrained=True,
                          optimizer='auto',# Options include SGD, Adam, AdamW, NAdam, RAdam, RMSProp etc.,
                          verbose=False,
                          deterministic = True,# Forces deterministic algorithm use, ensuring reproducibility but may affect performance and speed due to the restriction on non-deterministic algorithms.
                          amp = True,# Automatic mixed precision training (fp16/fp32) - faster and more memory efficient.
                          seed = 0,
                          lr0=lr0,
                          lrf=lrf,
                          freeze = None,# Freeze model (excluding tail)Freezes the first N layers of the model or specified layers by index, reducing the number of trainable parameters. Useful for fine-tuning or transfer learning.
                          plots = False, # Generates and saves plots of training and validation metrics, as well as prediction examples, providing visual insights into model performance and learning progression.
                          mask_ratio = 4,# Downsample ratio for segmentation masks, affecting the resolution of masks used during training.
                          nbs = 64,# Nominal batch size for normalization of loss.
                          dropout = dropout,
                          momentum=momentum,
                          weight_decay=weight_decay,
                          warmup_epochs=warmup_epochs,
                          warmup_momentum=warmup_momentum,
                          warmup_bias_lr=warmup_bias_lr,
                          box=box,
                          cls=cls,
                          dfl=dfl,
                          device=[0])

    # Evaluar el rendimiento del modelo en el conjunto de validación
    metrics = model.val()
    
    # Devolver la métrica de evaluación (por ejemplo, mAP)
    return metrics['map']

# Crear un estudio de Optuna y optimizar los hiperparámetros
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Imprimir los mejores hiperparámetros encontrados
print('Número de prueba: ', study.best_trial.number)
print('Valor: ', study.best_trial.value)
print('Mejores hiperparámetros: ', study.best_trial.params)

# USAR NETRON PARA VISUALIZAR ARQUITECTURAS DE RED https://netron.app/