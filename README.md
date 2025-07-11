# Proyecto: Detección de Incendios Forestales con Vision Transformers (ViT)

Este proyecto implementa un sistema de clasificación de imágenes aéreas para detectar incendios forestales, utilizando modelos Vision Transformer (ViT), como Swin-Tiny y Swin-Base, entrenados sobre datos reales de Arequipa (Perú) y un dataset público (FLAME).

---

## Características

- Clasificación de imágenes en dos clases: `Fire` y `No_Fire`.
- Soporte para entrenamiento con balanceo de clases.
- Evaluación cuantitativa: Accuracy, Precision, Recall, F1-score.
- Evaluación cualitativa: visualización de errores y aciertos.
- Inferencia en video frame a frame.
- Cálculo de curva ROC y AUC.

---

## Estructura del Proyecto

```
.
├── data/
│   ├── Training/
│   │   ├── Fire/
│   │   └── No_Fire/
│   ├── Test/
│   │   ├── Fire/
│   │   └── No_Fire/
│   └── mydata/
│       └── aplaco/  # Videos reales
├── models/
│   └── best_swin_model_50_tinny.pth
├── notebooks/
│   ├── Swin_FLAME_timm_example.ipynb
│   └── ResNet18_FLAME_timm_example.ipynb
├── scripts/
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── infer_on_video.py
│   └── roc_auc_curve.py
├── README.md
└── LICENSE
```

---

## Requisitos

- Python 3.9+
- PyTorch
- torchvision
- timm
- scikit-learn
- matplotlib
- opencv-python

Instalación recomendada:

```bash
pip install -r requirements.txt
```

---

## Instrucciones de Uso

### Entrenamiento

```bash
python scripts/train_model.py
```

### Inferencia en Video

```bash
python scripts/infer_on_video.py --video_path data/mydata/aplao/video.mp4
```

### Cálculo de Curva ROC y AUC

```bash
python scripts/roc_auc_curve.py
```

---

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.

---

## Contribuciones

Este repositorio forma parte de una tesis de maestría realizada en Perú. Se agradece cualquier contribución para mejorar el modelo, optimizar su inferencia o ampliar a tareas como segmentación de fuego. Puedes abrir un *issue* o enviar un *pull request*.

