
# CNN Anime Classifier

Este proyecto implementa una red neuronal convolucional (CNN) en PyTorch para **clasificar imágenes de personajes de anime** en dos categorías: `anastasia` y `takao`.

>  Ideal para quienes desean practicar clasificación de imágenes con datasets personalizados en formato `.zip`.

---

## Estructura del Proyecto

Tu directorio debe tener este aspecto:

```
CNN_anime/
│
├── data.zip                # Contiene las imágenes de entrenamiento
├── CNN_classifier.py       # Script principal del modelo
└── README.md               # Este archivo
```

El archivo `data.zip` debe contener dos carpetas:

```
data.zip
├── anastasia/              # Imágenes de la clase 0
└── takao/                  # Imágenes de la clase 1
```

---

## Requisitos

### ✅ Python

- Recomendado: **Python 3.10 o superior**

### ✅ Librerías (puedes usar `pip install`)

```bash
pip install torch torchvision matplotlib numpy
```

O con `conda`:

```bash
conda install pytorch torchvision matplotlib numpy -c pytorch
```

---

## Cómo Ejecutar el Script

1. Asegúrate de que `data.zip` está en la raíz del proyecto.
2. Ejecuta el script desde consola:

```bash
python CNN_classifier.py
```

El script hará lo siguiente:

- Cargará y preprocesará las imágenes desde el `.zip`.
- Dividirá el dataset en entrenamiento y validación (80/20).
- Entrenará una CNN durante 10 épocas.
- Mostrará una gráfica de pérdidas (`loss`) para entrenamiento y validación.
- Visualizará predicciones del modelo sobre imágenes reales.

---

## Resultado Esperado

Durante la ejecución verás algo como:

```
[INFO] Cargando imágenes desde data.zip...
[INFO] Tamaño del dataset: 100 imágenes (train=80, val=20)
[Época 1] Loss: train=0.67, val=0.63, acc=62.50%
...
```

Y al final:

- Una **gráfica de evolución del loss**
- Una **galería de imágenes** con predicción y etiqueta real (GT = Ground Truth)

---

##  Notas Técnicas

- Las imágenes se reescalan automáticamente a 64x64 píxeles.
- Se normalizan con media y desviación estándar `(0.5, 0.5, 0.5)` para cada canal RGB.
- El modelo utiliza dos capas convolucionales y dos densas.
- Se entrena con `CrossEntropyLoss` y `Adam`.

---

## ¿Por qué es interesante?

-  Puedes cambiar las carpetas dentro de `data.zip` por cualquier otro dataset personalizado.
-  Es una excelente base para explorar **transfer learning**, aumento de datos o regularización.

---

##  Autor

Creado por [Angel Calvar Pastoriza].  
Licencia: MIT.
