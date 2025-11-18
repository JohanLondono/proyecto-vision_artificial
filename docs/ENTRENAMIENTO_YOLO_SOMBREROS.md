# Entrenamiento de YOLO para Detección de Sombreros 🎩

## Universidad del Quindío - Visión Artificial
### Sistema de Detección de Sombreros con YOLO Fine-Tuning

---

## 📋 Tabla de Contenidos

1. [Introducción](#introducción)
2. [Requisitos del Sistema](#requisitos-del-sistema)
3. [Estructura del Dataset](#estructura-del-dataset)
4. [Preparación de Datos](#preparación-de-datos)
5. [Proceso de Entrenamiento](#proceso-de-entrenamiento)
6. [Uso del Modelo Entrenado](#uso-del-modelo-entrenado)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Introducción

### ¿Qué es Fine-Tuning?

**Fine-tuning** es el proceso de tomar un modelo preentrenado (como YOLOv8 entrenado en COCO) y **re-entrenarlo** con un dataset específico (sombreros) para especializarlo en una tarea particular.

### ¿Por qué YOLO?

- ✅ **Detección en tiempo real** (>30 FPS)
- ✅ **Alta precisión** para objetos pequeños
- ✅ **Fácil de entrenar** con pocos datos
- ✅ **Compatible** con cámara web y archivos de video
- ✅ **Formato simple** de anotación (YOLO txt)

### Modelos Actuales vs YOLO Fine-Tuned

| Característica | VGG16/ResNet (Actual) | YOLO Fine-Tuned |
|----------------|----------------------|-----------------|
| Tipo | Clasificación de imagen completa | Detección con bounding boxes |
| Salida | "Cowboy Hat 79%", "Sombrero 20%" | Coordenadas exactas del sombrero |
| Múltiples objetos | ❌ Solo uno por imagen | ✅ Detecta múltiples sombreros |
| Localización | ❌ No localiza | ✅ Dibuja cajas precisas |
| Tiempo real | ⚠️ Lento (~200ms) | ✅ Rápido (~30ms) |
| Clases personalizadas | ❌ 1000 clases fijas ImageNet | ✅ Solo tus clases |

---

## 💻 Requisitos del Sistema

### Hardware Requerido

#### Mínimo:
- **CPU**: Intel i5 o AMD Ryzen 5
- **RAM**: 8 GB
- **Disco**: 10 GB libres
- **Tiempo de entrenamiento**: ~2-4 horas (CPU)

#### Recomendado:
- **GPU**: NVIDIA GTX 1060 o superior (6GB VRAM)
- **RAM**: 16 GB
- **Disco**: 20 GB libres (SSD preferible)
- **Tiempo de entrenamiento**: ~20-40 minutos (GPU)

### Software Requerido

```bash
# Verificar instalación
python --version  # Python 3.8+
pip --version
nvcc --version    # CUDA (opcional, para GPU)
```

### Dependencias Python

Ya están en `requirements-full.txt`, pero específicamente necesitas:

```bash
pip install ultralytics  # YOLOv8
pip install opencv-python
pip install torch torchvision  # PyTorch
pip install pillow
pip install pyyaml
pip install pandas
pip install matplotlib
pip install tqdm
```

---

## 📁 Estructura del Dataset

### Formato YOLO

YOLO usa un formato específico de anotación:

```
dataset_sombreros_yolo/           ← Formato correcto YOLO
├── images/
│   ├── train/
│   │   ├── sombrero_001.jpg
│   │   ├── sombrero_002.jpg
│   │   └── ...
│   ├── val/
│   │   ├── sombrero_501.jpg
│   │   └── ...
│   └── test/
│       ├── sombrero_701.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── sombrero_001.txt
│   │   ├── sombrero_002.txt
│   │   └── ...
│   ├── val/
│   │   ├── sombrero_501.txt
│   │   └── ...
│   └── test/
│       ├── sombrero_701.txt
│       └── ...
└── data.yaml

dataset_sombreros/                ← Open Images (descargado)
├── cowboy hat/
│   ├── images/
│   └── darknet/
├── fedora/
│   ├── images/
│   └── darknet/
└── ...
```

**⚠️ IMPORTANTE**: Si descargaste de Open Images, necesitas reorganizar:

```bash
# Reorganizar automáticamente a formato YOLO
python preparar_dataset_sombreros.py \
    --input ./dataset_sombreros \
    --output ./dataset_sombreros_yolo \
    --split 0.7 0.2 0.1
```

```powershell
# PowerShell
python preparar_dataset_sombreros.py --input ./dataset_sombreros --output ./dataset_sombreros_yolo --split 0.7 0.2 0.1
```

### Formato de Anotación (`.txt`)

Cada imagen tiene un archivo `.txt` con el mismo nombre:

```
# sombrero_001.txt
# Formato: <clase> <x_centro> <y_centro> <ancho> <alto>
0 0.5 0.3 0.2 0.15
1 0.7 0.4 0.18 0.12
```

**Coordenadas normalizadas** (0.0 a 1.0):
- `clase`: ID de la clase (0 = cowboy_hat, 1 = sombrero, etc.)
- `x_centro`: Centro X del bbox / ancho_imagen
- `y_centro`: Centro Y del bbox / alto_imagen
- `ancho`: Ancho del bbox / ancho_imagen
- `alto`: Alto del bbox / alto_imagen

### Archivo `data.yaml`

```yaml
# Rutas del dataset
train: dataset_sombreros/images/train
val: dataset_sombreros/images/val
test: dataset_sombreros/images/test

# Número de clases
nc: 3

# Nombres de clases
names:
  0: cowboy_hat
  1: sombrero
  2: baseball_cap
```

---

## 🗃️ Preparación de Datos

### Opción 1: Recolección Manual

#### Paso 1: Capturar Imágenes

```python
# Usar el script de captura
python scripts/capturar_imagenes_sombreros.py
```

**Recomendaciones:**
- **Mínimo**: 100-200 imágenes por clase
- **Recomendado**: 500-1000 imágenes por clase
- **Óptimo**: 2000+ imágenes por clase

**Variaciones importantes:**
- ✅ Diferentes ángulos (frontal, lateral, superior)
- ✅ Diferentes iluminaciones (día, noche, sombra)
- ✅ Diferentes fondos
- ✅ Diferentes distancias (cerca, lejos)
- ✅ Múltiples sombreros en una imagen
- ✅ Sombreros parcialmente ocultos

#### Paso 2: Etiquetar Imágenes

**Herramientas recomendadas:**

1. **LabelImg** (Recomendado - Fácil)
   ```bash
   pip install labelImg
   labelImg
   ```
   - Interfaz gráfica simple
   - Exporta directamente a formato YOLO
   - Atajos de teclado rápidos

2. **Roboflow** (Online - Gratis hasta 1000 imágenes)
   - https://roboflow.com
   - Etiquetado colaborativo
   - Augmentación automática
   - Exporta a YOLO directamente

3. **CVAT** (Avanzado - Proyectos grandes)
   - https://cvat.org
   - Etiquetado en equipo
   - Tracking automático en video

**Tutorial LabelImg:**

1. Abrir LabelImg
2. "Open Dir" → Seleccionar carpeta con imágenes
3. "Change Save Dir" → Seleccionar carpeta de salida
4. "PascalVOC" → Cambiar a "YOLO"
5. Presionar `W` → Dibujar caja
6. Escribir nombre de clase
7. `Ctrl+S` → Guardar
8. `D` → Siguiente imagen

### Opción 2: Datasets Públicos

#### Datasets Disponibles:

1. **Open Images V7** (Google)
   ```bash
   # Instalar herramienta
   pip install openimages
   
   # Listar clases disponibles de sombreros
   # Hat, Helmet, Fedora, Sombrero, "Cowboy hat", "Sun hat", 
   # "Bicycle helmet", "Football helmet", "Swim cap"
   
   # Descargar clase "Hat" (SINTAXIS CORRECTA)
   oi_download_dataset --base_dir ./dataset_sombreros --labels Hat --format darknet --limit 500
   
   # Parámetros:
   # --base_dir: Carpeta destino (antes era --dest)
   # --labels: Clases a descargar (antes era --classes) - USAR NOMBRES EXACTOS
   # --format: darknet=YOLO, pascal=XML
   # --limit: Máximo de imágenes (opcional)
   
   # Descargar múltiples tipos de sombreros (NOMBRES EXACTOS)
   oi_download_dataset --base_dir ./dataset_sombreros --labels "Cowboy hat" Fedora Hat Helmet Sombrero "Sun hat" --format darknet --limit 500
   
   # IMPORTANTE: Usar comillas para nombres con espacios
   # Cap NO existe → usar Hat
   # Cowboy hat → usar "Cowboy hat" (con comillas)
   
   #Para imprimir clases on sombreros de openimage:
   python -c "import pandas as pd; import urllib.request; url='https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv'; urllib.request.urlretrieve(url, 'classes.csv'); df=pd.read_csv('classes.csv', header=None); hats=df[df[1].str.contains('hat|cap|helmet|fedora|sombrero|beret|cowboy', case=False, na=False)]; print('CLASES RELACIONADAS CON SOMBREROS:'); print(hats.to_string(index=False))"
   ```

2. **COCO Dataset** (Subset)
   - Contiene algunas imágenes con sombreros
   - Clase: "hat" (ID: 89)

3. **ImageNet** (Subset)
   - cowboy_hat (n03122748)
   - sombrero (n04208210)
   - bonnet (n02870526)

#### Conversión de Formatos:

```python
# Si tienes anotaciones en otro formato
python preparar_dataset_sombreros.py --input-format coco \
                                      --input-path ./coco_annotations.json \
                                      --output-path ./dataset_sombreros
```

### Opción 3: Reorganizar Dataset de Open Images

Si descargaste imágenes con `oi_download_dataset`, necesitas reorganizar de estructura Open Images a formato YOLO.

#### **¿Por qué reorganizar?**

Open Images descarga en esta estructura:
```
dataset_sombreros/
├── cowboy hat/          ← Organizado por clase
│   ├── images/
│   └── darknet/
├── fedora/
│   ├── images/
│   └── darknet/
└── ...
```

YOLO necesita esta estructura:
```
dataset_sombreros_yolo/
├── images/              ← Organizado por split
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

#### **Script: `preparar_dataset_sombreros.py`**

Este script automatiza la reorganización completa del dataset.

##### **Características:**

✅ **Reorganiza automáticamente** de Open Images a YOLO  
✅ **Split inteligente** (train/val/test configurable)  
✅ **Valida anotaciones** (formato, coordenadas, correspondencia)  
✅ **Genera `data.yaml`** automáticamente  
✅ **Mezcla aleatoria** (evita sesgos por orden)  
✅ **Muestra estadísticas** por clase y split  

##### **Uso Básico:**

```bash
# Reorganizar con split por defecto (70/20/10)
python preparar_dataset_sombreros.py \
    --input ./dataset_sombreros \
    --output ./dataset_sombreros_yolo
```

```powershell
# PowerShell
python preparar_dataset_sombreros.py --input ./dataset_sombreros --output ./dataset_sombreros_yolo

# Resultado:
# ✓ 2,670 imágenes organizadas
# ✓ Train: 1,868 imágenes (70%)
# ✓ Val: 534 imágenes (20%)
# ✓ Test: 268 imágenes (10%)
# ✓ data.yaml generado
```

##### **Comandos Disponibles:**

```bash
# 1. Reorganizar con split personalizado (80/15/5)
python preparar_dataset_sombreros.py \
    --input ./dataset_sombreros \
    --output ./dataset_sombreros_yolo \
    --split 0.8 0.15 0.05

# 2. Solo validar dataset existente (sin reorganizar)
python preparar_dataset_sombreros.py \
    --input ./dataset_sombreros_yolo \
    --validate-only

# 3. Solo mostrar estadísticas
python preparar_dataset_sombreros.py \
    --input ./dataset_sombreros_yolo \
    --stats-only

# 4. Reorganizar y validar en un solo paso
python preparar_dataset_sombreros.py \
    --input ./dataset_sombreros \
    --output ./dataset_sombreros_yolo
```

```powershell
# PowerShell (comandos en una línea)
# 1. Split personalizado
python preparar_dataset_sombreros.py --input ./dataset_sombreros --output ./dataset_sombreros_yolo --split 0.8 0.15 0.05

# 2. Solo validar
python preparar_dataset_sombreros.py --input ./dataset_sombreros_yolo --validate-only

# 3. Solo estadísticas
python preparar_dataset_sombreros.py --input ./dataset_sombreros_yolo --stats-only

# 4. Reorganizar y validar
python preparar_dataset_sombreros.py --input ./dataset_sombreros --output ./dataset_sombreros_yolo
```

##### **Parámetros:**

| Parámetro | Descripción | Obligatorio | Default |
|-----------|-------------|-------------|---------|
| `--input` | Carpeta de entrada (Open Images o YOLO) | ✅ Sí | - |
| `--output` | Carpeta de salida (formato YOLO) | ❌ No | `{input}_yolo` |
| `--split` | Proporción train/val/test | ❌ No | `0.7 0.2 0.1` |
| `--validate-only` | Solo validar sin reorganizar | ❌ No | `False` |
| `--stats-only` | Solo mostrar estadísticas | ❌ No | `False` |

##### **Proceso Completo:**

```bash
# Paso 1: Descargar dataset de Open Images
oi_download_dataset \
    --base_dir ./dataset_sombreros \
    --labels "Cowboy hat" Fedora Hat Helmet Sombrero "Sun hat" \
    --format darknet \
    --limit 500

# Paso 2: Reorganizar a formato YOLO
python preparar_dataset_sombreros.py \
    --input ./dataset_sombreros \
    --output ./dataset_sombreros_yolo \
    --split 0.7 0.2 0.1

# Paso 3: Validar resultado (automático en paso 2)
# O manualmente:
python preparar_dataset_sombreros.py \
    --input ./dataset_sombreros_yolo \
    --validate-only

# Paso 4: Ver estadísticas por clase
python preparar_dataset_sombreros.py \
    --input ./dataset_sombreros_yolo \
    --stats-only
```

```powershell
# PowerShell
# Paso 1: Descargar dataset
oi_download_dataset --base_dir ./dataset_sombreros --labels "Cowboy hat" Fedora Hat Helmet Sombrero "Sun hat" --format darknet --limit 500

# Paso 2: Reorganizar
python preparar_dataset_sombreros.py --input ./dataset_sombreros --output ./dataset_sombreros_yolo --split 0.7 0.2 0.1

# Paso 3: Validar manualmente
python preparar_dataset_sombreros.py --input ./dataset_sombreros_yolo --validate-only

# Paso 4: Ver estadísticas
python preparar_dataset_sombreros.py --input ./dataset_sombreros_yolo --stats-only
```




```powershell
# PowerShell
Get-ChildItem dataset_sombreros
Get-ChildItem "dataset_sombreros/cowboy hat/"
```

**Error: "Split debe sumar 1.0"**
```bash
# Incorrecto:
--split 0.7 0.2 0.2  # Suma 1.1

# Correcto:
--split 0.7 0.2 0.1  # Suma 1.0
--split 0.8 0.15 0.05  # Suma 1.0
```

**Advertencia: "Imagen sin label"**
```powershell
# Algunas imágenes de Open Images pueden no tener anotaciones
# Esto es normal, el script las copia de todas formas
# Si quieres eliminarlas:
python preparar_dataset_sombreros.py --input ./dataset_sombreros_yolo --validate-only
# Revisa las advertencias y elimina manualmente si es necesario
```

##### **Estadísticas Detalladas:**

```bash
# Ver distribución completa por clase
python preparar_dataset_sombreros.py \
    --input ./dataset_sombreros_yolo \
    --stats-only
```

```powershell
# PowerShell
python preparar_dataset_sombreros.py --input ./dataset_sombreros_yolo --stats-only

# Salida:
# - Número de imágenes por clase y split
# - Número de instancias (objetos) por clase
# - Balanceo del dataset
# - Totales generales
```

##### **Workflow Completo:**

```bash
# 1. Descargar dataset
oi_download_dataset --base_dir ./dataset_sombreros --labels Hat "Cowboy hat" Fedora --format darknet --limit 500

# 2. Reorganizar automáticamente
python preparar_dataset_sombreros.py --input ./dataset_sombreros --output ./dataset_yolo

# 3. Verificar estructura
ls dataset_yolo/
# Debe mostrar: images/, labels/, data.yaml

# 4. Entrenar modelo
python entrenador_yolo_sombreros.py --mode train --dataset dataset_yolo/data.yaml --epochs 100
```

```powershell
# PowerShell
# 1. Descargar dataset
oi_download_dataset --base_dir ./dataset_sombreros --labels Hat "Cowboy hat" Fedora --format darknet --limit 500

# 2. Reorganizar
python preparar_dataset_sombreros.py --input ./dataset_sombreros --output ./dataset_yolo

# 3. Verificar estructura
Get-ChildItem dataset_yolo

# 4. Entrenar modelo
python entrenador_yolo_sombreros.py --mode train --dataset dataset_yolo/data.yaml --epochs 100
```

##### **Ventajas del Script:**

✅ **Automatización completa** - Un solo comando reorganiza todo  
✅ **Validación integrada** - Detecta errores antes de entrenar  
✅ **Split aleatorio** - Evita sesgos por orden de descarga  
✅ **Estadísticas claras** - Sabes exactamente qué tienes  
✅ **Manejo de múltiples clases** - Soporta cualquier número de clases  
✅ **Compatible con Open Images** - Funciona directamente con `oi_download_dataset`  

---

---

## 🏋️ Proceso de Entrenamiento

### Paso 1: Verificar Dataset

```python
# Validar que el dataset está correcto
python entrenador_yolo_sombreros.py --mode validate \
                                    --dataset dataset_sombreros/data.yaml
                        
    EN PowerShell:
    
python entrenador_yolo_sombreros.py --mode validate --dataset dataset_sombreros_yolo/data.yaml
```

**Salida esperada:**
```
✓ Dataset válido
  Train: 700 imágenes
  Val: 200 imágenes
  Test: 100 imágenes
  Clases: 3 (cowboy_hat, sombrero, baseball_cap)
  Anotaciones: OK
```

### Paso 2: Configurar Entrenamiento

```python
# Editar configuración (opcional)
nano entrenador_yolo_sombreros.py

# Parámetros importantes:
EPOCHS = 100          # Número de épocas (50-200)
BATCH_SIZE = 16       # Tamaño de batch (8-32)
IMG_SIZE = 640        # Tamaño de imagen (416, 640, 1280)
LEARNING_RATE = 0.01  # Tasa de aprendizaje
```

### Paso 3: Iniciar Entrenamiento

```bash
# Entrenamiento básico
python entrenador_yolo_sombreros.py \
    --mode train \
    --dataset dataset_sombreros_yolo/data.yaml \
    --epochs 100 \
    --batch-size 16

# Entrenamiento con GPU (recomendado)
python entrenador_yolo_sombreros.py \
    --mode train \
    --dataset dataset_sombreros_yolo/data.yaml \
    --epochs 100 \
    --batch-size 16 \
    --device 0  # GPU 0

# Entrenamiento desde checkpoint (continuar)
python entrenador_yolo_sombreros.py \
    --mode train \
    --dataset dataset_sombreros_yolo/data.yaml \
    --resume runs/detect/train/weights/last.pt
```

```powershell
# PowerShell (una sola línea cada comando)
python entrenador_yolo_sombreros.py --mode train --dataset dataset_sombreros_yolo/data.yaml --epochs 100 --batch-size 16

# Con GPU
python entrenador_yolo_sombreros.py --mode train --dataset dataset_sombreros_yolo/data.yaml --epochs 100 --batch-size 16 --device 0

# Desde checkpoint
python entrenador_yolo_sombreros.py --mode train --dataset dataset_sombreros_yolo/data.yaml --resume runs/detect/train/weights/last.pt
```

### Paso 4: Monitorear Entrenamiento

Durante el entrenamiento verás:

```
Epoch 1/100: 100%|████████| 44/44 [00:15<00:00,  2.85it/s]
      Class     Images  Instances      P      R  mAP50  mAP50-95
        all        200        450  0.823  0.756  0.801     0.612
  cowboy_hat        200        150  0.850  0.780  0.820     0.630
   sombrero         200        180  0.810  0.750  0.795     0.605
baseball_cap        200        120  0.810  0.740  0.788     0.601

Epoch 2/100: ...
```

**Métricas importantes:**
- **P (Precision)**: Precisión (cuántos detectados son correctos)
- **R (Recall)**: Recall (cuántos objetos detecta del total)
- **mAP50**: Mean Average Precision al 50% IoU
- **mAP50-95**: mAP promedio de IoU 50% a 95%

**Valores objetivo:**
- mAP50 > 0.70 = Bueno
- mAP50 > 0.80 = Muy bueno
- mAP50 > 0.90 = Excelente

### Paso 5: Evaluar Resultados

```bash
# Evaluar en conjunto de test
python entrenador_yolo_sombreros.py \
    --mode test \
    --weights runs/detect/train/weights/best.pt \
    --dataset dataset_sombreros_yolo/data.yaml
```

```powershell
# PowerShell
python entrenador_yolo_sombreros.py --mode test --weights runs/detect/train/weights/best.pt --dataset dataset_sombreros_yolo/data.yaml
```

**Archivos generados:**
```
runs/detect/train/
├── weights/
│   ├── best.pt          # Mejor modelo (usa este)
│   └── last.pt          # Último checkpoint
├── confusion_matrix.png # Matriz de confusión
├── F1_curve.png        # Curva F1
├── P_curve.png         # Curva de precisión
├── R_curve.png         # Curva de recall
├── PR_curve.png        # Curva precision-recall
├── results.csv         # Métricas por época
└── results.png         # Gráficos de entrenamiento
```

---

## 🚀 Uso del Modelo Entrenado

### Integración con el Sistema Actual

El modelo entrenado se integra automáticamente:

```python
# En sistema_deteccion_mejorado.py
sistema = SistemaDeteccionSombrerosMejorado()
sistema.seleccionar_modelo()
# Seleccionar: "YOLO Custom (Sombreros)"
sistema.detectar_video_tiempo_real_mejorado()
```

### Detección en Imagen Individual

```python
from detectores.deteccion_video_modelos import DetectorVideoModelos

detector = DetectorVideoModelos()

# Cargar modelo custom
detector.cargar_modelo_yolo_custom('runs/detect/train/weights/best.pt')

# Detectar en imagen
import cv2
imagen = cv2.imread('test_sombrero.jpg')
detecciones = detector.detectar_en_frame(imagen, 'yolo_custom')

# Dibujar resultados
resultado = detector.dibujar_detecciones(imagen, detecciones)
cv2.imshow('Detección', resultado)
cv2.waitKey(0)
```

### Detección en Video

```python
# Desde línea de comandos
python main_deteccion_vehicular.py

# Menú:
# 1. Sistema de Detección con IA
# 2. Detección en Video
# 3. Seleccionar modelo: YOLO Custom
# 4. Fuente: Archivo de video o cámara
```

### Detección en Tiempo Real (Cámara Web)

```python
python scripts/detectar_tiempo_real_yolo.py \
    --weights runs/detect/train/weights/best.pt \
    --source 0  # Cámara web
```

---

## 🎨 Augmentación de Datos

YOLO aplica augmentación automáticamente durante el entrenamiento:

### Augmentaciones Incluidas:

1. **Geométricas:**
   - Rotación (±10°)
   - Escalado (0.5x - 1.5x)
   - Traslación (±10% de imagen)
   - Flip horizontal

2. **Fotométricas:**
   - Cambio de brillo (±30%)
   - Cambio de contraste (±30%)
   - Cambio de saturación (±30%)
   - Cambio de matiz (±5%)

3. **Específicas YOLO:**
   - Mosaic (combina 4 imágenes)
   - MixUp (mezcla 2 imágenes)
   - Copy-Paste (copia objetos entre imágenes)

### Configuración Personalizada:

```yaml
# En data.yaml agregar:
augmentation:
  hsv_h: 0.015  # Matiz
  hsv_s: 0.7    # Saturación
  hsv_v: 0.4    # Valor
  degrees: 10.0  # Rotación
  translate: 0.1 # Traslación
  scale: 0.5     # Escalado
  flipud: 0.5    # Flip vertical
  fliplr: 0.5    # Flip horizontal
  mosaic: 1.0    # Mosaic
  mixup: 0.1     # MixUp
```

---

## 📊 Interpretación de Resultados

### Matriz de Confusión

```
                Predicted
              CH    S    BC
         CH [ 85    3    2  ]
Actual    S [  4   88    3  ]
         BC [  2    5   88  ]

CH = cowboy_hat
S = sombrero
BC = baseball_cap
```

**Interpretación:**
- Diagonal principal = Predicciones correctas
- Fuera de diagonal = Confusiones entre clases

### Curvas de Aprendizaje

#### Pérdida (Loss):
```
Train Loss: Debe bajar constantemente
Val Loss: Debe bajar y estabilizarse

Si Val Loss sube mientras Train Loss baja = Overfitting
```

#### Soluciones Overfitting:
1. Más datos de entrenamiento
2. Más augmentación
3. Regularización (dropout, weight decay)
4. Early stopping

### mAP (Mean Average Precision)

```
mAP@0.5 = 0.85    # Excelente
mAP@0.75 = 0.72   # Bueno
mAP@0.5:0.95 = 0.65  # Promedio de todos los IoU
```

**Objetivo:** mAP@0.5 > 0.70

---

## 🔧 Troubleshooting

### Problema 1: "CUDA out of memory"

**Causa:** GPU sin memoria suficiente

**Solución:**
```bash
# Reducir batch size
python entrenador_yolo_sombreros.py --batch-size 8

# O usar CPU (más lento)
python entrenador_yolo_sombreros.py --device cpu
```

```powershell
# PowerShell
python entrenador_yolo_sombreros.py --mode train --dataset dataset_sombreros_yolo/data.yaml --batch-size 8
python entrenador_yolo_sombreros.py --mode train --dataset dataset_sombreros_yolo/data.yaml --device cpu
```

### Problema 2: "No labels found"

**Causa:** Formato de anotaciones incorrecto

**Solución:**
```bash
# Verificar dataset
python entrenador_yolo_sombreros.py --mode validate --dataset data.yaml
```

```powershell
# PowerShell
python entrenador_yolo_sombreros.py --mode validate --dataset dataset_sombreros_yolo/data.yaml

# Revisar que:
# 1. Carpetas train/val existen
# 2. Archivos .txt tienen mismo nombre que .jpg
# 3. Formato YOLO correcto en .txt
```

### Problema 3: mAP muy bajo (<0.30)

**Causas posibles:**
1. Dataset muy pequeño
2. Imágenes de mala calidad
3. Anotaciones incorrectas
4. Clases muy similares

**Soluciones:**
```bash
# 1. Más datos
# 2. Limpiar dataset
python preparar_dataset_sombreros.py --clean --validate
```

```powershell
# PowerShell
python preparar_dataset_sombreros.py --input dataset_sombreros_yolo --validate-only

# 3. Revisar anotaciones manualmente
# 4. Combinar clases similares
```

### Problema 4: Entrenamiento muy lento (CPU)

**Solución:**
```bash
# Reducir imagen y batch
python entrenador_yolo_sombreros.py \
    --img-size 416 \
    --batch-size 4 \
    --workers 4
```

```powershell
# PowerShell
python entrenador_yolo_sombreros.py --mode train --dataset dataset_sombreros_yolo/data.yaml --img-size 416 --batch-size 4
```

### Problema 5: PyTorch no encuentra CUDA

**Solución:**
```bash
# Verificar instalación CUDA
nvcc --version

# Reinstalar PyTorch con CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 📈 Mejores Prácticas

### 1. Dataset Balanceado

```python
# Verificar distribución de clases
python scripts/analizar_dataset.py --dataset data.yaml

# Objetivo: Similar número de instancias por clase
# cowboy_hat: 500 imágenes ✓
# sombrero: 480 imágenes ✓
# baseball_cap: 450 imágenes ✓
```

### 2. Validación Cruzada

```bash
# Entrenar múltiples veces con diferentes splits
for fold in {0..4}; do
    python entrenador_yolo_sombreros.py \
        --dataset data.yaml \
        --fold $fold
done
```

```powershell
# PowerShell
for ($fold = 0; $fold -lt 5; $fold++) {
    python entrenador_yolo_sombreros.py --mode train --dataset dataset_sombreros_yolo/data.yaml --epochs 100
}
```

### 3. Early Stopping

```bash
# Parar si no mejora en 50 épocas
python entrenador_yolo_sombreros.py \
    --patience 50
```

```powershell
# PowerShell (incluido por defecto en el script con patience=50)
python entrenador_yolo_sombreros.py --mode train --dataset dataset_sombreros_yolo/data.yaml --epochs 100
```

### 4. Transfer Learning

```bash
# Usar modelo preentrenado como base
python entrenador_yolo_sombreros.py \
    --weights yolov8n.pt  # Nano (más rápido)
    
python entrenador_yolo_sombreros.py \
    --weights yolov8s.pt  # Small
    
python entrenador_yolo_sombreros.py \
    --weights yolov8m.pt  # Medium
    
python entrenador_yolo_sombreros.py \
    --weights yolov8l.pt  # Large (más preciso)
```

```powershell
# PowerShell
python entrenador_yolo_sombreros.py --mode train --dataset dataset_sombreros_yolo/data.yaml --model-size n  # Nano
python entrenador_yolo_sombreros.py --mode train --dataset dataset_sombreros_yolo/data.yaml --model-size s  # Small
python entrenador_yolo_sombreros.py --mode train --dataset dataset_sombreros_yolo/data.yaml --model-size m  # Medium
python entrenador_yolo_sombreros.py --mode train --dataset dataset_sombreros_yolo/data.yaml --model-size l  # Large
```

---

## 🎯 Checklist de Preparación

Antes de entrenar, verifica:

- [ ] Python 3.8+ instalado
- [ ] PyTorch instalado (con CUDA si tienes GPU)
- [ ] Ultralytics instalado (`pip install ultralytics`)
- [ ] Dataset organizado (train/val/test)
- [ ] Imágenes en formato JPG/PNG
- [ ] Anotaciones en formato YOLO (.txt)
- [ ] Archivo `data.yaml` configurado
- [ ] Al menos 100 imágenes por clase
- [ ] Imágenes variadas (ángulos, iluminación, fondos)
- [ ] Anotaciones revisadas manualmente
- [ ] Dataset validado sin errores
- [ ] Espacio en disco suficiente (10+ GB)
- [ ] GPU configurada (opcional pero recomendado)

---

## 📚 Referencias y Recursos

### Documentación Oficial:
- **YOLOv8**: https://docs.ultralytics.com
- **PyTorch**: https://pytorch.org/docs
- **COCO Dataset**: https://cocodataset.org

### Tutoriales:
- **YOLOv8 Custom Training**: https://docs.ultralytics.com/modes/train
- **Dataset Preparation**: https://roboflow.com/formats/yolo-darknet-txt

### Herramientas:
- **LabelImg**: https://github.com/tzutalin/labelImg
- **Roboflow**: https://roboflow.com
- **CVAT**: https://cvat.org

---

## ✅ Siguiente Paso

Una vez completado el entrenamiento:

```bash
# 1. Copiar modelo entrenado
cp runs/detect/train/weights/best.pt modelos/yolo_sombreros_custom.pt

# 2. Probar en el sistema
python sistema_deteccion_mejorado.py

# 3. Seleccionar "YOLO Custom"

# 4. Detectar en video o tiempo real
```

```powershell
# PowerShell
Copy-Item runs/detect/train/weights/best.pt modelos/yolo_sombreros_custom.pt
python sistema_deteccion_mejorado.py
```

**¡Listo para detectar sombreros en tiempo real! 🎩🚀**
