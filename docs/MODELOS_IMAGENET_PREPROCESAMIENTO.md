# Modelos ImageNet: Preprocesamiento y Diferencias

## Universidad del Quindío - Visión Artificial
**Fecha:** Noviembre 2025

---

## 📋 Tabla de Contenidos

1. [Introducción a ImageNet](#introducción-a-imagenet)
2. [Modelos Implementados](#modelos-implementados)
3. [Diferencias Críticas de Preprocesamiento](#diferencias-críticas-de-preprocesamiento)
4. [Comparación Detallada](#comparación-detallada)
5. [Código de Implementación](#código-de-implementación)
6. [Errores Comunes](#errores-comunes)

---

## 🎯 Introducción a ImageNet

**ImageNet** es un proyecto de base de datos de imágenes a gran escala diseñado para investigación en reconocimiento visual de objetos. Contiene:

- **1000 categorías/clases** de objetos
- Más de **14 millones de imágenes** etiquetadas
- Base de datos estándar para entrenar y evaluar modelos de visión por computadora

### ¿Por qué ImageNet es importante?

Los modelos preentrenados en ImageNet han aprendido características visuales generales que pueden transferirse a otras tareas mediante **Transfer Learning**. Estos modelos pueden:

1. **Clasificar** imágenes en 1000 categorías
2. **Extraer características** útiles para otras tareas
3. **Servir como base** para fine-tuning en dominios específicos

---

## 🧠 Modelos Implementados

### 1. VGG16 (Visual Geometry Group - 16 capas)

**Características:**
- **Año:** 2014
- **Profundidad:** 16 capas con pesos entrenables
- **Parámetros:** ~138 millones
- **Arquitectura:** Bloques repetitivos de convoluciones 3x3

**Ventajas:**
- ✅ Arquitectura simple y uniforme
- ✅ Excelente para extracción de características
- ✅ Fácil de entender e implementar

**Desventajas:**
- ❌ Muy pesado (muchos parámetros)
- ❌ Lento en inferencia
- ❌ Consume mucha memoria

**Funcionamiento:**
```
Entrada (224x224x3)
    ↓
[Conv 3x3] × 2 → MaxPool → [Conv 3x3] × 2 → MaxPool
    ↓
[Conv 3x3] × 3 → MaxPool → [Conv 3x3] × 3 → MaxPool
    ↓
[Conv 3x3] × 3 → MaxPool → FC Layers → Softmax (1000 clases)
```

---

### 2. ResNet50 (Residual Network - 50 capas)

**Características:**
- **Año:** 2015
- **Profundidad:** 50 capas
- **Parámetros:** ~25 millones
- **Innovación:** Conexiones residuales (skip connections)

**Ventajas:**
- ✅ Resuelve el problema del desvanecimiento del gradiente
- ✅ Más profundo pero más eficiente que VGG
- ✅ Excelente balance precisión/velocidad
- ✅ Menos parámetros que VGG16

**Desventajas:**
- ❌ Más complejo de entender
- ❌ Arquitectura más sofisticada

**Funcionamiento con Bloques Residuales:**
```
Bloque Residual:
    x → [Conv] → [BN] → [ReLU] → [Conv] → [BN] → (+) → [ReLU]
    └────────────────────────────────────────────┘
             (Conexión de atajo/skip)
```

**¿Por qué funciona?**
Las conexiones residuales permiten que el gradiente fluya directamente hacia atrás, facilitando el entrenamiento de redes muy profundas.

---

### 3. ResNet101 V2 (Residual Network - 101 capas, Versión 2)

**Características:**
- **Año:** 2016 (mejora de ResNet original)
- **Profundidad:** 101 capas
- **Parámetros:** ~44 millones
- **Innovación:** Preactivación (BatchNorm → ReLU → Conv)

**Ventajas:**
- ✅ Mejor rendimiento que ResNet V1
- ✅ Entrenamiento más estable
- ✅ Mayor capacidad de representación
- ✅ Mejor propagación del gradiente

**Desventajas:**
- ❌ Más lento que ResNet50
- ❌ Más parámetros
- ❌ Requiere más memoria

**Diferencia clave con ResNet V1:**
```
ResNet V1:           x → [Conv] → [BN] → [ReLU] → ...
ResNet V2 (mejor):   x → [BN] → [ReLU] → [Conv] → ...
```

La preactivación (BN y ReLU antes de Conv) mejora el flujo de gradientes.

---

## ⚙️ Diferencias Críticas de Preprocesamiento

### 🔴 **ESTO ES CRÍTICO:** Orden de Canales de Color

Los modelos de ImageNet se entrenaron con diferentes bibliotecas que usan órdenes de canales distintos:

| Modelo | Framework Original | Orden de Canales | Modo |
|--------|-------------------|------------------|------|
| **VGG16** | Caffe | **BGR** | Caffe |
| **ResNet50** | Caffe | **BGR** | Caffe |
| **ResNet101 V2** | TensorFlow | **RGB** | Torch |

### 📊 Modos de Preprocesamiento

#### Modo Caffe (VGG16, ResNet50)

**Entrada esperada:** BGR [0, 255]

**Transformación:**
```python
# Substracción de medias de ImageNet (en BGR)
mean = [103.939, 116.779, 123.68]  # BGR
preprocessed = image - mean

# Resultado: valores aproximados en [-128, 128]
```

**Valores de media (calculados del dataset ImageNet):**
- Canal B (Blue): 103.939
- Canal G (Green): 116.779  
- Canal R (Red): 123.68

#### Modo Torch (ResNet V2)

**Entrada esperada:** RGB [0, 255]

**Transformación:**
```python
# Normalización a [-1, 1]
preprocessed = (image / 127.5) - 1

# Equivalente a:
# preprocessed = (image / 255.0) * 2 - 1

# Resultado: valores en [-1, 1]
```

---

## 📊 Comparación Detallada

### Tabla Comparativa Completa

| Característica | VGG16 | ResNet50 | ResNet101 V2 |
|---------------|-------|----------|--------------|
| **Año** | 2014 | 2015 | 2016 |
| **Capas** | 16 | 50 | 101 |
| **Parámetros** | ~138M | ~25M | ~44M |
| **Top-1 Accuracy** | 71.3% | 76.0% | 77.8% |
| **Top-5 Accuracy** | 90.1% | 93.0% | 93.8% |
| **Tamaño Modelo** | 528 MB | 98 MB | 171 MB |
| **Velocidad Inferencia** | Lenta | Media | Media-Lenta |
| **Preprocesamiento** | Caffe (BGR) | Caffe (BGR) | Torch (RGB) |
| **Rango Valores** | ~[-128, 128] | ~[-128, 128] | [-1, 1] |

### Rendimiento por Categorías

**VGG16:**
- 🎯 Bueno en: Objetos grandes, escenas simples
- ⚠️ Débil en: Objetos pequeños, escenas complejas

**ResNet50:**
- 🎯 Bueno en: Balance general, objetos variados
- ⚠️ Débil en: Detalles muy finos

**ResNet101 V2:**
- 🎯 Bueno en: Objetos complejos, detalles finos
- ⚠️ Débil en: Velocidad de inferencia

---

## 💻 Código de Implementación

### Preprocesamiento Correcto

```python
import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnetv2_preprocess

def preprocesar_imagen_modelo(imagen_rgb, modelo_nombre):
    """
    Preprocesa una imagen según el modelo específico.
    
    Args:
        imagen_rgb: Imagen en formato RGB [0, 255]
        modelo_nombre: 'vgg16', 'resnet50', o 'resnet101'
    
    Returns:
        Imagen preprocesada lista para el modelo
    """
    # Redimensionar a 224x224
    imagen_resized = cv2.resize(imagen_rgb, (224, 224))
    
    if modelo_nombre in ['vgg16', 'resnet50']:
        # Modo Caffe: Necesita BGR
        imagen_bgr = cv2.cvtColor(imagen_resized, cv2.COLOR_RGB2BGR)
        imagen_batch = np.expand_dims(imagen_bgr, axis=0)
        
        if modelo_nombre == 'vgg16':
            return vgg_preprocess(imagen_batch.copy())
        else:  # resnet50
            return resnet50_preprocess(imagen_batch.copy())
    
    else:  # resnet101 (V2)
        # Modo Torch: Usa RGB directamente
        imagen_batch = np.expand_dims(imagen_resized, axis=0)
        return resnetv2_preprocess(imagen_batch.copy())
```

### Carga de Modelos

```python
from tensorflow.keras.applications import VGG16, ResNet50, ResNet101V2
from tensorflow.keras.applications.imagenet_utils import decode_predictions

# Cargar modelos con pesos de ImageNet
vgg16 = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
resnet50 = ResNet50(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
resnet101 = ResNet101V2(weights='imagenet', include_top=True, input_shape=(224, 224, 3))

print("✓ Modelos cargados con 1000 clases de ImageNet")
```

### Predicción y Decodificación

```python
def predecir_imagenet(imagen_rgb, modelo, modelo_nombre):
    """
    Realiza predicción con un modelo de ImageNet.
    
    Args:
        imagen_rgb: Imagen en RGB [0, 255]
        modelo: Modelo cargado
        modelo_nombre: Nombre del modelo
    
    Returns:
        Diccionario con predicciones decodificadas
    """
    # Preprocesar
    imagen_prep = preprocesar_imagen_modelo(imagen_rgb, modelo_nombre)
    
    # Predecir
    predicciones = modelo.predict(imagen_prep)
    
    # Decodificar (convierte índices a nombres de clases)
    decoded = decode_predictions(predicciones, top=5)[0]
    
    # Formatear resultados
    resultados = []
    for id_clase, nombre_clase, confianza in decoded:
        resultados.append({
            'id': id_clase,
            'clase': nombre_clase.replace('_', ' ').title(),
            'confianza': float(confianza)
        })
    
    return resultados
```

---

## ⚠️ Errores Comunes

### Error #1: Usar mismo preprocesamiento para todos los modelos

**❌ INCORRECTO:**
```python
# Esto causará resultados erróneos en ResNet V2
imagen_prep = preprocess_input(imagen)  # ¿Qué función es esta?
pred_vgg = vgg16.predict(imagen_prep)
pred_resnet101 = resnet101.predict(imagen_prep)  # ❌ INCORRECTO
```

**✅ CORRECTO:**
```python
# Preprocesamiento específico para cada modelo
imagen_prep_vgg = vgg_preprocess(imagen_bgr)
imagen_prep_resnet101 = resnetv2_preprocess(imagen_rgb)

pred_vgg = vgg16.predict(imagen_prep_vgg)
pred_resnet101 = resnet101.predict(imagen_prep_resnet101)
```

### Error #2: Usar RGB para VGG16/ResNet50

**❌ INCORRECTO:**
```python
imagen = cv2.imread('foto.jpg')
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
# Usar imagen_rgb directamente para VGG16 ❌
pred = vgg16.predict(vgg_preprocess(np.expand_dims(imagen_rgb, 0)))
```

**✅ CORRECTO:**
```python
imagen = cv2.imread('foto.jpg')
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
# Convertir de vuelta a BGR para VGG16
imagen_bgr = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2BGR)
pred = vgg16.predict(vgg_preprocess(np.expand_dims(imagen_bgr, 0)))
```

### Error #3: Usar BGR para ResNet V2

**❌ INCORRECTO:**
```python
imagen = cv2.imread('foto.jpg')  # BGR por defecto
# Usar BGR directamente para ResNet V2 ❌
pred = resnet101.predict(resnetv2_preprocess(np.expand_dims(imagen, 0)))
```

**✅ CORRECTO:**
```python
imagen = cv2.imread('foto.jpg')
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
pred = resnet101.predict(resnetv2_preprocess(np.expand_dims(imagen_rgb, 0)))
```

### Error #4: No usar decode_predictions

**❌ INCORRECTO:**
```python
predicciones = modelo.predict(imagen_prep)
print(predicciones)  # Imprime array de 1000 valores
# Salida: [0.001, 0.002, 0.997, ...] ❌ No interpretable
```

**✅ CORRECTO:**
```python
predicciones = modelo.predict(imagen_prep)
decoded = decode_predictions(predicciones, top=5)[0]
for id_clase, nombre, conf in decoded:
    print(f"{nombre}: {conf:.3f}")
# Salida: "Golden Retriever: 0.997" ✅ Interpretable
```

---

## 🔍 Verificación de Preprocesamiento

### Script de Prueba

```python
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_prep
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_prep
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnetv2_prep

# Imagen de prueba [0-255]
test_img = np.array([[[[100, 150, 200]]]], dtype=np.float32)

print("Imagen original (RGB):", test_img[0,0,0])

# VGG16
vgg_out = vgg_prep(test_img.copy())
print(f"VGG16: {vgg_out[0,0,0]}")
print(f"  Rango: [{vgg_out.min():.1f}, {vgg_out.max():.1f}]")

# ResNet50
resnet50_out = resnet50_prep(test_img.copy())
print(f"ResNet50: {resnet50_out[0,0,0]}")
print(f"  Rango: [{resnet50_out.min():.1f}, {resnet50_out.max():.1f}]")

# ResNet V2
resnetv2_out = resnetv2_prep(test_img.copy())
print(f"ResNet V2: {resnetv2_out[0,0,0]}")
print(f"  Rango: [{resnetv2_out.min():.1f}, {resnetv2_out.max():.1f}]")
```

**Salida esperada:**
```
Imagen original (RGB): [100. 150. 200.]
VGG16: [ 96.061  33.221 -23.68 ]
  Rango: [-23.7, 96.1]
ResNet50: [ 96.061  33.221 -23.68 ]
  Rango: [-23.7, 96.1]
ResNet V2: [-0.216  0.176  0.569]
  Rango: [-1.0, 1.0]
```

---

## 📚 Referencias

1. **VGG16:** Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition.
2. **ResNet:** He, K., et al. (2015). Deep residual learning for image recognition.
3. **ResNet V2:** He, K., et al. (2016). Identity mappings in deep residual networks.
4. **ImageNet:** Deng, J., et al. (2009). ImageNet: A large-scale hierarchical image database.

---

## 🎓 Resumen Ejecutivo

### Lo que DEBES recordar:

1. **VGG16 y ResNet50** → Modo Caffe → **BGR** → Substracción de medias
2. **ResNet101 V2** → Modo Torch → **RGB** → Normalización [-1, 1]
3. Siempre usar el **preprocesamiento correcto** para cada modelo
4. **decode_predictions** convierte índices de clase a nombres legibles
5. **include_top=True** para usar las 1000 clases completas de ImageNet

### Flujo correcto:

```
OpenCV imread → BGR
    ↓
cv2.cvtColor → RGB
    ↓
┌───────────────┬──────────────────┬─────────────────┐
│   VGG16       │   ResNet50       │   ResNet101 V2  │
│   RGB → BGR   │   RGB → BGR      │   RGB (directo) │
│   vgg_prep    │   resnet50_prep  │   resnetv2_prep │
└───────────────┴──────────────────┴─────────────────┘
    ↓               ↓                   ↓
  Predicción    Predicción          Predicción
    ↓               ↓                   ↓
      decode_predictions (Top 5 clases)
```

---

## 🎨 Modelos de Segmentación

Además de los modelos de clasificación de ImageNet, el sistema incluye modelos de segmentación semántica y de instancias para detectar y delimitar objetos pixel por pixel.

---

### U-Net con Encoder Preentrenado

**Tipo:** Segmentación Semántica  
**Año:** 2015 (arquitectura original), mejorado con encoders modernos  
**Framework:** TensorFlow/Keras

**Descripción:**
U-Net es una arquitectura de red neuronal convolucional diseñada específicamente para segmentación de imágenes. Su nombre proviene de su forma de "U" cuando se visualiza la arquitectura.

**Implementación en este proyecto:**

1. **Opción Principal:** DeepLabV3 desde TensorFlow Hub
   - Modelo completo preentrenado en PASCAL VOC
   - Arquitectura de última generación con ASPP (Atrous Spatial Pyramid Pooling)
   - Segmentación semántica de alta calidad

2. **Opción Alternativa:** U-Net con ResNet50 preentrenado
   - Encoder: ResNet50 con pesos de ImageNet (congelado)
   - Decoder: Capas de upsampling con skip connections
   - Combina características de bajo y alto nivel

**Arquitectura U-Net:**
```
Entrada (224x224x3)
    ↓
┌─────────────────────────────────────┐
│ ENCODER (ResNet50 preentrenado)    │
│                                     │
│ Conv2D + BN + ReLU → 112x112 ──┐   │
│         ↓                       │   │
│ MaxPool → 56x56 ──┐            │   │
│         ↓         │            │   │
│ Conv Blocks → 28x28 ─┐         │   │
│         ↓           │         │   │
│ Conv Blocks → 14x14 ─┐         │   │
│         ↓           │         │   │
│ Bottleneck → 7x7    │         │   │
└──────────┬──────────┘         │   │
           ↓                    │   │
┌──────────────────────────────┼───┼┐
│ DECODER                      │   ││
│                              │   ││
│ UpSample + Concat ← ─────────┘   ││
│         ↓                        ││
│ Conv2D × 2 (14x14)               ││
│         ↓                        ││
│ UpSample + Concat ← ──────────── ┘│
│         ↓                          │
│ Conv2D × 2 (28x28)                 │
│         ↓                          │
│ UpSample + Concat ← ───────────────┘
│         ↓
│ Conv2D × 2 (56x56)
│         ↓
│ UpSample (112x112)
│         ↓
│ Conv2D × 2 (224x224)
└──────────┬─────────┘
           ↓
    Máscara (224x224x1)
```

**Características Clave:**

1. **Skip Connections:** Conecta capas del encoder con el decoder
   - Preserva detalles espaciales finos
   - Combina características de diferentes niveles
   - Mejora la precisión de los bordes

2. **Encoder Preentrenado:** Usa ResNet50 con pesos de ImageNet
   - Extrae características visuales robustas
   - Reduce el tiempo de entrenamiento
   - Mejora el rendimiento con pocos datos

3. **Decoder Simétrico:** Reconstruye la resolución espacial
   - Upsampling bilineal para suavidad
   - Convoluciones para refinar detalles
   - Salida con resolución completa

**Ventajas:**
- ✅ Excelente para segmentación de objetos
- ✅ Encoder preentrenado (no aleatorio)
- ✅ Funciona bien con pocos datos de entrenamiento
- ✅ Skip connections preservan detalles
- ✅ Salida de alta resolución

**Desventajas:**
- ❌ Más lento que modelos de clasificación
- ❌ Requiere más memoria
- ❌ Segmentación binaria (objeto vs. fondo)

**Métricas de Evaluación:**

```python
# Métricas proporcionadas por U-Net
{
    'tipo': 'segmentacion_unet',
    'clase': 'objeto_grande',  # grande/mediano/pequeño/sin_objeto
    'confianza': 0.89,
    'bbox': [x, y, width, height],
    'segmentacion': mascara_binaria,  # Array numpy
    'area_segmentada': 15420,  # píxeles
    'porcentaje': 30.1,  # % de la imagen
    'num_objetos': 1,
    'metricas': {
        'area_contorno_principal': 15300,
        'area_bbox': 18000,
        'densidad': 85.0,  # % de relleno del bbox
        'pixeles_totales': 50176
    }
}
```

**Preprocesamiento:**
```python
# U-Net usa el preprocesamiento de ResNet50 (modo Caffe)
imagen_bgr = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2BGR)
imagen_batch = np.expand_dims(cv2.resize(imagen_bgr, (224, 224)), axis=0)
imagen_prep = preprocess_input(imagen_batch)  # Substrae medias ImageNet
```

---

### Mask R-CNN / DeepLabV3+

**Tipo:** Segmentación de Instancias / Segmentación Semántica  
**Año:** 2017 (Mask R-CNN), 2018 (DeepLabV3+)  
**Framework:** Detectron2 / TensorFlow

**Descripción:**
El sistema intenta usar Mask R-CNN real (detectron2) para segmentación de instancias, pero si no está disponible, usa DeepLabV3+ como alternativa con resultados comparables.

**Implementación:**

#### Opción 1: Mask R-CNN (Detectron2)

Si detectron2 está instalado:
- Modelo completo preentrenado en COCO (80 clases)
- Detecta y segmenta múltiples instancias simultáneamente
- Proporciona bbox, clase y máscara para cada instancia
- Basado en Faster R-CNN + branch de segmentación

**Arquitectura Mask R-CNN:**
```
Entrada
    ↓
┌─────────────────────────┐
│ Backbone (ResNet50-FPN) │
│ Extracción de features  │
└──────────┬──────────────┘
           ↓
┌──────────────────────────┐
│ RPN (Region Proposal Net)│
│ Genera propuestas de bbox│
└──────────┬───────────────┘
           ↓
    ┌──────┴──────┐
    │             │
    ↓             ↓
┌─────────┐  ┌─────────┐
│ Box Head│  │Mask Head│
│ Clasif. │  │Segment. │
│ + Bbox  │  │  Máscara│
└────┬────┘  └────┬────┘
     │            │
     └─────┬──────┘
           ↓
   Detecciones con Máscaras
```

**Salida Mask R-CNN:**
```python
{
    'tipo': 'segmentacion_instancias',
    'clase': 'person',  # De 80 clases COCO
    'confianza': 0.95,
    'bbox': [x, y, w, h],
    'segmentacion': mascara_binaria,
    'num_instancias': 3,
    'instancias': [
        {
            'bbox': [100, 50, 80, 150],
            'clase_id': 0,
            'confianza': 0.95,
            'mascara': array(...)
        },
        # ... más instancias
    ]
}
```

#### Opción 2: DeepLabV3+ (TensorFlow)

Si detectron2 no está disponible (Windows):
- Segmentación semántica con arquitectura ASPP
- Preentrenado con ResNet50 en ImageNet
- 21 clases de PASCAL VOC incluyendo personas
- Alta calidad de bordes

**Arquitectura DeepLabV3+:**
```
Entrada (224x224x3)
    ↓
┌───────────────────────────┐
│ Encoder (ResNet50)        │
│ Extracción de features    │
└──────────┬────────────────┘
           ↓ (7x7)
┌──────────────────────────────────┐
│ ASPP (Atrous Spatial Pyramid)    │
│                                  │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐   │
│  │1x1 │ │3x3 │ │3x3 │ │3x3 │   │
│  │conv│ │r=6 │ │r=12│ │r=18│   │
│  └─┬──┘ └─┬──┘ └─┬──┘ └─┬──┘   │
│    │      │      │      │       │
│    └──────┴──────┴──────┘       │
│            ↓                     │
│     Concatenate + Conv           │
└──────────┬───────────────────────┘
           ↓
┌──────────────────────┐
│ Decoder              │
│ UpSample 4x → 28x28  │
│ Skip connection      │
│ UpSample 8x → 224x224│
└──────────┬───────────┘
           ↓
    Máscara de clases
    (21 canales)
```

**ASPP (Atrous Spatial Pyramid Pooling):**
- Múltiples convoluciones con diferentes tasas de dilatación
- Captura contexto en múltiples escalas
- Mejora la precisión sin perder resolución

**Clases PASCAL VOC (DeepLabV3+):**
```
 0: background       7: cat          14: motorbike
 1: aeroplane        8: chair        15: person ⭐
 2: bicycle          9: cow          16: pottedplant
 3: bird            10: diningtable  17: sheep
 4: boat            11: dog          18: sofa
 5: bottle          12: horse        19: train
 6: bus             13: motorbike    20: tvmonitor
```

**Salida DeepLabV3+:**
```python
{
    'tipo': 'segmentacion_semantica',
    'clase': 'person',
    'confianza': 0.87,
    'bbox': [x, y, w, h],
    'segmentacion': mascara_clases,  # Array con IDs de clase
    'area_segmentada': 24680,
    'porcentaje': 49.2,
    'clases_detectadas': ['person', 'chair', 'bottle']
}
```

**Ventajas de cada enfoque:**

| Característica | Mask R-CNN | DeepLabV3+ |
|---------------|------------|------------|
| Instancias separadas | ✅ Sí | ❌ No |
| Múltiples objetos | ✅ Excelente | ⚠️ Fusionados |
| Calidad de bordes | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Velocidad | Media | Rápida |
| Clases disponibles | 80 (COCO) | 21 (VOC) |
| Instalación | Difícil (Windows) | Fácil |
| Bbox automático | ✅ Sí | ⚠️ Calculado |

**Cuándo usar cada uno:**

**Mask R-CNN:**
- Necesitas distinguir instancias individuales (ej: 3 personas separadas)
- Trabajas en Linux/macOS (fácil instalación)
- Necesitas las 80 clases de COCO
- Precisión es más importante que velocidad

**DeepLabV3+:**
- Solo necesitas saber QUÉ objetos hay (no cuántos)
- Trabajas en Windows
- Necesitas alta calidad de bordes
- Velocidad es importante
- Las 21 clases de VOC son suficientes

**Instalación de Mask R-CNN (opcional):**

```bash
# En Linux/macOS (recomendado):
pip install torch torchvision
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# En Windows (complejo):
# Requiere compilación con Visual Studio
# Se recomienda usar WSL o Docker
# Documentación: https://detectron2.readthedocs.io/
```

**Preprocesamiento:**

Ambos modelos manejan su propio preprocesamiento internamente:

```python
# Mask R-CNN (detectron2)
# Espera BGR, lo convierte internamente
imagen_bgr = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2BGR)
outputs = predictor(imagen_bgr)

# DeepLabV3+ (TensorFlow)
# Usa preprocesamiento de ResNet50
imagen_bgr = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2BGR)
imagen_batch = np.expand_dims(cv2.resize(imagen_bgr, (224, 224)), axis=0)
prediccion = model.predict(imagen_batch)
```

---

## 🆚 Comparación: Clasificación vs. Segmentación

| Aspecto | Clasificación (VGG/ResNet) | Segmentación (U-Net/Mask R-CNN) |
|---------|---------------------------|----------------------------------|
| **Salida** | Etiqueta de clase | Máscara pixel por pixel |
| **Información** | "Qué hay" | "Qué hay y DÓNDE está" |
| **Precisión espacial** | Baja (solo imagen completa) | Alta (nivel de píxel) |
| **Velocidad** | Rápida | Media/Lenta |
| **Uso de memoria** | Bajo | Alto |
| **Clases** | 1000 (ImageNet) | Variable (21-80+) |
| **Aplicaciones** | Reconocimiento general | Edición, conteo, medición |

**Ejemplo práctico:**

Imagen: Persona con sombrero

**Clasificación (VGG16):**
```
Salida: "Cowboy Hat" (79.2% confianza)
Información: Hay un sombrero vaquero en la imagen
```

**Segmentación (U-Net):**
```
Salida: Máscara binaria mostrando píxeles del objeto
Información: 
  - Objeto está en coordenadas (120, 80)
  - Ocupa 15,420 píxeles (30% de la imagen)
  - Bounding box: 80x150 píxeles
  - Densidad: 85% (forma compacta)
```

---

## 📊 Resumen de Modelos Implementados

### Clasificación (ImageNet - 1000 clases)

1. **LeNet** - Arquitectura básica (no preentrenada)
2. **AlexNet** → Reemplazado por VGG16 preentrenado
3. **VGG16** - Modo Caffe, BGR, 138M parámetros
4. **ResNet50** - Modo Caffe, BGR, 25M parámetros
5. **ResNet101 V2** - Modo Torch, RGB, 44M parámetros

### Segmentación

6. **U-Net** - ResNet50 encoder + decoder personalizado
   - Alternativa: DeepLabV3 desde TF Hub
7. **Mask R-CNN** - Detectron2 (80 clases COCO)
   - Alternativa: DeepLabV3+ (21 clases VOC)

### Detección de Objetos

8. **YOLO** - YOLOv8 nano (ultralytics)

---

## 🎓 Conceptos Clave

### Transfer Learning
Usar modelos preentrenados en ImageNet para otras tareas:
- Encoder congelado preserva características aprendidas
- Solo se entrena el decoder/clasificador final
- Reduce drásticamente el tiempo y datos necesarios

### Skip Connections
Conexiones que saltan capas en redes profundas:
- Resuelven el problema del gradiente que desaparece
- Preservan información espacial en segmentación
- Combinan características de diferentes niveles

### Atrous Convolutions (Dilated Convolutions)
Convoluciones con "agujeros" entre píxeles:
- Aumentan el campo receptivo sin aumentar parámetros
- Capturan contexto a múltiples escalas
- Esenciales en DeepLabV3+

---

**Documento actualizado:** 17 de Noviembre de 2025  
**Proyecto:** Sistema de Detección y Segmentación  
**Universidad del Quindío** - Visión Artificial

