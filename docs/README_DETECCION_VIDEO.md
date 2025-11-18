# Sistema de Detección de Video con Modelos Preentrenados

## Descripción

Sistema completo de detección de sombreros en video tiempo real utilizando múltiples modelos de deep learning, incluyendo clasificación, detección y segmentación.

## Modelos Soportados

### Clasificación
- **LeNet**: Red neuronal clásica para clasificación básica
- **AlexNet**: Arquitectura profunda con capas convolucionales
- **VGG16**: Red con filtros pequeños y arquitectura profunda
- **ResNet50**: Red residual de 50 capas
- **ResNet101**: Red residual de 101 capas

### Detección de Objetos
- **YOLO (You Only Look Once)**: Detección en tiempo real
- **SSD (Single Shot MultiBox Detector)**: Detección eficiente
- **R-CNN (Regions with CNN features)**: Detección basada en regiones

### Segmentación
- **U-Net**: Segmentación semántica
- **Mask R-CNN**: Segmentación de instancias

## Instalación

### Dependencias Básicas
```bash
pip install opencv-python tensorflow numpy matplotlib
```

### Dependencias Opcionales (Para YOLO)
```bash
pip install torch torchvision ultralytics
```

### Archivo de Requerimientos
```bash
pip install -r requirements-full.txt
```

## Uso del Sistema

### 1. Inicialización
```bash
python sistema_deteccion_mejorado.py
```

### 2. Flujo Básico de Uso

1. **Seleccionar Modelo** (Opción 3):
   - Elija entre modelos de clasificación, detección o segmentación
   - El modelo se carga automáticamente

2. **Detección en Video** (Opción 2):
   - Seleccione fuente: cámara web o archivo de video
   - Configure parámetros si es necesario
   - Inicie la detección

3. **Controles Durante la Detección**:
   - `q`: Salir
   - `c`: Cambiar configuración en tiempo real

## Estructura de Archivos

```
proyecto-vision_artificial/
├── detectores/
│   └── deteccion_video_modelos.py    # Módulo principal de detección
├── sistema_deteccion_mejorado.py     # Sistema integrado
├── test_sistema_video.py             # Script de pruebas
└── requirements-full.txt             # Dependencias
```

## Configuración

### Parámetros Principales
- **Umbral de confianza**: 0.1 - 0.9 (default: 0.5)
- **FPS objetivo**: 10 - 60 (default: 30)
- **Tamaño de entrada**: Automático según modelo

### Configuración de Video
```python
configuracion = {
    'umbral_confianza': 0.5,
    'fps_objetivo': 30,
    'mostrar_confianza': True,
    'guardar_detecciones': False
}
```

## Ejemplos de Uso

### Detección Básica con VGG16
1. Ejecutar el sistema
2. Seleccionar modelo: `vgg16`
3. Elegir detección en video: cámara web
4. El sistema mostrará clasificación en tiempo real

### Detección de Objetos con YOLO
1. Asegurar que PyTorch está instalado
2. Seleccionar modelo: `yolo`
3. Elegir fuente de video
4. Ver detecciones con bounding boxes

### Segmentación con U-Net
1. Seleccionar modelo: `unet`
2. Usar video o cámara
3. Visualizar máscara de segmentación superpuesta

## API del Detector de Video

### Clase Principal: `DetectorVideoModelos`

```python
from detectores.deteccion_video_modelos import DetectorVideoModelos

# Inicializar
detector = DetectorVideoModelos()

# Cargar modelo
detector.cargar_modelo('vgg16')

# Procesar video
detector.procesar_video_tiempo_real(0)  # Cámara web
detector.procesar_video_tiempo_real('video.mp4')  # Archivo
```

### Métodos Principales

```python
# Cargar modelo específico
detector.cargar_modelo(nombre_modelo)

# Detectar en frame individual
resultado = detector.detectar_en_frame(frame, nombre_modelo)

# Procesar video completo
detector.procesar_video_tiempo_real(fuente)

# Configurar parámetros
detector.configuracion['umbral_confianza'] = 0.7
```

## Tipos de Resultado

### Clasificación
```python
{
    'tipo': 'clasificacion',
    'clase': 'con_sombrero' | 'sin_sombrero',
    'confianza': float,
    'bbox': None,
    'segmentacion': None
}
```

### Detección
```python
{
    'tipo': 'deteccion',
    'clase': 'con_sombrero' | 'sin_sombrero',
    'confianza': float,
    'bbox': [x1, y1, x2, y2],
    'segmentacion': None
}
```

### Segmentación
```python
{
    'tipo': 'segmentacion',
    'clase': 'con_sombrero' | 'sin_sombrero',
    'confianza': float,
    'bbox': None,
    'segmentacion': numpy_array,
    'area_segmentada': int
}
```

## Solución de Problemas

### Error de Cámara
- Verificar que la cámara no esté en uso por otra aplicación
- Probar con diferentes índices de cámara (0, 1, 2...)

### Error de Modelo
- Verificar que las dependencias estén instaladas
- Para YOLO: instalar PyTorch y ultralytics
- Verificar memoria disponible

### Rendimiento Lento
- Reducir FPS objetivo
- Usar modelos más ligeros (LeNet, MobileNet)
- Verificar disponibilidad de GPU

## Pruebas

Ejecutar script de pruebas:
```bash
python test_sistema_video.py
```

Este script verifica:
- Importación de módulos
- Inicialización del sistema
- Carga de modelos
- Disponibilidad de cámara
- Integración completa

## Características Técnicas

### Preprocesamiento Automático
- Redimensionamiento según modelo
- Normalización [0,1]
- Conversión de formato (BGR/RGB)
- Gestión de dimensiones de batch

### Optimizaciones
- Detección eficiente por frame
- Control de FPS adapatativo
- Gestión de memoria optimizada
- Procesamiento en tiempo real

### Visualización
- Bounding boxes para detección
- Máscaras de segmentación
- Información de confianza
- Estadísticas de rendimiento

## Extensiones Futuras

- Soporte para más arquitecturas
- Entrenamiento personalizado
- Detección multi-clase
- Análisis de video batch
- Exportación de resultados

## Compatibilidad

- **Python**: 3.7+
- **OpenCV**: 4.0+
- **TensorFlow**: 2.x
- **PyTorch**: 1.x (opcional)
- **Sistema**: Windows, Linux, macOS

## Licencia

Universidad del Quindío - Visión Artificial
Proyecto Académico - Noviembre 2025