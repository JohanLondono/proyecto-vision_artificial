# Sistema de Detección de Sombreros con Redes Neuronales

## Descripción General

Este proyecto implementa un sistema completo de detección de sombreros usando múltiples arquitecturas de redes neuronales, tanto personalizadas como preentrenadas. El sistema está integrado al proyecto de detección vehicular existente y proporciona capacidades avanzadas de procesamiento de imágenes y videos en tiempo real.

## 🚀 Características Principales

### Modelos Implementados

1. **Redes Neuronales Personalizadas** (`modules/redes_neuronales_custom.py`)
   - AlexNet: Arquitectura clásica para clasificación
   - VGG16/19: Capas convolucionales profundas
   - ResNet50/101: Skip connections para mejor flujo de gradientes

2. **Modelos Preentrenados** (`modules/modelos_preentrenados.py`)
   - YOLO v8: Detección rápida y eficiente
   - Faster R-CNN: Alta precisión con ResNet50 + FPN
   - SSD MobileNet: Balance entre velocidad y precisión

3. **Segmentación Neuronal** (`modules/segmentacion_neuronal.py`)
   - U-Net: Segmentación semántica pixel a pixel
   - Mask R-CNN: Segmentación de instancias individuales
   - DeepLabV3: Segmentación semántica avanzada con ASPP
   - FCN: Fully Convolutional Networks

### Capacidades del Sistema

- ✅ Detección en imágenes individuales
- ✅ Procesamiento por lotes de múltiples imágenes
- ✅ Procesamiento de video en tiempo real
- ✅ Comparación automática entre modelos
- ✅ Métricas de rendimiento detalladas
- ✅ Visualización de resultados
- ✅ Exportación de reportes
- ✅ Consolidación de datos de descriptores

## 📁 Estructura del Proyecto

```
proyecto-vision_artificial/
├── sistema_deteccion_sombreros.py          # Sistema principal integrado
├── modules/
│   ├── redes_neuronales_custom.py          # Redes personalizadas
│   ├── modelos_preentrenados.py            # Modelos preentrenados
│   └── segmentacion_neuronal.py            # Redes de segmentación
├── consolidador_descriptores.py            # Consolidador avanzado
├── consolidador_rapido.py                  # Consolidador rápido
├── main_deteccion_vehicular.py             # Sistema principal (actualizado)
├── resultados_deteccion/
│   ├── hat_detection/                      # Resultados de detección IA
│   ├── custom_networks/                    # Resultados redes custom
│   ├── pretrained_models/                  # Resultados modelos preentrenados
│   └── neural_segmentation/               # Resultados segmentación
└── README_NEURAL_SYSTEM.md                # Este archivo
```

## 🛠️ Instalación y Configuración

### Dependencias Requeridas

```bash
# Dependencias básicas
pip install opencv-python numpy matplotlib pillow scikit-image

# Deep Learning frameworks
pip install torch torchvision tensorflow

# Modelos preentrenados
pip install ultralytics  # Para YOLO

# Análisis de datos
pip install pandas seaborn

# Opcional: GPU support
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

### Verificación de Instalación

```python
# Ejecutar para verificar dependencias
import torch
import torchvision
import tensorflow as tf
from ultralytics import YOLO

print(f"PyTorch: {torch.__version__}")
print(f"TorchVision: {torchvision.__version__}")
print(f"TensorFlow: {tf.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
```

## 🎯 Uso del Sistema

### 1. Acceso desde el Menú Principal

```
1. Ejecutar: python main_deteccion_vehicular.py
2. Seleccionar opción 9: "Sistema de Detección con IA (Sombreros)"
3. El sistema inicializará automáticamente todos los módulos
```

### 2. Detección en Imagen Individual

```python
# Desde el sistema principal
from sistema_deteccion_sombreros import SistemaDeteccionSombreros

# Inicializar
sistema = SistemaDeteccionSombreros()
sistema.inicializar_modulos()
sistema.preparar_modelos()

# Detectar sombreros
resultado = sistema.detectar_sombreros_imagen(
    imagen_path="path/to/image.jpg",
    metodos=['yolo', 'faster_rcnn'],
    mostrar_resultados=True,
    guardar_resultados=True
)
```

### 3. Procesamiento por Lotes

```python
# Procesar múltiples imágenes
resultado_lote = sistema.procesar_lote_imagenes(
    directorio_imagenes="./images/",
    metodos=['todos'],
    extensiones=['.jpg', '.png']
)

print(f"Procesadas: {resultado_lote['imagenes_procesadas']}")
print(f"Con detecciones: {resultado_lote['imagenes_con_sombreros']}")
```

### 4. Video en Tiempo Real

```python
# Procesar video o cámara en tiempo real
estadisticas = sistema.procesar_video_tiempo_real(
    usar_camara=True,  # o video_path="path/to/video.mp4"
    metodos=['yolo'],
    output_path="output_video.mp4"
)
```

## 📊 Interpretación de Resultados

### Estructura del Resultado de Detección

```python
resultado = {
    'imagen': 'nombre_imagen.jpg',
    'timestamp': '2025-11-XX...',
    'metodos_utilizados': ['YOLO', 'FasterRCNN'],
    'resultados_por_metodo': {
        'YOLO': {
            'encontrado': True,
            'detecciones_por_modelo': {
                'YOLO': {
                    'num_detecciones': 2,
                    'detecciones': [...],
                    'confianza_maxima': 0.85
                }
            }
        }
    },
    'resumen_detecciones': {
        'metodos_exitosos': ['YOLO'],
        'detecciones_totales': 2,
        'confianza_promedio': 0.75,
        'mejor_resultado': 'YOLO'
    },
    'estadisticas_rendimiento': {
        'tiempo_total': 1.25,
        'metodos_utilizados_count': 1
    }
}
```

### Métricas de Evaluación

- **Confianza**: 0.0 - 1.0 (mayor valor = mayor certeza)
- **IoU**: Intersection over Union para segmentación
- **Tiempo de inferencia**: Velocidad del modelo en segundos
- **Throughput**: Imágenes procesadas por segundo
- **F1-Score**: Balance entre precisión y recall

## 🎛️ Configuración Avanzada

### Ajuste de Umbrales

```python
# Modificar umbrales de confianza
sistema.configuracion['umbral_confianza_deteccion'] = 0.7  # Mayor precisión
sistema.configuracion['umbral_segmentacion'] = 0.5         # Segmentación
```

### Selección de Dispositivo

```python
# El sistema detecta automáticamente GPU/CPU
# Para forzar CPU:
import torch
torch.device('cpu')
```

### Configuración de Modelos

```python
# Cargar modelos específicos
sistema.modelos_preentrenados.cargar_yolo('yolov8s')  # Modelo más grande
sistema.segmentacion_neuronal.cargar_unet(num_clases=5)  # Más clases
```

## 📈 Rendimiento y Optimización

### Benchmarks Típicos (GPU Tesla T4)

| Modelo | Tiempo/Imagen | FPS Video | Precisión | Uso de Memoria |
|--------|---------------|-----------|-----------|----------------|
| YOLO v8n | 0.02s | 45 FPS | Alta | 2GB |
| Faster R-CNN | 0.15s | 6 FPS | Muy Alta | 4GB |
| U-Net | 0.08s | 12 FPS | Media | 3GB |
| Mask R-CNN | 0.25s | 4 FPS | Muy Alta | 6GB |

### Recomendaciones de Uso

**Para Tiempo Real (>20 FPS):**
- YOLO v8n o v8s
- Resolución máxima 640x640

**Para Máxima Precisión:**
- Faster R-CNN + Mask R-CNN
- Resolución completa

**Para Balance:**
- YOLO v8m + U-Net
- Resolución 1024x1024

## 🔧 Solución de Problemas

### Errores Comunes

1. **"CUDA out of memory"**
   ```python
   # Reducir batch size o usar CPU
   torch.device('cpu')
   ```

2. **"ModuleNotFoundError: No module named 'ultralytics'"**
   ```bash
   pip install ultralytics
   ```

3. **"Sistema de IA no disponible"**
   - Verificar instalación de PyTorch y TensorFlow
   - Revisar dependencias en requirements.txt

4. **Video no se reproduce en tiempo real**
   - Usar solo YOLO para video en tiempo real
   - Reducir resolución del video
   - Verificar capacidad de hardware

### Logs y Debugging

```python
# Activar logs detallados
import logging
logging.basicConfig(level=logging.DEBUG)

# Ver información del sistema
sistema.mostrar_informacion_sistema()
sistema.mostrar_estadisticas_globales()
```

## 📝 Extensiones y Personalización

### Agregar Nuevo Modelo

1. Implementar en módulo correspondiente
2. Agregar método de carga
3. Implementar método de detección
4. Integrar en sistema principal

### Entrenar Modelo Personalizado

```python
# Para entrenar U-Net personalizada
segmentador = SegmentacionNeuronal()
unet = segmentador.cargar_unet(num_clases=2)

# Entrenar (requiere dataset organizado)
historial = segmentador.entrenar_unet(
    dataset_path="./dataset/",
    num_epochs=50,
    learning_rate=0.001
)
```

### Nuevos Tipos de Objeto

Modificar las clases de detección en cada módulo para detectar otros objetos además de sombreros.

## 📄 Reportes y Exportación

### Formatos de Salida

- **JSON**: Datos estructurados completos
- **TXT**: Resúmenes legibles
- **CSV**: Para análisis estadístico
- **PNG**: Visualizaciones

### Consolidación de Datos

```python
# Usar consolidadores incluidos
from consolidador_descriptores import ConsolidadorDescriptores

consolidador = ConsolidadorDescriptores()
resultado = consolidador.consolidar_todo("./resultados_deteccion/")
```

## 🤝 Contribución

Para contribuir al proyecto:

1. Fork del repositorio
2. Crear rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## 📞 Soporte

Para preguntas o problemas:
- Revisar esta documentación
- Verificar logs de error
- Consultar código fuente con comentarios detallados
- Probar con imágenes de ejemplo incluidas

## 🔄 Actualizaciones Futuras

### Próximas Características

- [ ] Entrenamiento automático con datos propios
- [ ] Modelos de detección de múltiples objetos
- [ ] API REST para uso remoto
- [ ] Interfaz gráfica de usuario (GUI)
- [ ] Optimizaciones para edge computing
- [ ] Soporte para más formatos de video

### Roadmap de Modelos

- [ ] YOLO v9/v10 cuando estén disponibles
- [ ] Transformer-based models (DETR, ViT)
- [ ] Modelos específicos para sombreros entrenados desde cero
- [ ] Quantización para dispositivos móviles

---

**Versión**: 1.0.0  
**Fecha**: Noviembre 2025  
**Autor**: Sistema de Detección Vehicular  
**Licencia**: MIT