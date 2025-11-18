# 🎩 Guía de Uso del Modelo YOLO Custom (Sombreros)

## 📋 Descripción

Esta guía explica cómo usar el modelo YOLO Custom entrenado para detección de sombreros en el sistema de detección de video.

---

## ✅ Verificación del Entrenamiento

### 1. Confirmar que el entrenamiento terminó

El entrenamiento está completo cuando veas este mensaje en la consola:

```
100 epochs completed in X.XXX hours.
Optimizer stripped from runs/detect/train/weights/last.pt, XX.XMB
Optimizer stripped from runs/detect/train/weights/best.pt, XX.XMB

Validating runs/detect/train/weights/best.pt...
Ultralytics YOLOv8.X.X 🚀 Python-3.XX.X torch-X.X.X+cpu CPU (...)
Model summary (fused): XXX layers, X parameters, X gradients, XX.X GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):
                   all        XXX        XXX      X.XXX      X.XXX      X.XXX     X.XXX
            cowboy_hat        XXX        XXX      X.XXX      X.XXX      X.XXX     X.XXX
                fedora        XXX        XXX      X.XXX      X.XXX      X.XXX     X.XXX
                   hat        XXX        XXX      X.XXX      X.XXX      X.XXX     X.XXX
                helmet        XXX        XXX      X.XXX      X.XXX      X.XXX     X.XXX
              sombrero        XXX        XXX      X.XXX      X.XXX      X.XXX     X.XXX
               sun_hat        XXX        XXX      X.XXX      X.XXX      X.XXX     X.XXX
Speed: X.Xms preprocess, XX.Xms inference, X.Xms postprocess per image
Results saved to runs/detect/train
```

### 2. Verificar archivos generados

Revisa que existan estos archivos:

**Bash:**
```bash
ls -la runs/detect/train/weights/
# Deberías ver: best.pt, last.pt
```

**PowerShell:**
```powershell
Get-ChildItem runs\detect\train\weights\
# Deberías ver: best.pt, last.pt
```

---

## 🚀 Copiar Modelo a Ubicación Permanente

### Opción 1: Copiar a carpeta de modelos (Recomendado)

**Bash:**
```bash
# Crear carpeta de modelos si no existe
mkdir -p modelos

# Copiar modelo entrenado
cp runs/detect/train/weights/best.pt modelos/yolo_sombreros_custom.pt

# Verificar
ls -l modelos/yolo_sombreros_custom.pt
```

**PowerShell:**
```powershell
# Crear carpeta de modelos si no existe
New-Item -ItemType Directory -Force -Path modelos

# Copiar modelo entrenado
Copy-Item runs\detect\train\weights\best.pt modelos\yolo_sombreros_custom.pt

# Verificar
Get-Item modelos\yolo_sombreros_custom.pt
```

### Opción 2: Dejar en carpeta de entrenamiento

Si prefieres usar el modelo directamente desde la carpeta de entrenamiento, puedes omitir el paso de copia. El sistema buscará automáticamente en:

1. `runs/detect/train/weights/best.pt` (ubicación predeterminada)
2. `runs/detect/train2/weights/best.pt` (si existe segunda ejecución)
3. `modelos/yolo_sombreros_custom.pt` (ubicación recomendada)

---

## 🎯 Usar el Modelo en Detección de Video

### 1. Iniciar el sistema principal

**Bash:**
```bash
python sistema_deteccion_mejorado.py
```

**PowerShell:**
```powershell
python sistema_deteccion_mejorado.py
```

### 2. Seleccionar detección en video

En el menú principal, selecciona:

```
====================================
SISTEMA DE DETECCIÓN Y CLASIFICACIÓN
====================================

1. Entrenar/Cargar Modelo de Redes Neuronales 🧠
2. Detección en Video/Tiempo Real 📹
3. Detección Individual de Objetos 🔍
4. Procesamiento de Imágenes 🖼️
5. Evaluación y Comparación de Algoritmos 📊
6. Salir 🚪

Seleccione una opción: 2
```

### 3. Seleccionar YOLO Custom

El sistema te mostrará los modelos disponibles:

```
DETECCIÓN EN VIDEO/TIEMPO REAL
========================================

Modelo activo: ninguno

¿Desea seleccionar otro modelo? (s/n): s

MODELOS DISPONIBLES PARA DETECCIÓN DE VIDEO:
---------------------------------------------
1. LENET - Modelo de clasificación LeNet
2. ALEXNET - Modelo de clasificación AlexNet
3. VGG16 - Modelo de clasificación VGG16
4. RESNET50 - Modelo de clasificación ResNet50
5. RESNET101 - Modelo de clasificación ResNet101
6. YOLO - Modelo de detección YOLO (COCO)
7. SSD - Modelo de detección SSD
8. RCNN - Modelo de detección RCNN
9. UNET - Modelo de segmentación U-Net
10. MASK_RCNN - Modelo de segmentación Mask R-CNN
11. YOLO_CUSTOM - YOLO Custom - Modelo entrenado para sombreros 🎩

Seleccione modelo (1-11): 11
```

### 4. El sistema carga el modelo automáticamente

Verás este mensaje:

```
Cargando modelo: yolo_custom
🔍 Buscando modelo YOLO Custom entrenado...

✅ Modelo encontrado: runs/detect/train/weights/best.pt

📋 Clases del modelo:
   0: cowboy_hat
   1: fedora
   2: hat
   3: helmet
   4: sombrero
   5: sun_hat

Modelo yolo_custom cargado exitosamente
```

### 5. Configurar parámetros (Opcional)

El sistema te preguntará si deseas configurar parámetros:

```
¿Desea configurar parámetros de detección? (s/n): s

CONFIGURACIÓN DE PARÁMETROS DE VIDEO:
----------------------------------------
Umbral de confianza actual: 0.5
Nuevo umbral (0.1-0.9) [Enter para mantener]: 0.6

Umbral actualizado a 0.6
FPS objetivo actual: 30
Nuevo FPS (10-60) [Enter para mantener]: 

Configuración actualizada
```

**Recomendaciones:**
- **Umbral de confianza**: `0.6` (detecta sombreros con >60% confianza)
- **FPS objetivo**: `30` (para tiempo real fluido)

### 6. Seleccionar fuente de video

```
Seleccione fuente de video:
1. Cámara web
2. Archivo de video
0. Volver

Seleccione opción: 1
```

### 7. Ver resultados en tiempo real

El sistema mostrará:
- Cuadros delimitadores (bounding boxes) en diferentes colores
- Etiquetas con la clase detectada y confianza
- Información en consola sobre detecciones

**Atajos de teclado:**
- `q`: Salir
- `p`: Pausar/Reanudar
- `s`: Guardar frame actual
- `r`: Resetear estadísticas

---

## 📊 Interpretación de Resultados

### Clases detectadas

El modelo puede detectar 6 tipos de sombreros:

| Clase | Descripción |
|-------|-------------|
| `cowboy_hat` | Sombrero de vaquero/cowboy |
| `fedora` | Sombrero fedora clásico |
| `hat` | Sombrero genérico |
| `helmet` | Casco de seguridad/deportivo |
| `sombrero` | Sombrero mexicano tradicional |
| `sun_hat` | Sombrero de sol/playa |

### Umbral de confianza

- **0.5-0.6**: Detección balanceada (recomendado)
- **0.7-0.8**: Detección conservadora (menos falsos positivos)
- **0.3-0.4**: Detección agresiva (más detecciones pero más falsos positivos)

### Ejemplo de detección

```
🔍 Detectando con modelo: yolo_custom
✅ Frame procesado - Detecciones: 2
   - cowboy_hat (conf: 0.87) en bbox [120, 45, 280, 195]
   - sun_hat (conf: 0.72) en bbox [450, 60, 590, 210]
```

---

## 🔧 Solución de Problemas

### Problema: "Modelo yolo_custom no está cargado"

**Causa**: El modelo no se encuentra en ninguna de las rutas esperadas.

**Solución 1** - Copiar modelo:
```powershell
Copy-Item runs\detect\train\weights\best.pt modelos\yolo_sombreros_custom.pt
```

**Solución 2** - Verificar ubicación:
```powershell
Get-ChildItem runs\detect\train\weights\best.pt
```

### Problema: "Error loading yolo_custom: No such file or directory"

**Causa**: El entrenamiento no ha completado o los archivos se movieron.

**Solución**: Verifica que el entrenamiento haya terminado:
```powershell
Get-ChildItem runs\detect\train -Recurse -Filter *.pt
```

### Problema: "No se detectan sombreros en el video"

**Posibles causas:**

1. **Umbral de confianza muy alto**
   - Reducir a 0.5 o menos
   
2. **Iluminación inadecuada**
   - Mejorar condiciones de luz
   - Ajustar brillo/contraste de cámara
   
3. **Sombreros muy pequeños en el frame**
   - Acercarse más a la cámara
   - Usar resolución mayor
   
4. **Clases no representadas en entrenamiento**
   - Verificar que el tipo de sombrero esté en el dataset

### Problema: "CUDA out of memory"

**Causa**: Intentando usar GPU sin suficiente memoria.

**Solución**: El modelo ya fue entrenado en CPU. Para predicción:
```python
# En deteccion_video_modelos.py, método cargar_modelo_yolo_custom()
# Ya está configurado para CPU automáticamente
```

### Problema: "RuntimeError: CUDA error: device-side assert triggered"

**Causa**: Clase ID fuera de rango en el modelo.

**Solución**: Verificar que el modelo se entrenó correctamente:
```powershell
python entrenador_yolo_sombreros.py --mode test --model runs\detect\train\weights\best.pt
```

---

## 📈 Optimización de Rendimiento

### Para mejorar FPS en detección

1. **Reducir resolución de entrada**:
   ```python
   # En configuración del detector
   self.configuracion['max_resolucion'] = (640, 480)  # En vez de (1280, 720)
   ```

2. **Procesar cada N frames**:
   ```python
   # En configuración
   self.configuracion['skip_frames'] = 2  # Procesar 1 de cada 2 frames
   ```

3. **Usar modelo más pequeño**:
   - YOLOv8n (actual): Más rápido, menos preciso
   - YOLOv8s: Balanceado
   - YOLOv8m: Más lento, más preciso

### Para mejorar precisión

1. **Aumentar umbral de confianza**: `0.7` o `0.8`
2. **Aplicar supresión de no-máximos (NMS)**: Ya incluido en YOLO
3. **Post-procesamiento temporal**: Filtrar detecciones inestables

---

## 🎥 Ejemplos de Uso

### Caso 1: Detección en cámara web

```
1. Ejecutar: python sistema_deteccion_mejorado.py
2. Seleccionar: 2 (Detección en Video/Tiempo Real)
3. Seleccionar modelo: 11 (YOLO_CUSTOM)
4. Configurar umbral: 0.6
5. Seleccionar fuente: 1 (Cámara web)
6. ¡Usar sombreros frente a la cámara!
```

### Caso 2: Detección en archivo de video

```
1. Ejecutar: python sistema_deteccion_mejorado.py
2. Seleccionar: 2 (Detección en Video/Tiempo Real)
3. Seleccionar modelo: 11 (YOLO_CUSTOM)
4. Seleccionar fuente: 2 (Archivo de video)
5. Ingresar ruta: videos/personas_con_sombreros.mp4
```

### Caso 3: Evaluación en dataset de prueba

```powershell
# Ejecutar evaluación automática
python entrenador_yolo_sombreros.py --mode test --model modelos/yolo_sombreros_custom.pt

# Ver resultados
Get-Content runs\detect\test\results.txt
```

---

## 📝 Notas Importantes

### Rendimiento esperado

Basado en entrenamiento de 100 épocas con 2,670 imágenes:

- **mAP@0.5**: ~0.50-0.70 (depende de la clase)
- **Velocidad**: ~20-30 FPS en CPU (Intel i5/i7 moderno)
- **Precisión**: Variable según clase (helmet > hat > sombrero)

### Limitaciones conocidas

1. **Sombreros muy pequeños**: Difíciles de detectar (<32x32 píxeles)
2. **Oclusión parcial**: Puede confundir clases similares
3. **Ángulos extremos**: Mejor detección frontal/lateral
4. **Iluminación extrema**: Contraluces/sombras fuertes afectan

### Mejoras futuras

- [ ] Aumentar dataset con más variaciones
- [ ] Entrenar con imágenes de mayor resolución
- [ ] Aplicar data augmentation más agresivo
- [ ] Usar YOLOv8m o YOLOv8l para mayor precisión
- [ ] Implementar tracking entre frames
- [ ] Agregar clasificación de colores/materiales

---

## 📚 Referencias

- [Documentación YOLOv8](https://docs.ultralytics.com/)
- [Open Images V7 Dataset](https://storage.googleapis.com/openimages/web/index.html)
- [ENTRENAMIENTO_YOLO_SOMBREROS.md](ENTRENAMIENTO_YOLO_SOMBREROS.md) - Guía de entrenamiento

---

## 🆘 Soporte

Si tienes problemas:

1. Revisar logs en `runs/detect/train/` y `resultados_deteccion/logs/`
2. Verificar versiones de dependencias: `pip list | grep -E "torch|ultralytics|opencv"`
3. Consultar sección de **Solución de Problemas** arriba
4. Revisar issues en el repositorio del proyecto

---

**Última actualización**: 2024-01-XX  
**Versión del sistema**: 2.0  
**Modelo YOLO**: YOLOv8n Custom (6 clases de sombreros)
