# ⚡ Guía Rápida: Post-Entrenamiento YOLO Custom

## ✅ Cuando el entrenamiento termine (100/100 épocas):

### 1️⃣ Verificar que el modelo se entrenó correctamente

**PowerShell:**
```powershell
# Ver estadísticas finales
Get-Content runs\detect\train\results.csv | Select-Object -Last 5

# Verificar que existe el modelo
Get-Item runs\detect\train\weights\best.pt
```

Deberías ver métricas como:
- `mAP@0.5`: ~0.50-0.70
- `precision`: ~0.60-0.80
- `recall`: ~0.50-0.70

---

### 2️⃣ OPCIÓN A: Copiar modelo a ubicación permanente (Recomendado)

**PowerShell:**
```powershell
# Crear carpeta si no existe
New-Item -ItemType Directory -Force -Path modelos

# Copiar modelo
Copy-Item runs\detect\train\weights\best.pt modelos\yolo_sombreros_custom.pt

# Verificar
Get-Item modelos\yolo_sombreros_custom.pt
```

---

### 2️⃣ OPCIÓN B: Dejar el modelo en carpeta de entrenamiento

No necesitas hacer nada. El sistema buscará automáticamente en:
1. `runs/detect/train/weights/best.pt`
2. `modelos/yolo_sombreros_custom.pt`

---

### 3️⃣ Verificar que el modelo funciona

**PowerShell:**
```powershell
# Verificación rápida
python verificar_yolo_custom.py

# Verificación con webcam
python verificar_yolo_custom.py --webcam

# Verificación con imagen
python verificar_yolo_custom.py --image images\senal6.avif
```

Deberías ver:
```
✅ Modelo encontrado: runs/detect/train/weights/best.pt
✅ Modelo cargado exitosamente
📋 Clases del modelo:
   0: cowboy_hat
   1: fedora
   2: hat
   3: helmet
   4: sombrero
   5: sun_hat
🧪 Probando inferencia con imagen sintética...
✅ Inferencia exitosa
```

---

### 4️⃣ Usar el modelo en detección de video

**PowerShell:**
```powershell
python sistema_deteccion_mejorado.py
```

**Pasos en el menú:**
```
1. Seleccionar opción: 2
   (Detección en Video/Tiempo Real 📹)

2. ¿Desea seleccionar otro modelo? s

3. Seleccione modelo (1-11): 11
   (YOLO_CUSTOM - YOLO Custom - Modelo entrenado para sombreros 🎩)

4. ¿Desea configurar parámetros de detección? s
   - Umbral de confianza: 0.6
   - FPS objetivo: 30

5. Seleccione fuente de video: 1
   (Cámara web)

6. ¡Usar sombreros frente a la cámara! 🎩

   Atajos:
   - 'q': Salir
   - 'p': Pausar
   - 's': Guardar frame
```

---

### 5️⃣ Evaluar rendimiento en dataset de prueba

**PowerShell:**
```powershell
# Evaluar en test set
python entrenador_yolo_sombreros.py --mode test --model runs\detect\train\weights\best.pt

# Ver resultados detallados
Get-Content runs\detect\test\results.txt
```

Métricas importantes:
- **Precision**: Qué tan exactas son las detecciones
- **Recall**: Cuántos sombreros detecta de los que hay
- **mAP@0.5**: Precisión promedio (más alto = mejor)

---

## 🎯 Qué esperar del modelo entrenado

### Clases detectables:
| Clase | Descripción | Rendimiento esperado |
|-------|-------------|---------------------|
| `cowboy_hat` | Sombrero de vaquero | ⭐⭐⭐⭐ Bueno |
| `fedora` | Sombrero fedora | ⭐⭐⭐ Medio |
| `hat` | Sombrero genérico | ⭐⭐⭐⭐ Bueno |
| `helmet` | Casco/helmet | ⭐⭐⭐⭐⭐ Excelente |
| `sombrero` | Sombrero mexicano | ⭐⭐⭐ Medio |
| `sun_hat` | Sombrero de sol | ⭐⭐⭐⭐ Bueno |

### Rendimiento:
- **Velocidad**: ~20-30 FPS en CPU (Intel i5/i7)
- **Precisión**: Variable según clase (helmet > hat > sombrero)
- **Confianza recomendada**: 0.5-0.7

---

## 🔧 Solución de problemas comunes

### Problema: "No se detectan sombreros"

**Soluciones:**
1. **Reducir umbral de confianza**:
   - Cambiar de 0.6 → 0.4 o 0.3
   
2. **Verificar iluminación**:
   - Mejorar luz en la escena
   - Evitar contraluces
   
3. **Acercarse más**:
   - Sombreros muy pequeños son difíciles de detectar

### Problema: "Demasiados falsos positivos"

**Soluciones:**
1. **Aumentar umbral de confianza**:
   - Cambiar de 0.6 → 0.7 o 0.8
   
2. **Aplicar filtrado temporal**:
   - Solo mostrar detecciones que persistan por varios frames

### Problema: "FPS muy bajos"

**Soluciones:**
1. **Procesar cada N frames**:
   ```python
   if frame_count % 2 == 0:  # Procesar 1 de cada 2
       detecciones = modelo(frame)
   ```
   
2. **Reducir resolución**:
   - Cambiar de 1280x720 → 640x480

---

## 📊 Visualizar métricas de entrenamiento

### Ver gráficas de entrenamiento

**PowerShell:**
```powershell
# Las gráficas se guardaron automáticamente
explorer runs\detect\train\
```

Archivos importantes:
- `results.png`: Gráficas de todas las métricas
- `confusion_matrix.png`: Matriz de confusión
- `F1_curve.png`: Curva F1 por confianza
- `PR_curve.png`: Curva Precision-Recall
- `results.csv`: Datos numéricos de cada época

### Interpretar resultados

**results.png** muestra 10 gráficas:
1. `train/box_loss` ⬇️ = Mejora en localización de cajas
2. `train/cls_loss` ⬇️ = Mejora en clasificación
3. `val/box_loss` ⬇️ = Validación de localización
4. `val/cls_loss` ⬇️ = Validación de clasificación
5. `metrics/precision` ⬆️ = Menos falsos positivos
6. `metrics/recall` ⬆️ = Detecta más sombreros
7. `metrics/mAP50` ⬆️ = Precisión general
8. `metrics/mAP50-95` ⬆️ = Precisión estricta

**¿Qué valores son buenos?**
- `mAP@0.5` > 0.50 = Aceptable
- `mAP@0.5` > 0.60 = Bueno
- `mAP@0.5` > 0.70 = Excelente

---

## 🎓 Próximos pasos

### Mejorar el modelo

1. **Aumentar dataset**:
   - Descargar más imágenes de Open Images
   - Agregar imágenes propias con sombreros
   
2. **Entrenar más épocas**:
   ```powershell
   python entrenador_yolo_sombreros.py --mode train --epochs 200
   ```
   
3. **Usar modelo más grande**:
   ```powershell
   python entrenador_yolo_sombreros.py --mode train --model yolov8s.pt
   ```

### Exportar para producción

**PowerShell:**
```powershell
# Exportar a ONNX (compatible con muchas plataformas)
python entrenador_yolo_sombreros.py --mode export --model runs\detect\train\weights\best.pt
```

---

## 📚 Documentación completa

Para más detalles, consultar:
- 📖 [ENTRENAMIENTO_YOLO_SOMBREROS.md](ENTRENAMIENTO_YOLO_SOMBREROS.md) - Guía completa de entrenamiento
- 🚀 [USO_MODELO_YOLO_CUSTOM.md](USO_MODELO_YOLO_CUSTOM.md) - Guía de uso detallada
- 📝 [README_SISTEMA_MEJORADO.md](README_SISTEMA_MEJORADO.md) - Información del sistema

---

## ✅ Checklist Final

Después del entrenamiento, verificar:

- [ ] Modelo existe en `runs/detect/train/weights/best.pt`
- [ ] Copiado a `modelos/yolo_sombreros_custom.pt` (opcional)
- [ ] Verificación con `verificar_yolo_custom.py` exitosa
- [ ] Modelo carga correctamente en sistema principal
- [ ] Detección funciona en webcam
- [ ] Métricas de evaluación satisfactorias
- [ ] Documentación revisada

---

**¡Listo! Tu modelo YOLO Custom está entrenado y funcionando! 🎩🚀**
