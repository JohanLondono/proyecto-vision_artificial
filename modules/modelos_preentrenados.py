#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Modelos Preentrenados para Detección de Objetos
========================================================

Implementación de modelos preentrenados (YOLO, SSD, R-CNN) 
para detección de sombreros en tiempo real.

Modelos implementados:
- YOLO (YOLOv5, YOLOv8)
- SSD (Single Shot MultiBox Detector)
- R-CNN (Faster R-CNN)

Autor: Sistema de Detección Vehicular
Fecha: Noviembre 2025
"""

import os
import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import pickle
import json
from datetime import datetime
import time

class ModelosPreentrenados:
    """
    Clase para manejar modelos preentrenados para detección de objetos.
    """
    
    def __init__(self, directorio_resultados="./resultados_deteccion/pretrained_models"):
        """
        Inicializa el módulo de modelos preentrenados.
        
        Args:
            directorio_resultados: Directorio para guardar resultados
        """
        self.directorio_resultados = directorio_resultados
        self.modelos_cargados = {}
        self.configuraciones = {}
        
        # Crear directorio si no existe
        os.makedirs(self.directorio_resultados, exist_ok=True)
        
        # Configurar device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Usando dispositivo: {self.device}")
        
        # Clases COCO (para modelos preentrenados)
        self.coco_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
            20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
            25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
            30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
            35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
            39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
            44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
            49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
            54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
            59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
            64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
            69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
            74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
            79: 'toothbrush'
        }
        
        print("🤖 Módulo de Modelos Preentrenados inicializado")
    
    def cargar_yolo(self, modelo='yolov8n'):
        """
        Carga un modelo YOLO preentrenado.
        
        Args:
            modelo: Versión del modelo YOLO (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            
        Returns:
            Modelo YOLO cargado
        """
        try:
            print(f"📥 Cargando modelo YOLO: {modelo}...")
            
            model = YOLO(modelo)  # Descarga automáticamente si no existe
            self.modelos_cargados['YOLO'] = model
            
            self.configuraciones['YOLO'] = {
                'modelo': modelo,
                'tipo': 'YOLO',
                'clases_detectables': list(self.coco_classes.values()),
                'fecha_carga': datetime.now().isoformat()
            }
            
            print(f"✅ YOLO {modelo} cargado exitosamente")
            return model
            
        except Exception as e:
            print(f"❌ Error cargando YOLO: {e}")
            return None
    
    def cargar_faster_rcnn(self, preentrenado=True):
        """
        Carga un modelo Faster R-CNN preentrenado.
        
        Args:
            preentrenado: Si usar pesos preentrenados
            
        Returns:
            Modelo Faster R-CNN cargado
        """
        try:
            print("📥 Cargando Faster R-CNN...")
            
            model = fasterrcnn_resnet50_fpn(pretrained=preentrenado)
            model.to(self.device)
            model.eval()
            
            self.modelos_cargados['FasterRCNN'] = model
            
            self.configuraciones['FasterRCNN'] = {
                'tipo': 'Faster R-CNN',
                'backbone': 'ResNet50 + FPN',
                'preentrenado': preentrenado,
                'clases_detectables': list(self.coco_classes.values()),
                'fecha_carga': datetime.now().isoformat()
            }
            
            print("✅ Faster R-CNN cargado exitosamente")
            return model
            
        except Exception as e:
            print(f"❌ Error cargando Faster R-CNN: {e}")
            return None
    
    def cargar_ssd_mobilenet(self):
        """
        Carga un modelo SSD MobileNet desde TensorFlow Hub.
        
        Returns:
            Modelo SSD cargado
        """
        try:
            print("📥 Cargando SSD MobileNet...")
            
            # Descargar modelo preentrenado
            model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
            model = tf.keras.utils.get_file(
                fname="ssd_mobilenet_v2",
                origin=model_url,
                extract=True
            )
            
            # Cargar modelo
            detector = tf.saved_model.load(model)
            
            self.modelos_cargados['SSD'] = detector
            
            self.configuraciones['SSD'] = {
                'tipo': 'SSD MobileNet',
                'backbone': 'MobileNetV2',
                'clases_detectables': list(self.coco_classes.values()),
                'fecha_carga': datetime.now().isoformat()
            }
            
            print("✅ SSD MobileNet cargado exitosamente")
            return detector
            
        except Exception as e:
            print(f"❌ Error cargando SSD: {e}")
            print("💡 Intentando método alternativo...")
            return self._cargar_ssd_alternativo()
    
    def _cargar_ssd_alternativo(self):
        """Método alternativo para cargar SSD."""
        try:
            # Implementación alternativa usando OpenCV DNN
            config_file = "ssd_mobilenet_v2_coco.pbtxt"
            frozen_model = "ssd_mobilenet_v2_coco.pb"
            
            # Nota: En un caso real, estos archivos deberían descargarse
            print("⚠️  Para usar SSD, descarga los archivos del modelo desde:")
            print("   Config: https://github.com/opencv/opencv_extra/tree/master/testdata/dnn")
            print("   Modelo: http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz")
            
            return None
            
        except Exception as e:
            print(f"❌ Error en método alternativo: {e}")
            return None
    
    def detectar_yolo(self, imagen_path, confianza=0.5, mostrar_resultado=True):
        """
        Realiza detección usando YOLO.
        
        Args:
            imagen_path: Ruta a la imagen
            confianza: Umbral de confianza mínimo
            mostrar_resultado: Si mostrar la imagen con detecciones
            
        Returns:
            Resultados de detección
        """
        if 'YOLO' not in self.modelos_cargados:
            print("❌ Modelo YOLO no cargado. Usa cargar_yolo() primero.")
            return None
        
        try:
            model = self.modelos_cargados['YOLO']
            
            # Realizar detección
            inicio = time.time()
            resultados = model(imagen_path, conf=confianza)
            tiempo_inferencia = time.time() - inicio
            
            # Procesar resultados
            imagen = cv2.imread(imagen_path)
            altura, ancho = imagen.shape[:2]
            
            detecciones = []
            
            for r in resultados:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # Coordenadas de la caja
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        
                        if conf >= confianza:
                            deteccion = {
                                'clase': self.coco_classes.get(cls, 'unknown'),
                                'confianza': conf,
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'centro': [int((x1+x2)/2), int((y1+y2)/2)],
                                'area': int((x2-x1) * (y2-y1))
                            }
                            detecciones.append(deteccion)
                            
                            # Dibujar en la imagen si se requiere
                            if mostrar_resultado:
                                cv2.rectangle(imagen, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                label = f"{deteccion['clase']}: {conf:.2f}"
                                cv2.putText(imagen, label, (int(x1), int(y1)-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            resultado_final = {
                'imagen': os.path.basename(imagen_path),
                'modelo': 'YOLO',
                'tiempo_inferencia': tiempo_inferencia,
                'num_detecciones': len(detecciones),
                'detecciones': detecciones,
                'imagen_con_detecciones': imagen if mostrar_resultado else None
            }
            
            if mostrar_resultado and len(detecciones) > 0:
                self._mostrar_imagen_con_detecciones(imagen, f"YOLO - {len(detecciones)} detecciones")
            
            return resultado_final
            
        except Exception as e:
            print(f"❌ Error en detección YOLO: {e}")
            return None
    
    def detectar_faster_rcnn(self, imagen_path, confianza=0.5, mostrar_resultado=True):
        """
        Realiza detección usando Faster R-CNN.
        
        Args:
            imagen_path: Ruta a la imagen
            confianza: Umbral de confianza mínimo
            mostrar_resultado: Si mostrar la imagen con detecciones
            
        Returns:
            Resultados de detección
        """
        if 'FasterRCNN' not in self.modelos_cargados:
            print("❌ Modelo Faster R-CNN no cargado. Usa cargar_faster_rcnn() primero.")
            return None
        
        try:
            model = self.modelos_cargados['FasterRCNN']
            
            # Cargar y preprocesar imagen
            imagen = Image.open(imagen_path).convert("RGB")
            imagen_cv = cv2.imread(imagen_path)
            
            # Convertir a tensor
            img_tensor = F.to_tensor(imagen).unsqueeze(0).to(self.device)
            
            # Realizar detección
            inicio = time.time()
            with torch.no_grad():
                predictions = model(img_tensor)
            tiempo_inferencia = time.time() - inicio
            
            # Procesar resultados
            pred = predictions[0]
            detecciones = []
            
            for idx, score in enumerate(pred['scores']):
                if score.item() >= confianza:
                    box = pred['boxes'][idx].cpu().numpy()
                    label_idx = pred['labels'][idx].item()
                    
                    deteccion = {
                        'clase': self.coco_classes.get(label_idx, 'unknown'),
                        'confianza': float(score.item()),
                        'bbox': [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                        'centro': [int((box[0]+box[2])/2), int((box[1]+box[3])/2)],
                        'area': int((box[2]-box[0]) * (box[3]-box[1]))
                    }
                    detecciones.append(deteccion)
                    
                    # Dibujar en la imagen si se requiere
                    if mostrar_resultado:
                        cv2.rectangle(imagen_cv, 
                                    (int(box[0]), int(box[1])), 
                                    (int(box[2]), int(box[3])), 
                                    (0, 0, 255), 2)
                        label = f"{deteccion['clase']}: {score.item():.2f}"
                        cv2.putText(imagen_cv, label, 
                                  (int(box[0]), int(box[1])-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            resultado_final = {
                'imagen': os.path.basename(imagen_path),
                'modelo': 'Faster R-CNN',
                'tiempo_inferencia': tiempo_inferencia,
                'num_detecciones': len(detecciones),
                'detecciones': detecciones,
                'imagen_con_detecciones': imagen_cv if mostrar_resultado else None
            }
            
            if mostrar_resultado and len(detecciones) > 0:
                self._mostrar_imagen_con_detecciones(imagen_cv, f"Faster R-CNN - {len(detecciones)} detecciones")
            
            return resultado_final
            
        except Exception as e:
            print(f"❌ Error en detección Faster R-CNN: {e}")
            return None
    
    def detectar_ssd(self, imagen_path, confianza=0.5, mostrar_resultado=True):
        """
        Realiza detección usando SSD.
        
        Args:
            imagen_path: Ruta a la imagen
            confianza: Umbral de confianza mínimo
            mostrar_resultado: Si mostrar la imagen con detecciones
            
        Returns:
            Resultados de detección
        """
        if 'SSD' not in self.modelos_cargados:
            print("❌ Modelo SSD no cargado. Usa cargar_ssd_mobilenet() primero.")
            return None
        
        # Implementación placeholder - requiere modelo SSD funcional
        print("⚠️  Función SSD en desarrollo. Use YOLO o Faster R-CNN por ahora.")
        return None
    
    def comparar_modelos(self, imagen_path, confianza=0.5):
        """
        Compara el rendimiento de todos los modelos cargados en una imagen.
        
        Args:
            imagen_path: Ruta a la imagen
            confianza: Umbral de confianza mínimo
            
        Returns:
            Comparación de resultados
        """
        print(f"🔍 Comparando modelos en: {os.path.basename(imagen_path)}")
        
        resultados_comparacion = {
            'imagen': os.path.basename(imagen_path),
            'fecha_analisis': datetime.now().isoformat(),
            'modelos_comparados': [],
            'resumen': {}
        }
        
        # Probar YOLO
        if 'YOLO' in self.modelos_cargados:
            print("Testing YOLO...")
            resultado_yolo = self.detectar_yolo(imagen_path, confianza, mostrar_resultado=False)
            if resultado_yolo:
                resultados_comparacion['modelos_comparados'].append(resultado_yolo)
        
        # Probar Faster R-CNN
        if 'FasterRCNN' in self.modelos_cargados:
            print("Testing Faster R-CNN...")
            resultado_rcnn = self.detectar_faster_rcnn(imagen_path, confianza, mostrar_resultado=False)
            if resultado_rcnn:
                resultados_comparacion['modelos_comparados'].append(resultado_rcnn)
        
        # Probar SSD
        if 'SSD' in self.modelos_cargados:
            print("Testing SSD...")
            resultado_ssd = self.detectar_ssd(imagen_path, confianza, mostrar_resultado=False)
            if resultado_ssd:
                resultados_comparacion['modelos_comparados'].append(resultado_ssd)
        
        # Generar resumen
        if resultados_comparacion['modelos_comparados']:
            self._generar_resumen_comparacion(resultados_comparacion)
            self._visualizar_comparacion(resultados_comparacion, imagen_path)
        
        return resultados_comparacion
    
    def _generar_resumen_comparacion(self, resultados_comparacion):
        """Genera resumen estadístico de la comparación."""
        resumen = {}
        
        for resultado in resultados_comparacion['modelos_comparados']:
            modelo = resultado['modelo']
            resumen[modelo] = {
                'tiempo_inferencia': resultado['tiempo_inferencia'],
                'num_detecciones': resultado['num_detecciones'],
                'clases_detectadas': list(set([det['clase'] for det in resultado['detecciones']])),
                'confianza_promedio': np.mean([det['confianza'] for det in resultado['detecciones']]) if resultado['detecciones'] else 0
            }
        
        resultados_comparacion['resumen'] = resumen
        
        # Guardar comparación
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archivo_comparacion = os.path.join(
            self.directorio_resultados,
            f'comparacion_modelos_{timestamp}.json'
        )
        
        with open(archivo_comparacion, 'w') as f:
            # Convertir numpy arrays a listas para JSON
            resultados_json = self._convertir_para_json(resultados_comparacion)
            json.dump(resultados_json, f, indent=2)
        
        print(f"📊 Comparación guardada en: {archivo_comparacion}")
    
    def _convertir_para_json(self, obj):
        """Convierte objetos numpy a tipos serializables."""
        if isinstance(obj, dict):
            return {k: self._convertir_para_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convertir_para_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        else:
            return obj
    
    def _visualizar_comparacion(self, resultados_comparacion, imagen_path):
        """Visualiza la comparación de modelos."""
        modelos = resultados_comparacion['modelos_comparados']
        
        if len(modelos) == 0:
            return
        
        # Crear figura para comparación
        fig, axes = plt.subplots(1, len(modelos), figsize=(5*len(modelos), 5))
        if len(modelos) == 1:
            axes = [axes]
        
        imagen_original = cv2.imread(imagen_path)
        imagen_original = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2RGB)
        
        for i, resultado in enumerate(modelos):
            ax = axes[i]
            
            # Mostrar imagen original
            ax.imshow(imagen_original)
            
            # Dibujar detecciones
            for det in resultado['detecciones']:
                bbox = det['bbox']
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]),
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                    linewidth=2,
                    edgecolor='red',
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                # Etiqueta
                ax.text(bbox[0], bbox[1]-5, 
                       f"{det['clase']}: {det['confianza']:.2f}",
                       color='red', fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
            
            ax.set_title(f"{resultado['modelo']}\n{resultado['num_detecciones']} detecciones\n{resultado['tiempo_inferencia']:.3f}s")
            ax.axis('off')
        
        plt.tight_layout()
        
        # Guardar comparación visual
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(os.path.join(
            self.directorio_resultados,
            f'comparacion_visual_{timestamp}.png'
        ), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _mostrar_imagen_con_detecciones(self, imagen, titulo="Detecciones"):
        """Muestra imagen con detecciones."""
        plt.figure(figsize=(12, 8))
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        plt.imshow(imagen_rgb)
        plt.title(titulo)
        plt.axis('off')
        plt.show()
    
    def buscar_objeto_especifico(self, imagen_path, objeto_buscado, confianza=0.5):
        """
        Busca un objeto específico en la imagen usando todos los modelos disponibles.
        
        Args:
            imagen_path: Ruta a la imagen
            objeto_buscado: Nombre del objeto a buscar (ej: 'person', 'car', etc.)
            confianza: Umbral de confianza mínimo
            
        Returns:
            Resultados de búsqueda específica
        """
        print(f"🔍 Buscando '{objeto_buscado}' en {os.path.basename(imagen_path)}")
        
        # Realizar detección con todos los modelos
        resultados_completos = self.comparar_modelos(imagen_path, confianza)
        
        # Filtrar por objeto específico
        resultados_filtrados = {
            'objeto_buscado': objeto_buscado,
            'imagen': os.path.basename(imagen_path),
            'encontrado': False,
            'detecciones_por_modelo': {}
        }
        
        for resultado in resultados_completos['modelos_comparados']:
            modelo = resultado['modelo']
            detecciones_objeto = [
                det for det in resultado['detecciones'] 
                if objeto_buscado.lower() in det['clase'].lower()
            ]
            
            if detecciones_objeto:
                resultados_filtrados['encontrado'] = True
                resultados_filtrados['detecciones_por_modelo'][modelo] = {
                    'num_detecciones': len(detecciones_objeto),
                    'detecciones': detecciones_objeto,
                    'confianza_maxima': max([det['confianza'] for det in detecciones_objeto])
                }
        
        # Mostrar resultados
        if resultados_filtrados['encontrado']:
            print(f"✅ '{objeto_buscado}' encontrado!")
            for modelo, info in resultados_filtrados['detecciones_por_modelo'].items():
                print(f"   {modelo}: {info['num_detecciones']} detección(es), "
                      f"confianza máx: {info['confianza_maxima']:.3f}")
        else:
            print(f"❌ '{objeto_buscado}' no encontrado en ningún modelo")
        
        return resultados_filtrados
    
    def procesar_video(self, video_path, modelo='YOLO', confianza=0.5, 
                      output_path=None, mostrar_tiempo_real=False):
        """
        Procesa un video completo aplicando detección de objetos.
        
        Args:
            video_path: Ruta al video de entrada
            modelo: Modelo a usar ('YOLO', 'FasterRCNN', 'SSD')
            confianza: Umbral de confianza
            output_path: Ruta para guardar video procesado
            mostrar_tiempo_real: Si mostrar el video en tiempo real
            
        Returns:
            Estadísticas del procesamiento
        """
        if modelo not in self.modelos_cargados:
            print(f"❌ Modelo {modelo} no cargado")
            return None
        
        # Abrir video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ No se pudo abrir el video: {video_path}")
            return None
        
        # Propiedades del video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Procesando video: {os.path.basename(video_path)}")
        print(f"   Resolución: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Configurar salida si se especifica
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Estadísticas
        estadisticas = {
            'video_original': os.path.basename(video_path),
            'modelo_usado': modelo,
            'frames_procesados': 0,
            'detecciones_totales': 0,
            'tiempo_total': 0,
            'fps_promedio': 0
        }
        
        frame_count = 0
        inicio_total = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Guardar frame temporal para procesamiento
                temp_path = os.path.join(self.directorio_resultados, "temp_frame.jpg")
                cv2.imwrite(temp_path, frame)
                
                # Procesar frame según modelo
                if modelo == 'YOLO':
                    resultado = self.detectar_yolo(temp_path, confianza, False)
                elif modelo == 'FasterRCNN':
                    resultado = self.detectar_faster_rcnn(temp_path, confianza, False)
                else:
                    print(f"⚠️  Modelo {modelo} no implementado para video")
                    continue
                
                # Dibujar detecciones en el frame
                if resultado and resultado['detecciones']:
                    for det in resultado['detecciones']:
                        bbox = det['bbox']
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        label = f"{det['clase']}: {det['confianza']:.2f}"
                        cv2.putText(frame, label, (bbox[0], bbox[1]-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    estadisticas['detecciones_totales'] += len(resultado['detecciones'])
                
                # Guardar frame procesado
                if writer:
                    writer.write(frame)
                
                # Mostrar en tiempo real si se solicita
                if mostrar_tiempo_real:
                    cv2.imshow('Detección en Tiempo Real', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
                estadisticas['frames_procesados'] = frame_count
                
                # Mostrar progreso
                if frame_count % 30 == 0:  # Cada 30 frames
                    progreso = (frame_count / total_frames) * 100
                    print(f"   Progreso: {progreso:.1f}% ({frame_count}/{total_frames})")
                
                # Limpiar archivo temporal
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        except KeyboardInterrupt:
            print("\n⚠️  Procesamiento interrumpido por el usuario")
        
        finally:
            # Limpiar recursos
            cap.release()
            if writer:
                writer.release()
            if mostrar_tiempo_real:
                cv2.destroyAllWindows()
            
            # Calcular estadísticas finales
            tiempo_total = time.time() - inicio_total
            estadisticas['tiempo_total'] = tiempo_total
            estadisticas['fps_promedio'] = frame_count / tiempo_total if tiempo_total > 0 else 0
            
            print(f"\n✅ Procesamiento completado!")
            print(f"   Frames procesados: {frame_count}")
            print(f"   Detecciones totales: {estadisticas['detecciones_totales']}")
            print(f"   Tiempo total: {tiempo_total:.2f} segundos")
            print(f"   FPS promedio: {estadisticas['fps_promedio']:.2f}")
            
            if output_path:
                print(f"   Video guardado en: {output_path}")
        
        return estadisticas
    
    def mostrar_info_modelos(self):
        """Muestra información de todos los modelos cargados."""
        print("\n📋 MODELOS CARGADOS")
        print("=" * 50)
        
        if not self.modelos_cargados:
            print("❌ No hay modelos cargados")
            return
        
        for nombre, config in self.configuraciones.items():
            print(f"\n🤖 {nombre}")
            print(f"   Tipo: {config['tipo']}")
            if 'modelo' in config:
                print(f"   Versión: {config['modelo']}")
            if 'backbone' in config:
                print(f"   Backbone: {config['backbone']}")
            print(f"   Clases detectables: {len(config['clases_detectables'])}")
            print(f"   Fecha de carga: {config['fecha_carga']}")

def main():
    """Función principal para probar el módulo."""
    print("🤖 MÓDULO DE MODELOS PREENTRENADOS")
    print("=" * 50)
    
    # Inicializar módulo
    modelos = ModelosPreentrenados()
    
    # Cargar modelos de ejemplo
    print("\n📥 Cargando modelos...")
    yolo = modelos.cargar_yolo('yolov8n')
    rcnn = modelos.cargar_faster_rcnn()
    
    # Mostrar información
    modelos.mostrar_info_modelos()
    
    print("\n✅ Módulo listo para detección!")
    print("💡 Para usar:")
    print("   1. Carga una imagen con detectar_yolo() o detectar_faster_rcnn()")
    print("   2. Compara modelos con comparar_modelos()")
    print("   3. Procesa videos con procesar_video()")

if __name__ == "__main__":
    main()