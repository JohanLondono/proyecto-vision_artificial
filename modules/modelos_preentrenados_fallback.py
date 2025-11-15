#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Fallback para Modelos Preentrenados
=============================================

Este módulo se activa cuando no están disponibles las dependencias completas
de deep learning, proporcionando funcionalidad básica usando solo OpenCV.

Autor: Sistema de Detección Vehicular
Fecha: Noviembre 2025
"""

import os
import cv2
import numpy as np
import time
from datetime import datetime
import json

class ModelosPreentrenados:
    """
    Versión simplificada que usa solo OpenCV para detección básica.
    """
    
    def __init__(self, directorio_resultados="./resultados_deteccion/pretrained_models"):
        """
        Inicializa el módulo de modelos preentrenados básicos.
        
        Args:
            directorio_resultados: Directorio para guardar resultados
        """
        self.directorio_resultados = directorio_resultados
        self.modelos_cargados = {}
        self.configuraciones = {}
        
        # Crear directorio si no existe
        os.makedirs(self.directorio_resultados, exist_ok=True)
        
        print("⚠️  Usando módulo de fallback - funcionalidad limitada")
        print("💡 Para funcionalidad completa, instale: pip install torch ultralytics")
    
    def cargar_yolo(self, modelo='yolov8n'):
        """
        Simula carga de YOLO con detector de personas básico usando OpenCV.
        """
        print(f"⚠️  YOLO no disponible, usando detector básico de OpenCV")
        
        # Cargar clasificador de caras como proxy
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.modelos_cargados['YOLO'] = face_cascade
            
            self.configuraciones['YOLO'] = {
                'modelo': 'OpenCV_HaarCascade',
                'tipo': 'Fallback',
                'descripcion': 'Detector básico de caras usando Haar Cascades',
                'fecha_carga': datetime.now().isoformat()
            }
            
            print("✅ Detector básico cargado (Haar Cascades)")
            return face_cascade
            
        except Exception as e:
            print(f"❌ Error cargando detector básico: {e}")
            return None
    
    def cargar_faster_rcnn(self, preentrenado=True):
        """
        Placeholder para Faster R-CNN.
        """
        print("❌ Faster R-CNN no disponible sin PyTorch")
        return None
    
    def detectar_yolo(self, imagen_path, confianza=0.5, mostrar_resultado=True):
        """
        Realiza detección básica usando OpenCV.
        """
        if 'YOLO' not in self.modelos_cargados:
            print("❌ Detector básico no cargado")
            return None
        
        try:
            face_cascade = self.modelos_cargados['YOLO']
            
            # Cargar imagen
            imagen = cv2.imread(imagen_path)
            if imagen is None:
                print(f"❌ No se pudo cargar la imagen: {imagen_path}")
                return None
            
            gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            
            # Detectar caras
            inicio = time.time()
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            tiempo_inferencia = time.time() - inicio
            
            # Convertir resultados a formato similar a YOLO
            detecciones = []
            for (x, y, w, h) in faces:
                deteccion = {
                    'clase': 'person',  # Proxy: cara = persona
                    'confianza': 0.8,   # Confianza simulada
                    'bbox': [int(x), int(y), int(x+w), int(y+h)],
                    'centro': [int(x+w/2), int(y+h/2)],
                    'area': int(w * h)
                }
                detecciones.append(deteccion)
                
                # Dibujar detección si se requiere
                if mostrar_resultado:
                    cv2.rectangle(imagen, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(imagen, f"Persona: 0.80", (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            resultado_final = {
                'imagen': os.path.basename(imagen_path),
                'modelo': 'OpenCV_Fallback',
                'tiempo_inferencia': tiempo_inferencia,
                'num_detecciones': len(detecciones),
                'detecciones': detecciones,
                'imagen_con_detecciones': imagen if mostrar_resultado else None,
                'nota': 'Detección básica usando OpenCV Haar Cascades'
            }
            
            if mostrar_resultado and len(detecciones) > 0:
                print(f"🔍 Detectadas {len(detecciones)} persona(s) con detector básico")
                
                # Mostrar imagen con detecciones
                import matplotlib.pyplot as plt
                imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(10, 8))
                plt.imshow(imagen_rgb)
                plt.title(f"Detección Básica - {len(detecciones)} persona(s)")
                plt.axis('off')
                plt.show()
            
            return resultado_final
            
        except Exception as e:
            print(f"❌ Error en detección básica: {e}")
            return None
    
    def detectar_faster_rcnn(self, imagen_path, confianza=0.5, mostrar_resultado=True):
        """
        Placeholder para Faster R-CNN.
        """
        print("❌ Faster R-CNN no disponible - use detector básico")
        return None
    
    def buscar_objeto_especifico(self, imagen_path, objeto_buscado, confianza=0.5):
        """
        Busca objeto específico usando detector básico.
        """
        print(f"🔍 Buscando '{objeto_buscado}' con detector básico...")
        
        if objeto_buscado.lower() in ['person', 'persona', 'people']:
            resultado = self.detectar_yolo(imagen_path, confianza, False)
            
            if resultado and resultado['detecciones']:
                return {
                    'objeto_buscado': objeto_buscado,
                    'imagen': os.path.basename(imagen_path),
                    'encontrado': True,
                    'detecciones_por_modelo': {
                        'OpenCV_Fallback': {
                            'num_detecciones': resultado['num_detecciones'],
                            'detecciones': resultado['detecciones'],
                            'confianza_maxima': max([det['confianza'] for det in resultado['detecciones']])
                        }
                    }
                }
            else:
                return {
                    'objeto_buscado': objeto_buscado,
                    'imagen': os.path.basename(imagen_path),
                    'encontrado': False,
                    'detecciones_por_modelo': {}
                }
        else:
            print(f"⚠️  Objeto '{objeto_buscado}' no soportado por detector básico")
            return {
                'objeto_buscado': objeto_buscado,
                'imagen': os.path.basename(imagen_path),
                'encontrado': False,
                'detecciones_por_modelo': {}
            }
    
    def mostrar_info_modelos(self):
        """Muestra información de modelos básicos cargados."""
        print("\n📋 MODELOS BÁSICOS CARGADOS (FALLBACK)")
        print("=" * 50)
        
        if not self.modelos_cargados:
            print("❌ No hay modelos cargados")
            return
        
        for nombre, config in self.configuraciones.items():
            print(f"\n🔧 {nombre} (Básico)")
            print(f"   Tipo: {config['tipo']}")
            print(f"   Modelo: {config['modelo']}")
            print(f"   Descripción: {config['descripcion']}")
            print(f"   Fecha de carga: {config['fecha_carga']}")
            print(f"   ⚠️  NOTA: Funcionalidad limitada sin dependencias completas")

def main():
    """Función principal para probar el módulo."""
    print("🔧 MÓDULO DE MODELOS PREENTRENADOS (FALLBACK)")
    print("=" * 50)
    
    # Inicializar módulo
    modelos = ModelosPreentrenados()
    
    # Cargar detector básico
    detector = modelos.cargar_yolo()
    
    # Mostrar información
    modelos.mostrar_info_modelos()
    
    print("\n✅ Módulo básico listo!")
    print("💡 Para funcionalidad completa, instale dependencias de deep learning")

if __name__ == "__main__":
    main()