#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Preprocesamiento Avanzado para CNNs
============================================

Implementa técnicas avanzadas de preprocesamiento específicas para 
redes neuronales convolucionales, incluyendo augmentación de datos,
normalización específica y transformaciones para mejorar el rendimiento.

Universidad del Quindío - Visión Artificial
Autor: Sistema de Visión Artificial
Fecha: Noviembre 2024
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random


class PreprocesadorAvanzadoCNN:
    """
    Clase para preprocesamiento avanzado específico para CNNs.
    """
    
    def __init__(self):
        """Inicializa el preprocesador avanzado."""
        self.estadisticas_normalizacion = {
            'imagenet': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        }
    
    def normalizar_imagenet(self, imagen):
        """
        Normaliza la imagen usando estadísticas de ImageNet.
        
        Args:
            imagen: Imagen en formato RGB con valores [0,1]
            
        Returns:
            Imagen normalizada con estadísticas de ImageNet
        """
        try:
            # Asegurar que la imagen está en [0,1]
            if imagen.max() > 1.0:
                imagen = imagen / 255.0
            
            imagen_norm = imagen.copy().astype(np.float32)
            
            # Aplicar normalización por canal
            mean = np.array(self.estadisticas_normalizacion['imagenet']['mean'])
            std = np.array(self.estadisticas_normalizacion['imagenet']['std'])
            
            for i in range(3):  # RGB
                imagen_norm[:, :, i] = (imagen_norm[:, :, i] - mean[i]) / std[i]
            
            return imagen_norm
            
        except Exception as e:
            print(f"Error en normalización ImageNet: {e}")
            return None
    
    def augmentacion_basica(self, imagen):
        """
        Aplica augmentación básica de datos.
        
        Args:
            imagen: Imagen original
            
        Returns:
            tuple: (imagen_augmentada, descripcion_transformaciones)
        """
        try:
            transformaciones = []
            imagen_aug = imagen.copy()
            
            # Rotación aleatoria
            if random.random() > 0.5:
                angulo = random.randint(-15, 15)
                h, w = imagen_aug.shape[:2]
                matriz_rot = cv2.getRotationMatrix2D((w//2, h//2), angulo, 1.0)
                imagen_aug = cv2.warpAffine(imagen_aug, matriz_rot, (w, h))
                transformaciones.append(f"Rotación: {angulo}°")
            
            # Flip horizontal
            if random.random() > 0.5:
                imagen_aug = cv2.flip(imagen_aug, 1)
                transformaciones.append("Flip horizontal")
            
            # Ajuste de brillo
            if random.random() > 0.5:
                factor_brillo = random.uniform(0.8, 1.2)
                imagen_aug = np.clip(imagen_aug * factor_brillo, 0, 255).astype(np.uint8)
                transformaciones.append(f"Brillo: {factor_brillo:.2f}")
            
            # Ajuste de contraste
            if random.random() > 0.5:
                alpha = random.uniform(0.8, 1.2)  # Factor de contraste
                imagen_aug = np.clip(alpha * imagen_aug, 0, 255).astype(np.uint8)
                transformaciones.append(f"Contraste: {alpha:.2f}")
            
            return imagen_aug, transformaciones
            
        except Exception as e:
            print(f"Error en augmentación: {e}")
            return imagen, []
    
    def preprocesamiento_completo_cnn(self, imagen, incluir_augmentacion=False, 
                                    normalizacion='imagenet'):
        """
        Aplica preprocesamiento completo para CNNs.
        
        Args:
            imagen: Imagen original
            incluir_augmentacion: Si aplicar augmentación de datos
            normalizacion: Tipo de normalización ('imagenet', 'basica')
            
        Returns:
            dict: Resultados del preprocesamiento
        """
        try:
            resultados = {
                'imagen_original': imagen.copy(),
                'transformaciones_aplicadas': [],
                'estadisticas': {}
            }
            
            imagen_procesada = imagen.copy()
            
            # 1. Redimensionar a 224x224
            imagen_procesada = cv2.resize(imagen_procesada, (224, 224))
            resultados['transformaciones_aplicadas'].append("Redimensionado a 224x224")
            
            # 2. Augmentación (opcional)
            if incluir_augmentacion:
                imagen_procesada, augmentaciones = self.augmentacion_basica(imagen_procesada)
                resultados['transformaciones_aplicadas'].extend(augmentaciones)
            
            # 3. Normalización
            if normalizacion == 'imagenet':
                # Primero normalizar a [0,1] si es necesario
                if imagen_procesada.max() > 1.0:
                    imagen_procesada = imagen_procesada / 255.0
                
                imagen_procesada = self.normalizar_imagenet(imagen_procesada)
                resultados['transformaciones_aplicadas'].append("Normalización ImageNet")
                
            elif normalizacion == 'basica':
                if imagen_procesada.max() > 1.0:
                    imagen_procesada = imagen_procesada / 255.0
                resultados['transformaciones_aplicadas'].append("Normalización [0,1]")
            
            # Estadísticas finales
            resultados['estadisticas'] = {
                'forma_final': imagen_procesada.shape,
                'tipo_datos': imagen_procesada.dtype,
                'min_valor': float(np.min(imagen_procesada)),
                'max_valor': float(np.max(imagen_procesada)),
                'media': float(np.mean(imagen_procesada)),
                'std': float(np.std(imagen_procesada))
            }
            
            resultados['imagen_final'] = imagen_procesada
            
            return resultados
            
        except Exception as e:
            print(f"Error en preprocesamiento completo CNN: {e}")
            return None
    
    def preparar_batch(self, imagen):
        """
        Prepara la imagen para procesamiento en batch (añade dimensión).
        
        Args:
            imagen: Imagen preprocesada
            
        Returns:
            Imagen con dimensión de batch (1, H, W, C) o (1, C, H, W)
        """
        try:
            if len(imagen.shape) == 3:  # (H, W, C)
                # Formato PyTorch: (1, C, H, W)
                batch_pytorch = np.transpose(imagen, (2, 0, 1))[np.newaxis, :]
                # Formato TensorFlow: (1, H, W, C)
                batch_tensorflow = imagen[np.newaxis, :]
                
                return {
                    'pytorch_format': batch_pytorch,
                    'tensorflow_format': batch_tensorflow,
                    'shape_pytorch': batch_pytorch.shape,
                    'shape_tensorflow': batch_tensorflow.shape
                }
            else:
                print("La imagen debe tener 3 dimensiones (H, W, C)")
                return None
                
        except Exception as e:
            print(f"Error preparando batch: {e}")
            return None
    
    def aplicar_filtros_cnn(self, imagen):
        """
        Aplica filtros que pueden mejorar la detección de características.
        
        Args:
            imagen: Imagen original
            
        Returns:
            dict: Diferentes versiones filtradas de la imagen
        """
        try:
            resultados = {}
            
            # Conversión a escala de grises para algunos filtros
            gray = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY) if len(imagen.shape) == 3 else imagen
            
            # Filtro Gaussiano (suavizado)
            gaussian = cv2.GaussianBlur(imagen, (5, 5), 0)
            resultados['gaussian'] = gaussian
            
            # Detección de bordes Canny
            edges = cv2.Canny(gray, 50, 150)
            # Convertir a 3 canales para consistencia
            edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            resultados['edges'] = edges_3ch
            
            # Filtro de realce de bordes
            kernel_sharpen = np.array([[-1,-1,-1],
                                     [-1, 9,-1],
                                     [-1,-1,-1]])
            sharpened = cv2.filter2D(imagen, -1, kernel_sharpen)
            resultados['sharpened'] = np.clip(sharpened, 0, 255)
            
            # Ecualización de histograma (solo para visualización)
            if len(imagen.shape) == 3:
                # Ecualizar en espacio LAB
                lab = cv2.cvtColor(imagen, cv2.COLOR_RGB2LAB)
                lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
                equalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                resultados['equalized'] = equalized
            
            return resultados
            
        except Exception as e:
            print(f"Error aplicando filtros: {e}")
            return None
    
    def visualizar_comparacion_multiple(self, imagen_original, resultados_preprocesamiento):
        """
        Visualiza múltiples versiones de la imagen en una sola figura.
        
        Args:
            imagen_original: Imagen original
            resultados_preprocesamiento: Resultados del preprocesamiento
        """
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.ravel()
            
            # Imagen original
            axes[0].imshow(imagen_original)
            axes[0].set_title('Original')
            axes[0].axis('off')
            
            # Imagen redimensionada
            img_224 = cv2.resize(imagen_original, (224, 224))
            axes[1].imshow(img_224)
            axes[1].set_title('Redimensionada 224x224')
            axes[1].axis('off')
            
            # Con augmentación
            img_aug, _ = self.augmentacion_basica(img_224)
            axes[2].imshow(img_aug)
            axes[2].set_title('Con Augmentación')
            axes[2].axis('off')
            
            # Filtros
            filtros = self.aplicar_filtros_cnn(img_224)
            if filtros:
                axes[3].imshow(filtros.get('gaussian', img_224))
                axes[3].set_title('Filtro Gaussiano')
                axes[3].axis('off')
                
                axes[4].imshow(filtros.get('edges', img_224), cmap='gray')
                axes[4].set_title('Detección de Bordes')
                axes[4].axis('off')
                
                axes[5].imshow(filtros.get('sharpened', img_224))
                axes[5].set_title('Realce de Bordes')
                axes[5].axis('off')
            
            plt.tight_layout()
            plt.suptitle('Comparación de Técnicas de Preprocesamiento para CNN', 
                        fontsize=16, y=1.02)
            plt.show()
            
        except Exception as e:
            print(f"Error en visualización múltiple: {e}")
    
    def generar_reporte_preprocesamiento(self, resultados):
        """
        Genera un reporte detallado del preprocesamiento aplicado.
        
        Args:
            resultados: Resultados del preprocesamiento
            
        Returns:
            str: Reporte formateado
        """
        try:
            reporte = []
            reporte.append("="*60)
            reporte.append("    REPORTE DE PREPROCESAMIENTO AVANZADO CNN")
            reporte.append("="*60)
            reporte.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            reporte.append("")
            
            reporte.append("TRANSFORMACIONES APLICADAS:")
            reporte.append("-"*30)
            for i, transform in enumerate(resultados.get('transformaciones_aplicadas', []), 1):
                reporte.append(f"{i}. {transform}")
            
            reporte.append("")
            reporte.append("ESTADÍSTICAS FINALES:")
            reporte.append("-"*30)
            stats = resultados.get('estadisticas', {})
            reporte.append(f"Forma final: {stats.get('forma_final', 'N/A')}")
            reporte.append(f"Tipo de datos: {stats.get('tipo_datos', 'N/A')}")
            reporte.append(f"Rango: [{stats.get('min_valor', 0):.4f}, {stats.get('max_valor', 0):.4f}]")
            reporte.append(f"Media: {stats.get('media', 0):.4f}")
            reporte.append(f"Desviación estándar: {stats.get('std', 0):.4f}")
            
            reporte.append("")
            reporte.append("RECOMENDACIONES PARA CNN:")
            reporte.append("-"*30)
            reporte.append("• La imagen está lista para ser utilizada en modelos preentrenados")
            reporte.append("• Si usa normalización ImageNet, compatible con ResNet, VGG, etc.")
            reporte.append("• Para entrenamiento, considere augmentación adicional")
            reporte.append("• Verifique que el formato coincida con el framework (TF/PyTorch)")
            
            return "\\n".join(reporte)
            
        except Exception as e:
            print(f"Error generando reporte: {e}")
            return "Error al generar el reporte"