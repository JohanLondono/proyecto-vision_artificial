#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Filtros para el Sistema de Detección Vehicular
=======================================================

Contiene todas las operaciones de filtrado de imágenes necesarias
para el preprocesamiento en el sistema de detección vehicular.

Autor: Sistema de Detección Vehicular
Fecha: Octubre 2025
"""

import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
import os
import pandas as pd
from datetime import datetime

class Filtros:
    """
    Clase para operaciones de filtrado de imágenes.
    """
    
    @staticmethod
    def aplicar_filtro_desenfoque(imagen, kernel_size=(5, 5)):
        """
        Aplica un filtro de desenfoque promedio a la imagen.
        
        Args:
            imagen: Imagen de entrada
            kernel_size: Tamaño del kernel para el desenfoque
            
        Returns:
            Imagen con filtro de desenfoque aplicado
        """
        return cv2.blur(imagen, kernel_size)
    
    @staticmethod
    def aplicar_filtro_gaussiano(imagen, kernel_size=(5, 5), sigma=0):
        """
        Aplica un filtro gaussiano a la imagen.
        
        Args:
            imagen: Imagen de entrada
            kernel_size: Tamaño del kernel para el filtro
            sigma: Desviación estándar en X e Y
            
        Returns:
            Imagen con filtro gaussiano aplicado
        """
        return cv2.GaussianBlur(imagen, kernel_size, sigma)
    
    @staticmethod
    def aplicar_filtro_mediana(imagen, kernel_size=5):
        """
        Aplica un filtro de mediana a la imagen.
        
        Args:
            imagen: Imagen de entrada
            kernel_size: Tamaño del kernel para el filtro
            
        Returns:
            Imagen con filtro de mediana aplicado
        """
        return cv2.medianBlur(imagen, kernel_size)
    
    @staticmethod
    def aplicar_filtro_nitidez(imagen):
        """
        Aplica un filtro de nitidez a la imagen.
        
        Args:
            imagen: Imagen de entrada
            
        Returns:
            Imagen con filtro de nitidez aplicado
        """
        # Kernel para nitidez
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        
        return cv2.filter2D(imagen, -1, kernel)
    
    @staticmethod
    def detectar_bordes_canny(imagen, umbral1=100, umbral2=200):
        """
        Detecta bordes usando el algoritmo Canny.
        
        Args:
            imagen: Imagen en escala de grises
            umbral1: Primer umbral para la detección
            umbral2: Segundo umbral para la detección
            
        Returns:
            Imagen con bordes detectados
        """
        return cv2.Canny(imagen, umbral1, umbral2)
    
    @staticmethod
    def ecualizar_histograma(imagen):
        """
        Ecualiza el histograma de una imagen en escala de grises.
        
        Args:
            imagen: Imagen en escala de grises
            
        Returns:
            Imagen con histograma ecualizado
        """
        return cv2.equalizeHist(imagen)
    
    @staticmethod
    def aplicar_filtro_bilateral(imagen, d=9, sigma_color=75, sigma_space=75):
        """
        Aplica un filtro bilateral para reducir ruido preservando bordes.
        
        Args:
            imagen: Imagen de entrada
            d: Diámetro de cada vecindad de píxeles
            sigma_color: Valor sigma en el espacio de color
            sigma_space: Valor sigma en el espacio de coordenadas
            
        Returns:
            Imagen con filtro bilateral aplicado
        """
        return cv2.bilateralFilter(imagen, d, sigma_color, sigma_space)
    
    @staticmethod
    def aplicar_filtro_sobel(imagen, direccion='ambas'):
        """
        Aplica el filtro Sobel para detección de bordes.
        
        Args:
            imagen: Imagen en escala de grises
            direccion: 'x', 'y' o 'ambas'
            
        Returns:
            Imagen con filtro Sobel aplicado
        """
        if len(imagen.shape) > 2:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        if direccion == 'x':
            return cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
        elif direccion == 'y':
            return cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)
        else:  # ambas
            sobel_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)
            return cv2.magnitude(sobel_x, sobel_y)
    
    @staticmethod
    def aplicar_filtro_laplaciano(imagen):
        """
        Aplica el filtro Laplaciano para detección de bordes.
        
        Args:
            imagen: Imagen de entrada
            
        Returns:
            Imagen con filtro Laplaciano aplicado
        """
        if len(imagen.shape) > 2:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        return cv2.Laplacian(imagen, cv2.CV_64F)
    
    @staticmethod
    def aplicar_laplaciano_gauss(imagen, sigma=2.0, umbral_factor=0.1):
        """
        Aplica el filtro Laplaciano de Gauss (LoG) para detección de bordes.
        
        Args:
            imagen: Imagen de entrada
            sigma: Desviación estándar para el filtro Gaussiano
            umbral_factor: Factor para el umbralizado de la respuesta LoG
            
        Returns:
            Imagen con bordes detectados usando LoG
        """
        try:
            # Intentar usar skimage (más preciso)
            from scipy import ndimage
            from skimage.filters import gaussian, laplace
            
            # Convertir a escala de grises si es necesario
            if len(imagen.shape) == 3:
                imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            else:
                imagen_gris = imagen.copy()
            
            # Normalizar la imagen
            imagen_norm = imagen_gris.astype(np.float64) / 255.0
            
            # Aplicar filtro Gaussiano
            imagen_suavizada = gaussian(imagen_norm, sigma=sigma)
            
            # Aplicar operador Laplaciano
            respuesta_log = laplace(imagen_suavizada)
            
            # Calcular umbral dinámico
            umbral = umbral_factor * np.max(np.abs(respuesta_log))
            
            # Crear imagen binaria de bordes
            bordes = np.abs(respuesta_log) > umbral
            
            # Convertir a imagen de 8 bits para visualización
            imagen_bordes = (bordes * 255).astype(np.uint8)
            
            return imagen_bordes
            
        except ImportError:
            # Implementación alternativa usando solo OpenCV y NumPy
            print("Usando implementación alternativa con OpenCV...")
            
            # Convertir a escala de grises si es necesario
            if len(imagen.shape) == 3:
                imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            else:
                imagen_gris = imagen.copy()
            
            # Calcular tamaño del kernel basado en sigma
            kernel_size = int(6 * sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Aplicar filtro Gaussiano
            imagen_suavizada = cv2.GaussianBlur(imagen_gris, (kernel_size, kernel_size), sigma)
            
            # Aplicar operador Laplaciano
            respuesta_log = cv2.Laplacian(imagen_suavizada, cv2.CV_64F)
            
            # Calcular umbral dinámico
            umbral = umbral_factor * np.max(np.abs(respuesta_log))
            
            # Crear imagen binaria de bordes
            bordes = np.abs(respuesta_log) > umbral
            
            # Convertir a imagen de 8 bits para visualización
            imagen_bordes = (bordes * 255).astype(np.uint8)
            
            return imagen_bordes
    
    @staticmethod
    def aplicar_canny_con_estadisticas(imagen, umbral1=100, umbral2=200, guardar_estadisticas=True, 
                                      directorio_salida="./resultados_deteccion/estadisticas_bordes", 
                                      nombre_archivo=None):
        """
        Aplica detector Canny y calcula estadísticas de magnitud del gradiente.
        
        Args:
            imagen: Imagen de entrada
            umbral1: Primer umbral para Canny
            umbral2: Segundo umbral para Canny
            guardar_estadisticas: Si guardar las estadísticas en archivos
            directorio_salida: Directorio donde guardar los resultados
            nombre_archivo: Nombre base para los archivos (si None, usa timestamp)
            
        Returns:
            tuple: (imagen_bordes, estadisticas_dict)
        """
        # Convertir a escala de grises si es necesario
        if len(imagen.shape) > 2:
            imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gris = imagen.copy()
        
        # Aplicar filtro Gaussiano para reducir ruido
        imagen_suavizada = cv2.GaussianBlur(imagen_gris, (5, 5), 1.4)
        
        # Calcular gradientes con Sobel
        grad_x = cv2.Sobel(imagen_suavizada, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(imagen_suavizada, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calcular magnitud y dirección del gradiente
        magnitud = np.sqrt(grad_x**2 + grad_y**2)
        direccion = np.arctan2(grad_y, grad_x) * 180 / np.pi
        
        # Aplicar Canny
        bordes_canny = cv2.Canny(imagen_gris, umbral1, umbral2)
        
        # Calcular estadísticas de magnitud
        estadisticas = {
            'Metodo': 'Canny',
            'Umbral_Inferior': umbral1,
            'Umbral_Superior': umbral2,
            'Fecha_Analisis': datetime.now().isoformat(),
            'Dimensiones_Imagen': imagen_gris.shape,
            
            # Estadísticas de magnitud del gradiente
            'Magnitud_Media': np.mean(magnitud),
            'Magnitud_Mediana': np.median(magnitud),
            'Magnitud_Std': np.std(magnitud),
            'Magnitud_Min': np.min(magnitud),
            'Magnitud_Max': np.max(magnitud),
            'Magnitud_Percentil_25': np.percentile(magnitud, 25),
            'Magnitud_Percentil_75': np.percentile(magnitud, 75),
            'Magnitud_Percentil_90': np.percentile(magnitud, 90),
            'Magnitud_Percentil_95': np.percentile(magnitud, 95),
            
            # Estadísticas de dirección del gradiente
            'Direccion_Media': np.mean(direccion),
            'Direccion_Std': np.std(direccion),
            'Direccion_Min': np.min(direccion),
            'Direccion_Max': np.max(direccion),
            
            # Estadísticas de bordes detectados
            'Pixeles_Borde': np.sum(bordes_canny > 0),
            'Porcentaje_Bordes': (np.sum(bordes_canny > 0) / bordes_canny.size) * 100,
            'Densidad_Bordes': np.sum(bordes_canny > 0) / (imagen_gris.shape[0] * imagen_gris.shape[1]),
            
            # Estadísticas de intensidad en bordes
            'Intensidad_Bordes_Media': np.mean(imagen_gris[bordes_canny > 0]) if np.sum(bordes_canny > 0) > 0 else 0,
            'Intensidad_Bordes_Std': np.std(imagen_gris[bordes_canny > 0]) if np.sum(bordes_canny > 0) > 0 else 0,
            
            # Análisis de conectividad
            'Num_Componentes_Conexas': Filtros._contar_componentes_conexas(bordes_canny),
            
            # Relación de aspecto y características geométricas
            'Relacion_Aspecto': imagen_gris.shape[1] / imagen_gris.shape[0],
            'Area_Total': imagen_gris.shape[0] * imagen_gris.shape[1]
        }
        
        # Guardar estadísticas si se solicita
        if guardar_estadisticas:
            Filtros._guardar_estadisticas_bordes(estadisticas, directorio_salida, nombre_archivo, 'canny')
        
        return bordes_canny, estadisticas
    
    @staticmethod
    def aplicar_sobel_con_estadisticas(imagen, direccion='ambas', ksize=3, guardar_estadisticas=True,
                                      directorio_salida="./resultados_deteccion/estadisticas_bordes",
                                      nombre_archivo=None):
        """
        Aplica detector Sobel y calcula estadísticas de magnitud del gradiente.
        
        Args:
            imagen: Imagen de entrada
            direccion: 'x', 'y' o 'ambas'
            ksize: Tamaño del kernel Sobel
            guardar_estadisticas: Si guardar las estadísticas en archivos
            directorio_salida: Directorio donde guardar los resultados
            nombre_archivo: Nombre base para los archivos
            
        Returns:
            tuple: (imagen_bordes, estadisticas_dict)
        """
        # Convertir a escala de grises si es necesario
        if len(imagen.shape) > 2:
            imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gris = imagen.copy()
        
        # Calcular gradientes con Sobel
        if direccion == 'x':
            grad_x = cv2.Sobel(imagen_gris, cv2.CV_64F, 1, 0, ksize=ksize)
            grad_y = np.zeros_like(grad_x)
            resultado_sobel = np.abs(grad_x)
        elif direccion == 'y':
            grad_y = cv2.Sobel(imagen_gris, cv2.CV_64F, 0, 1, ksize=ksize)
            grad_x = np.zeros_like(grad_y)
            resultado_sobel = np.abs(grad_y)
        else:  # ambas
            grad_x = cv2.Sobel(imagen_gris, cv2.CV_64F, 1, 0, ksize=ksize)
            grad_y = cv2.Sobel(imagen_gris, cv2.CV_64F, 0, 1, ksize=ksize)
            resultado_sobel = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calcular magnitud y dirección total
        magnitud = np.sqrt(grad_x**2 + grad_y**2)
        if direccion == 'ambas':
            direccion_grad = np.arctan2(grad_y, grad_x) * 180 / np.pi
        else:
            direccion_grad = np.zeros_like(magnitud)
        
        # Normalizar resultado para visualización
        resultado_normalizado = cv2.normalize(resultado_sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Crear imagen binaria de bordes (umbralización automática)
        umbral_otsu, bordes_binarios = cv2.threshold(resultado_normalizado, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Calcular estadísticas
        estadisticas = {
            'Metodo': f'Sobel_{direccion}',
            'Kernel_Size': ksize,
            'Direccion_Analizada': direccion,
            'Umbral_Otsu_Calculado': umbral_otsu,
            'Fecha_Analisis': datetime.now().isoformat(),
            'Dimensiones_Imagen': imagen_gris.shape,
            
            # Estadísticas de magnitud del gradiente
            'Magnitud_Media': np.mean(magnitud),
            'Magnitud_Mediana': np.median(magnitud),
            'Magnitud_Std': np.std(magnitud),
            'Magnitud_Min': np.min(magnitud),
            'Magnitud_Max': np.max(magnitud),
            'Magnitud_Percentil_25': np.percentile(magnitud, 25),
            'Magnitud_Percentil_75': np.percentile(magnitud, 75),
            'Magnitud_Percentil_90': np.percentile(magnitud, 90),
            'Magnitud_Percentil_95': np.percentile(magnitud, 95),
            
            # Estadísticas específicas del gradiente en X e Y
            'Gradiente_X_Media': np.mean(grad_x),
            'Gradiente_X_Std': np.std(grad_x),
            'Gradiente_X_Min': np.min(grad_x),
            'Gradiente_X_Max': np.max(grad_x),
            'Gradiente_Y_Media': np.mean(grad_y),
            'Gradiente_Y_Std': np.std(grad_y),
            'Gradiente_Y_Min': np.min(grad_y),
            'Gradiente_Y_Max': np.max(grad_y),
            
            # Estadísticas de dirección (solo si direccion='ambas')
            'Direccion_Media': np.mean(direccion_grad) if direccion == 'ambas' else 0,
            'Direccion_Std': np.std(direccion_grad) if direccion == 'ambas' else 0,
            'Direccion_Min': np.min(direccion_grad) if direccion == 'ambas' else 0,
            'Direccion_Max': np.max(direccion_grad) if direccion == 'ambas' else 0,
            
            # Estadísticas de bordes detectados
            'Pixeles_Borde': np.sum(bordes_binarios > 0),
            'Porcentaje_Bordes': (np.sum(bordes_binarios > 0) / bordes_binarios.size) * 100,
            'Densidad_Bordes': np.sum(bordes_binarios > 0) / (imagen_gris.shape[0] * imagen_gris.shape[1]),
            
            # Estadísticas de intensidad en bordes
            'Intensidad_Bordes_Media': np.mean(imagen_gris[bordes_binarios > 0]) if np.sum(bordes_binarios > 0) > 0 else 0,
            'Intensidad_Bordes_Std': np.std(imagen_gris[bordes_binarios > 0]) if np.sum(bordes_binarios > 0) > 0 else 0,
            
            # Análisis de conectividad
            'Num_Componentes_Conexas': Filtros._contar_componentes_conexas(bordes_binarios),
            
            # Características específicas de Sobel
            'Respuesta_Sobel_Media': np.mean(resultado_sobel),
            'Respuesta_Sobel_Max': np.max(resultado_sobel),
            'Respuesta_Sobel_Std': np.std(resultado_sobel),
            
            # Relación de aspecto y características geométricas
            'Relacion_Aspecto': imagen_gris.shape[1] / imagen_gris.shape[0],
            'Area_Total': imagen_gris.shape[0] * imagen_gris.shape[1]
        }
        
        # Guardar estadísticas si se solicita
        if guardar_estadisticas:
            Filtros._guardar_estadisticas_bordes(estadisticas, directorio_salida, nombre_archivo, 'sobel')
        
        return resultado_normalizado, estadisticas
    
    @staticmethod
    def _contar_componentes_conexas(imagen_binaria):
        """Cuenta el número de componentes conexas en una imagen binaria."""
        try:
            num_labels, _ = cv2.connectedComponents(imagen_binaria)
            return num_labels - 1  # Restar 1 porque el fondo también se cuenta
        except:
            return 0
    
    @staticmethod
    def _guardar_estadisticas_bordes(estadisticas, directorio_salida, nombre_archivo, metodo):
        """
        Guarda las estadísticas de bordes en archivos TXT y CSV.
        
        Args:
            estadisticas: Diccionario con estadísticas
            directorio_salida: Directorio donde guardar
            nombre_archivo: Nombre base del archivo
            metodo: 'canny' o 'sobel'
        """
        try:
            # Crear directorio si no existe
            os.makedirs(directorio_salida, exist_ok=True)
            
            # Generar nombre de archivo si no se proporciona
            if nombre_archivo is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"estadisticas_{metodo}_{timestamp}"
            
            # Guardar en TXT
            ruta_txt = os.path.join(directorio_salida, f"{nombre_archivo}.txt")
            with open(ruta_txt, 'w', encoding='utf-8') as f:
                f.write(f"ESTADÍSTICAS DE DETECCIÓN DE BORDES - {metodo.upper()}\n")
                f.write("=" * 60 + "\n\n")
                
                f.write("INFORMACIÓN GENERAL:\n")
                f.write("-" * 30 + "\n")
                for key in ['Metodo', 'Fecha_Analisis', 'Dimensiones_Imagen']:
                    if key in estadisticas:
                        f.write(f"{key.replace('_', ' ')}: {estadisticas[key]}\n")
                
                if metodo == 'canny':
                    f.write(f"Umbral Inferior: {estadisticas.get('Umbral_Inferior', 'N/A')}\n")
                    f.write(f"Umbral Superior: {estadisticas.get('Umbral_Superior', 'N/A')}\n")
                elif metodo == 'sobel':
                    f.write(f"Tamaño Kernel: {estadisticas.get('Kernel_Size', 'N/A')}\n")
                    f.write(f"Dirección: {estadisticas.get('Direccion_Analizada', 'N/A')}\n")
                    umbral_otsu = estadisticas.get('Umbral_Otsu_Calculado', 0)
                    if isinstance(umbral_otsu, (int, float)):
                        f.write(f"Umbral Otsu: {umbral_otsu:.2f}\n")
                    else:
                        f.write(f"Umbral Otsu: {umbral_otsu}\n")
                
                f.write("\nESTADÍSTICAS DE MAGNITUD DEL GRADIENTE:\n")
                f.write("-" * 40 + "\n")
                magnitud_keys = [k for k in estadisticas.keys() if k.startswith('Magnitud_')]
                for key in magnitud_keys:
                    value = estadisticas[key]
                    if isinstance(value, (int, float)):
                        f.write(f"{key.replace('_', ' ')}: {value:.4f}\n")
                    else:
                        f.write(f"{key.replace('_', ' ')}: {value}\n")
                
                f.write("\nESTADÍSTICAS DE BORDES DETECTADOS:\n")
                f.write("-" * 40 + "\n")
                border_keys = ['Pixeles_Borde', 'Porcentaje_Bordes', 'Densidad_Bordes', 
                              'Intensidad_Bordes_Media', 'Intensidad_Bordes_Std', 'Num_Componentes_Conexas']
                for key in border_keys:
                    if key in estadisticas:
                        value = estadisticas[key]
                        if isinstance(value, (int, float)):
                            f.write(f"{key.replace('_', ' ')}: {value:.4f}\n")
                        else:
                            f.write(f"{key.replace('_', ' ')}: {value}\n")
                
                if metodo == 'sobel':
                    f.write("\nESTADÍSTICAS DE GRADIENTES X e Y:\n")
                    f.write("-" * 40 + "\n")
                    grad_keys = [k for k in estadisticas.keys() if k.startswith('Gradiente_')]
                    for key in grad_keys:
                        value = estadisticas[key]
                        if isinstance(value, (int, float)):
                            f.write(f"{key.replace('_', ' ')}: {value:.4f}\n")
                        else:
                            f.write(f"{key.replace('_', ' ')}: {value}\n")
                
                f.write("\nESTADÍSTICAS DE DIRECCIÓN:\n")
                f.write("-" * 40 + "\n")
                dir_keys = [k for k in estadisticas.keys() if k.startswith('Direccion_')]
                for key in dir_keys:
                    value = estadisticas[key]
                    if isinstance(value, (int, float)):
                        f.write(f"{key.replace('_', ' ')}: {value:.4f}\n")
                    else:
                        f.write(f"{key.replace('_', ' ')}: {value}\n")
            
            print(f"Estadísticas TXT guardadas en: {ruta_txt}")
            
            # Guardar en CSV
            ruta_csv = os.path.join(directorio_salida, f"{nombre_archivo}.csv")
            df = pd.DataFrame([estadisticas])
            df.to_csv(ruta_csv, index=False, encoding='utf-8')
            print(f"Estadísticas CSV guardadas en: {ruta_csv}")
            
            return True
            
        except Exception as e:
            print(f"Error guardando estadísticas: {e}")
            return False