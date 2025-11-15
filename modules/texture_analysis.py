#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Análisis de Texturas para Identificación de Objetos en Tráfico Vehicular
==================================================================================

Sistema avanzado de análisis de texturas con capacidades de procesamiento por lotes,
estadísticas de primer y segundo orden, visualizaciones y exportación de resultados.

Características implementadas:
- Estadísticas de primer orden (Media, Varianza, Desviación, Entropía)
- Estadísticas de segundo orden basadas en GLCM
- Procesamiento por lotes de imágenes
- Exportación a CSV, TXT y visualizaciones
- Matrices de correlación entre características
- Análisis comparativo entre imágenes
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage import img_as_ubyte
from datetime import datetime
import seaborn as sns
import glob

class TextureAnalyzer:
    """Analizador avanzado de texturas para imágenes de tráfico vehicular."""
    
    def __init__(self, output_dir="./resultados"):
        """
        Inicializar el analizador de texturas.
        
        Args:
            output_dir (str): Directorio de salida para resultados
        """
        self.output_dir = output_dir
        self.results_dir = os.path.join(output_dir, "texture_analysis")
        os.makedirs(self.results_dir, exist_ok=True)
        self.current_results = []
        
    def convertir_a_gris(self, imagen):
        """
        Convierte una imagen a escala de grises si es necesario.
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            
        Returns:
            np.ndarray: Imagen en escala de grises
        """
        if len(imagen.shape) == 3:
            return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        return imagen
    
    def estadisticas_primer_orden(self, imagen):
        """
        Calcula estadísticas de primer orden: Media, Varianza, Desviación Estándar y Entropía.
        
        Args:
            imagen (np.ndarray): Imagen en escala de grises
            
        Returns:
            dict: Diccionario con estadísticas de primer orden
        """
        # 1. Media
        media = np.mean(imagen)
        
        # 2. Varianza
        varianza = np.var(imagen)
        
        # 3. Desviación estándar
        desviacion = np.std(imagen)
        
        # 4. Entropía
        # Preparar imagen para cálculo de entropía (evitando log(0))
        hist = cv2.calcHist([imagen], [0], None, [256], [0, 256])
        hist = hist / hist.sum()  # Normalizar para obtener probabilidades
        hist_no_ceros = hist[hist > 0]  # Eliminar probabilidades cero
        entropia = -np.sum(hist_no_ceros * np.log2(hist_no_ceros))
        
        return {
            'Media': media,
            'Varianza': varianza,
            'Desviación_Estándar': desviacion,
            'Entropía': entropia
        }
    
    def estadisticas_segundo_orden(self, imagen):
        """
        Calcula estadísticas de segundo orden basadas en la matriz de co-ocurrencia (GLCM).
        
        Args:
            imagen (np.ndarray): Imagen en escala de grises
            
        Returns:
            dict: Diccionario con estadísticas de segundo orden
        """
        # Asegurar que la imagen es de tipo uint8 para GLCM
        imagen = img_as_ubyte(imagen)
        
        # Calcular la matriz de co-ocurrencia con distancia 1 en dirección horizontal (0°)
        # También calculamos para otras direcciones (45°, 90°, 135°) para tener un análisis más completo
        distancias = [1]
        angulos = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 0°, 45°, 90°, 135°
        
        glcm = graycomatrix(imagen, 
                           distances=distancias, 
                           angles=angulos, 
                           symmetric=True, 
                           normed=True)
        
        # Calcular propiedades de la GLCM
        contraste = graycoprops(glcm, 'contrast').mean()
        disimilitud = graycoprops(glcm, 'dissimilarity').mean()
        homogeneidad = graycoprops(glcm, 'homogeneity').mean()
        energia = graycoprops(glcm, 'energy').mean()
        correlacion = graycoprops(glcm, 'correlation').mean()
        
        # Media y desviación estándar de la GLCM
        media_glcm = np.mean(glcm)
        desviacion_glcm = np.std(glcm)
        
        # Entropía de la GLCM
        # Aplanar la GLCM y eliminar valores cero para evitar log(0)
        glcm_flat = glcm.flatten()
        glcm_flat = glcm_flat[glcm_flat > 0]
        entropia_glcm = -np.sum(glcm_flat * np.log2(glcm_flat)) if len(glcm_flat) > 0 else 0
        
        return {
            'Contraste': contraste,
            'Disimilitud': disimilitud,
            'Homogeneidad': homogeneidad,
            'Energía': energia,
            'Correlación': correlacion,
            'Media_GLCM': media_glcm,
            'Desviación_Estándar_GLCM': desviacion_glcm,
            'Entropía_GLCM': entropia_glcm
        }
    
    def estadisticas_lbp(self, imagen, radio=1, n_puntos=8, metodo='uniform'):
        """
        Calcula estadísticas basadas en Local Binary Patterns (LBP).
        
        Args:
            imagen (np.ndarray): Imagen en escala de grises
            radio (int): Radio para calcular LBP
            n_puntos (int): Número de puntos de muestreo
            metodo (str): Método LBP ('uniform', 'default', 'ror', 'var')
            
        Returns:
            dict: Diccionario con estadísticas LBP
        """
        # Asegurar que la imagen es de tipo uint8
        if imagen.dtype != np.uint8:
            imagen = img_as_ubyte(imagen)
        
        # Calcular LBP
        lbp = local_binary_pattern(imagen, n_puntos, radio, method=metodo)
        
        # Calcular histograma de LBP
        if metodo == 'uniform':
            # Para uniform LBP, el número de bins es n_puntos + 2
            n_bins = n_puntos + 2
        else:
            # Para otros métodos, el número de bins es 2^n_puntos
            n_bins = 2 ** n_puntos
        
        hist_lbp, _ = np.histogram(lbp.flatten(), bins=n_bins, range=(0, n_bins))
        hist_lbp = hist_lbp.astype(float)
        
        # Normalizar histograma para obtener probabilidades
        hist_lbp /= (hist_lbp.sum() + 1e-10)  # Agregar epsilon para evitar división por cero
        
        # Calcular estadísticas del histograma LBP
        # 1. Entropía LBP
        hist_no_ceros = hist_lbp[hist_lbp > 0]
        entropia_lbp = -np.sum(hist_no_ceros * np.log2(hist_no_ceros)) if len(hist_no_ceros) > 0 else 0
        
        # 2. Uniformidad (Energy) del histograma LBP
        uniformidad_lbp = np.sum(hist_lbp ** 2)
        
        # 3. Contraste LBP (dispersión del histograma)
        media_lbp = np.sum(np.arange(len(hist_lbp)) * hist_lbp)
        contraste_lbp = np.sum(((np.arange(len(hist_lbp)) - media_lbp) ** 2) * hist_lbp)
        
        # 4. Homogeneidad LBP
        indices = np.arange(len(hist_lbp))
        homogeneidad_lbp = np.sum(hist_lbp / (1 + np.abs(indices - media_lbp)))
        
        # 5. Disimilitud LBP
        disimilitud_lbp = np.sum(np.abs(indices - media_lbp) * hist_lbp)
        
        # 6. Correlación LBP (momento de segundo orden)
        varianza_lbp = np.sum(((indices - media_lbp) ** 2) * hist_lbp)
        if varianza_lbp > 0:
            correlacion_lbp = np.sum(((indices - media_lbp) ** 2) * hist_lbp) / varianza_lbp
        else:
            correlacion_lbp = 0
        
        # 7. Estadísticas adicionales de la imagen LBP
        media_imagen_lbp = np.mean(lbp)
        varianza_imagen_lbp = np.var(lbp)
        desviacion_imagen_lbp = np.std(lbp)
        
        # 8. Porcentaje de patrones uniformes (solo para método uniform)
        if metodo == 'uniform':
            # Los patrones uniformes están en los primeros n_puntos bins
            patrones_uniformes = np.sum(hist_lbp[:n_puntos])
            porcentaje_uniformes = patrones_uniformes * 100
        else:
            porcentaje_uniformes = 0
        
        return {
            'LBP_Entropía': entropia_lbp,
            'LBP_Uniformidad': uniformidad_lbp,
            'LBP_Contraste': contraste_lbp,
            'LBP_Homogeneidad': homogeneidad_lbp,
            'LBP_Disimilitud': disimilitud_lbp,
            'LBP_Correlación': correlacion_lbp,
            'LBP_Media_Imagen': media_imagen_lbp,
            'LBP_Varianza_Imagen': varianza_imagen_lbp,
            'LBP_Desviación_Imagen': desviacion_imagen_lbp,
            'LBP_Porcentaje_Uniformes': porcentaje_uniformes,
            'LBP_Radio': radio,
            'LBP_Puntos': n_puntos,
            'LBP_Método': metodo
        }
    
    def analizar_lbp_completo(self, imagen, radios=[1, 2, 3], n_puntos=[8, 16, 24], metodos=['uniform', 'default']):
        """
        Realiza un análisis completo de LBP con diferentes parámetros.
        
        Args:
            imagen (np.ndarray): Imagen en escala de grises
            radios (list): Lista de radios a probar
            n_puntos (list): Lista de números de puntos a probar
            metodos (list): Lista de métodos LBP a probar
            
        Returns:
            dict: Resultados del análisis completo
        """
        resultados_completos = {}
        
        # Convertir a escala de grises si es necesario
        imagen_gris = self.convertir_a_gris(imagen)
        
        for radio in radios:
            for puntos in n_puntos:
                for metodo in metodos:
                    clave = f"LBP_R{radio}_P{puntos}_{metodo}"
                    try:
                        stats = self.estadisticas_lbp(imagen_gris, radio, puntos, metodo)
                        resultados_completos[clave] = stats
                    except Exception as e:
                        print(f"Error calculando {clave}: {str(e)}")
                        continue
        
        return resultados_completos
    
    def visualizar_lbp_imagen(self, imagen, radio=1, n_puntos=8, metodo='uniform'):
        """
        Visualiza la imagen LBP junto con la imagen original.
        
        Args:
            imagen (np.ndarray): Imagen original
            radio (int): Radio para LBP
            n_puntos (int): Número de puntos para LBP
            metodo (str): Método LBP
        """
        # Convertir a escala de grises
        imagen_gris = self.convertir_a_gris(imagen)
        
        # Calcular LBP
        lbp = local_binary_pattern(imagen_gris, n_puntos, radio, method=metodo)
        
        # Crear visualización
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Imagen original
        if len(imagen.shape) == 3:
            axes[0].imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
        else:
            axes[0].imshow(imagen, cmap='gray')
        axes[0].set_title('Imagen Original')
        axes[0].axis('off')
        
        # Imagen en escala de grises
        axes[1].imshow(imagen_gris, cmap='gray')
        axes[1].set_title('Escala de Grises')
        axes[1].axis('off')
        
        # Imagen LBP
        axes[2].imshow(lbp, cmap='hot')
        axes[2].set_title(f'LBP (R={radio}, P={n_puntos}, {metodo})')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Guardar visualización
        ruta_viz = os.path.join(self.results_dir, 'visualizacion_lbp.png')
        os.makedirs(os.path.dirname(ruta_viz), exist_ok=True)
        plt.savefig(ruta_viz, dpi=300, bbox_inches='tight')
        print(f"Visualización LBP guardada en {ruta_viz}")
        plt.show()
        
        # Mostrar histograma LBP
        self._mostrar_histograma_lbp(lbp, metodo, n_puntos)
    
    def _mostrar_histograma_lbp(self, lbp, metodo, n_puntos):
        """Muestra el histograma de la imagen LBP."""
        plt.figure(figsize=(10, 6))
        
        if metodo == 'uniform':
            n_bins = n_puntos + 2
        else:
            n_bins = min(2 ** n_puntos, 256)  # Limitar bins para evitar histogramas muy grandes
        
        plt.hist(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True, alpha=0.7)
        plt.xlabel('Valor LBP')
        plt.ylabel('Densidad de Probabilidad')
        plt.title(f'Histograma LBP ({metodo}, {n_puntos} puntos)')
        plt.grid(True, alpha=0.3)
        
        # Guardar histograma
        ruta_hist = os.path.join(self.results_dir, 'histograma_lbp.png')
        plt.savefig(ruta_hist, dpi=300, bbox_inches='tight')
        print(f"Histograma LBP guardado en {ruta_hist}")
        plt.show()
    
    def _calcular_skewness(self, imagen):
        """Calcular skewness de la imagen."""
        media = np.mean(imagen)
        std = np.std(imagen)
        if std == 0:
            return 0
        return np.mean(((imagen - media) / std) ** 3)
    
    def _calcular_kurtosis(self, imagen):
        """Calcular kurtosis de la imagen."""
        media = np.mean(imagen)
        std = np.std(imagen)
        if std == 0:
            return 0
        return np.mean(((imagen - media) / std) ** 4) - 3
    
    def _calcular_uniformidad_local(self, glcm):
        """Calcular índice de uniformidad local."""
        return np.sum(glcm ** 2)
    
    def _calcular_textura_direccional(self, glcm, angulos):
        """Calcular índice de textura direccional."""
        # Variabilidad entre diferentes direcciones
        valores_direccionales = []
        for i, angulo in enumerate(angulos):
            energia_direccional = np.sum(glcm[:, :, 0, i] ** 2)
            valores_direccionales.append(energia_direccional)
        return np.std(valores_direccionales)
    
    def analizar_regiones_vehiculares(self, imagen, mostrar_regiones=True):
        """
        Analiza diferentes regiones de la imagen que pueden contener vehículos.
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            mostrar_regiones (bool): Si mostrar las regiones analizadas
            
        Returns:
            dict: Resultados por región
        """
        # Convertir a escala de grises
        imagen_gris = self.convertir_a_gris(imagen)
        h, w = imagen_gris.shape
        
        # Definir regiones típicas donde aparecen vehículos
        regiones = {
            'Centro': imagen_gris[h//4:3*h//4, w//4:3*w//4],
            'Inferior': imagen_gris[2*h//3:h, :],
            'Superior': imagen_gris[:h//3, :],
            'Izquierda': imagen_gris[:, :w//2],
            'Derecha': imagen_gris[:, w//2:],
            'Completa': imagen_gris
        }
        
        resultados_regiones = {}
        
        for nombre_region, region in regiones.items():
            if region.size > 0:
                stats_1er = self.estadisticas_primer_orden(region)
                stats_2do = self.estadisticas_segundo_orden(region)
                stats_lbp = self.estadisticas_lbp(region)
                
                resultados_regiones[nombre_region] = {
                    **stats_1er,
                    **stats_2do,
                    **stats_lbp,
                    'Tamaño_Region': region.shape
                }
        
        if mostrar_regiones:
            self.visualizar_regiones(imagen, regiones)
        
        return resultados_regiones
    
    def procesar_imagen_completa(self, imagen_input, nombre_imagen=None):
        """
        Procesa una imagen completa y extrae todas las características de textura.
        
        Args:
            imagen_input (str o np.ndarray): Ruta a la imagen o array numpy de la imagen
            nombre_imagen (str): Nombre personalizado para la imagen
            
        Returns:
            dict: Resultados completos del análisis
        """
        try:
            # Determinar si es ruta o array
            if isinstance(imagen_input, str):
                # Es una ruta de archivo
                imagen_path = imagen_input
                imagen = cv2.imread(imagen_path)
                if imagen is None:
                    raise ValueError(f"No se pudo cargar la imagen: {imagen_path}")
                
                # Nombre de la imagen
                if nombre_imagen is None:
                    nombre_imagen = os.path.basename(imagen_path)
            else:
                # Es un array numpy
                imagen = imagen_input
                imagen_path = nombre_imagen if nombre_imagen else "imagen_array"
                
                # Nombre de la imagen
                if nombre_imagen is None:
                    nombre_imagen = "imagen_procesada"
            
            print(f"Analizando texturas en: {nombre_imagen}")
            
            # Análisis de la imagen completa
            imagen_gris = self.convertir_a_gris(imagen)
            stats_1er_orden = self.estadisticas_primer_orden(imagen_gris)
            stats_2do_orden = self.estadisticas_segundo_orden(imagen_gris)
            stats_lbp = self.estadisticas_lbp(imagen_gris)
            
            # Análisis por regiones
            stats_regiones = self.analizar_regiones_vehiculares(imagen, mostrar_regiones=False)
            
            # Combinar resultados
            resultado_completo = {
                'Imagen': nombre_imagen,
                'Ruta': imagen_path,
                'Dimensiones': imagen.shape,
                'Fecha_Analisis': datetime.now().isoformat(),
                **stats_1er_orden,
                **stats_2do_orden,
                **stats_lbp,
                'Regiones': stats_regiones
            }
            
            self.current_results.append(resultado_completo)
            
            print(f"Análisis completado para: {nombre_imagen}")
            return resultado_completo
            
        except Exception as e:
            print(f"Error al procesar {imagen_path}: {str(e)}")
            return None
    
    def procesar_lote_imagenes(self, carpeta_imagenes, patron="*.jpg,*.png,*.tif"):
        """
        Procesa múltiples imágenes en lote.
        
        Args:
            carpeta_imagenes (str): Ruta a la carpeta con imágenes
            patron (str): Patrones de archivos separados por comas
            
        Returns:
            list: Lista de resultados
        """
        import glob
        
        # Expandir patrones
        patrones = patron.split(',')
        archivos = []
        
        for p in patrones:
            patron_ruta = os.path.join(carpeta_imagenes, p.strip())
            archivos.extend(glob.glob(patron_ruta))
        
        if not archivos:
            print(f"No se encontraron imágenes en {carpeta_imagenes}")
            return []
        
        print(f"Procesando {len(archivos)} imágenes...")
        
        resultados_lote = []
        for i, archivo in enumerate(archivos, 1):
            print(f"Progreso: {i}/{len(archivos)}")
            resultado = self.procesar_imagen_completa(archivo)
            if resultado:
                resultados_lote.append(resultado)
        
        self.current_results.extend(resultados_lote)
        return resultados_lote
    
    def guardar_resultados(self, formato='csv'):
        """
        Guarda los resultados del análisis.
        
        Args:
            formato (str): Formato de salida ('csv', 'json', 'txt')
        """
        if not self.current_results:
            print("No hay resultados para guardar.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if formato.lower() == 'csv':
            self._guardar_csv(timestamp)
        elif formato.lower() == 'json':
            self._guardar_json(timestamp)
        elif formato.lower() == 'txt':
            self._guardar_txt(timestamp)
        else:
            print(f"Formato no soportado: {formato}")
    
    def _guardar_csv(self, timestamp):
        """Guardar resultados en formato CSV."""
        # Preparar datos para CSV (sin regiones por simplicidad)
        datos_csv = []
        for resultado in self.current_results:
            fila = {k: v for k, v in resultado.items() if k != 'Regiones'}
            datos_csv.append(fila)
        
        df = pd.DataFrame(datos_csv)
        archivo_csv = os.path.join(self.results_dir, f'texturas_vehicular_{timestamp}.csv')
        os.makedirs(os.path.dirname(archivo_csv), exist_ok=True)
        df.to_csv(archivo_csv, index=False)
        print(f"Resultados CSV guardados: {archivo_csv}")
        return archivo_csv
    
    def _guardar_json(self, timestamp):
        """Guardar resultados en formato JSON."""
        import json
        archivo_json = os.path.join(self.results_dir, f'texturas_vehicular_{timestamp}.json')
        with open(archivo_json, 'w', encoding='utf-8') as f:
            json.dump(self.current_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"Resultados JSON guardados: {archivo_json}")
        return archivo_json
    
    def _guardar_txt(self, timestamp):
        """Guardar resultados en formato texto."""
        archivo_txt = os.path.join(self.results_dir, f'texturas_vehicular_{timestamp}.txt')
        
        with open(archivo_txt, 'w', encoding='utf-8') as f:
            f.write("ANÁLISIS DE TEXTURAS - IDENTIFICACIÓN VEHICULAR\n")
            f.write("=" * 60 + "\n\n")
            
            for resultado in self.current_results:
                f.write(f"IMAGEN: {resultado['Imagen']}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Ruta: {resultado['Ruta']}\n")
                f.write(f"Dimensiones: {resultado['Dimensiones']}\n")
                f.write(f"Fecha: {resultado['Fecha_Analisis']}\n\n")
                
                # ESTADÍSTICAS DE PRIMER ORDEN (claves correctas)
                f.write("ESTADÍSTICAS DE PRIMER ORDEN:\n")
                primer_orden_keys = ['Media', 'Varianza', 'Desviación_Estándar', 'Entropía']
                for k in primer_orden_keys:
                    if k in resultado:
                        f.write(f"  {k}: {resultado[k]:.6f}\n")
                
                # ESTADÍSTICAS DE SEGUNDO ORDEN (claves correctas)
                f.write("\nESTADÍSTICAS DE SEGUNDO ORDEN:\n")
                segundo_orden_keys = ['Contraste', 'Disimilitud', 'Homogeneidad', 'Energía', 
                                    'Correlación', 'Media_GLCM', 'Desviación_Estándar_GLCM', 'Entropía_GLCM']
                for k in segundo_orden_keys:
                    if k in resultado:
                        f.write(f"  {k}: {resultado[k]:.6f}\n")
                
                # ESTADÍSTICAS LBP (LOCAL BINARY PATTERNS) - NUEVA SECCIÓN
                f.write("\nESTADÍSTICAS LBP (LOCAL BINARY PATTERNS):\n")
                lbp_keys = ['LBP_Entropía', 'LBP_Uniformidad', 'LBP_Contraste', 'LBP_Homogeneidad',
                           'LBP_Disimilitud', 'LBP_Correlación', 'LBP_Media_Imagen', 'LBP_Varianza_Imagen',
                           'LBP_Desviación_Imagen', 'LBP_Porcentaje_Uniformes', 'LBP_Radio', 'LBP_Puntos', 'LBP_Método']
                for k in lbp_keys:
                    if k in resultado:
                        if k in ['LBP_Radio', 'LBP_Puntos']:
                            f.write(f"  {k}: {resultado[k]}\n")  # Valores enteros sin decimales
                        elif k == 'LBP_Método':
                            f.write(f"  {k}: {resultado[k]}\n")  # Valor string
                        else:
                            f.write(f"  {k}: {resultado[k]:.6f}\n")  # Valores con decimales
                
                f.write("\n" + "=" * 60 + "\n\n")
        
        print(f"Resultados TXT guardados: {archivo_txt}")
        return archivo_txt
    
    def guardar_resultados_csv(self):
        """Método específico para guardar solo en formato CSV."""
        if not self.current_results:
            print("No hay resultados para guardar.")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self._guardar_csv(timestamp)
    
    def guardar_resultados_txt(self):
        """Método específico para guardar solo en formato TXT."""
        if not self.current_results:
            print("No hay resultados para guardar.")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self._guardar_txt(timestamp)
    
    def guardar_resultados_json(self):
        """Método específico para guardar solo en formato JSON."""
        if not self.current_results:
            print("No hay resultados para guardar.")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self._guardar_json(timestamp)
    
    def visualizar_regiones(self, imagen, regiones):
        """Visualizar las regiones analizadas."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Convertir a RGB para matplotlib
        if len(imagen.shape) == 3:
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        else:
            imagen_rgb = imagen
        
        nombres_regiones = list(regiones.keys())
        
        for i, (nombre, region) in enumerate(regiones.items()):
            if i < len(axes):
                if len(region.shape) == 3:
                    axes[i].imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
                else:
                    axes[i].imshow(region, cmap='gray')
                axes[i].set_title(f'Región: {nombre}')
                axes[i].axis('off')
        
        plt.tight_layout()
        ruta_imagen = os.path.join(self.results_dir, 'regiones_analisis.png')
        os.makedirs(os.path.dirname(ruta_imagen), exist_ok=True)
        plt.savefig(ruta_imagen, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualizar_resultados(self, top_n=5):
        """
        Visualiza los resultados del análisis de texturas.
        
        Args:
            top_n (int): Número de imágenes principales a mostrar
        """
        if not self.current_results:
            print("No hay resultados para visualizar.")
            return
        
        # Preparar datos para visualización
        datos_viz = []
        for resultado in self.current_results[:top_n]:
            fila = {k: v for k, v in resultado.items() 
                   if isinstance(v, (int, float)) and k != 'Dimensiones'}
            fila['Imagen'] = resultado['Imagen']
            datos_viz.append(fila)
        
        df_viz = pd.DataFrame(datos_viz)
        
        # Crear visualizaciones
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Comparación de estadísticas de primer orden
        ax1 = axes[0, 0]
        caracteristicas_1er = ['Media', 'Varianza', 'Desviacion_Estandar', 'Entropia']
        x_pos = np.arange(len(df_viz))
        width = 0.2
        
        for i, caract in enumerate(caracteristicas_1er):
            if caract in df_viz.columns:
                ax1.bar(x_pos + i*width, df_viz[caract], width, label=caract)
        
        ax1.set_xlabel('Imágenes')
        ax1.set_ylabel('Valores')
        ax1.set_title('Estadísticas de Primer Orden')
        ax1.set_xticks(x_pos + width*1.5)
        ax1.set_xticklabels([img[:10] + '...' if len(img) > 10 else img 
                            for img in df_viz['Imagen']], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Comparación de estadísticas de segundo orden
        ax2 = axes[0, 1]
        caracteristicas_2do = ['Contraste', 'Homogeneidad', 'Energia', 'Correlacion']
        
        for i, caract in enumerate(caracteristicas_2do):
            if caract in df_viz.columns:
                ax2.bar(x_pos + i*width, df_viz[caract], width, label=caract)
        
        ax2.set_xlabel('Imágenes')
        ax2.set_ylabel('Valores')
        ax2.set_title('Estadísticas de Segundo Orden (GLCM)')
        ax2.set_xticks(x_pos + width*1.5)
        ax2.set_xticklabels([img[:10] + '...' if len(img) > 10 else img 
                            for img in df_viz['Imagen']], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Comparación de estadísticas LBP
        ax3 = axes[0, 2]
        caracteristicas_lbp = ['LBP_Entropía', 'LBP_Uniformidad', 'LBP_Contraste', 'LBP_Homogeneidad']
        
        for i, caract in enumerate(caracteristicas_lbp):
            if caract in df_viz.columns:
                ax3.bar(x_pos + i*width, df_viz[caract], width, label=caract.replace('LBP_', ''))
        
        ax3.set_xlabel('Imágenes')
        ax3.set_ylabel('Valores')
        ax3.set_title('Estadísticas LBP (Local Binary Patterns)')
        ax3.set_xticks(x_pos + width*1.5)
        ax3.set_xticklabels([img[:10] + '...' if len(img) > 10 else img 
                            for img in df_viz['Imagen']], rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Mapa de calor de correlaciones
        ax4 = axes[1, 0]
        # 4. Mapa de calor de correlaciones
        ax4 = axes[1, 0]
        columnas_numericas = df_viz.select_dtypes(include=[np.number]).columns
        if len(columnas_numericas) > 1:
            corr_matrix = df_viz[columnas_numericas].corr()
            im = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax4.set_xticks(range(len(columnas_numericas)))
            ax4.set_yticks(range(len(columnas_numericas)))
            ax4.set_xticklabels([col[:8] + '...' if len(col) > 8 else col 
                                for col in columnas_numericas], rotation=90)
            ax4.set_yticklabels([col[:8] + '...' if len(col) > 8 else col 
                                for col in columnas_numericas])
            ax4.set_title('Matriz de Correlación')
            plt.colorbar(im, ax=ax4)
        
        # 5. Distribución de entropía vs contraste
        ax5 = axes[1, 1]
        if 'Entropia' in df_viz.columns and 'Contraste' in df_viz.columns:
            scatter = ax5.scatter(df_viz['Entropia'], df_viz['Contraste'], 
                                 c=range(len(df_viz)), cmap='viridis', s=100)
            ax5.set_xlabel('Entropía')
            ax5.set_ylabel('Contraste')
            ax5.set_title('Entropía vs Contraste')
            ax5.grid(True, alpha=0.3)
            
            # Agregar etiquetas de imágenes
            for i, txt in enumerate(df_viz['Imagen']):
                ax5.annotate(txt[:5], (df_viz['Entropia'].iloc[i], df_viz['Contraste'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 6. Comparación LBP vs GLCM
        ax6 = axes[1, 2]
        if 'LBP_Entropía' in df_viz.columns and 'Entropía_GLCM' in df_viz.columns:
            scatter = ax6.scatter(df_viz['LBP_Entropía'], df_viz['Entropía_GLCM'], 
                                 c=range(len(df_viz)), cmap='plasma', s=100)
            ax6.set_xlabel('Entropía LBP')
            ax6.set_ylabel('Entropía GLCM')
            ax6.set_title('Comparación: LBP vs GLCM')
            ax6.grid(True, alpha=0.3)
            
            # Agregar etiquetas de imágenes
            for i, txt in enumerate(df_viz['Imagen']):
                ax6.annotate(txt[:5], (df_viz['LBP_Entropía'].iloc[i], df_viz['Entropía_GLCM'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        archivo_viz = os.path.join(self.results_dir, 'visualizacion_texturas.png')
        os.makedirs(os.path.dirname(archivo_viz), exist_ok=True)
        plt.savefig(archivo_viz, dpi=300, bbox_inches='tight')
        print(f"Visualización guardada: {archivo_viz}")
        plt.show()
    
    def generar_reporte_resumen(self):
        """Genera un reporte resumen del análisis."""
        if not self.current_results:
            print("No hay resultados para el reporte.")
            return
        
        print("\nREPORTE RESUMEN - ANÁLISIS DE TEXTURAS VEHICULARES")
        print("=" * 60)
        print(f"Total de imágenes analizadas: {len(self.current_results)}")
        
        # Estadísticas generales
        caracteristicas_numericas = []
        for resultado in self.current_results:
            valores = [v for k, v in resultado.items() 
                      if isinstance(v, (int, float)) and k not in ['Dimensiones']]
            caracteristicas_numericas.extend(valores)
        
        if caracteristicas_numericas:
            print(f"Rango de valores: {min(caracteristicas_numericas):.3f} - {max(caracteristicas_numericas):.3f}")
            print(f"Promedio general: {np.mean(caracteristicas_numericas):.3f}")
            print(f"Desviación estándar: {np.std(caracteristicas_numericas):.3f}")
        
        # Top imágenes por entropía (mayor complejidad de textura)
        imagenes_entropia = [(r['Imagen'], r.get('Entropia', 0)) for r in self.current_results]
        imagenes_entropia.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTOP 3 - MAYOR COMPLEJIDAD DE TEXTURA (Entropía):")
        for i, (imagen, entropia) in enumerate(imagenes_entropia[:3], 1):
            print(f"   {i}. {imagen}: {entropia:.4f}")
        
        # Top imágenes por contraste
        imagenes_contraste = [(r['Imagen'], r.get('Contraste', 0)) for r in self.current_results]
        imagenes_contraste.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTOP 3 - MAYOR CONTRASTE (Variación local):")
        for i, (imagen, contraste) in enumerate(imagenes_contraste[:3], 1):
            print(f"   {i}. {imagen}: {contraste:.4f}")
        
        print("\n" + "=" * 60)

    def procesar_imagen_individual(self, ruta_imagen):
        """Procesa una imagen individual y devuelve sus estadísticas de textura."""
        # Leer la imagen
        imagen = cv2.imread(ruta_imagen)
        
        if imagen is None:
            print(f"Error: No se pudo cargar la imagen {ruta_imagen}")
            return None
        
        # Convertir a escala de grises si es necesario
        imagen_gris = self.convertir_a_gris(imagen)
        
        # Calcular estadísticas
        stats_1er_orden = self.estadisticas_primer_orden(imagen_gris)
        stats_2do_orden = self.estadisticas_segundo_orden(imagen_gris)
        stats_lbp = self.estadisticas_lbp(imagen_gris)
        
        # Combinar resultados
        nombre_imagen = os.path.basename(ruta_imagen)
        resultados = {'Imagen': nombre_imagen}
        resultados.update(stats_1er_orden)
        resultados.update(stats_2do_orden)
        resultados.update(stats_lbp)
        
        return resultados

    def procesar_carpeta(self, ruta_carpeta, patron="*.png,*.jpg,*.jpeg,*.tif,*.tiff"):
        """Procesa todas las imágenes en una carpeta que coincidan con los patrones dados."""
        # Expandir los patrones separados por comas
        patrones = patron.split(',')
        archivos = []
        
        # Obtener todos los archivos que coinciden con los patrones
        for p in patrones:
            patron_ruta = os.path.join(ruta_carpeta, p.strip())
            archivos.extend(glob.glob(patron_ruta))
        
        if not archivos:
            print(f"No se encontraron imágenes en {ruta_carpeta} con el patrón {patron}")
            return None
        
        # Procesar cada imagen
        resultados = []
        # String para el archivo TXT
        txt_resultados = "ANÁLISIS DE TEXTURAS - RESULTADOS\n"
        txt_resultados += "=" * 50 + "\n\n"
        
        for archivo in archivos:
            print(f"Procesando {archivo}...")
            resultado = self.procesar_imagen_individual(archivo)
            if resultado:
                resultados.append(resultado)
                self.current_results.append(resultado)
                
                # Agregar resultados al TXT de forma organizada
                txt_resultados += f"IMAGEN: {os.path.basename(archivo)}\n"
                txt_resultados += "-" * 30 + "\n"
                txt_resultados += "ESTADÍSTICAS DE PRIMER ORDEN:\n"
                for k, v in {k: resultado[k] for k in ['Media', 'Varianza', 'Desviación_Estándar', 'Entropía']}.items():
                    txt_resultados += f"  {k}: {v:.6f}\n"
                
                txt_resultados += "\nESTADÍSTICAS DE SEGUNDO ORDEN:\n"
                for k, v in {k: resultado[k] for k in ['Contraste', 'Disimilitud', 'Homogeneidad', 
                                                      'Energía', 'Correlación', 'Media_GLCM',
                                                      'Desviación_Estándar_GLCM', 'Entropía_GLCM']}.items():
                    txt_resultados += f"  {k}: {v:.6f}\n"
                
                txt_resultados += "\nESTADÍSTICAS LBP (LOCAL BINARY PATTERNS):\n"
                lbp_keys = [k for k in resultado.keys() if k.startswith('LBP_')]
                for k in lbp_keys:
                    if k not in ['LBP_Radio', 'LBP_Puntos', 'LBP_Método']:
                        txt_resultados += f"  {k}: {resultado[k]:.6f}\n"
                    else:
                        txt_resultados += f"  {k}: {resultado[k]}\n"
                
                txt_resultados += "\n" + "=" * 50 + "\n\n"
        
        # Crear DataFrame con los resultados
        if resultados:
            df = pd.DataFrame(resultados)
            
            # Guardar resultados en CSV
            ruta_salida_csv = os.path.join(self.results_dir, 'resultados_texturas.csv')
            os.makedirs(os.path.dirname(ruta_salida_csv), exist_ok=True)
            df.to_csv(ruta_salida_csv, index=False)
            print(f"Resultados CSV guardados en {ruta_salida_csv}")
            
            # Guardar resultados en TXT
            ruta_salida_txt = os.path.join(self.results_dir, 'resultados_texturas.txt')
            with open(ruta_salida_txt, 'w', encoding='utf-8') as f:
                f.write(txt_resultados)
            print(f"Resultados TXT guardados en {ruta_salida_txt}")
            
            return df
        
        return None

    def visualizar_resultados_batch(self, df, ruta_carpeta=None):
        """Genera visualizaciones de los resultados del procesamiento por lotes."""
        if df is None or len(df) == 0:
            print("No hay datos para visualizar")
            return
        
        # 1. Gráfico de barras para características seleccionadas de segundo orden
        plt.figure(figsize=(18, 12))
        
        # Subplot 1: Características de segundo orden (GLCM)
        plt.subplot(2, 2, 1)
        caracteristicas = ['Contraste', 'Homogeneidad', 'Energía', 'Correlación']
        
        plt.subplot(2, 1, 1)
        for i, caract in enumerate(caracteristicas):
            plt.bar(np.arange(len(df)) + i*0.2, df[caract], width=0.2, label=caract)
        
        plt.xticks(np.arange(len(df)) + 0.3, df['Imagen'], rotation=45)
        plt.title('Características de Segundo Orden')
        plt.legend()
        plt.tight_layout()
        
        # 2. Gráfico de características de primer orden
        plt.subplot(2, 1, 2)
        primer_orden = ['Media', 'Varianza', 'Desviación_Estándar', 'Entropía']
        
        for i, caract in enumerate(primer_orden):
            plt.bar(np.arange(len(df)) + i*0.2, df[caract], width=0.2, label=caract)
        
        plt.xticks(np.arange(len(df)) + 0.3, df['Imagen'], rotation=45)
        plt.title('Características de Primer Orden')
        plt.legend()
        
        plt.tight_layout()
        ruta_grafico = os.path.join(self.results_dir, 'comparacion_texturas.png')
        os.makedirs(os.path.dirname(ruta_grafico), exist_ok=True)
        plt.savefig(ruta_grafico)
        print(f"Gráfico comparativo guardado en {ruta_grafico}")
        plt.show()
        
        # 2.5. Gráfico adicional para características LBP
        if any(col.startswith('LBP_') for col in df.columns):
            plt.figure(figsize=(16, 10))
            
            # Subplot 1: Características principales LBP
            plt.subplot(2, 2, 1)
            caracteristicas_lbp = ['LBP_Entropía', 'LBP_Uniformidad', 'LBP_Contraste', 'LBP_Homogeneidad']
            
            for i, caract in enumerate(caracteristicas_lbp):
                if caract in df.columns:
                    plt.bar(np.arange(len(df)) + i*0.2, df[caract], width=0.18, 
                            label=caract.replace('LBP_', ''))
            
            plt.xticks(np.arange(len(df)) + 0.3, df['Imagen'], rotation=45)
            plt.title('Características Principales LBP')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Subplot 2: Estadísticas de imagen LBP
            plt.subplot(2, 2, 2)
            stats_lbp = ['LBP_Media_Imagen', 'LBP_Varianza_Imagen', 'LBP_Desviación_Imagen']
            
            for i, caract in enumerate(stats_lbp):
                if caract in df.columns:
                    plt.bar(np.arange(len(df)) + i*0.25, df[caract], width=0.22, 
                            label=caract.replace('LBP_', '').replace('_', ' '))
            
            plt.xticks(np.arange(len(df)) + 0.25, df['Imagen'], rotation=45)
            plt.title('Estadísticas de Imagen LBP')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Subplot 3: Comparación de entropías
            plt.subplot(2, 2, 3)
            entropias = []
            labels_entropias = []
            
            if 'Entropía' in df.columns:
                entropias.append(df['Entropía'])
                labels_entropias.append('Primer Orden')
            if 'Entropía_GLCM' in df.columns:
                entropias.append(df['Entropía_GLCM'])
                labels_entropias.append('GLCM')
            if 'LBP_Entropía' in df.columns:
                entropias.append(df['LBP_Entropía'])
                labels_entropias.append('LBP')
            
            for i, entropia in enumerate(entropias):
                plt.bar(np.arange(len(df)) + i*0.25, entropia, width=0.22, label=labels_entropias[i])
            
            plt.xticks(np.arange(len(df)) + 0.25, df['Imagen'], rotation=45)
            plt.title('Comparación de Entropías')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Subplot 4: Porcentaje de patrones uniformes
            plt.subplot(2, 2, 4)
            if 'LBP_Porcentaje_Uniformes' in df.columns:
                plt.bar(np.arange(len(df)), df['LBP_Porcentaje_Uniformes'], color='skyblue')
                plt.xticks(np.arange(len(df)), df['Imagen'], rotation=45)
                plt.title('Porcentaje de Patrones Uniformes LBP')
                plt.ylabel('Porcentaje (%)')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            ruta_grafico_lbp = os.path.join(self.results_dir, 'analisis_lbp.png')
            plt.savefig(ruta_grafico_lbp, dpi=300, bbox_inches='tight')
            print(f"Análisis LBP guardado en {ruta_grafico_lbp}")
            plt.show()
        
        # 3. Matriz de correlación entre todas las características
        plt.figure(figsize=(12, 10))
        caracteristicas_todas = df.columns.drop('Imagen')
        corr = df[caracteristicas_todas].corr()
        
        plt.imshow(corr, cmap='coolwarm')
        plt.colorbar()
        plt.xticks(range(len(caracteristicas_todas)), caracteristicas_todas, rotation=90)
        plt.yticks(range(len(caracteristicas_todas)), caracteristicas_todas)
        plt.title('Matriz de Correlación entre Características')
        
        for i in range(len(caracteristicas_todas)):
            for j in range(len(caracteristicas_todas)):
                plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', 
                         color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black')
        
        plt.tight_layout()
        ruta_matriz = os.path.join(self.results_dir, 'matriz_correlacion.png')
        os.makedirs(os.path.dirname(ruta_matriz), exist_ok=True)
        plt.savefig(ruta_matriz)
        print(f"Matriz de correlación guardada en {ruta_matriz}")
        plt.show()

# Función de utilidad para uso directo - imagen individual
def analizar_imagen_vehicular(imagen_path, output_dir="./resultados"):
    """
    Función de conveniencia para analizar una sola imagen.
    
    Args:
        imagen_path (str): Ruta a la imagen
        output_dir (str): Directorio de salida
        
    Returns:
        dict: Resultados del análisis
    """
    analyzer = TextureAnalyzer(output_dir)
    resultado = analyzer.procesar_imagen_completa(imagen_path)
    if resultado:
        analyzer.visualizar_resultados()
        analyzer.guardar_resultados('csv')
        analyzer.generar_reporte_resumen()
    return resultado

# Función de utilidad para procesamiento por lotes
def analizar_carpeta_texturas(carpeta_imagenes, output_dir="./resultados", patron="*.png,*.jpg,*.jpeg,*.tif,*.tiff"):
    """
    Función de conveniencia para analizar todas las imágenes en una carpeta.
    
    Args:
        carpeta_imagenes (str): Ruta a la carpeta de imágenes
        output_dir (str): Directorio de salida
        patron (str): Patrones de archivos a procesar
        
    Returns:
        DataFrame: Resultados del análisis por lotes
    """
    # Verificar que la carpeta existe
    if not os.path.exists(carpeta_imagenes):
        print(f"La carpeta {carpeta_imagenes} no existe")
        return None
    
    analyzer = TextureAnalyzer(output_dir)
    
    # Procesar las imágenes
    resultados = analyzer.procesar_carpeta(carpeta_imagenes, patron)
    
    # Visualizar resultados si existen
    if resultados is not None:
        analyzer.visualizar_resultados_batch(resultados, carpeta_imagenes)
        
        # Mostrar tabla de resultados en consola
        print("\nRESULTADOS ESTADÍSTICOS DE TEXTURA:")
        print("=" * 80)
        print(resultados)
    else:
        print("No se pudieron obtener resultados")
    
    return resultados