#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidador de Descriptores de Características
==============================================

Script para consolidar automáticamente todos los descriptores de características
extraídos por diferentes algoritmos (SURF, ORB, SIFT, HOG, KAZE, etc.) en una
hoja de cálculo unificada.

Cada fila representa una imagen analizada y cada columna una característica específica.

Autor: Sistema de Detección Vehicular
Fecha: Octubre 2025
"""

import pandas as pd
import numpy as np
import os
import glob
import json
from datetime import datetime
import csv
import re

class ConsolidadorDescriptores:
    """
    Clase para consolidar descriptores de características de múltiples algoritmos
    en una estructura unificada tipo hoja de cálculo.
    """
    
    def __init__(self, directorio_resultados="./resultados_deteccion"):
        """
        Inicializa el consolidador.
        
        Args:
            directorio_resultados: Directorio donde están los archivos de resultados
        """
        self.directorio_resultados = directorio_resultados
        self.datos_consolidados = []
        self.columnas_finales = []
        
        # Definir estructura de columnas base
        self.columnas_base = [
            'imagen', 'algoritmo_extraccion', 'dimensiones_imagen', 
            'ancho_px', 'alto_px', 'area_total_px', 'fecha_analisis'
        ]
        
        # Columnas específicas por tipo de algoritmo
        self.columnas_puntos_clave = [
            'num_puntos_clave', 'dimension_descriptor', 'densidad_puntos_clave',
            'respuesta_detector_promedio', 'respuesta_detector_max', 'respuesta_detector_std'
        ]
        
        self.columnas_descriptores_continuos = [
            'valor_min_descriptores', 'valor_max_descriptores', 'promedio_descriptores',
            'mediana_descriptores', 'desviacion_std_descriptores', 'entropia_descriptores',
            'diversidad_descriptores', 'varianza_descriptores'
        ]
        
        self.columnas_descriptores_binarios = [
            'hamming_distance_promedio', 'hamming_distance_std', 'hamming_distance_mediana',
            'ratio_bits_activos', 'entropia_binaria', 'unicidad_descriptores'
        ]
        
        self.columnas_textura = [
            'homogeneidad_glcm', 'contraste_glcm', 'correlacion_glcm', 'energia_glcm',
            'entropia_glcm', 'varianza_local', 'uniformidad_local', 'momento_angular'
        ]
        
        self.columnas_bordes = [
            'magnitud_borde_promedio', 'magnitud_borde_max', 'magnitud_borde_std',
            'densidad_bordes', 'coherencia_bordes', 'orientacion_dominante',
            'energia_gradiente', 'longitud_contorno_total'
        ]
        
        self.columnas_geometricas = [
            'area_forma', 'perimetro_forma', 'compacidad', 'excentricidad',
            'solidez', 'convexidad', 'momento_hu_1', 'momento_hu_2', 'momento_hu_3',
            'centroide_x', 'centroide_y', 'eje_mayor', 'eje_menor', 'angulo_orientacion'
        ]
        
        self.columnas_rendimiento = [
            'tiempo_procesamiento_ms', 'memoria_utilizada_mb', 'eficiencia_computacional',
            'calidad_descriptores', 'discriminabilidad', 'repeatabilidad'
        ]
        
    def buscar_archivos_resultados(self):
        """
        Busca todos los archivos de resultados en el directorio especificado.
        
        Returns:
            dict: Diccionario con archivos organizados por tipo
        """
        archivos = {
            'surf_estadisticas': [],
            'orb_estadisticas': [],
            'sift_estadisticas': [],
            'hog_resultados': [],
            'kaze_resultados': [],
            'freak_resultados': [],
            'akaze_resultados': [],
            'texture_analysis': [],
            'hough_analysis': [],
            'reportes_json': []
        }
        
        # Buscar archivos SURF/ORB
        surf_orb_dir = os.path.join(self.directorio_resultados, 'surf_orb_analysis')
        if os.path.exists(surf_orb_dir):
            archivos['surf_estadisticas'] = glob.glob(os.path.join(surf_orb_dir, 'surf_estadisticas_*.csv'))
            archivos['orb_estadisticas'] = glob.glob(os.path.join(surf_orb_dir, 'orb_estadisticas_*.csv'))
        
        # Buscar archivos HOG/KAZE
        hog_kaze_dir = os.path.join(self.directorio_resultados, 'hog_kaze_analysis')
        if os.path.exists(hog_kaze_dir):
            archivos['hog_resultados'] = glob.glob(os.path.join(hog_kaze_dir, '*hog*.csv'))
            archivos['kaze_resultados'] = glob.glob(os.path.join(hog_kaze_dir, '*kaze*.csv'))
        
        # Buscar archivos de análisis avanzado
        advanced_dir = os.path.join(self.directorio_resultados, 'advanced_analysis')
        if os.path.exists(advanced_dir):
            archivos['freak_resultados'] = glob.glob(os.path.join(advanced_dir, '*freak*.csv'))
            archivos['akaze_resultados'] = glob.glob(os.path.join(advanced_dir, '*akaze*.csv'))
        
        # Buscar archivos de textura
        texture_dir = os.path.join(self.directorio_resultados, 'texture_analysis')
        if os.path.exists(texture_dir):
            archivos['texture_analysis'] = glob.glob(os.path.join(texture_dir, '*.csv'))
        
        # Buscar archivos Hough
        hough_dir = os.path.join(self.directorio_resultados, 'hough_analysis')
        if os.path.exists(hough_dir):
            archivos['hough_analysis'] = glob.glob(os.path.join(hough_dir, '**/*.csv'), recursive=True)
        
        # Buscar reportes JSON
        reportes_dir = os.path.join(self.directorio_resultados, 'reportes')
        if os.path.exists(reportes_dir):
            archivos['reportes_json'] = glob.glob(os.path.join(reportes_dir, '*.json'))
        
        return archivos
    
    def extraer_nombre_imagen(self, archivo):
        """
        Extrae el nombre de la imagen del nombre del archivo.
        
        Args:
            archivo: Ruta del archivo
            
        Returns:
            str: Nombre de la imagen
        """
        nombre_archivo = os.path.basename(archivo)
        
        # Patrones comunes para extraer nombre de imagen
        patrones = [
            r'surf_estadisticas_(.+?)_\d{8}_\d{6}\.csv',
            r'orb_estadisticas_(.+?)_\d{8}_\d{6}\.csv',
            r'hog_descriptores_(.+?)_\d{8}_\d{6}\.csv',
            r'kaze_estadisticas_(.+?)_\d{8}_\d{6}\.csv',
            r'(.+?)_\d{8}_\d{6}\.csv',
            r'(.+?)\.csv'
        ]
        
        for patron in patrones:
            match = re.search(patron, nombre_archivo)
            if match:
                return match.group(1)
        
        # Si no encuentra patrón, usar nombre sin extensión
        return os.path.splitext(nombre_archivo)[0]
    
    def procesar_surf_orb(self, archivos_surf, archivos_orb):
        """
        Procesa archivos de estadísticas SURF y ORB.
        
        Args:
            archivos_surf: Lista de archivos SURF
            archivos_orb: Lista de archivos ORB
        """
        # Procesar SURF
        for archivo in archivos_surf:
            try:
                df = pd.read_csv(archivo)
                for _, fila in df.iterrows():
                    registro = self.crear_registro_base(fila.get('imagen', self.extraer_nombre_imagen(archivo)), 'SURF')
                    
                    # Agregar datos específicos SURF
                    registro.update({
                        'dimensiones_imagen': fila.get('dimensiones', 'N/A'),
                        'num_puntos_clave': fila.get('num_puntos_clave', 0),
                        'dimension_descriptor': fila.get('dimension_descriptor', 0),
                        'densidad_puntos_clave': fila.get('densidad_puntos', 0),
                        'fecha_analisis': fila.get('fecha_analisis', ''),
                        'valor_min_descriptores': fila.get('valor_min_descriptores', 0),
                        'valor_max_descriptores': fila.get('valor_max_descriptores', 0),
                        'promedio_descriptores': fila.get('promedio_descriptores', 0),
                        'entropia_descriptores': fila.get('entropia_descriptores', 0),
                        'diversidad_descriptores': fila.get('diversidad_descriptores', 0),
                        'algoritmo_usado': fila.get('algoritmo_usado', 'SURF')
                    })
                    
                    self.datos_consolidados.append(registro)
                    
            except Exception as e:
                print(f"Error procesando archivo SURF {archivo}: {e}")
        
        # Procesar ORB
        for archivo in archivos_orb:
            try:
                df = pd.read_csv(archivo)
                for _, fila in df.iterrows():
                    registro = self.crear_registro_base(fila.get('imagen', self.extraer_nombre_imagen(archivo)), 'ORB')
                    
                    # Agregar datos específicos ORB
                    registro.update({
                        'dimensiones_imagen': fila.get('dimensiones', 'N/A'),
                        'num_puntos_clave': fila.get('num_puntos_clave', 0),
                        'dimension_descriptor_bytes': fila.get('dimension_descriptor_bytes', 0),
                        'dimension_descriptor_bits': fila.get('dimension_descriptor_bits', 0),
                        'densidad_puntos_clave': fila.get('densidad_puntos', 0),
                        'fecha_analisis': fila.get('fecha_analisis', ''),
                        'hamming_distance_promedio': fila.get('hamming_distance_promedio', 0),
                        'hamming_distance_std': fila.get('hamming_distance_std', 0),
                        'ratio_bits_activos': fila.get('ratio_bits_activos', 0),
                        'entropia_descriptores': fila.get('entropia_descriptores', 0),
                        'unicidad_descriptores': fila.get('unicidad_descriptores', 0)
                    })
                    
                    self.datos_consolidados.append(registro)
                    
            except Exception as e:
                print(f"Error procesando archivo ORB {archivo}: {e}")
    
    def procesar_otros_algoritmos(self, archivos):
        """
        Procesa archivos de otros algoritmos (HOG, KAZE, FREAK, AKAZE).
        
        Args:
            archivos: Diccionario con archivos por tipo
        """
        # Procesar HOG
        for archivo in archivos.get('hog_resultados', []):
            self.procesar_archivo_generico(archivo, 'HOG')
        
        # Procesar KAZE
        for archivo in archivos.get('kaze_resultados', []):
            self.procesar_archivo_generico(archivo, 'KAZE')
        
        # Procesar FREAK
        for archivo in archivos.get('freak_resultados', []):
            self.procesar_archivo_generico(archivo, 'FREAK')
        
        # Procesar AKAZE
        for archivo in archivos.get('akaze_resultados', []):
            self.procesar_archivo_generico(archivo, 'AKAZE')
    
    def procesar_archivo_generico(self, archivo, algoritmo):
        """
        Procesa un archivo genérico de resultados.
        
        Args:
            archivo: Ruta del archivo
            algoritmo: Nombre del algoritmo
        """
        try:
            df = pd.read_csv(archivo)
            nombre_imagen = self.extraer_nombre_imagen(archivo)
            
            if len(df) > 0:
                registro = self.crear_registro_base(nombre_imagen, algoritmo)
                
                # Mapear columnas comunes
                for columna in df.columns:
                    if columna.lower() in ['puntos_clave', 'num_puntos_clave', 'keypoints']:
                        registro['num_puntos_clave'] = df[columna].iloc[0] if len(df) > 0 else 0
                    elif columna.lower() in ['dimension_descriptor', 'descriptor_size']:
                        registro['dimension_descriptor'] = df[columna].iloc[0] if len(df) > 0 else 0
                    elif 'tiempo' in columna.lower() or 'time' in columna.lower():
                        registro['tiempo_procesamiento_ms'] = df[columna].iloc[0] if len(df) > 0 else 0
                
                self.datos_consolidados.append(registro)
                
        except Exception as e:
            print(f"Error procesando archivo {algoritmo} {archivo}: {e}")
    
    def procesar_analisis_textura(self, archivos):
        """
        Procesa archivos de análisis de textura.
        
        Args:
            archivos: Lista de archivos de textura
        """
        for archivo in archivos:
            try:
                df = pd.read_csv(archivo)
                nombre_imagen = self.extraer_nombre_imagen(archivo)
                
                # Buscar registro existente o crear nuevo
                registro_existente = None
                for registro in self.datos_consolidados:
                    if registro['imagen'] == nombre_imagen:
                        registro_existente = registro
                        break
                
                if registro_existente is None:
                    registro_existente = self.crear_registro_base(nombre_imagen, 'TEXTURE_ANALYSIS')
                    self.datos_consolidados.append(registro_existente)
                
                # Agregar características de textura
                if len(df) > 0:
                    for columna in df.columns:
                        if 'glcm' in columna.lower():
                            registro_existente[f'textura_{columna}'] = df[columna].iloc[0]
                        elif 'lbp' in columna.lower():
                            registro_existente[f'textura_{columna}'] = df[columna].iloc[0]
                        elif any(x in columna.lower() for x in ['contraste', 'homogeneidad', 'energia', 'correlacion']):
                            registro_existente[f'textura_{columna}'] = df[columna].iloc[0]
                
            except Exception as e:
                print(f"Error procesando archivo de textura {archivo}: {e}")
    
    def procesar_reportes_json(self, archivos):
        """
        Procesa archivos de reportes en formato JSON.
        
        Args:
            archivos: Lista de archivos JSON
        """
        for archivo in archivos:
            try:
                with open(archivo, 'r', encoding='utf-8') as f:
                    datos_json = json.load(f)
                
                # Extraer información relevante del JSON
                if isinstance(datos_json, dict):
                    for imagen, datos in datos_json.items():
                        if isinstance(datos, dict):
                            # Buscar registro existente
                            registro_existente = None
                            for registro in self.datos_consolidados:
                                if registro['imagen'] == imagen:
                                    registro_existente = registro
                                    break
                            
                            if registro_existente is None:
                                registro_existente = self.crear_registro_base(imagen, 'MIXED')
                                self.datos_consolidados.append(registro_existente)
                            
                            # Agregar datos del JSON
                            for clave, valor in datos.items():
                                if isinstance(valor, (int, float, str)):
                                    registro_existente[f'json_{clave}'] = valor
                
            except Exception as e:
                print(f"Error procesando archivo JSON {archivo}: {e}")
    
    def crear_registro_base(self, nombre_imagen, algoritmo):
        """
        Crea un registro base con campos comunes.
        
        Args:
            nombre_imagen: Nombre de la imagen
            algoritmo: Algoritmo utilizado
            
        Returns:
            dict: Registro base
        """
        # Extraer dimensiones si están en el nombre
        dimensiones = self.extraer_dimensiones(nombre_imagen)
        
        return {
            'imagen': nombre_imagen,
            'algoritmo_extraccion': algoritmo,
            'dimensiones_imagen': dimensiones['texto'],
            'ancho_px': dimensiones['ancho'],
            'alto_px': dimensiones['alto'],
            'area_total_px': dimensiones['area'],
            'fecha_analisis': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_puntos_clave': 0,
            'dimension_descriptor': 0,
            'densidad_puntos_clave': 0,
            'tiempo_procesamiento_ms': 0,
            'memoria_utilizada_mb': 0,
            'calidad_descriptores': 0
        }
    
    def extraer_dimensiones(self, nombre_imagen):
        """
        Extrae dimensiones de la imagen si están disponibles.
        
        Args:
            nombre_imagen: Nombre de la imagen
            
        Returns:
            dict: Información de dimensiones
        """
        # Patrones comunes de dimensiones
        patrones_dim = [
            r'(\d+)x(\d+)',
            r'(\d+)_(\d+)',
        ]
        
        for patron in patrones_dim:
            match = re.search(patron, nombre_imagen)
            if match:
                ancho = int(match.group(1))
                alto = int(match.group(2))
                return {
                    'texto': f'{ancho}x{alto}',
                    'ancho': ancho,
                    'alto': alto,
                    'area': ancho * alto
                }
        
        # Valores por defecto
        return {
            'texto': 'N/A',
            'ancho': 0,
            'alto': 0,
            'area': 0
        }
    
    def generar_columnas_finales(self):
        """
        Genera la lista final de columnas basada en los datos consolidados.
        """
        todas_las_claves = set()
        for registro in self.datos_consolidados:
            todas_las_claves.update(registro.keys())
        
        # Ordenar columnas: base primero, luego alfabéticamente
        columnas_ordenadas = []
        
        # Agregar columnas base en orden específico
        for col in self.columnas_base:
            if col in todas_las_claves:
                columnas_ordenadas.append(col)
                todas_las_claves.remove(col)
        
        # Agregar columnas de puntos clave
        for col in self.columnas_puntos_clave:
            if col in todas_las_claves:
                columnas_ordenadas.append(col)
                todas_las_claves.remove(col)
        
        # Agregar columnas de descriptores
        for col in self.columnas_descriptores_continuos + self.columnas_descriptores_binarios:
            if col in todas_las_claves:
                columnas_ordenadas.append(col)
                todas_las_claves.remove(col)
        
        # Agregar columnas de textura
        for col in self.columnas_textura:
            if col in todas_las_claves:
                columnas_ordenadas.append(col)
                todas_las_claves.remove(col)
        
        # Agregar columnas de bordes
        for col in self.columnas_bordes:
            if col in todas_las_claves:
                columnas_ordenadas.append(col)
                todas_las_claves.remove(col)
        
        # Agregar columnas geométricas
        for col in self.columnas_geometricas:
            if col in todas_las_claves:
                columnas_ordenadas.append(col)
                todas_las_claves.remove(col)
        
        # Agregar columnas de rendimiento
        for col in self.columnas_rendimiento:
            if col in todas_las_claves:
                columnas_ordenadas.append(col)
                todas_las_claves.remove(col)
        
        # Agregar columnas restantes alfabéticamente
        columnas_ordenadas.extend(sorted(todas_las_claves))
        
        self.columnas_finales = columnas_ordenadas
    
    def completar_datos_faltantes(self):
        """
        Completa datos faltantes en los registros consolidados.
        """
        for registro in self.datos_consolidados:
            for columna in self.columnas_finales:
                if columna not in registro:
                    # Asignar valores por defecto según el tipo de columna
                    if any(x in columna.lower() for x in ['tiempo', 'memoria', 'area', 'num_', 'dimension']):
                        registro[columna] = 0
                    elif any(x in columna.lower() for x in ['ratio', 'densidad', 'promedio', 'std', 'entropia']):
                        registro[columna] = 0.0
                    elif 'fecha' in columna.lower():
                        registro[columna] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        registro[columna] = 'N/A'
    
    def consolidar_descriptores(self):
        """
        Ejecuta el proceso completo de consolidación.
        
        Returns:
            str: Ruta del archivo CSV generado
        """
        print("🔍 Iniciando consolidación de descriptores de características...")
        
        # Buscar archivos
        print("📁 Buscando archivos de resultados...")
        archivos = self.buscar_archivos_resultados()
        
        total_archivos = sum(len(lista) for lista in archivos.values())
        print(f"📊 Encontrados {total_archivos} archivos para procesar")
        
        # Procesar cada tipo de archivo
        print("🔄 Procesando archivos SURF/ORB...")
        self.procesar_surf_orb(archivos['surf_estadisticas'], archivos['orb_estadisticas'])
        
        print("🔄 Procesando otros algoritmos...")
        self.procesar_otros_algoritmos(archivos)
        
        print("🔄 Procesando análisis de textura...")
        self.procesar_analisis_textura(archivos['texture_analysis'])
        
        print("🔄 Procesando reportes JSON...")
        self.procesar_reportes_json(archivos['reportes_json'])
        
        # Generar estructura final
        print("📋 Generando estructura final de columnas...")
        self.generar_columnas_finales()
        
        print("🔧 Completando datos faltantes...")
        self.completar_datos_faltantes()
        
        # Crear DataFrame y guardar
        print("💾 Generando archivo CSV consolidado...")
        df_consolidado = pd.DataFrame(self.datos_consolidados, columns=self.columnas_finales)
        
        # Generar nombre de archivo con timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archivo_salida = f'DESCRIPTORES_CARACTERISTICAS_CONSOLIDADOS_{timestamp}.csv'
        ruta_completa = os.path.join(self.directorio_resultados, archivo_salida)
        
        # Guardar CSV
        df_consolidado.to_csv(ruta_completa, index=False, encoding='utf-8')
        
        # Generar estadísticas
        self.generar_reporte_consolidacion(df_consolidado, ruta_completa)
        
        print(f"✅ Consolidación completada!")
        print(f"📄 Archivo generado: {ruta_completa}")
        print(f"📊 Total de registros: {len(df_consolidado)}")
        print(f"📊 Total de características: {len(self.columnas_finales)}")
        
        return ruta_completa
    
    def generar_reporte_consolidacion(self, df, archivo_csv):
        """
        Genera un reporte detallado del proceso de consolidación.
        
        Args:
            df: DataFrame consolidado
            archivo_csv: Ruta del archivo CSV generado
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archivo_reporte = os.path.join(self.directorio_resultados, f'REPORTE_CONSOLIDACION_{timestamp}.txt')
        
        with open(archivo_reporte, 'w', encoding='utf-8') as f:
            f.write("REPORTE DE CONSOLIDACIÓN DE DESCRIPTORES DE CARACTERÍSTICAS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Archivo CSV generado: {os.path.basename(archivo_csv)}\n\n")
            
            f.write("ESTADÍSTICAS GENERALES:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total de imágenes procesadas: {len(df)}\n")
            f.write(f"Total de características extraídas: {len(df.columns)}\n")
            f.write(f"Algoritmos utilizados: {', '.join(df['algoritmo_extraccion'].unique())}\n\n")
            
            f.write("DISTRIBUCIÓN POR ALGORITMO:\n")
            f.write("-" * 40 + "\n")
            conteo_algoritmos = df['algoritmo_extraccion'].value_counts()
            for algoritmo, cantidad in conteo_algoritmos.items():
                f.write(f"{algoritmo}: {cantidad} registros\n")
            f.write("\n")
            
            f.write("CARACTERÍSTICAS EXTRAÍDAS:\n")
            f.write("-" * 40 + "\n")
            for i, columna in enumerate(df.columns, 1):
                f.write(f"{i:3d}. {columna}\n")
            f.write("\n")
            
            f.write("ESTADÍSTICAS NUMÉRICAS:\n")
            f.write("-" * 40 + "\n")
            columnas_numericas = df.select_dtypes(include=[np.number]).columns
            if len(columnas_numericas) > 0:
                stats = df[columnas_numericas].describe()
                f.write(stats.to_string())
            else:
                f.write("No se encontraron columnas numéricas para estadísticas.\n")
        
        print(f"📋 Reporte generado: {archivo_reporte}")

def main():
    """Función principal para ejecutar el consolidador."""
    print("🚀 CONSOLIDADOR DE DESCRIPTORES DE CARACTERÍSTICAS")
    print("=" * 60)
    
    # Solicitar directorio de resultados
    directorio = input("Ingrese el directorio de resultados [./resultados_deteccion]: ").strip()
    if not directorio:
        directorio = "./resultados_deteccion"
    
    if not os.path.exists(directorio):
        print(f"❌ Error: El directorio {directorio} no existe")
        return
    
    # Crear consolidador y ejecutar
    consolidador = ConsolidadorDescriptores(directorio)
    
    try:
        archivo_generado = consolidador.consolidar_descriptores()
        
        print("\n🎉 PROCESO COMPLETADO EXITOSAMENTE")
        print(f"📊 Archivo Excel/CSV listo para análisis: {archivo_generado}")
        print("\n💡 Recomendaciones:")
        print("   • Abrir el archivo CSV en Excel o Google Sheets")
        print("   • Cada fila representa una imagen analizada")
        print("   • Cada columna representa una característica extraída")
        print("   • Usar para análisis estadístico, ML, o clasificación")
        
    except Exception as e:
        print(f"❌ Error durante la consolidación: {e}")
        import traceback
        print("Detalles del error:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()