#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidador Rápido de Descriptores
==================================

Script simplificado para consolidar rápidamente archivos CSV específicos
de descriptores de características en una hoja de cálculo unificada.

Uso:
    python utils/consolidador_rapido.py

O importar:
    from utils.consolidador_rapido import ConsolidadorRapido

Autor: Sistema de Detección Vehicular
Fecha: Octubre 2025
"""

import pandas as pd
import os
import glob
from datetime import datetime

class ConsolidadorRapido:
    """
    Clase para consolidación rápida de archivos CSV.
    """
    
    def __init__(self, directorio_resultados="./resultados_deteccion"):
        """
        Inicializa el consolidador rápido.
        
        Args:
            directorio_resultados: Directorio base para buscar archivos
        """
        self.directorio_resultados = directorio_resultados
    
    def consolidar(self):
        """
        Ejecuta la consolidación rápida.
        
        Returns:
            str: Ruta del archivo consolidado
        """
        return consolidar_csv_rapido(self.directorio_resultados)

def consolidar_csv_rapido(directorio_resultados="./resultados_deteccion"):
    """
    Consolida rápidamente todos los archivos CSV encontrados.
    
    Args:
        directorio_resultados: Directorio donde buscar archivos CSV
        
    Returns:
        str: Ruta del archivo consolidado
    """
    print("🔍 Buscando archivos CSV...")
    
    # Buscar todos los archivos CSV en subdirectorios
    patron_busqueda = os.path.join(directorio_resultados, "**", "*.csv")
    archivos_csv = glob.glob(patron_busqueda, recursive=True)
    
    if not archivos_csv:
        print("❌ No se encontraron archivos CSV")
        return None
    
    print(f"📊 Encontrados {len(archivos_csv)} archivos CSV")
    
    datos_consolidados = []
    
    for archivo in archivos_csv:
        try:
            print(f"📄 Procesando: {os.path.basename(archivo)}")
            
            # Leer archivo CSV
            df = pd.read_csv(archivo)
            
            if len(df) == 0:
                continue
            
            # Agregar metadatos
            df['archivo_origen'] = os.path.basename(archivo)
            df['directorio_origen'] = os.path.basename(os.path.dirname(archivo))
            df['ruta_completa'] = archivo
            df['fecha_procesamiento'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Detectar tipo de archivo por nombre
            nombre_archivo = os.path.basename(archivo).lower()
            if 'surf' in nombre_archivo:
                df['tipo_algoritmo'] = 'SURF'
            elif 'orb' in nombre_archivo:
                df['tipo_algoritmo'] = 'ORB'
            elif 'hog' in nombre_archivo:
                df['tipo_algoritmo'] = 'HOG'
            elif 'kaze' in nombre_archivo:
                df['tipo_algoritmo'] = 'KAZE'
            elif 'sift' in nombre_archivo:
                df['tipo_algoritmo'] = 'SIFT'
            elif 'freak' in nombre_archivo:
                df['tipo_algoritmo'] = 'FREAK'
            elif 'akaze' in nombre_archivo:
                df['tipo_algoritmo'] = 'AKAZE'
            elif 'texture' in nombre_archivo:
                df['tipo_algoritmo'] = 'TEXTURE'
            elif 'hough' in nombre_archivo:
                df['tipo_algoritmo'] = 'HOUGH'
            else:
                df['tipo_algoritmo'] = 'UNKNOWN'
            
            datos_consolidados.append(df)
            
        except Exception as e:
            print(f"⚠️  Error procesando {archivo}: {e}")
            continue
    
    if not datos_consolidados:
        print("❌ No se pudo procesar ningún archivo")
        return None
    
    print("🔄 Consolidando datos...")
    
    # Concatenar todos los DataFrames
    df_final = pd.concat(datos_consolidados, ignore_index=True, sort=False)
    
    # Reorganizar columnas: metadatos primero
    columnas_meta = ['tipo_algoritmo', 'archivo_origen', 'directorio_origen', 'fecha_procesamiento']
    otras_columnas = [col for col in df_final.columns if col not in columnas_meta + ['ruta_completa']]
    
    columnas_ordenadas = columnas_meta + sorted(otras_columnas) + ['ruta_completa']
    df_final = df_final[columnas_ordenadas]
    
    # Generar archivo de salida
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archivo_salida = f'DESCRIPTORES_CONSOLIDADOS_RAPIDO_{timestamp}.csv'
    ruta_completa = os.path.join(directorio_resultados, archivo_salida)
    
    # Guardar
    df_final.to_csv(ruta_completa, index=False, encoding='utf-8')
    
    print(f"✅ Consolidación completada!")
    print(f"📄 Archivo generado: {ruta_completa}")
    print(f"📊 Total de registros: {len(df_final)}")
    print(f"📊 Total de columnas: {len(df_final.columns)}")
    print(f"📊 Algoritmos detectados: {', '.join(df_final['tipo_algoritmo'].unique())}")
    
    # Generar resumen rápido
    generar_resumen_rapido(df_final, ruta_completa)
    
    return ruta_completa

def generar_resumen_rapido(df, archivo_csv):
    """
    Genera un resumen rápido de la consolidación.
    
    Args:
        df: DataFrame consolidado
        archivo_csv: Ruta del archivo CSV
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    directorio = os.path.dirname(archivo_csv)
    archivo_resumen = os.path.join(directorio, f'RESUMEN_CONSOLIDACION_{timestamp}.txt')
    
    with open(archivo_resumen, 'w', encoding='utf-8') as f:
        f.write("RESUMEN RÁPIDO - CONSOLIDACIÓN DE DESCRIPTORES\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Archivo CSV: {os.path.basename(archivo_csv)}\n\n")
        
        f.write("ESTADÍSTICAS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total registros: {len(df)}\n")
        f.write(f"Total columnas: {len(df.columns)}\n\n")
        
        f.write("DISTRIBUCIÓN POR ALGORITMO:\n")
        f.write("-" * 30 + "\n")
        conteo = df['tipo_algoritmo'].value_counts()
        for alg, count in conteo.items():
            f.write(f"{alg}: {count} registros\n")
        f.write("\n")
        
        f.write("ARCHIVOS PROCESADOS:\n")
        f.write("-" * 30 + "\n")
        archivos_unicos = df['archivo_origen'].unique()
        for i, archivo in enumerate(sorted(archivos_unicos), 1):
            f.write(f"{i:3d}. {archivo}\n")
    
    print(f"📋 Resumen guardado: {archivo_resumen}")

def consolidar_por_tipo():
    """
    Consolida archivos agrupándolos por tipo de algoritmo.
    """
    directorio = input("Directorio de resultados [./resultados_deteccion]: ").strip()
    if not directorio:
        directorio = "./resultados_deteccion"
    
    print("🔍 Consolidación por tipo de algoritmo...")
    
    # Patrones específicos para cada tipo
    patrones = {
        'SURF': '**/surf_*.csv',
        'ORB': '**/orb_*.csv',
        'HOG': '**/hog_*.csv',
        'KAZE': '**/kaze_*.csv',
        'SIFT': '**/sift_*.csv',
        'TEXTURE': '**/texture_*.csv',
        'HOUGH': '**/hough_*.csv'
    }
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for tipo, patron in patrones.items():
        archivos = glob.glob(os.path.join(directorio, patron), recursive=True)
        
        if not archivos:
            print(f"⚠️  No se encontraron archivos para {tipo}")
            continue
        
        print(f"📊 Procesando {len(archivos)} archivos de {tipo}...")
        
        dfs = []
        for archivo in archivos:
            try:
                df = pd.read_csv(archivo)
                df['archivo_origen'] = os.path.basename(archivo)
                dfs.append(df)
            except Exception as e:
                print(f"⚠️  Error en {archivo}: {e}")
        
        if dfs:
            df_consolidado = pd.concat(dfs, ignore_index=True, sort=False)
            archivo_salida = os.path.join(directorio, f'CONSOLIDADO_{tipo}_{timestamp}.csv')
            df_consolidado.to_csv(archivo_salida, index=False, encoding='utf-8')
            print(f"✅ {tipo}: {archivo_salida} ({len(df_consolidado)} registros)")

def main():
    """Función principal."""
    print("🚀 CONSOLIDADOR RÁPIDO DE DESCRIPTORES")
    print("=" * 50)
    print("1. Consolidación rápida (todos los CSV)")
    print("2. Consolidación por tipo de algoritmo")
    print("3. Salir")
    
    opcion = input("\nSeleccione opción [1]: ").strip()
    
    if opcion == "2":
        consolidar_por_tipo()
    elif opcion == "3":
        print("👋 ¡Hasta luego!")
        return
    else:
        # Opción 1 (por defecto)
        directorio = input("Directorio de resultados [./resultados_deteccion]: ").strip()
        if not directorio:
            directorio = "./resultados_deteccion"
        
        if not os.path.exists(directorio):
            print(f"❌ Error: El directorio {directorio} no existe")
            return
        
        archivo_generado = consolidar_csv_rapido(directorio)
        
        if archivo_generado:
            print("\n🎉 ¡PROCESO COMPLETADO!")
            print("💡 El archivo CSV está listo para:")
            print("   • Abrir en Excel o Google Sheets")
            print("   • Análisis estadístico")
            print("   • Machine Learning")
            print("   • Visualización de datos")

if __name__ == "__main__":
    main()