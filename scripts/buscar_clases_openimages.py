#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para buscar clases disponibles en Open Images V7
Útil para encontrar nombres exactos antes de descargar
"""

import pandas as pd
import urllib.request
import sys

def buscar_clases(termino_busqueda=''):
    """
    Busca clases en Open Images que coincidan con el término.
    
    Args:
        termino_busqueda: Palabra clave a buscar (ej: 'hat', 'car', 'dog')
    """
    print("=" * 60)
    print("BUSCADOR DE CLASES - OPEN IMAGES V7")
    print("=" * 60)
    
    # Descargar catálogo
    print("\n📥 Descargando catálogo de clases...")
    url = 'https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv'
    
    try:
        urllib.request.urlretrieve(url, 'temp_classes.csv')
        df = pd.read_csv('temp_classes.csv', header=None)
        df.columns = ['ID', 'Nombre']
        
        print(f"✓ {len(df)} clases disponibles\n")
        
        if termino_busqueda:
            # Buscar coincidencias
            resultados = df[df['Nombre'].str.contains(termino_busqueda, case=False, na=False)]
            
            print(f"🔍 Búsqueda: '{termino_busqueda}'")
            print("=" * 60)
            
            if len(resultados) > 0:
                print(f"\n✅ {len(resultados)} clases encontradas:\n")
                
                for idx, row in resultados.iterrows():
                    nombre = row['Nombre']
                    # Mostrar si necesita comillas
                    if ' ' in nombre:
                        print(f'  • "{nombre}"  ← Usar con comillas')
                    else:
                        print(f'  • {nombre}')
                
                # Mostrar ejemplo de comando
                print("\n" + "=" * 60)
                print("💡 Ejemplo de descarga:")
                print("=" * 60)
                
                nombres = []
                for idx, row in resultados.iterrows():
                    nombre = row['Nombre']
                    if ' ' in nombre:
                        nombres.append(f'"{nombre}"')
                    else:
                        nombres.append(nombre)
                
                # Limitar a primeras 5 clases
                nombres_sample = ' '.join(nombres[:5])
                print(f'\noi_download_dataset --base_dir ./dataset --labels {nombres_sample} --format darknet --limit 500\n')
                
            else:
                print(f"\n❌ No se encontraron clases con '{termino_busqueda}'")
                print("\n💡 Sugerencias:")
                print("  • Intenta con términos en inglés")
                print("  • Prueba sinónimos (ej: car, automobile, vehicle)")
                print("  • Busca términos más generales")
        
        else:
            # Mostrar todas las clases
            print("📋 TODAS LAS CLASES DISPONIBLES:")
            print("=" * 60)
            print("\nMostrando primeras 50 clases (de 600):\n")
            
            for idx, row in df.head(50).iterrows():
                nombre = row['Nombre']
                if ' ' in nombre:
                    print(f'  • "{nombre}"')
                else:
                    print(f'  • {nombre}')
            
            print("\n... (550 clases más)")
            print("\n💡 Usa: python buscar_clases_openimages.py <término>")
            print("   Ejemplo: python buscar_clases_openimages.py car")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Limpiar archivo temporal
        try:
            import os
            if os.path.exists('temp_classes.csv'):
                os.remove('temp_classes.csv')
        except:
            pass

if __name__ == '__main__':
    if len(sys.argv) > 1:
        termino = ' '.join(sys.argv[1:])
        buscar_clases(termino)
    else:
        print("\nUso: python buscar_clases_openimages.py <término_búsqueda>")
        print("\nEjemplos:")
        print("  python buscar_clases_openimages.py hat")
        print("  python buscar_clases_openimages.py car")
        print("  python buscar_clases_openimages.py animal")
        print("  python buscar_clases_openimages.py food")
        print("\nMostrando clases populares de sombreros...\n")
        buscar_clases('hat')
