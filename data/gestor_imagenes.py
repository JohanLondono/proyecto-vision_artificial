#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gestor de Imágenes para Visión Artificial
========================================

Clase para gestionar la carga, visualización y manejo de imágenes
del dataset para el sistema de visión artificial.

Universidad del Quindío - Visión Artificial
Fecha: Noviembre 2024
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime


class GestorImagenes:
    """Clase para gestionar operaciones con imágenes del dataset."""
    
    def __init__(self, ruta_imagenes="images"):
        """
        Inicializa el gestor de imágenes.
        
        Args:
            ruta_imagenes (str): Ruta a la carpeta de imágenes
        """
        self.ruta_imagenes = ruta_imagenes
        self.extensiones_validas = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        self._verificar_directorio()
        
    def _verificar_directorio(self):
        """Verifica que el directorio de imágenes existe."""
        if not os.path.exists(self.ruta_imagenes):
            print(f"⚠️  Directorio {self.ruta_imagenes} no encontrado")
            # Intentar crear el directorio
            try:
                os.makedirs(self.ruta_imagenes, exist_ok=True)
                print(f"✅ Directorio {self.ruta_imagenes} creado")
            except Exception as e:
                print(f"❌ Error creando directorio: {e}")
    
    def listar_imagenes(self):
        """
        Lista todas las imágenes disponibles en el directorio.
        
        Returns:
            list: Lista de rutas de imágenes válidas
        """
        imagenes = []
        
        if not os.path.exists(self.ruta_imagenes):
            print(f"❌ Directorio {self.ruta_imagenes} no existe")
            return imagenes
            
        try:
            for archivo in os.listdir(self.ruta_imagenes):
                if any(archivo.lower().endswith(ext) for ext in self.extensiones_validas):
                    ruta_completa = os.path.join(self.ruta_imagenes, archivo)
                    imagenes.append(ruta_completa)
            
            imagenes.sort()  # Ordenar alfabéticamente
            
        except Exception as e:
            print(f"❌ Error listando imágenes: {e}")
        
        return imagenes
    
    def cargar_imagen(self, ruta_imagen):
        """
        Carga una imagen desde archivo.
        
        Args:
            ruta_imagen (str): Ruta a la imagen
            
        Returns:
            tuple: (imagen_rgb, información)
        """
        try:
            # Cargar imagen
            imagen = cv2.imread(ruta_imagen)
            
            if imagen is None:
                raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")
            
            # Convertir de BGR a RGB para matplotlib
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            
            # Obtener información de la imagen
            info = {
                'nombre': os.path.basename(ruta_imagen),
                'ruta': ruta_imagen,
                'dimensiones': imagen_rgb.shape,
                'ancho': imagen_rgb.shape[1],
                'alto': imagen_rgb.shape[0],
                'canales': imagen_rgb.shape[2] if len(imagen_rgb.shape) == 3 else 1,
                'tipo_datos': imagen_rgb.dtype,
                'tamaño_archivo': os.path.getsize(ruta_imagen),
                'min_valor': np.min(imagen_rgb),
                'max_valor': np.max(imagen_rgb),
                'promedio': np.mean(imagen_rgb)
            }
            
            return imagen_rgb, info
            
        except Exception as e:
            print(f"❌ Error cargando imagen {ruta_imagen}: {e}")
            return None, None
    
    def mostrar_imagen(self, imagen, titulo="Imagen", info=None):
        """
        Muestra una imagen usando matplotlib.
        
        Args:
            imagen (numpy.ndarray): Imagen a mostrar
            titulo (str): Título de la ventana
            info (dict): Información adicional de la imagen
        """
        try:
            plt.figure(figsize=(10, 8))
            
            if len(imagen.shape) == 3:
                plt.imshow(imagen)
            else:
                plt.imshow(imagen, cmap='gray')
            
            plt.title(titulo, fontsize=14, fontweight='bold')
            plt.axis('off')
            
            # Agregar información si está disponible
            if info:
                info_text = (
                    f"Dimensiones: {info['ancho']}×{info['alto']}\n"
                    f"Canales: {info['canales']}\n"
                    f"Rango: [{info['min_valor']}, {info['max_valor']}]\n"
                    f"Promedio: {info['promedio']:.2f}\n"
                    f"Archivo: {info['nombre']}"
                )
                plt.figtext(0.02, 0.02, info_text, fontsize=10, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Error mostrando imagen: {e}")
    
    def mostrar_lista_imagenes(self):
        """
        Muestra la lista de imágenes disponibles con estadísticas.
        
        Returns:
            list: Lista de imágenes encontradas
        """
        imagenes = self.listar_imagenes()
        
        print(f"\n📁 IMÁGENES DISPONIBLES EN '{self.ruta_imagenes}'")
        print("=" * 60)
        
        if not imagenes:
            print("❌ No se encontraron imágenes en el directorio")
            print(f"   Extensiones soportadas: {', '.join(self.extensiones_validas)}")
            return []
        
        # Mostrar lista numerada
        for i, ruta in enumerate(imagenes, 1):
            nombre = os.path.basename(ruta)
            try:
                tamaño = os.path.getsize(ruta)
                tamaño_mb = tamaño / (1024 * 1024)
                print(f"{i:2d}. {nombre:<30} ({tamaño_mb:.2f} MB)")
            except:
                print(f"{i:2d}. {nombre}")
        
        print("-" * 60)
        print(f"📊 Total: {len(imagenes)} imágenes")
        
        return imagenes
    
    def obtener_imagen_aleatoria(self):
        """
        Selecciona y carga una imagen aleatoria.
        
        Returns:
            tuple: (imagen_rgb, información)
        """
        imagenes = self.listar_imagenes()
        
        if not imagenes:
            print("❌ No hay imágenes disponibles")
            return None, None
        
        # Seleccionar imagen aleatoria
        ruta_seleccionada = random.choice(imagenes)
        print(f"🎲 Imagen seleccionada aleatoriamente: {os.path.basename(ruta_seleccionada)}")
        
        return self.cargar_imagen(ruta_seleccionada)
    
    def obtener_estadisticas_imagen(self, imagen, nombre="imagen"):
        """
        Obtiene estadísticas detalladas de una imagen.
        
        Args:
            imagen (numpy.ndarray): Imagen a analizar
            nombre (str): Nombre de la imagen
            
        Returns:
            dict: Estadísticas de la imagen
        """
        try:
            estadisticas = {
                'nombre': nombre,
                'forma': imagen.shape,
                'tipo_datos': str(imagen.dtype),
                'tamaño_memoria': imagen.nbytes,
                'min_global': float(np.min(imagen)),
                'max_global': float(np.max(imagen)),
                'promedio_global': float(np.mean(imagen)),
                'desviacion_estandar': float(np.std(imagen)),
                'mediana': float(np.median(imagen))
            }
            
            # Si es imagen en color, analizar por canales
            if len(imagen.shape) == 3:
                estadisticas['canales'] = imagen.shape[2]
                estadisticas['por_canal'] = {}
                
                nombres_canales = ['Rojo', 'Verde', 'Azul'] if imagen.shape[2] == 3 else [f'Canal_{i}' for i in range(imagen.shape[2])]
                
                for i in range(imagen.shape[2]):
                    canal = imagen[:,:,i]
                    estadisticas['por_canal'][nombres_canales[i]] = {
                        'min': float(np.min(canal)),
                        'max': float(np.max(canal)),
                        'promedio': float(np.mean(canal)),
                        'desviacion': float(np.std(canal))
                    }
            else:
                estadisticas['canales'] = 1
            
            return estadisticas
            
        except Exception as e:
            print(f"❌ Error calculando estadísticas: {e}")
            return {}
    
    def mostrar_estadisticas(self, estadisticas):
        """
        Muestra las estadísticas de una imagen de forma legible.
        
        Args:
            estadisticas (dict): Estadísticas calculadas
        """
        if not estadisticas:
            return
        
        print(f"\n📊 ESTADÍSTICAS DE '{estadisticas['nombre']}'")
        print("=" * 50)
        print(f"Dimensiones: {estadisticas['forma']}")
        print(f"Tipo de datos: {estadisticas['tipo_datos']}")
        print(f"Memoria utilizada: {estadisticas['tamaño_memoria']/1024:.2f} KB")
        print(f"Canales: {estadisticas['canales']}")
        
        print(f"\n📈 Valores globales:")
        print(f"   Mínimo: {estadisticas['min_global']:.2f}")
        print(f"   Máximo: {estadisticas['max_global']:.2f}")
        print(f"   Promedio: {estadisticas['promedio_global']:.2f}")
        print(f"   Desv. estándar: {estadisticas['desviacion_estandar']:.2f}")
        print(f"   Mediana: {estadisticas['mediana']:.2f}")
        
        # Si hay estadísticas por canal
        if 'por_canal' in estadisticas:
            print(f"\n🎨 Por canal:")
            for canal, stats in estadisticas['por_canal'].items():
                print(f"   {canal}:")
                print(f"     Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
                print(f"     Promedio: {stats['promedio']:.2f}, Desv: {stats['desviacion']:.2f}")
    
    def comparar_imagenes(self, imagen1, imagen2, titulo1="Imagen 1", titulo2="Imagen 2"):
        """
        Compara dos imágenes lado a lado.
        
        Args:
            imagen1, imagen2 (numpy.ndarray): Imágenes a comparar
            titulo1, titulo2 (str): Títulos para las imágenes
        """
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Primera imagen
            if len(imagen1.shape) == 3:
                axes[0].imshow(imagen1)
            else:
                axes[0].imshow(imagen1, cmap='gray')
            axes[0].set_title(titulo1, fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            # Segunda imagen
            if len(imagen2.shape) == 3:
                axes[1].imshow(imagen2)
            else:
                axes[1].imshow(imagen2, cmap='gray')
            axes[1].set_title(titulo2, fontsize=12, fontweight='bold')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Error comparando imágenes: {e}")
    
    def validar_imagen(self, ruta_imagen):
        """
        Valida que un archivo sea una imagen válida.
        
        Args:
            ruta_imagen (str): Ruta al archivo
            
        Returns:
            bool: True si es válida, False si no
        """
        try:
            # Verificar existencia
            if not os.path.exists(ruta_imagen):
                return False
            
            # Verificar extensión
            _, extension = os.path.splitext(ruta_imagen.lower())
            if extension not in self.extensiones_validas:
                return False
            
            # Intentar cargar con OpenCV
            imagen = cv2.imread(ruta_imagen)
            if imagen is None:
                return False
            
            return True
            
        except:
            return False