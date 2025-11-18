#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Preprocesamiento para Parcial de Visión Artificial
===========================================================

Implementa las funciones específicas de preprocesamiento requeridas
para el parcial: redimensionamiento a 224x224 y normalización [0,1].

Autor: Sistema de Visión Artificial
Fecha: Noviembre 2024
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from .operaciones_aritmeticas import OperacionesAritmeticas
from .operaciones_geometricas import OperacionesGeometricas

class PreprocesadorParcial:
    """
    Clase especializada para el preprocesamiento del parcial.
    """
    
    def __init__(self):
        """
        Inicializa el preprocesador.
        """
        self.ops_aritmeticas = OperacionesAritmeticas()
        self.ops_geometricas = OperacionesGeometricas()
    
    def preprocesar_imagen_completo(self, imagen):
        """
        Aplica el preprocesamiento completo requerido en el parcial:
        1. Redimensiona a 224x224
        2. Normaliza al rango [0,1]
        
        Args:
            imagen: Imagen de entrada (numpy array)
            
        Returns:
            tuple: (imagen_preprocesada, info_preprocesamiento)
        """
        if imagen is None:
            raise ValueError("La imagen de entrada no puede ser None")
        
        # Guardar información original
        forma_original = imagen.shape
        tipo_original = imagen.dtype
        rango_original = (np.min(imagen), np.max(imagen))
        
        # Paso 1: Redimensionar a 224x224
        imagen_redimensionada = self.ops_geometricas.redimensionar_224x224(imagen)
        
        # Paso 2: Normalizar al rango [0,1]
        imagen_normalizada = self.ops_aritmeticas.normalizar_imagen(imagen_redimensionada)
        
        # Información del preprocesamiento
        info = {
            'forma_original': forma_original,
            'forma_final': imagen_normalizada.shape,
            'tipo_original': tipo_original,
            'tipo_final': imagen_normalizada.dtype,
            'rango_original': rango_original,
            'rango_final': (np.min(imagen_normalizada), np.max(imagen_normalizada)),
            'pasos_aplicados': ['Redimensionado a 224x224', 'Normalizado a [0,1]']
        }
        
        return imagen_normalizada, info
    
    def mostrar_comparacion_preprocesamiento(self, imagen_original, imagen_preprocesada, 
                                           nombre_imagen="Imagen", info=None):
        """
        Muestra una comparación antes y después del preprocesamiento.
        
        Args:
            imagen_original: Imagen antes del preprocesamiento
            imagen_preprocesada: Imagen después del preprocesamiento
            nombre_imagen: Nombre de la imagen para el título
            info: Información del preprocesamiento
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Imagen original
        axes[0].imshow(imagen_original)
        axes[0].set_title(f'Original: {nombre_imagen}\nDimensiones: {imagen_original.shape}', 
                         fontsize=12)
        axes[0].axis('off')
        
        # Imagen preprocesada (desnormalizar para visualización)
        imagen_vis = self.ops_aritmeticas.desnormalizar_imagen(imagen_preprocesada)
        axes[1].imshow(imagen_vis)
        axes[1].set_title(f'Preprocesada: {nombre_imagen}\nDimensiones: {imagen_preprocesada.shape}', 
                         fontsize=12)
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Mostrar información adicional si está disponible
        if info:
            print("\n📊 Información del Preprocesamiento:")
            print("-" * 40)
            print(f"Forma original: {info['forma_original']}")
            print(f"Forma final: {info['forma_final']}")
            print(f"Rango original: [{info['rango_original'][0]:.0f}, {info['rango_original'][1]:.0f}]")
            print(f"Rango final: [{info['rango_final'][0]:.3f}, {info['rango_final'][1]:.3f}]")
            print("Pasos aplicados:")
            for paso in info['pasos_aplicados']:
                print(f"  ✓ {paso}")
        
        plt.show()
    
    def preprocesar_solo_redimensionar(self, imagen):
        """
        Solo redimensiona la imagen a 224x224 sin normalizar.
        
        Args:
            imagen: Imagen de entrada
            
        Returns:
            Imagen redimensionada
        """
        return self.ops_geometricas.redimensionar_224x224(imagen)
    
    def preprocesar_solo_normalizar(self, imagen):
        """
        Solo normaliza la imagen al rango [0,1] sin redimensionar.
        
        Args:
            imagen: Imagen de entrada
            
        Returns:
            Imagen normalizada
        """
        return self.ops_aritmeticas.normalizar_imagen(imagen)
    
    def verificar_preprocesamiento(self, imagen_preprocesada):
        """
        Verifica que el preprocesamiento se aplicó correctamente.
        
        Args:
            imagen_preprocesada: Imagen después del preprocesamiento
            
        Returns:
            dict: Resultados de la verificación
        """
        verificacion = {
            'dimensiones_correctas': imagen_preprocesada.shape[:2] == (224, 224),
            'rango_correcto': 0.0 <= np.min(imagen_preprocesada) and np.max(imagen_preprocesada) <= 1.0,
            'tipo_datos': imagen_preprocesada.dtype,
            'forma': imagen_preprocesada.shape,
            'rango_actual': (np.min(imagen_preprocesada), np.max(imagen_preprocesada))
        }
        
        verificacion['preprocesamiento_correcto'] = (
            verificacion['dimensiones_correctas'] and 
            verificacion['rango_correcto']
        )
        
        return verificacion
    
    def mostrar_verificacion(self, verificacion):
        """
        Muestra los resultados de la verificación del preprocesamiento.
        
        Args:
            verificacion: Resultados de la verificación
        """
        print("\n✅ Verificación del Preprocesamiento:")
        print("-" * 40)
        
        # Verificar dimensiones
        if verificacion['dimensiones_correctas']:
            print("✓ Dimensiones correctas: 224x224")
        else:
            print(f"❌ Dimensiones incorrectas: {verificacion['forma'][:2]} (esperado: 224x224)")
        
        # Verificar rango
        if verificacion['rango_correcto']:
            print("✓ Rango correcto: [0,1]")
        else:
            print(f"❌ Rango incorrecto: [{verificacion['rango_actual'][0]:.3f}, {verificacion['rango_actual'][1]:.3f}] (esperado: [0,1])")
        
        print(f"Tipo de datos: {verificacion['tipo_datos']}")
        print(f"Forma completa: {verificacion['forma']}")
        
        if verificacion['preprocesamiento_correcto']:
            print("\n🎉 ¡Preprocesamiento aplicado correctamente!")
        else:
            print("\n⚠️ Hay problemas con el preprocesamiento")
    
    def preparar_para_cnn(self, imagen):
        """
        Prepara una imagen para ser utilizada en modelos CNN preentrenados.
        Aplica el preprocesamiento estándar y añade dimensión de batch.
        
        Args:
            imagen: Imagen de entrada
            
        Returns:
            tuple: (imagen_preparada, info_preparacion)
        """
        # Aplicar preprocesamiento completo
        imagen_prep, info = self.preprocesar_imagen_completo(imagen)
        
        # Añadir dimensión de batch (1, 224, 224, 3)
        imagen_batch = np.expand_dims(imagen_prep, axis=0)
        
        info['forma_final_con_batch'] = imagen_batch.shape
        info['pasos_aplicados'].append('Añadida dimensión de batch')
        
        return imagen_batch, info
    
    def procesar_lote_imagenes(self, lista_imagenes, mostrar_progreso=True):
        """
        Procesa un lote de imágenes aplicando el preprocesamiento.
        
        Args:
            lista_imagenes: Lista de imágenes a procesar
            mostrar_progreso: Si mostrar el progreso del procesamiento
            
        Returns:
            list: Lista de imágenes preprocesadas
        """
        imagenes_procesadas = []
        total = len(lista_imagenes)
        
        if mostrar_progreso:
            print(f"\n⚙️ Procesando lote de {total} imágenes...")
        
        for i, imagen in enumerate(lista_imagenes):
            try:
                imagen_prep, _ = self.preprocesar_imagen_completo(imagen)
                imagenes_procesadas.append(imagen_prep)
                
                if mostrar_progreso and (i + 1) % 10 == 0:
                    print(f"   Procesadas: {i + 1}/{total}")
                    
            except Exception as e:
                print(f"❌ Error procesando imagen {i + 1}: {e}")
                continue
        
        if mostrar_progreso:
            print(f"✅ Lote procesado: {len(imagenes_procesadas)}/{total} imágenes exitosas")
        
        return imagenes_procesadas