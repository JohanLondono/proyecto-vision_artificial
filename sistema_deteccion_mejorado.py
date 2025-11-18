#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Detección de Sombreros - Versión Mejorada y Corregida
===============================================================

Sistema completo con:
- Selección interactiva de modelos
- Detección en tiempo real mejorada
- Configuración avanzada
- Integración con detectores de video

Universidad del Quindío - Visión Artificial
Fecha: Noviembre 2025
"""

import os
import cv2
import numpy as np
import time
from datetime import datetime
import json
from detectores.deteccion_video_modelos import DetectorVideoModelos

class PreprocesadorInteractivo:
    """Clase para manejar preprocesamiento interactivo de imágenes."""
    
    def __init__(self):
        """Inicializa el preprocesador interactivo."""
        self.tecnicas_disponibles = {
            'ecualizacion_histograma': 'Ecualización del histograma (normalizar iluminación)',
            'clahe': 'CLAHE - Ecualización adaptativa (mejorar contraste local)',
            'mejorar_saturacion': 'Mejorar saturación de colores',
            'filtro_bilateral': 'Filtro bilateral (reducir ruido preservando bordes)',
            'filtro_gaussiano': 'Filtro Gaussiano (suavizado)',
            'ajuste_brillo': 'Ajustar brillo',
            'ajuste_contraste': 'Ajustar contraste global',
            'gamma_correction': 'Corrección Gamma'
        }
    
    def mostrar_opciones_preprocesamiento(self):
        """Muestra las opciones de preprocesamiento disponibles."""
        print("\nOPCIONES DE PREPROCESAMIENTO DISPONIBLES:")
        print("=" * 50)
        for i, (key, desc) in enumerate(self.tecnicas_disponibles.items(), 1):
            print(f"{i}. {desc}")
        print("0. Continuar sin preprocesamiento")
        print("99. Aplicar todas las técnicas (automático)")
    
    def seleccionar_tecnicas(self):
        """Permite al usuario seleccionar qué técnicas aplicar."""
        self.mostrar_opciones_preprocesamiento()
        
        tecnicas_seleccionadas = []
        keys_list = list(self.tecnicas_disponibles.keys())
        
        print("\nSeleccione las técnicas que desea aplicar (separadas por comas):")
        print("Ejemplo: 1,3,4 para aplicar ecualización, saturación y filtro bilateral")
        
        try:
            entrada = input("Opciones seleccionadas: ").strip()
            
            if entrada == '0':
                return []
            elif entrada == '99':
                return keys_list
            
            indices = [int(x.strip()) for x in entrada.split(',') if x.strip()]
            
            for indice in indices:
                if 1 <= indice <= len(keys_list):
                    tecnica = keys_list[indice - 1]
                    tecnicas_seleccionadas.append(tecnica)
                    print(f"✓ Seleccionado: {self.tecnicas_disponibles[tecnica]}")
                else:
                    print(f"⚠️ Índice inválido ignorado: {indice}")
            
            if tecnicas_seleccionadas:
                print(f"\nTotal de técnicas seleccionadas: {len(tecnicas_seleccionadas)}")
            else:
                print("\nNo se seleccionaron técnicas de preprocesamiento")
                
            return tecnicas_seleccionadas
            
        except ValueError:
            print("Entrada inválida. No se aplicará preprocesamiento.")
            return []
    
    def aplicar_ecualizacion_histograma(self, imagen):
        """Aplica ecualización del histograma para normalizar iluminación."""
        try:
            if len(imagen.shape) == 3:
                # Convertir a LAB y ecualizar el canal L
                lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
                lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
                resultado = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                # Imagen en escala de grises
                resultado = cv2.equalizeHist(imagen)
            
            print("✓ Ecualización del histograma aplicada")
            return resultado
        except Exception as e:
            print(f"Error en ecualización del histograma: {e}")
            return imagen
    
    def aplicar_clahe(self, imagen, clip_limit=3.0, tile_grid_size=(8,8)):
        """Aplica CLAHE para mejorar contraste local."""
        try:
            if len(imagen.shape) == 3:
                # Aplicar CLAHE al canal L en espacio LAB
                lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                lab[:,:,0] = clahe.apply(lab[:,:,0])
                resultado = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                # Imagen en escala de grises
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                resultado = clahe.apply(imagen)
            
            print("✓ CLAHE aplicado para mejorar contraste local")
            return resultado
        except Exception as e:
            print(f"Error en CLAHE: {e}")
            return imagen
    
    def mejorar_saturacion(self, imagen, factor=1.3):
        """Mejora la saturación de colores."""
        try:
            if len(imagen.shape) == 3:
                # Convertir a HSV y ajustar saturación
                hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:,:,1] = hsv[:,:,1] * factor
                hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
                resultado = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                
                print(f"✓ Saturación mejorada (factor: {factor})")
                return resultado
            else:
                print("⚠️ No se puede mejorar saturación en imagen en escala de grises")
                return imagen
        except Exception as e:
            print(f"Error mejorando saturación: {e}")
            return imagen
    
    def aplicar_filtro_bilateral(self, imagen, d=9, sigma_color=75, sigma_space=75):
        """Aplica filtro bilateral para reducir ruido preservando bordes."""
        try:
            resultado = cv2.bilateralFilter(imagen, d, sigma_color, sigma_space)
            print(f"✓ Filtro bilateral aplicado (d={d}, σ_color={sigma_color}, σ_space={sigma_space})")
            return resultado
        except Exception as e:
            print(f"Error en filtro bilateral: {e}")
            return imagen
    
    def aplicar_filtro_gaussiano(self, imagen, kernel_size=(5,5), sigma=0):
        """Aplica filtro Gaussiano para suavizado."""
        try:
            resultado = cv2.GaussianBlur(imagen, kernel_size, sigma)
            print(f"✓ Filtro Gaussiano aplicado (kernel: {kernel_size})")
            return resultado
        except Exception as e:
            print(f"Error en filtro Gaussiano: {e}")
            return imagen
    
    def ajustar_brillo(self, imagen, incremento=20):
        """Ajusta el brillo de la imagen."""
        try:
            resultado = cv2.add(imagen, np.ones(imagen.shape, dtype=np.uint8) * incremento)
            resultado = np.clip(resultado, 0, 255)
            print(f"✓ Brillo ajustado (incremento: {incremento})")
            return resultado
        except Exception as e:
            print(f"Error ajustando brillo: {e}")
            return imagen
    
    def ajustar_contraste(self, imagen, factor=1.2):
        """Ajusta el contraste global de la imagen."""
        try:
            resultado = cv2.multiply(imagen, factor)
            resultado = np.clip(resultado, 0, 255).astype(np.uint8)
            print(f"✓ Contraste ajustado (factor: {factor})")
            return resultado
        except Exception as e:
            print(f"Error ajustando contraste: {e}")
            return imagen
    
    def correccion_gamma(self, imagen, gamma=1.2):
        """Aplica corrección gamma."""
        try:
            # Crear tabla de lookup para corrección gamma
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            resultado = cv2.LUT(imagen, table)
            print(f"✓ Corrección gamma aplicada (γ={gamma})")
            return resultado
        except Exception as e:
            print(f"Error en corrección gamma: {e}")
            return imagen
    
    def aplicar_tecnicas_seleccionadas(self, imagen, tecnicas_seleccionadas, mostrar_progreso=True):
        """Aplica las técnicas seleccionadas en secuencia."""
        if not tecnicas_seleccionadas:
            print("No hay técnicas de preprocesamiento seleccionadas")
            return imagen
        
        imagen_procesada = imagen.copy()
        print(f"\n🔄 Aplicando {len(tecnicas_seleccionadas)} técnicas de preprocesamiento...")
        print("-" * 60)
        
        for i, tecnica in enumerate(tecnicas_seleccionadas, 1):
            if mostrar_progreso:
                print(f"[{i}/{len(tecnicas_seleccionadas)}] {self.tecnicas_disponibles.get(tecnica, tecnica)}...")
            
            if tecnica == 'ecualizacion_histograma':
                imagen_procesada = self.aplicar_ecualizacion_histograma(imagen_procesada)
            elif tecnica == 'clahe':
                imagen_procesada = self.aplicar_clahe(imagen_procesada)
            elif tecnica == 'mejorar_saturacion':
                imagen_procesada = self.mejorar_saturacion(imagen_procesada)
            elif tecnica == 'filtro_bilateral':
                imagen_procesada = self.aplicar_filtro_bilateral(imagen_procesada)
            elif tecnica == 'filtro_gaussiano':
                imagen_procesada = self.aplicar_filtro_gaussiano(imagen_procesada)
            elif tecnica == 'ajuste_brillo':
                imagen_procesada = self.ajustar_brillo(imagen_procesada)
            elif tecnica == 'ajuste_contraste':
                imagen_procesada = self.ajustar_contraste(imagen_procesada)
            elif tecnica == 'gamma_correction':
                imagen_procesada = self.correccion_gamma(imagen_procesada)
            else:
                print(f"⚠️ Técnica no reconocida: {tecnica}")
        
        print("-" * 60)
        print("✅ Preprocesamiento completado")
        return imagen_procesada
    
    def mostrar_comparacion(self, imagen_original, imagen_procesada, tecnicas_aplicadas):
        """Muestra comparación antes y después del preprocesamiento."""
        try:
            # Mostrar imágenes lado a lado
            altura = max(imagen_original.shape[0], imagen_procesada.shape[0])
            
            # Redimensionar para comparación si es necesario
            if imagen_original.shape[:2] != imagen_procesada.shape[:2]:
                imagen_original_resized = cv2.resize(imagen_original, imagen_procesada.shape[1::-1])
            else:
                imagen_original_resized = imagen_original
            
            # Crear imagen combinada
            combinada = np.hstack([imagen_original_resized, imagen_procesada])
            
            # Agregar texto informativo
            cv2.putText(combinada, 'ORIGINAL', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combinada, 'PROCESADA', (imagen_original_resized.shape[1] + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Comparación: Original vs Procesada - Presione cualquier tecla para cerrar', combinada)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            print(f"\n📊 Técnicas aplicadas: {', '.join([self.tecnicas_disponibles.get(t, t) for t in tecnicas_aplicadas])}")
            
        except Exception as e:
            print(f"Error mostrando comparación: {e}")

# Configuración silenciosa
try:
    from utils.tensorflow_quiet_config import configure_libraries
    configure_libraries()
except ImportError:
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class SistemaDeteccionSombrerosMejorado:
    """
    Sistema mejorado de detección de sombreros con interfaz completa.
    """
    
    def __init__(self):
        """Inicializa el sistema mejorado."""
        self.detector_video = DetectorVideoModelos()
        self.preprocesador = PreprocesadorInteractivo()  # Agregar preprocesador interactivo
        self.modelo_activo = None
        self.configuracion = {
            'modelo_seleccionado': None,
            'umbral_confianza': 0.5,
            'procesamiento_tiempo_real': {
                'fps_objetivo': 30,
                'escala_deteccion': 1.0,
                'mostrar_confianza': True,
                'guardar_video': False
            },
            'entrenamiento': {
                'epochs': 50,
                'batch_size': 32,
                'learning_rate': 0.001,
                'validacion_split': 0.2
            }
        }
        
        self.modelos_disponibles = {}
        self.inicializar_sistema()
    
    def inicializar_sistema(self):
        """Inicializa el sistema base independiente."""
        print("INICIALIZANDO SISTEMA MEJORADO")
        print("=" * 40)
        
        try:
            # Catalogar modelos disponibles
            self.catalogar_modelos_disponibles_independiente()
            print("Sistema inicializado correctamente")
            return True
            
        except Exception as e:
            print(f"Error en inicialización: {e}")
            return False
    
    def catalogar_modelos_disponibles_independiente(self):
        """Cataloga todos los modelos disponibles utilizando el detector de video."""
        print("\nCatalogando modelos disponibles")
        print("-" * 35)
        
        # Obtener modelos del detector de video
        modelos_video = self.detector_video.modelos_disponibles
        
        # Catalogar modelos de clasificación
        for modelo_key, modelo_desc in modelos_video['clasificacion'].items():
            self.modelos_disponibles[modelo_key] = {
                'tipo': 'clasificacion',
                'nombre': modelo_key,
                'objeto': None,
                'entrenado': False,
                'descripcion': f'Modelo de clasificación {modelo_desc}',
                'uso': 'clasificacion',
                'requiere_entrenamiento': False
            }
            print(f"   Clasificación: {modelo_key.upper()} - {modelo_desc}")
        
        # Catalogar modelos de detección
        for modelo_key, modelo_desc in modelos_video['deteccion'].items():
            self.modelos_disponibles[modelo_key] = {
                'tipo': 'deteccion',
                'nombre': modelo_key,
                'objeto': None,
                'entrenado': True,
                'descripcion': f'Modelo de detección {modelo_desc}',
                'uso': 'deteccion',
                'requiere_entrenamiento': False
            }
            print(f"   Detección: {modelo_key.upper()} - {modelo_desc}")
        
        # Agregar YOLO Custom (modelo entrenado personalizado)
        self.modelos_disponibles['yolo_custom'] = {
            'tipo': 'deteccion',
            'nombre': 'yolo_custom',
            'objeto': None,
            'entrenado': True,
            'descripcion': 'YOLO Custom - Modelo entrenado para sombreros 🎩',
            'uso': 'deteccion',
            'requiere_entrenamiento': False
        }
        print(f"   Detección: YOLO_CUSTOM - Modelo personalizado de sombreros 🎩")
        
        # Catalogar modelos de segmentación
        for modelo_key, modelo_desc in modelos_video['segmentacion'].items():
            self.modelos_disponibles[modelo_key] = {
                'tipo': 'segmentacion',
                'nombre': modelo_key,
                'objeto': None,
                'entrenado': False,
                'descripcion': f'Modelo de segmentación {modelo_desc}',
                'uso': 'segmentacion',
                'requiere_entrenamiento': False
            }
            print(f"   Segmentación: {modelo_key.upper()} - {modelo_desc}")
        
        print(f"Total de modelos disponibles: {len(self.modelos_disponibles)}")
    
    def mostrar_menu_principal(self):
        """Muestra el menú principal mejorado."""
        print(f"\nSISTEMA DE DETECCIÓN DE SOMBREROS - MEJORADO v2.0")
        print("=" * 55)
        print("1. Detección en Imagen Individual 🔍 [NUEVO: Preprocesamiento]")
        print("2. Detección en Video/Tiempo Real 📹")
        print("3. Gestión de Modelos 🧠")
        print("4. Comparativa de Modelos 📊 [NUEVO: Análisis Completo]")
        print("5. Entrenar Modelo desde Cero 🎯")
        print("6. Configuración del Sistema ⚙️")
        print("7. Estadísticas y Reportes 📊")
        print("8. Herramientas Avanzadas 🔧")
        print("9. Ayuda y Documentación ❓")
        print("0. Salir")
        print("=" * 55)
        print("✨ NUEVO: Preprocesamiento interactivo disponible")
        print("   • Ecualización de histograma • CLAHE • Mejora de saturación")
        print("   • Filtro bilateral • Ajustes de brillo/contraste • Y más...")
        print("🆕 NUEVO: Comparativa de modelos con estadísticas y gráficos")
        print("-" * 55)
        
        if self.modelo_activo:
            print(f"📍 Modelo activo: {self.modelo_activo.upper()}")
        else:
            print("⚠️  No hay modelo activo - Seleccione un modelo en la opción 3")
    
    def seleccionar_modelo(self):
        """Permite al usuario seleccionar un modelo específico."""
        print(f"\nSELECCIONAR MODELO PARA DETECCIÓN")
        print("=" * 40)
        
        if not self.modelos_disponibles:
            print("No hay modelos disponibles")
            return None
        
        # Mostrar modelos por tipo
        modelos_por_tipo = {
            'clasificacion': [],
            'deteccion': [],
            'segmentacion': []
        }
        
        for key, modelo in self.modelos_disponibles.items():
            tipo = modelo['tipo']
            if tipo in modelos_por_tipo:
                modelos_por_tipo[tipo].append((key, modelo))
        
        # Mostrar opciones
        opciones = []
        indice = 1
        
        for tipo, lista in modelos_por_tipo.items():
            if lista:
                print(f"\n{tipo.upper()}:")
                for key, modelo in lista:
                    print(f"  {indice}. {key.upper()} - {modelo['descripcion']}")
                    opciones.append(key)
                    indice += 1
        
        print(f"\n0. Volver al menú principal")
        
        try:
            seleccion = int(input("Seleccione modelo: "))
            if seleccion == 0:
                return None
            elif 1 <= seleccion <= len(opciones):
                modelo_seleccionado = opciones[seleccion - 1]
                self.modelo_activo = modelo_seleccionado
                print(f"Modelo {modelo_seleccionado} seleccionado")
                return modelo_seleccionado
            else:
                print("Selección inválida")
                return None
        except ValueError:
            print("Entrada inválida")
            return None
    
    def configurar_parametros_deteccion(self):
        """Configura parámetros específicos para la detección."""
        print(f"\nCONFIGURACIÓN DE PARÁMETROS")
        print("=" * 35)
        
        print(f"Configuración actual:")
        print(f"  - Umbral de confianza: {self.configuracion['umbral_confianza']}")
        print(f"  - FPS objetivo: {self.configuracion['procesamiento_tiempo_real']['fps_objetivo']}")
        print(f"  - Escala de detección: {self.configuracion['procesamiento_tiempo_real']['escala_deteccion']}")
        
        print(f"\n¿Qué desea configurar?")
        print("1. Umbral de confianza")
        print("2. Parámetros de tiempo real")
        print("3. Parámetros de entrenamiento")
        print("0. Volver")
        
        try:
            opcion = int(input("\nSeleccione opción: "))
            
            if opcion == 1:
                nuevo_umbral = float(input(f"Nuevo umbral de confianza (0.0-1.0): "))
                if 0.0 <= nuevo_umbral <= 1.0:
                    self.configuracion['umbral_confianza'] = nuevo_umbral
                    print(f"Umbral actualizado a {nuevo_umbral}")
                else:
                    print("Valor inválido")
            
            elif opcion == 2:
                print("Configuración de tiempo real:")
                fps = int(input("FPS objetivo (10-60): "))
                if 10 <= fps <= 60:
                    self.configuracion['procesamiento_tiempo_real']['fps_objetivo'] = fps
                    print("FPS actualizado")
                
                escala = float(input("Escala de detección (0.1-2.0): "))
                if 0.1 <= escala <= 2.0:
                    self.configuracion['procesamiento_tiempo_real']['escala_deteccion'] = escala
                    print("Escala actualizada")
                
            elif opcion == 3:
                print("Configuración de entrenamiento:")
                epochs = int(input(f"Épocas ({self.configuracion['entrenamiento']['epochs']}): ") or self.configuracion['entrenamiento']['epochs'])
                batch_size = int(input(f"Batch size ({self.configuracion['entrenamiento']['batch_size']}): ") or self.configuracion['entrenamiento']['batch_size'])
                
                self.configuracion['entrenamiento']['epochs'] = epochs
                self.configuracion['entrenamiento']['batch_size'] = batch_size
                print("Configuración de entrenamiento actualizada")
                
        except ValueError:
            print("Entrada inválida")
    
    def detectar_video_tiempo_real_mejorado(self):
        """Detección en video con selección de modelo usando el sistema integrado."""
        print(f"\nDETECCIÓN EN VIDEO/TIEMPO REAL")
        print("=" * 40)
        
        # Verificar si hay modelo activo
        if not self.modelo_activo:
            print("Debe seleccionar un modelo primero")
            self._seleccionar_modelo_para_video()
            if not self.modelo_activo:
                return
        
        # Cargar el modelo en el detector de video
        if not self.detector_video.cargar_modelo(self.modelo_activo):
            print(f"Error cargando modelo {self.modelo_activo}")
            return
        
        # Configurar parámetros si es necesario
        print(f"\n¿Desea configurar parámetros de detección? (s/n): ", end="")
        if input().lower() == 's':
            self._configurar_parametros_video()
        
        # Seleccionar fuente
        print(f"\nSeleccione fuente de video:")
        print("1. Cámara web")
        print("2. Archivo de video")
        print("0. Volver")
        
        try:
            opcion = int(input("Seleccione opción: "))
            
            if opcion == 1:
                print("Iniciando detección desde cámara web...")
                self.detector_video.procesar_video_tiempo_real(0)
                
            elif opcion == 2:
                # Llamar al método del detector que maneja la selección de video
                self.detector_video._procesar_archivo_video()
                    
            elif opcion == 0:
                return
                
        except ValueError:
            print("Entrada inválida")
        except KeyboardInterrupt:
            print("\nDetección interrumpida por el usuario")
        except Exception as e:
            print(f"Error en detección de video: {e}")
    
    def _seleccionar_modelo_para_video(self):
        """Permite seleccionar un modelo específicamente para video."""
        print("\nMODELOS DISPONIBLES PARA DETECCIÓN DE VIDEO:")
        print("-" * 45)
        
        modelos_listados = []
        contador = 1
        
        for key, info in self.modelos_disponibles.items():
            print(f"{contador}. {key.upper()} - {info['descripcion']}")
            modelos_listados.append(key)
            contador += 1
        
        if not modelos_listados:
            print("No hay modelos disponibles")
            return
        
        try:
            seleccion = int(input(f"\nSeleccione modelo (1-{len(modelos_listados)}): "))
            if 1 <= seleccion <= len(modelos_listados):
                modelo_seleccionado = modelos_listados[seleccion - 1]
                self.modelo_activo = modelo_seleccionado
                print(f"Modelo {modelo_seleccionado} seleccionado")
            else:
                print("Selección inválida")
        except ValueError:
            print("Entrada inválida")
    
    def _configurar_parametros_video(self):
        """Configura parámetros específicos para detección de video."""
        try:
            print("\nCONFIGURACIÓN DE PARÁMETROS DE VIDEO:")
            print("-" * 40)
            
            # Umbral de confianza
            umbral_actual = self.detector_video.configuracion['umbral_confianza']
            print(f"Umbral de confianza actual: {umbral_actual}")
            nuevo_umbral = input(f"Nuevo umbral (0.1-0.9) [Enter para mantener]: ").strip()
            
            if nuevo_umbral:
                umbral_float = float(nuevo_umbral)
                if 0.1 <= umbral_float <= 0.9:
                    self.detector_video.configuracion['umbral_confianza'] = umbral_float
                    self.configuracion['umbral_confianza'] = umbral_float
                    print(f"Umbral actualizado a {umbral_float}")
                else:
                    print("Umbral fuera de rango, manteniendo valor actual")
            
            # FPS objetivo
            fps_actual = self.detector_video.configuracion['fps_objetivo']
            print(f"FPS objetivo actual: {fps_actual}")
            nuevo_fps = input(f"Nuevo FPS (10-60) [Enter para mantener]: ").strip()
            
            if nuevo_fps:
                fps_int = int(nuevo_fps)
                if 10 <= fps_int <= 60:
                    self.detector_video.configuracion['fps_objetivo'] = fps_int
                    self.configuracion['procesamiento_tiempo_real']['fps_objetivo'] = fps_int
                    print(f"FPS actualizado a {fps_int}")
                else:
                    print("FPS fuera de rango, manteniendo valor actual")
            
            print("Configuración actualizada")
            
        except ValueError:
            print("Valor inválido, manteniendo configuración actual")
        except Exception as e:
            print(f"Error en configuración: {e}")
    
    def ejecutar_sistema(self):
        """Ejecuta el sistema principal con menú interactivo."""
        print("SISTEMA DE DETECCIÓN DE SOMBREROS - VERSIÓN MEJORADA")
        print("Universidad del Quindío - Visión Artificial")
        print("=" * 60)
        
        print("Sistema mejorado independiente inicializado")
        
        while True:
            self.mostrar_menu_principal()
            
            try:
                opcion = input("\nSeleccione una opción: ").strip()
                
                if opcion == '0':
                    print("Hasta luego!")
                    break
                
                elif opcion == '1':
                    # Detección en imagen individual
                    if not self.modelo_activo:
                        print("ERROR: Seleccione un modelo primero (opción 3)")
                        continue
                    self.detectar_imagen_individual_mejorada()
                
                elif opcion == '2':
                    # Detección en video/tiempo real
                    self.detectar_video_tiempo_real_mejorado()
                
                elif opcion == '3':
                    # Gestión de modelos
                    self.seleccionar_modelo()
                
                elif opcion == '4':
                    # Comparativa de modelos
                    self.comparativa_modelos()
                
                elif opcion == '5':
                    # Entrenar modelo
                    print("Funcionalidad de entrenamiento disponible en versión completa")
                
                elif opcion == '6':
                    # Configuración del sistema
                    self._mostrar_configuracion_sistema()
                
                elif opcion == '7':
                    # Estadísticas y reportes
                    self._mostrar_estadisticas()
                
                elif opcion == '8':
                    # Herramientas avanzadas
                    print("Herramientas avanzadas disponibles en versión completa")
                
                elif opcion == '9':
                    # Ayuda
                    self.mostrar_ayuda()
                
                else:
                    print("Opción inválida. Seleccione una opción válida.")
                    
            except KeyboardInterrupt:
                print("\nSaliendo del sistema...")
                break
            except Exception as e:
                print(f"Error inesperado: {e}")
                print("Continuando con la ejecución...")
    
    def _mostrar_configuracion_sistema(self):
        """Muestra y permite modificar la configuración del sistema."""
        print("\nCONFIGURACIÓN DEL SISTEMA:")
        print("-" * 30)
        print(f"Umbral de confianza: {self.configuracion['umbral_confianza']}")
        print(f"FPS objetivo: {self.configuracion['procesamiento_tiempo_real']['fps_objetivo']}")
        print(f"Mostrar confianza: {self.configuracion['procesamiento_tiempo_real']['mostrar_confianza']}")
        print("\n1. Cambiar umbral de confianza")
        print("2. Cambiar FPS objetivo")
        print("3. Toggle mostrar confianza")
        print("0. Volver")
        
        try:
            opcion = input("Seleccione opción: ").strip()
            
            if opcion == '1':
                nuevo_umbral = float(input("Nuevo umbral (0.1-0.9): "))
                if 0.1 <= nuevo_umbral <= 0.9:
                    self.configuracion['umbral_confianza'] = nuevo_umbral
                    print("Umbral actualizado")
                    
            elif opcion == '2':
                nuevo_fps = int(input("Nuevo FPS (10-60): "))
                if 10 <= nuevo_fps <= 60:
                    self.configuracion['procesamiento_tiempo_real']['fps_objetivo'] = nuevo_fps
                    print("FPS actualizado")
                    
            elif opcion == '3':
                self.configuracion['procesamiento_tiempo_real']['mostrar_confianza'] = \
                    not self.configuracion['procesamiento_tiempo_real']['mostrar_confianza']
                estado = "activado" if self.configuracion['procesamiento_tiempo_real']['mostrar_confianza'] else "desactivado"
                print(f"Mostrar confianza {estado}")
                
        except ValueError:
            print("Valor inválido")
    
    def _mostrar_estadisticas(self):
        """Muestra estadísticas del sistema."""
        print("\nESTADÍSTICAS DEL SISTEMA:")
        print("-" * 25)
        print(f"Modelos disponibles: {len(self.modelos_disponibles)}")
        print(f"Modelo activo: {self.modelo_activo or 'Ninguno'}")
        
        if hasattr(self.detector_video, 'modelos_cargados'):
            print(f"Modelos cargados en memoria: {len(self.detector_video.modelos_cargados)}")
        
        print("\nModelos por tipo:")
        tipos = {}
        for modelo in self.modelos_disponibles.values():
            tipo = modelo['tipo']
            tipos[tipo] = tipos.get(tipo, 0) + 1
        
        for tipo, cantidad in tipos.items():
            print(f"  {tipo}: {cantidad}")
        
        input("\nPresione Enter para continuar...")
    
    def detectar_imagen_individual_mejorada(self):
        """Detección en imagen individual usando el detector de video."""
        print("\nDETECCIÓN EN IMAGEN INDIVIDUAL")
        print("=" * 35)
        
        if not self.modelo_activo:
            print("Error: Seleccione un modelo primero")
            return
        
        # Cargar el modelo en el detector de video
        if not self.detector_video.cargar_modelo(self.modelo_activo):
            print(f"Error cargando modelo {self.modelo_activo}")
            return
        
        print("Seleccione fuente de imagen:")
        print("1. Seleccionar de carpeta de imágenes")
        print("2. Ruta específica de archivo")
        print("3. Captura desde cámara")
        print("0. Volver")
        
        try:
            opcion = input("Seleccione opción: ").strip()
            
            if opcion == '1':
                self._seleccionar_imagen_carpeta()
            elif opcion == '2':
                ruta_imagen = input("Ruta completa de la imagen: ").strip()
                if os.path.exists(ruta_imagen) and os.path.isfile(ruta_imagen):
                    self._procesar_imagen_archivo(ruta_imagen)
                else:
                    print("Archivo no encontrado o no es un archivo válido")
            elif opcion == '3':
                self._capturar_desde_camara()
                
        except Exception as e:
            print(f"Error en detección de imagen: {e}")
    
    def _seleccionar_imagen_carpeta(self, solo_ruta=False):
        """Permite seleccionar una imagen de la carpeta de imágenes.
        
        Args:
            solo_ruta: Si es True, solo retorna la ruta sin procesarla.
                      Si es False, procesa la imagen directamente.
        
        Returns:
            str o None: Ruta de la imagen si solo_ruta=True, None en caso contrario.
        """
        from utils.image_utils import ImageHandler
        
        directorio_imagenes = "images"  # Carpeta por defecto
        
        # Obtener imágenes disponibles
        imagenes = ImageHandler.obtener_imagenes_carpeta(directorio_imagenes)
        
        if not imagenes:
            print(f"\nNo hay imágenes disponibles en {directorio_imagenes}")
            return None
        
        # Mostrar lista de imágenes
        print(f"\nImágenes disponibles en {directorio_imagenes}:")
        print("-" * 60)
        for i, ruta_imagen in enumerate(imagenes, 1):
            nombre = os.path.basename(ruta_imagen)
            print(f"{i:2d}. {nombre}")
        print("-" * 60)
        print(f"Total: {len(imagenes)} imágenes")
        
        try:
            seleccion = input("\nNúmero de imagen a procesar (0 para cancelar): ").strip()
            
            if seleccion == '0':
                return None
            
            indice = int(seleccion) - 1
            
            if 0 <= indice < len(imagenes):
                ruta_imagen = imagenes[indice]
                
                if solo_ruta:
                    return ruta_imagen
                else:
                    self._procesar_imagen_archivo(ruta_imagen)
                    return None
            else:
                print("Número de imagen no válido")
                return None
                
        except ValueError:
            print("Por favor, ingrese un número válido")
            return None
    
    def _procesar_imagen_archivo(self, ruta_imagen):
        """Procesa una imagen desde archivo con preprocesamiento interactivo opcional."""
        try:
            # Verificar que el archivo existe y es un archivo
            if not os.path.exists(ruta_imagen):
                print(f"Error: El archivo no existe: {ruta_imagen}")
                return
            
            if not os.path.isfile(ruta_imagen):
                print(f"Error: La ruta no es un archivo: {ruta_imagen}")
                return
            
            # Cargar imagen usando ImageHandler para consistencia
            from utils.image_utils import ImageHandler
            imagen_original = ImageHandler.cargar_imagen(ruta_imagen)
            
            if imagen_original is None:
                print("Error cargando la imagen")
                return
            
            print(f"Procesando imagen: {os.path.basename(ruta_imagen)}")
            print(f"Dimensiones: {imagen_original.shape[1]}x{imagen_original.shape[0]}")
            
            # ===== PREPROCESAMIENTO INTERACTIVO =====
            print("\n🔧 PREPROCESAMIENTO DE IMAGEN")
            print("=" * 50)
            print("¿Desea aplicar técnicas de preprocesamiento a la imagen?")
            print("Esto puede mejorar la calidad de detección en ciertos casos.")
            
            aplicar_preprocesamiento = input("¿Aplicar preprocesamiento? (s/n): ").lower().strip()
            
            imagen_a_procesar = imagen_original.copy()
            tecnicas_aplicadas = []
            
            if aplicar_preprocesamiento in ['s', 'si', 'sí', 'y', 'yes']:
                # Permitir seleccionar técnicas de preprocesamiento
                tecnicas_seleccionadas = self.preprocesador.seleccionar_tecnicas()
                
                if tecnicas_seleccionadas:
                    # Aplicar técnicas seleccionadas
                    imagen_a_procesar = self.preprocesador.aplicar_tecnicas_seleccionadas(
                        imagen_original, tecnicas_seleccionadas
                    )
                    tecnicas_aplicadas = tecnicas_seleccionadas
                    
                    # Preguntar si quiere ver comparación
                    ver_comparacion = input("\n¿Desea ver comparación antes/después? (s/n): ").lower().strip()
                    if ver_comparacion in ['s', 'si', 'sí', 'y', 'yes']:
                        self.preprocesador.mostrar_comparacion(
                            imagen_original, imagen_a_procesar, tecnicas_aplicadas
                        )
                else:
                    print("✅ Continuando sin preprocesamiento")
            else:
                print("✅ Continuando sin preprocesamiento")
            
            # ===== DETECCIÓN =====
            print("\n🔍 INICIANDO DETECCIÓN...")
            print("-" * 30)
            
            # Realizar detección en la imagen (original o preprocesada)
            print(f"   Usando modelo: {self.modelo_activo}")
            print(f"   Imagen shape: {imagen_a_procesar.shape}")
            
            resultado = self.detector_video.detectar_en_frame(imagen_a_procesar, self.modelo_activo)
            
            if resultado:
                print(f"✅ Resultado obtenido: tipo={resultado.get('tipo', 'desconocido')}")
                # Dibujar resultado sobre la imagen procesada
                imagen_resultado = self.detector_video.dibujar_detecciones(imagen_a_procesar, resultado)
                
                # Crear ventana con información sobre preprocesamiento
                window_title = 'Resultado de Detección'
                if tecnicas_aplicadas:
                    tecnicas_str = ', '.join([self.preprocesador.tecnicas_disponibles.get(t, t) for t in tecnicas_aplicadas[:3]])
                    if len(tecnicas_aplicadas) > 3:
                        tecnicas_str += f" (+{len(tecnicas_aplicadas)-3} más)"
                    window_title += f" [Preprocesado: {tecnicas_str}]"
                window_title += " - Presione cualquier tecla para cerrar"
                
                # Mostrar resultado
                cv2.imshow(window_title, imagen_resultado)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                # ===== REPORTE DE RESULTADOS =====
                print(f"\n📊 REPORTE DE DETECCIÓN")
                print("=" * 50)
                print(f"Archivo: {os.path.basename(ruta_imagen)}")
                print(f"Modelo usado: {self.modelo_activo.upper()}")
                if tecnicas_aplicadas:
                    print(f"Preprocesamiento aplicado: {len(tecnicas_aplicadas)} técnicas")
                    for i, tecnica in enumerate(tecnicas_aplicadas, 1):
                        print(f"  {i}. {self.preprocesador.tecnicas_disponibles.get(tecnica, tecnica)}")
                else:
                    print("Preprocesamiento: Ninguno")
                print("-" * 50)
                
                # Si es clasificación ImageNet, mostrar detalles
                if resultado.get('tipo') == 'clasificacion_imagenet':
                    print(f"🏷️  CLASIFICACIÓN IMAGENET (1000 clases)")
                    clase_principal = resultado['clase']
                    print(f"Clase principal: {clase_principal}")
                    print(f"Confianza: {resultado['confianza']:.3f} ({resultado['confianza']*100:.1f}%)")
                    print("\n🏆 Top 5 clases detectadas:")
                    for i, pred in enumerate(resultado.get('top_5_clases', [])[:5]):
                        nombre_limpio = pred['clase']
                        print(f"  {i+1}. {nombre_limpio}: {pred['confianza']:.3f} ({pred['confianza']*100:.1f}%)")
                    
                    deteccion_sombrero = resultado.get('deteccion_sombrero', 'sin_sombrero')
                    mejor_sombrero = resultado.get('mejor_sombrero')
                    
                    print(f"\n👒 Análisis de sombrero: {deteccion_sombrero}")
                    if mejor_sombrero:
                        nombre_sombrero = mejor_sombrero['clase']
                        print(f"Mejor detección de sombrero: {nombre_sombrero} ({mejor_sombrero['confianza']:.3f})")
                    
                elif resultado.get('tipo') == 'segmentacion_unet':
                    # Segmentación U-Net
                    print(f"🔍 SEGMENTACIÓN U-NET")
                    print(f"Clase: {resultado['clase']}")
                    print(f"Confianza: {resultado['confianza']:.3f} ({resultado['confianza']*100:.1f}%)")
                    print(f"Área segmentada: {resultado.get('area_segmentada', 0)} píxeles")
                    print(f"Porcentaje: {resultado.get('porcentaje', 0):.2f}% del frame")
                    print(f"Objetos detectados: {resultado.get('num_objetos', 0)}")
                    
                    if 'metricas' in resultado:
                        metricas = resultado['metricas']
                        print(f"Área contorno principal: {metricas.get('area_contorno_principal', 0)}")
                        print(f"Densidad: {metricas.get('densidad', 0):.2f}%")
                    
                    # Mostrar BBox si existe
                    bbox = resultado.get('bbox')
                    if bbox:
                        print(f"Bounding Box: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")
                
                elif resultado.get('tipo') == 'segmentacion_mask_rcnn':
                    # Segmentación Mask R-CNN
                    print(f"🔍 SEGMENTACIÓN MASK R-CNN")
                    print(f"Clase: {resultado['clase']}")
                    print(f"Confianza: {resultado['confianza']:.3f} ({resultado['confianza']*100:.1f}%)")
                    
                    instancias = resultado.get('instancias', [])
                    print(f"Instancias detectadas: {len(instancias)}")
                    
                    for i, inst in enumerate(instancias[:5]):  # Mostrar máximo 5
                        print(f"  Instancia {i+1}: {inst.get('clase', 'N/A')} (conf: {inst.get('confianza', 0):.2f})")
                
                else:
                    # Para otros tipos de detección (YOLO, etc.)
                    print(f"🔍 DETECCIÓN GENERAL")
                    print(f"Tipo: {resultado.get('tipo', 'desconocido')}")
                    print(f"Clase: {resultado['clase']}")
                    print(f"Confianza: {resultado['confianza']:.3f}")
                    print(f"Tipo: {resultado['tipo']}")
                    
                    # Mostrar estadísticas adicionales para segmentación
                    if resultado['tipo'] in ['segmentacion_unet', 'segmentacion_semantica', 'segmentacion_instancias']:
                        print(f"\n📊 Estadísticas de Segmentación:")
                        print(f"Área segmentada: {resultado.get('area_segmentada', 0):,} píxeles")
                        print(f"Porcentaje de imagen: {resultado.get('porcentaje', 0):.2f}%")
                        
                        if 'num_objetos' in resultado:
                            print(f"Objetos detectados: {resultado['num_objetos']}")
                        
                        if 'bbox' in resultado and resultado['bbox']:
                            bbox = resultado['bbox']
                            print(f"Bounding Box: ({bbox[0]}, {bbox[1]}) - {bbox[2]}x{bbox[3]}")
                        
                        # Métricas adicionales de U-Net
                        if 'metricas' in resultado:
                            metricas = resultado['metricas']
                            print(f"\n🔬 Métricas Detalladas:")
                            print(f"  • Área contorno principal: {metricas.get('area_contorno_principal', 0):,} px")
                            print(f"  • Densidad de segmentación: {metricas.get('densidad', 0):.1f}%")
                            print(f"  • Resolución: {metricas.get('pixeles_totales', 0):,} px totales")
                        
                        # Clases detectadas (para segmentación semántica)
                        if 'clases_detectadas' in resultado:
                            print(f"\n🏷️  Clases detectadas:")
                            for clase in resultado['clases_detectadas']:
                                print(f"  • {clase}")
                
                # Guardar información del procesamiento
                if tecnicas_aplicadas:
                    info_archivo = {
                        'archivo': ruta_imagen,
                        'modelo': self.modelo_activo,
                        'tecnicas_preprocesamiento': tecnicas_aplicadas,
                        'resultado': resultado,
                        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                    }
                    
                    # Opcional: guardar reporte
                    guardar_reporte = input("\n💾 ¿Desea guardar un reporte de este procesamiento? (s/n): ").lower().strip()
                    if guardar_reporte in ['s', 'si', 'sí', 'y', 'yes']:
                        self._guardar_reporte_procesamiento(info_archivo)
                
                # Preguntar si guardar resultado
                guardar = input("\n¿Guardar imagen con detecciones? (s/N): ").strip().lower()
                if guardar == 's':
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    nombre_archivo = f"deteccion_{self.modelo_activo}_{timestamp}.jpg"
                    ruta_salida = os.path.join("resultados_deteccion", "neural_networks", nombre_archivo)
                    
                    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
                    cv2.imwrite(ruta_salida, imagen_resultado)
                    print(f"✅ Imagen guardada en: {ruta_salida}")
                
            else:
                print("❌ No se detectaron objetos en la imagen")
                
        except Exception as e:
            print(f"Error procesando imagen: {e}")
            import traceback
            traceback.print_exc()
    
    def _capturar_desde_camara(self):
        """Captura una imagen desde la cámara para análisis."""
        print("Capturando desde cámara. Presione ESPACIO para capturar, 'q' para salir")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            return
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error capturando frame")
                    break
                
                cv2.imshow('Captura de Imagen - ESPACIO: capturar, Q: salir', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    # Procesar el frame capturado
                    resultado = self.detector_video.detectar_en_frame(frame, self.modelo_activo)
                    
                    if resultado:
                        frame_resultado = self.detector_video.dibujar_detecciones(frame, resultado)
                        cv2.imshow('Resultado - Presione cualquier tecla para continuar', frame_resultado)
                        cv2.waitKey(0)
                        
                        print("\nResultado de la captura:")
                        print(f"Clase: {resultado['clase']}")
                        print(f"Confianza: {resultado['confianza']:.3f}")
                    else:
                        print("No se detectaron objetos en la captura")
                    
                elif key == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def _guardar_reporte_procesamiento(self, info_archivo):
        """Guarda un reporte detallado del procesamiento realizado."""
        try:
            # Crear directorio de reportes si no existe
            directorio_reportes = os.path.join("resultados_deteccion", "reportes_preprocesamiento")
            os.makedirs(directorio_reportes, exist_ok=True)
            
            # Generar nombre de archivo único
            timestamp = info_archivo['timestamp']
            nombre_base = os.path.splitext(os.path.basename(info_archivo['archivo']))[0]
            nombre_reporte = f"reporte_{nombre_base}_{timestamp}.json"
            ruta_reporte = os.path.join(directorio_reportes, nombre_reporte)
            
            # Preparar datos para el reporte
            reporte = {
                'metadata': {
                    'archivo_origen': info_archivo['archivo'],
                    'nombre_archivo': os.path.basename(info_archivo['archivo']),
                    'timestamp': timestamp,
                    'fecha_procesamiento': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'modelo_usado': info_archivo['modelo']
                },
                'preprocesamiento': {
                    'tecnicas_aplicadas': info_archivo['tecnicas_preprocesamiento'],
                    'descripcion_tecnicas': [
                        self.preprocesador.tecnicas_disponibles.get(t, t) 
                        for t in info_archivo['tecnicas_preprocesamiento']
                    ]
                },
                'resultado_deteccion': info_archivo['resultado'],
                'configuracion_sistema': {
                    'umbral_confianza': self.detector_video.configuracion.get('umbral_confianza', 0.5),
                    'version_sistema': 'mejorado_v2.0'
                }
            }
            
            # Guardar reporte en JSON
            with open(ruta_reporte, 'w', encoding='utf-8') as f:
                json.dump(reporte, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Reporte guardado: {ruta_reporte}")
            
            # También crear un reporte en texto plano más legible
            nombre_txt = f"reporte_{nombre_base}_{timestamp}.txt"
            ruta_txt = os.path.join(directorio_reportes, nombre_txt)
            
            with open(ruta_txt, 'w', encoding='utf-8') as f:
                f.write("REPORTE DE PROCESAMIENTO DE IMAGEN CON PREPROCESAMIENTO\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Archivo: {os.path.basename(info_archivo['archivo'])}\n")
                f.write(f"Ruta completa: {info_archivo['archivo']}\n")
                f.write(f"Modelo usado: {info_archivo['modelo'].upper()}\n\n")
                
                f.write("TÉCNICAS DE PREPROCESAMIENTO APLICADAS:\n")
                f.write("-" * 40 + "\n")
                if info_archivo['tecnicas_preprocesamiento']:
                    for i, tecnica in enumerate(info_archivo['tecnicas_preprocesamiento'], 1):
                        desc = self.preprocesador.tecnicas_disponibles.get(tecnica, tecnica)
                        f.write(f"{i}. {desc}\n")
                else:
                    f.write("Ninguna técnica aplicada\n")
                
                f.write("\nRESULTADO DE LA DETECCIÓN:\n")
                f.write("-" * 30 + "\n")
                resultado = info_archivo['resultado']
                
                if resultado.get('tipo') == 'clasificacion_imagenet':
                    f.write(f"Tipo: Clasificación ImageNet (1000 clases)\n")
                    f.write(f"Clase principal: {resultado['clase']}\n")
                    f.write(f"Confianza: {resultado['confianza']:.3f} ({resultado['confianza']*100:.1f}%)\n\n")
                    
                    f.write("Top 5 clases detectadas:\n")
                    for i, pred in enumerate(resultado.get('top_5_clases', [])[:5]):
                        nombre = pred['clase']
                        f.write(f"  {i+1}. {nombre}: {pred['confianza']:.3f} ({pred['confianza']*100:.1f}%)\n")
                    
                    if resultado.get('mejor_sombrero'):
                        f.write(f"\nAnálisis de sombrero: {resultado.get('deteccion_sombrero', 'sin_sombrero')}\n")
                        mejor = resultado['mejor_sombrero']
                        f.write(f"Mejor detección de sombrero: {mejor['clase']} ({mejor['confianza']:.3f})\n")
                else:
                    f.write(f"Clase: {resultado['clase']}\n")
                    f.write(f"Confianza: {resultado['confianza']:.3f}\n")
                    f.write(f"Tipo: {resultado['tipo']}\n")
                
                f.write(f"\nConfiguración del sistema:\n")
                f.write(f"Umbral de confianza: {self.detector_video.configuracion.get('umbral_confianza', 0.5)}\n")
                f.write(f"Versión: Sistema Mejorado v2.0\n")
            
            print(f"✅ Reporte de texto guardado: {ruta_txt}")
            
        except Exception as e:
            print(f"❌ Error guardando reporte: {e}")

    def comparativa_modelos(self):
        """
        Realiza una comparativa completa de todos los modelos disponibles.
        Similar al notebook, con estadísticas y gráficos de las detecciones.
        """
        print("\n" + "=" * 70)
        print("COMPARATIVA DE MODELOS - ANÁLISIS COMPLETO")
        print("=" * 70)
        
        # Seleccionar imagen
        print("\n📷 SELECCIÓN DE IMAGEN")
        print("-" * 40)
        ruta_imagen = self._seleccionar_imagen_carpeta(solo_ruta=True)
        
        if not ruta_imagen:
            print("❌ No se seleccionó ninguna imagen")
            return
        
        # Cargar imagen
        try:
            imagen_original = cv2.imread(ruta_imagen)
            if imagen_original is None:
                print(f"❌ Error: No se pudo cargar la imagen desde {ruta_imagen}")
                return
            
            imagen_original = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2RGB)
            print(f"✅ Imagen cargada: {os.path.basename(ruta_imagen)}")
            print(f"   Tamaño: {imagen_original.shape}")
            
        except Exception as e:
            print(f"❌ Error cargando imagen: {e}")
            return
        
        # Preguntar por preprocesamiento
        print("\n" + "=" * 70)
        print("🔧 PREPROCESAMIENTO OPCIONAL")
        print("=" * 70)
        print("El preprocesamiento puede mejorar la calidad de detección.")
        print("\nTécnicas disponibles:")
        print("  1. Ecualización de histograma (mejorar iluminación)")
        print("  2. CLAHE (mejorar contraste local)")
        print("  3. Mejora de saturación")
        print("  4. Filtro bilateral (reducir ruido)")
        
        aplicar_prep = input("\n¿Aplicar preprocesamiento? (s/n): ").lower().strip()
        
        imagen_a_procesar = imagen_original.copy()
        tecnicas_aplicadas = []
        
        if aplicar_prep in ['s', 'si', 'sí', 'y', 'yes']:
            tecnicas_seleccionadas = self.preprocesador.seleccionar_tecnicas()
            
            if tecnicas_seleccionadas:
                imagen_a_procesar = self.preprocesador.aplicar_tecnicas_seleccionadas(
                    imagen_original, tecnicas_seleccionadas
                )
                tecnicas_aplicadas = tecnicas_seleccionadas
                
                # Mostrar comparación
                ver_comp = input("\n¿Ver comparación antes/después? (s/n): ").lower().strip()
                if ver_comp in ['s', 'si', 'sí', 'y', 'yes']:
                    self.preprocesador.mostrar_comparacion(
                        imagen_original, imagen_a_procesar, tecnicas_aplicadas
                    )
        
        # Seleccionar modelos para comparar
        print("\n" + "=" * 70)
        print("🧠 SELECCIÓN DE MODELOS")
        print("=" * 70)
        
        # Modelos de ImageNet disponibles para comparación
        modelos_imagenet = ['vgg16', 'resnet50', 'resnet101']
        
        print(f"Modelos de ImageNet disponibles: {len(modelos_imagenet)}")
        for i, modelo in enumerate(modelos_imagenet, 1):
            print(f"  {i}. {modelo.upper()}")
        
        seleccionar = input("\n¿Analizar todos los modelos de ImageNet? (s/n): ").lower().strip()
        
        if seleccionar not in ['s', 'si', 'sí', 'y', 'yes']:
            print("Comparativa cancelada")
            return
        
        # Ejecutar análisis con todos los modelos
        print("\n" + "=" * 70)
        print("🔍 EJECUTANDO ANÁLISIS COMPARATIVO")
        print("=" * 70)
        
        resultados_comparativa = {}
        
        for modelo_nombre in modelos_imagenet:
            print(f"\nAnalizando con {modelo_nombre.upper()}...")
            
            try:
                # Cargar modelo
                if not self.detector_video.cargar_modelo(modelo_nombre):
                    print(f"  ❌ Error cargando {modelo_nombre}")
                    continue
                
                # Realizar detección
                resultado = self.detector_video.detectar_en_frame(imagen_a_procesar, modelo_nombre)
                
                if resultado and resultado.get('tipo') == 'clasificacion_imagenet':
                    resultados_comparativa[modelo_nombre] = resultado
                    top_clase = resultado['clase']
                    top_conf = resultado['confianza']
                    print(f"  ✅ Clase principal: {top_clase} ({top_conf:.3f})")
                else:
                    print(f"  ⚠️  No se obtuvieron resultados válidos")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        if not resultados_comparativa:
            print("\n❌ No se pudieron obtener resultados de ningún modelo")
            return
        
        # Generar visualizaciones y estadísticas
        self._generar_visualizaciones_comparativa(
            imagen_original,
            imagen_a_procesar,
            resultados_comparativa,
            tecnicas_aplicadas,
            os.path.basename(ruta_imagen)
        )
        
        # Generar reporte
        self._generar_reporte_comparativa(
            resultados_comparativa,
            tecnicas_aplicadas,
            ruta_imagen
        )
        
        print("\n" + "=" * 70)
        print("✅ COMPARATIVA COMPLETADA")
        print("=" * 70)

    def _generar_visualizaciones_comparativa(self, imagen_original, imagen_procesada, 
                                            resultados, tecnicas, nombre_archivo):
        """Genera visualizaciones estilo notebook para la comparativa."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            
            num_modelos = len(resultados)
            
            # Crear figura grande con grid personalizado
            fig = plt.figure(figsize=(20, 12))
            gs = gridspec.GridSpec(3, num_modelos + 1, figure=fig, hspace=0.3, wspace=0.3)
            
            # Fila 1: Imagen original y procesada
            ax_orig = fig.add_subplot(gs[0, 0])
            ax_orig.imshow(imagen_original)
            ax_orig.set_title(f"📷 Imagen Original\n{nombre_archivo}", fontsize=10, fontweight='bold')
            ax_orig.axis('off')
            
            if tecnicas:
                ax_proc = fig.add_subplot(gs[0, 1])
                ax_proc.imshow(imagen_procesada)
                tecnicas_str = ', '.join([self.preprocesador.tecnicas_disponibles.get(t, t) for t in tecnicas[:2]])
                if len(tecnicas) > 2:
                    tecnicas_str += f" (+{len(tecnicas)-2})"
                ax_proc.set_title(f"🔧 Preprocesada\n{tecnicas_str}", fontsize=10, fontweight='bold')
                ax_proc.axis('off')
            
            # Fila 2 y 3: Resultados por modelo
            colores = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
            
            for idx, (modelo_nombre, resultado) in enumerate(resultados.items()):
                # Top 5 clases en gráfico de barras
                ax_bar = fig.add_subplot(gs[1, idx])
                
                top_5 = resultado.get('top_5_clases', [])[:5]
                nombres = [pred['clase'][:20] for pred in top_5]
                confianzas = [pred['confianza'] * 100 for pred in top_5]
                
                bars = ax_bar.barh(range(len(nombres)), confianzas, color=colores[:len(nombres)])
                ax_bar.set_yticks(range(len(nombres)))
                ax_bar.set_yticklabels(nombres, fontsize=8)
                ax_bar.set_xlabel('Confianza (%)', fontsize=9)
                ax_bar.set_title(f'{modelo_nombre.upper()}\nTop 5 Predicciones', 
                                fontsize=10, fontweight='bold')
                ax_bar.set_xlim(0, 100)
                ax_bar.invert_yaxis()
                
                # Agregar valores
                for i, (bar, conf) in enumerate(zip(bars, confianzas)):
                    ax_bar.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                              f'{conf:.1f}%', va='center', fontsize=8, fontweight='bold')
                
                # Información detallada
                ax_info = fig.add_subplot(gs[2, idx])
                ax_info.axis('off')
                
                info_texto = f"🏆 Clase Principal:\n{top_5[0]['clase']}\n\n"
                info_texto += f"📊 Confianza: {top_5[0]['confianza']*100:.2f}%\n\n"
                info_texto += "📋 Top 3:\n"
                for i, pred in enumerate(top_5[:3], 1):
                    info_texto += f"{i}. {pred['clase'][:25]}\n   {pred['confianza']*100:.1f}%\n"
                
                ax_info.text(0.1, 0.9, info_texto, transform=ax_info.transAxes,
                           fontsize=8, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            # Título general
            prep_info = f" [Preprocesamiento: {len(tecnicas)} técnicas]" if tecnicas else ""
            fig.suptitle(f'Comparativa de Modelos CNN - ImageNet{prep_info}',
                        fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            
            # Guardar figura
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ruta_salida = os.path.join("resultados_deteccion", "comparativas", 
                                      f"comparativa_{timestamp}.png")
            os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
            plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
            print(f"\n📊 Visualización guardada: {ruta_salida}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Error generando visualizaciones: {e}")
            import traceback
            traceback.print_exc()
    
    def _generar_reporte_comparativa(self, resultados, tecnicas, ruta_imagen):
        """Genera un reporte detallado de la comparativa."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ruta_reporte = os.path.join("resultados_deteccion", "comparativas",
                                       f"reporte_comparativa_{timestamp}.txt")
            os.makedirs(os.path.dirname(ruta_reporte), exist_ok=True)
            
            with open(ruta_reporte, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("REPORTE DE COMPARATIVA DE MODELOS CNN\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Imagen analizada: {os.path.basename(ruta_imagen)}\n")
                f.write(f"Ruta completa: {ruta_imagen}\n\n")
                
                if tecnicas:
                    f.write("PREPROCESAMIENTO APLICADO:\n")
                    f.write("-" * 40 + "\n")
                    for i, tecnica in enumerate(tecnicas, 1):
                        nombre_tecnica = self.preprocesador.tecnicas_disponibles.get(tecnica, tecnica)
                        f.write(f"  {i}. {nombre_tecnica}\n")
                    f.write("\n")
                else:
                    f.write("PREPROCESAMIENTO: Ninguno\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("RESULTADOS POR MODELO\n")
                f.write("=" * 80 + "\n\n")
                
                for modelo_nombre, resultado in resultados.items():
                    f.write(f"{'─' * 80}\n")
                    f.write(f"MODELO: {modelo_nombre.upper()}\n")
                    f.write(f"{'─' * 80}\n\n")
                    
                    top_5 = resultado.get('top_5_clases', [])
                    
                    f.write(f"Clase principal: {top_5[0]['clase']}\n")
                    f.write(f"Confianza: {top_5[0]['confianza']:.4f} ({top_5[0]['confianza']*100:.2f}%)\n\n")
                    
                    f.write("Top 5 Predicciones:\n")
                    for i, pred in enumerate(top_5, 1):
                        f.write(f"  {i}. {pred['clase']:<40} {pred['confianza']:.4f} ({pred['confianza']*100:.2f}%)\n")
                    
                    # Análisis de sombrero
                    if resultado.get('deteccion_sombrero') == 'con_sombrero':
                        f.write(f"\n👒 Detección de sombrero: SÍ\n")
                        if resultado.get('mejor_sombrero'):
                            mejor = resultado['mejor_sombrero']
                            f.write(f"   Mejor detección: {mejor['clase']} ({mejor['confianza']:.4f})\n")
                    else:
                        f.write(f"\n👒 Detección de sombrero: NO\n")
                    
                    f.write("\n")
                
                f.write("=" * 80 + "\n")
                f.write("ANÁLISIS COMPARATIVO\n")
                f.write("=" * 80 + "\n\n")
                
                # Comparar predicciones principales
                clases_principales = {}
                for modelo, resultado in resultados.items():
                    clase = resultado.get('top_5_clases', [{}])[0].get('clase', 'N/A')
                    conf = resultado.get('top_5_clases', [{}])[0].get('confianza', 0)
                    clases_principales[modelo] = (clase, conf)
                
                f.write("Predicciones principales por modelo:\n")
                for modelo, (clase, conf) in sorted(clases_principales.items(), 
                                                    key=lambda x: x[1][1], reverse=True):
                    f.write(f"  • {modelo.upper():<15} → {clase:<40} ({conf*100:.2f}%)\n")
                
                # Consenso
                clases_unicas = set(c for c, _ in clases_principales.values())
                f.write(f"\nClases únicas predichas: {len(clases_unicas)}\n")
                
                if len(clases_unicas) == 1:
                    f.write("✅ CONSENSO TOTAL: Todos los modelos predicen la misma clase\n")
                elif len(clases_unicas) == len(clases_principales):
                    f.write("⚠️  SIN CONSENSO: Cada modelo predice una clase diferente\n")
                else:
                    f.write("🔶 CONSENSO PARCIAL: Algunos modelos coinciden\n")
                
                f.write(f"\n{'=' * 80}\n")
                f.write("Reporte generado por Sistema de Detección Mejorado v2.0\n")
                f.write("Universidad del Quindío - Visión Artificial\n")
                f.write("=" * 80 + "\n")
            
            print(f"📄 Reporte guardado: {ruta_reporte}")
            
        except Exception as e:
            print(f"❌ Error generando reporte: {e}")

    def mostrar_ayuda(self):
        """Muestra información de ayuda del sistema."""
        print("\nAYUDA - SISTEMA DE DETECCIÓN DE SOMBREROS")
        print("=" * 50)
        print("\nFuncionalidades principales:")
        print("1. DETECCIÓN EN IMAGEN:")
        print("   - Seleccione un modelo (opción 3)")
        print("   - Vaya a detección en imagen (opción 1)")
        print("   - Opción de aplicar preprocesamiento interactivo")
        print("   - Proporcione la ruta de la imagen")
        
        print("\n2. DETECCIÓN EN VIDEO:")
        print("   - Seleccione un modelo (opción 3)")
        print("   - Vaya a detección en video (opción 2)")
        print("   - Elija entre cámara web o archivo de video")
        print("   - Use 'q' para salir durante la detección")
        
        print("\n3. MODELOS DISPONIBLES:")
        print("   Clasificación: LeNet, AlexNet, VGG16, ResNet50, ResNet101")
        print("   Detección: YOLO, SSD, R-CNN")
        print("   Segmentación: U-Net, Mask R-CNN")
        
        print("\n4. COMPARATIVA DE MODELOS (NUEVO):")
        print("   - Analiza una imagen con todos los modelos de ImageNet")
        print("   - Genera gráficos comparativos estilo notebook")
        print("   - Muestra Top 5 clases para cada modelo")
        print("   - Opción de preprocesamiento antes del análisis")
        print("   - Reporte detallado con análisis de consenso")
        
        print("\n5. PREPROCESAMIENTO:")
        print("   - Ecualización de histograma (mejorar iluminación)")
        print("   - CLAHE (mejorar contraste local)")
        print("   - Mejora de saturación de colores")
        print("   - Filtro bilateral (reducir ruido)")
        print("   - Ajustes de brillo y contraste")
        print("   - Corrección gamma")
        
        print("\n6. CONTROLES DE VIDEO:")
        print("   - 'q': Salir")
        print("   - 'c': Cambiar configuración en tiempo real")
        
        print("\n7. CONFIGURACIÓN:")
        print("   - Umbral de confianza: Ajusta sensibilidad")
        print("   - FPS objetivo: Controla velocidad de procesamiento")
        
        print("\nNOTA: Los modelos se cargan automáticamente cuando se seleccionan")
        print("      Algunos modelos requieren PyTorch instalado")
        print("      La comparativa usa modelos VGG16, ResNet50 y ResNet101")
        
        input("\nPresione Enter para continuar...")

def main():
    """Función principal."""
    sistema = SistemaDeteccionSombrerosMejorado()
    sistema.ejecutar_sistema()

if __name__ == "__main__":
    main()