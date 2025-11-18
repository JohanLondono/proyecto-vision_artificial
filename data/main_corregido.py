#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema Principal para Parcial de Visión Artificial
=================================================

Implementa la Parte I del parcial: Exploración y preprocesamiento
- Carga y visualización de imágenes del dataset
- Preprocesamiento: redimensionamiento a 224x224 y normalización [0,1]
- Análisis del problema de clasificación
- Gestión de estados guardados

Universidad del Quindío - Visión Artificial
Autor: Sistema de Visión Artificial  
Fecha: Noviembre 2024
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configurar matplotlib para mostrar gráficos
try:
    plt.ion()  # Modo interactivo
except:
    pass

# Importar módulos locales
from gestor_imagenes import GestorImagenes
from preprocesador_parcial import PreprocesadorParcial
from preprocesador_avanzado_cnn import PreprocesadorAvanzadoCNN
from redes_preentrenadas import RedesPreentrenadas


class SistemaVisionArtificialParcial:
    """
    Sistema principal para el parcial de visión artificial.
    Implementa carga, visualización y preprocesamiento de imágenes.
    """
    
    def __init__(self):
        """Inicializar el sistema con los componentes necesarios."""
        # Obtener directorio de trabajo
        self.directorio_trabajo = os.path.dirname(os.path.abspath(__file__))
        
        # Configurar rutas
        self.ruta_imagenes = os.path.join(self.directorio_trabajo, "images")
        
        # Inicializar módulos
        self.gestor_imagenes = GestorImagenes(self.ruta_imagenes)
        self.preprocesador = PreprocesadorParcial()
        self.preprocesador_avanzado = PreprocesadorAvanzadoCNN()
        self.redes_cnn = None  # Se inicializará cuando se acceda al menú CNN
        
        # Variables de estado
        self.imagen_actual = None
        self.imagen_preprocesada = None
        self.nombre_actual = ""
        self.info_preprocesamiento = None
        
    def mostrar_encabezado(self):
        """Muestra el encabezado del sistema."""
        print("\n" + "="*80)
        print("         SISTEMA DE PREPROCESAMIENTO DE IMÁGENES")
        print("              Universidad del Quindío - Visión Artificial")  
        print("                     Parcial 3 - Parte I")
        print("=" * 80)
        
    def mostrar_menu_principal(self):
        """Muestra el menú principal del sistema."""
        print("\nMENÚ PRINCIPAL")
        print("-" * 40)
        print("1. Carga y Visualización de Imágenes")
        print("2. Preprocesamiento de Imágenes") 
        print("3. Redes CNN Preentrenadas")
        print("4. Análisis del Dataset")
        print("5. Gestionar Estados Guardados")
        print("6. Información del Sistema")
        print("7. Configurar Ruta de Imágenes")
        print("8. Salir")
        print("-" * 40)
        
    def ejecutar_sistema(self):
        """Ejecuta el bucle principal del sistema."""
        self.mostrar_encabezado()
        
        while True:
            try:
                self.mostrar_menu_principal()
                opcion = input("\nSeleccione una opción (1-8): ").strip()
                
                if opcion == "1":
                    self.menu_carga_imagenes()
                elif opcion == "2":
                    self.menu_preprocesamiento()
                elif opcion == "3":
                    self.menu_redes_cnn()
                elif opcion == "4":
                    self.analisis_dataset()
                elif opcion == "5":
                    self.gestionar_estados_guardados()
                elif opcion == "6":
                    self.mostrar_info_sistema()
                elif opcion == "7":
                    self.configurar_ruta_imagenes()
                elif opcion == "8":
                    self.salir_sistema()
                    break
                else:
                    print("Opción no válida. Por favor seleccione 1-8.")
                    
            except KeyboardInterrupt:
                print("\n\nPrograma interrumpido por el usuario.")
                break
            except Exception as e:
                print(f"Error: {e}")
                input("Presione Enter para continuar...")

    def menu_carga_imagenes(self):
        """Maneja el menú de carga de imágenes."""
        while True:
            print("\n" + "="*50)
            print("         CARGA Y VISUALIZACIÓN")
            print("="*50)
            print("1. Cargar imagen específica")
            print("2. Mostrar imágenes disponibles")
            print("3. Cargar imagen aleatoria")
            print("4. Mostrar estadísticas de imagen actual")
            print("5. Volver al menú principal")
            
            opcion = input("\\nSeleccione una opción (1-5): ").strip()
            
            if opcion == "1":
                self.cargar_imagen_especifica()
            elif opcion == "2":
                self.mostrar_imagenes_disponibles()
            elif opcion == "3":
                self.cargar_imagen_aleatoria()
            elif opcion == "4":
                self.mostrar_estadisticas_actual()
            elif opcion == "5":
                break
            else:
                print("Opción no válida.")
                
    def cargar_imagen_especifica(self):
        """Carga una imagen específica."""
        imagenes = self.gestor_imagenes.listar_imagenes()
        if not imagenes:
            print("\\nNo se encontraron imágenes en el directorio.")
            input("Presione Enter para continuar...")
            return
        
        try:
            seleccion = input("\\nIngrese el número de la imagen (o Enter para cancelar): ").strip()
            if not seleccion:
                return
                
            indice = int(seleccion) - 1
            if 0 <= indice < len(imagenes):
                # Usar el nombre del archivo de la lista ya obtenida
                nombre_archivo = imagenes[indice]
                self.imagen_actual = self.gestor_imagenes.cargar_imagen(nombre_archivo=nombre_archivo)
                if self.imagen_actual is not None:
                    self.nombre_actual = nombre_archivo
                    print(f"\nImagen cargada: {self.nombre_actual}")
                    
                    # Mostrar automáticamente
                    self.gestor_imagenes.visualizar_imagen(self.imagen_actual, self.nombre_actual)
                    
            else:
                print("Selección inválida.")
                
        except ValueError:
            print("Entrada inválida. Ingrese un número.")
        except Exception as e:
            print(f"Error: {e}")
            
        input("\nPresione Enter para continuar...")
        
    def cargar_imagen_aleatoria(self):
        """Carga una imagen aleatoria."""
        import random
        imagenes = self.gestor_imagenes.listar_imagenes()
        if imagenes:
            imagen_seleccionada = random.choice(imagenes)
            self.imagen_actual = self.gestor_imagenes.cargar_imagen(nombre_archivo=os.path.basename(imagen_seleccionada))
            if self.imagen_actual is not None:
                self.nombre_actual = os.path.basename(imagen_seleccionada)
                print(f"\nImagen aleatoria cargada: {self.nombre_actual}")
                
                # Mostrar automáticamente
                self.gestor_imagenes.visualizar_imagen(self.imagen_actual, self.nombre_actual)
            else:
                print("\nError cargando la imagen.")
        else:
            print("\nNo hay imágenes disponibles.")
            
        input("\nPresione Enter para continuar...")
        
    def mostrar_imagenes_disponibles(self):
        """Muestra la lista de imágenes disponibles."""
        imagenes = self.gestor_imagenes.listar_imagenes()
        if imagenes:
            print(f"\nTotal: {len(imagenes)} imágenes encontradas")
        input("\nPresione Enter para continuar...")
        
    def mostrar_estadisticas_actual(self):
        """Muestra estadísticas de la imagen actual."""
        if self.imagen_actual is None:
            print("\nNo hay imagen cargada.")
        else:
            print(f"\nEstadísticas de: {self.nombre_actual}")
            estadisticas = self.gestor_imagenes.obtener_estadisticas_imagen(self.imagen_actual)
            if estadisticas:
                for clave, valor in estadisticas.items():
                    print(f"{clave}: {valor}")
            
        input("\\nPresione Enter para continuar...")

    def menu_preprocesamiento(self):
        """Maneja el menú de preprocesamiento."""
        while True:
            print("\n" + "="*50)
            print("         PREPROCESAMIENTO")
            print("="*50)
            print("1. Preprocesamiento completo (224x224 + [0,1])")
            print("2. Solo redimensionar a 224x224")
            print("3. Solo normalizar a [0,1]")
            print("4. Preprocesamiento avanzado para CNN")
            print("5. Augmentación de datos")
            print("6. Aplicar filtros para CNNs")
            print("7. Comparar antes y después")
            print("8. Volver al menú principal")
            
            opcion = input("\\nSeleccione una opción (1-8): ").strip()
            
            if opcion == "1":
                self.preprocesamiento_completo()
            elif opcion == "2":
                self.solo_redimensionar()
            elif opcion == "3":
                self.solo_normalizar()
            elif opcion == "4":
                self.preprocesamiento_avanzado_cnn()
            elif opcion == "5":
                self.augmentacion_datos()
            elif opcion == "6":
                self.aplicar_filtros_cnn()
            elif opcion == "7":
                self.comparar_preprocesamiento()
            elif opcion == "8":
                break
            else:
                print("Opción no válida.")
                
    def preprocesamiento_completo(self):
        """Aplica el preprocesamiento completo."""
        if self.imagen_actual is None:
            print("\\nNo hay imagen cargada. Cargue una imagen primero.")
            input("Presione Enter para continuar...")
            return
            
        print("\\nAplicando preprocesamiento completo...")
        print("   1. Redimensionando a 224x224 píxeles")
        print("   2. Normalizando valores al rango [0,1]")
        
        try:
            resultado = self.preprocesador.preprocesar_imagen_completo(self.imagen_actual)
            if resultado is None:
                print("Error en el preprocesamiento.")
                input("Presione Enter para continuar...")
                return
            
            self.imagen_preprocesada, self.info_preprocesamiento = resultado
            
            print("\\nPreprocesamiento completo aplicado exitosamente")
            
            # Mostrar comparación automáticamente
            print("\nMostrando comparación antes/después...")
            self.gestor_imagenes.mostrar_comparacion_con_info(
                self.imagen_actual,
                self.imagen_preprocesada,
                f"Original ({self.nombre_actual})",
                "Preprocesada",
                self.info_preprocesamiento
            )
            
            # Opciones de guardado
            self._opciones_guardado_preprocesamiento()
                
        except Exception as e:
            print(f"Error durante el preprocesamiento: {e}")
            
        input("\\nPresione Enter para continuar...")
        
    def solo_redimensionar(self):
        """Solo redimensiona la imagen."""
        if self.imagen_actual is None:
            print("\\nNo hay imagen cargada.")
            input("Presione Enter para continuar...")
            return
            
        try:
            from modules.operaciones_geometricas import OperacionesGeometricas
            ops_geo = OperacionesGeometricas()
            
            print("\\nRedimensionando a 224x224...")
            imagen_redimensionada = ops_geo.redimensionar_224x224(self.imagen_actual)
            
            # Crear información del preprocesamiento para la comparación
            info_redim = {
                'dimension_original': self.imagen_actual.shape[:2][::-1],  # (width, height)
                'dimension_final': (224, 224),
                'transformacion': 'Redimensionamiento'
            }
            
            # Mostrar comparación
            self.gestor_imagenes.mostrar_comparacion_con_info(
                self.imagen_actual,
                imagen_redimensionada,
                f"Original {self.imagen_actual.shape}",
                "Redimensionada",
                info_redim
            )
            
            # Preguntar si quiere guardar
            respuesta = input("\\n¿Desea mantener la imagen redimensionada? (s/n): ").lower()
            if respuesta == 's':
                self.imagen_actual = imagen_redimensionada
                print("Imagen redimensionada guardada como actual.")
                
        except Exception as e:
            print(f"Error: {e}")
            
        input("\\nPresione Enter para continuar...")
        
    def solo_normalizar(self):
        """Solo normaliza la imagen."""
        if self.imagen_actual is None:
            print("\\nNo hay imagen cargada.")
            input("Presione Enter para continuar...")
            return
            
        try:
            from modules.operaciones_aritmeticas import OperacionesAritmeticas
            ops_arit = OperacionesAritmeticas()
            
            print("\\nNormalizando al rango [0,1]...")
            imagen_normalizada = ops_arit.normalizar_imagen(self.imagen_actual)
            
            print(f"Rango original: [{np.min(self.imagen_actual):.3f}, {np.max(self.imagen_actual):.3f}]")
            print(f"Rango normalizado: [{np.min(imagen_normalizada):.3f}, {np.max(imagen_normalizada):.3f}]")
            
            # Crear información del preprocesamiento para la comparación
            info_norm = {
                'normalizacion': '[0,1]',
                'estadisticas': {
                    'rango_valores': [np.min(imagen_normalizada), np.max(imagen_normalizada)],
                    'media': np.mean(imagen_normalizada),
                    'desviacion': np.std(imagen_normalizada)
                },
                'transformacion': 'Normalización'
            }
            
            # Mostrar comparación
            self.gestor_imagenes.mostrar_comparacion_con_info(
                self.imagen_actual,
                imagen_normalizada,
                "Original",
                "Normalizada",
                info_norm
            )
            
            # Preguntar si quiere guardar
            respuesta = input("\\n¿Desea mantener la imagen normalizada? (s/n): ").lower()
            if respuesta == 's':
                self.imagen_actual = imagen_normalizada
                print("Imagen normalizada guardada como actual.")
                
        except Exception as e:
            print(f"Error: {e}")
            
        input("\\nPresione Enter para continuar...")
        
    def comparar_preprocesamiento(self):
        """Compara imagen actual con preprocesada."""
        if self.imagen_actual is None:
            print("\\nNo hay imagen actual cargada.")
        elif self.imagen_preprocesada is None:
            print("\\nNo hay imagen preprocesada. Aplique preprocesamiento primero.")
        else:
            print("\nMostrando comparación...")
            self.gestor_imagenes.mostrar_comparacion_con_info(
                self.imagen_actual,
                self.imagen_preprocesada,
                f"Original ({self.nombre_actual})",
                "Preprocesada",
                self.info_preprocesamiento if hasattr(self, 'info_preprocesamiento') else None
            )
            
        input("\\nPresione Enter para continuar...")
    
    def preprocesamiento_avanzado_cnn(self):
        """Aplica preprocesamiento avanzado específico para CNNs."""
        if self.imagen_actual is None:
            print("\\nNo hay imagen cargada. Cargue una imagen primero.")
            input("Presione Enter para continuar...")
            return
            
        print("\\n" + "="*60)
        print("       PREPROCESAMIENTO AVANZADO PARA CNN")
        print("="*60)
        
        # Opciones de normalización
        print("\\nTipo de normalización:")
        print("1. Normalización ImageNet (recomendado para modelos preentrenados)")
        print("2. Normalización básica [0,1]")
        
        norm_opcion = input("\\nSeleccione tipo de normalización (1-2): ").strip()
        normalizacion = 'imagenet' if norm_opcion == '1' else 'basica'
        
        # Augmentación
        aug_opcion = input("¿Incluir augmentación de datos? (s/n): ").strip().lower()
        incluir_aug = aug_opcion in ['s', 'si', 'y', 'yes']
        
        try:
            print("\\nProcesando imagen...")
            resultados = self.preprocesador_avanzado.preprocesamiento_completo_cnn(
                self.imagen_actual, 
                incluir_augmentacion=incluir_aug,
                normalizacion=normalizacion
            )
            
            if resultados:
                self.imagen_preprocesada = resultados['imagen_final']
                self.info_preprocesamiento = resultados
                
                print("\\n✅ Preprocesamiento avanzado completado!")
                
                # Mostrar reporte
                reporte = self.preprocesador_avanzado.generar_reporte_preprocesamiento(resultados)
                print("\\n" + reporte)
                
                # Mostrar comparación
                mostrar = input("\\n¿Desea ver la comparación visual? (s/n): ").strip().lower()
                if mostrar in ['s', 'si', 'y', 'yes']:
                    self.preprocesador_avanzado.visualizar_comparacion_multiple(
                        self.imagen_actual, resultados)
                
                # Preparar para batch
                batch_info = self.preprocesador_avanzado.preparar_batch(self.imagen_preprocesada)
                if batch_info:
                    print("\\n📦 INFORMACIÓN DE BATCH:")
                    print(f"Formato PyTorch: {batch_info['shape_pytorch']}")
                    print(f"Formato TensorFlow: {batch_info['shape_tensorflow']}")
                
                # Opciones de guardado
                self._opciones_guardado_preprocesamiento()
            else:
                print("\\nError en el preprocesamiento avanzado.")
                
        except Exception as e:
            print(f"\\nError: {e}")
            
        input("\\nPresione Enter para continuar...")
        
    def augmentacion_datos(self):
        """Aplica augmentación de datos a la imagen."""
        if self.imagen_actual is None:
            print("\\nNo hay imagen cargada.")
            input("Presione Enter para continuar...")
            return
            
        print("\\n" + "="*50)
        print("       AUGMENTACIÓN DE DATOS")
        print("="*50)
        
        try:
            # Aplicar augmentación múltiples veces para mostrar variedad
            print("\\nGenerando múltiples versiones augmentadas...")
            
            plt.figure(figsize=(15, 10))
            
            # Imagen original
            plt.subplot(2, 3, 1)
            plt.imshow(self.imagen_actual)
            plt.title('Original')
            plt.axis('off')
            
            # Generar 5 versiones augmentadas
            for i in range(5):
                imagen_aug, transformaciones = self.preprocesador_avanzado.augmentacion_basica(
                    self.imagen_actual)
                
                plt.subplot(2, 3, i+2)
                plt.imshow(imagen_aug)
                plt.title(f'Augmentación {i+1}\\n' + ', '.join(transformaciones[:2]))
                plt.axis('off')
            
            plt.tight_layout()
            plt.suptitle('Ejemplos de Augmentación de Datos', fontsize=16, y=1.02)
            plt.show()
            
            # Preguntar si quiere aplicar augmentación a la imagen actual
            aplicar = input("\\n¿Aplicar una augmentación a la imagen actual? (s/n): ").strip().lower()
            if aplicar in ['s', 'si', 'y', 'yes']:
                imagen_aug, transformaciones = self.preprocesador_avanzado.augmentacion_basica(
                    self.imagen_actual)
                
                print(f"\\nTransformaciones aplicadas: {', '.join(transformaciones)}")
                
                # Mostrar comparación
                self.gestor_imagenes.mostrar_comparacion(
                    self.imagen_actual, imagen_aug,
                    "Original", "Con Augmentación"
                )
                
                # Preguntar si mantener
                mantener = input("\\n¿Mantener la imagen augmentada como actual? (s/n): ").strip().lower()
                if mantener in ['s', 'si', 'y', 'yes']:
                    self.imagen_actual = imagen_aug
                    print("Imagen augmentada guardada como actual.")
            
        except Exception as e:
            print(f"\\nError en augmentación: {e}")
            
        input("\\nPresione Enter para continuar...")
        
    def aplicar_filtros_cnn(self):
        """Aplica filtros específicos para mejorar el rendimiento en CNNs."""
        if self.imagen_actual is None:
            print("\\nNo hay imagen cargada.")
            input("Presione Enter para continuar...")
            return
            
        print("\\n" + "="*50)
        print("     FILTROS PARA CNNs")
        print("="*50)
        
        try:
            print("\\nAplicando filtros especializados...")
            filtros = self.preprocesador_avanzado.aplicar_filtros_cnn(self.imagen_actual)
            
            if filtros:
                # Mostrar todos los filtros
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.ravel()
                
                # Original
                axes[0].imshow(self.imagen_actual)
                axes[0].set_title('Original')
                axes[0].axis('off')
                
                # Filtros aplicados
                titulos = ['Filtro Gaussiano', 'Detección de Bordes', 
                          'Realce de Bordes', 'Ecualización', 'Sin usar']
                
                i = 1
                for nombre, imagen_filtrada in filtros.items():
                    if i < 6:
                        if nombre == 'edges':
                            axes[i].imshow(imagen_filtrada, cmap='gray')
                        else:
                            axes[i].imshow(imagen_filtrada)
                        axes[i].set_title(titulos[i-1])
                        axes[i].axis('off')
                        i += 1
                
                # Ocultar el último subplot si no se usa
                if i <= 5:
                    axes[5].axis('off')
                
                plt.tight_layout()
                plt.suptitle('Filtros Especializados para CNNs', fontsize=16, y=1.02)
                plt.show()
                
                # Seleccionar filtro
                print("\\nFiltros disponibles:")
                opciones = list(filtros.keys())
                for i, filtro in enumerate(opciones, 1):
                    print(f"{i}. {filtro}")
                print(f"{len(opciones)+1}. No aplicar ninguno")
                
                seleccion = input(f"\\nSeleccione un filtro (1-{len(opciones)+1}): ").strip()
                try:
                    idx = int(seleccion) - 1
                    if 0 <= idx < len(opciones):
                        filtro_elegido = opciones[idx]
                        imagen_filtrada = filtros[filtro_elegido]
                        
                        # Mostrar comparación
                        self.gestor_imagenes.mostrar_comparacion(
                            self.imagen_actual, imagen_filtrada,
                            "Original", f"Con {filtro_elegido}"
                        )
                        
                        # Preguntar si mantener
                        mantener = input("\\n¿Mantener imagen filtrada como actual? (s/n): ").strip().lower()
                        if mantener in ['s', 'si', 'y', 'yes']:
                            self.imagen_actual = imagen_filtrada
                            print(f"Imagen con filtro {filtro_elegido} guardada como actual.")
                    elif idx == len(opciones):
                        print("No se aplicó ningún filtro.")
                    else:
                        print("Selección inválida.")
                except ValueError:
                    print("Entrada inválida.")
            else:
                print("\\nError aplicando filtros.")
                
        except Exception as e:
            print(f"\\nError: {e}")
            
        input("\\nPresione Enter para continuar...")
        
    def _opciones_guardado_preprocesamiento(self):
        """Muestra opciones para guardar el estado y la imagen preprocesada."""
        if self.imagen_preprocesada is None:
            return
        
        print("\\nOpciones de guardado:")
        print("1. Guardar imagen preprocesada")
        print("2. Guardar estado del procesamiento")
        print("3. Guardar ambos")
        print("4. No guardar")
        
        opcion = input("\\nSeleccione una opción (1-4): ").strip()
        
        if opcion == "1":
            self._guardar_imagen_preprocesada()
        elif opcion == "2":
            self._guardar_estado_preprocesamiento()
        elif opcion == "3":
            self._guardar_imagen_preprocesada()
            self._guardar_estado_preprocesamiento()
        elif opcion == "4":
            print("No se guardó nada.")
        else:
            print("Opción no válida.")
    
    def _guardar_imagen_preprocesada(self):
        """Guarda la imagen preprocesada."""
        try:
            nombre_base = os.path.splitext(self.nombre_actual)[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_preprocesada = f"{nombre_base}_preprocesada_{timestamp}"
            
            ruta_guardada = self.gestor_imagenes.guardar_imagen(
                self.imagen_preprocesada,
                nombre_preprocesada
            )
            
            if ruta_guardada:
                print(f"Imagen preprocesada guardada en: {ruta_guardada}")
            else:
                print("Error al guardar la imagen preprocesada.")
                
        except Exception as e:
            print(f"Error guardando imagen: {e}")
    
    def _guardar_estado_preprocesamiento(self):
        """Guarda el estado del preprocesamiento usando el nuevo sistema."""
        try:
            nombre_base = os.path.splitext(self.nombre_actual)[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_archivo = f"{nombre_base}_estado_{timestamp}"
            
            # Obtener estadísticas de la imagen preprocesada
            estadisticas = {
                "dimensiones": str(self.imagen_preprocesada.shape),
                "tipo_datos": str(self.imagen_preprocesada.dtype),
                "min_valor": float(np.min(self.imagen_preprocesada)),
                "max_valor": float(np.max(self.imagen_preprocesada)),
                "media": float(np.mean(self.imagen_preprocesada)),
                "fecha_procesamiento": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Guardar estado usando el nuevo sistema
            resultado = self.gestor_imagenes.guardar_estado_procesamiento(
                self.imagen_actual, 
                self.imagen_preprocesada, 
                estadisticas, 
                nombre_archivo
            )
            
            if resultado:
                print("\\nEstado guardado correctamente.")
                print("Podrá cargar este estado más tarde para continuar trabajando.")
                
        except Exception as e:
            print(f"Error guardando estado: {e}")
    
    def gestionar_estados_guardados(self):
        """Gestiona los estados de procesamiento guardados."""
        print("\\n" + "="*60)
        print("           GESTIÓN DE ESTADOS GUARDADOS")
        print("="*60)
        
        while True:
            print("\\nOpciones disponibles:")
            print("1. Listar estados guardados")
            print("2. Cargar estado guardado")
            print("3. Mostrar comparación (original vs preprocesada)")
            print("4. Restaurar a imagen original")
            print("5. Volver al menú principal")
            
            opcion = input("\\nSeleccione una opción (1-5): ").strip()
            
            if opcion == "1":
                self._listar_estados()
            elif opcion == "2":
                self._cargar_estado()
            elif opcion == "3":
                self._mostrar_comparacion_estados()
            elif opcion == "4":
                self._restaurar_original()
            elif opcion == "5":
                break
            else:
                print("Opción no válida.")
                
    def _listar_estados(self):
        """Lista todos los estados disponibles."""
        estados = self.gestor_imagenes.listar_estados_disponibles()
        if not estados:
            print("\\nNo hay estados guardados disponibles.")
        input("\\nPresione Enter para continuar...")
        
    def _cargar_estado(self):
        """Carga un estado guardado."""
        estados = self.gestor_imagenes.listar_estados_disponibles()
        if not estados:
            return
            
        try:
            seleccion = input("\\nIngrese el número del estado a cargar (Enter para cancelar): ").strip()
            if not seleccion:
                return
                
            indice = int(seleccion) - 1
            if 0 <= indice < len(estados):
                nombre_estado = estados[indice]
                imagen_original, imagen_preprocesada = self.gestor_imagenes.cargar_estado_procesamiento(nombre_estado)
                
                if imagen_original is not None and imagen_preprocesada is not None:
                    self.imagen_actual = imagen_original
                    self.imagen_preprocesada = imagen_preprocesada
                    self.nombre_actual = f"estado_cargado_{nombre_estado}"
                    print("\\nEstado cargado correctamente.")
                    print("Ahora puede trabajar con la imagen preprocesada o restaurar la original.")
                else:
                    print("\\nError cargando el estado.")
            else:
                print("\\nSelección inválida.")
                
        except ValueError:
            print("\\nEntrada inválida. Ingrese un número.")
        except Exception as e:
            print(f"\\nError: {e}")
            
        input("\\nPresione Enter para continuar...")
        
    def _mostrar_comparacion_estados(self):
        """Muestra comparación entre original y preprocesada si están cargadas."""
        if self.imagen_actual is None or self.imagen_preprocesada is None:
            print("\\nNo hay imágenes cargadas para comparar.")
            print("Primero cargue un estado guardado.")
        else:
            print("\\nMostrando comparación...")
            self.gestor_imagenes.mostrar_comparacion(
                self.imagen_actual, 
                self.imagen_preprocesada,
                "Imagen Original",
                "Imagen Preprocesada"
            )
        input("\\nPresione Enter para continuar...")
        
    def _restaurar_original(self):
        """Restaura la imagen a su estado original."""
        if self.imagen_actual is None:
            print("\\nNo hay imagen original disponible.")
            print("Primero cargue un estado guardado.")
        else:
            self.imagen_preprocesada = self.imagen_actual.copy()
            print("\\nImagen restaurada al estado original.")
            print("La imagen preprocesada ahora es igual a la original.")
        input("\\nPresione Enter para continuar...")

    def analisis_dataset(self):
        """Realiza análisis del dataset."""
        print("\\n" + "="*60)
        print("           ANÁLISIS DEL DATASET")
        print("="*60)
        
        self.gestor_imagenes.analizar_dataset()
        input("\\nPresione Enter para continuar...")
        
    def mostrar_info_sistema(self):
        """Muestra información del sistema."""
        print("\\n" + "="*60)
        print("           INFORMACIÓN DEL SISTEMA")
        print("="*60)
        print(f"Directorio de trabajo: {self.directorio_trabajo}")
        print(f"Ruta de imágenes: {self.ruta_imagenes}")
        
        if self.imagen_actual is not None:
            print(f"\\nImagen actual: {self.nombre_actual}")
            print(f"Dimensiones: {self.imagen_actual.shape}")
            print(f"Tipo de datos: {self.imagen_actual.dtype}")
        else:
            print("\\nNo hay imagen cargada actualmente.")
            
        if self.imagen_preprocesada is not None:
            print(f"\\nImagen preprocesada disponible: {self.imagen_preprocesada.shape}")
        
        input("\\nPresione Enter para continuar...")
    
    def menu_redes_cnn(self):
        """Menú para trabajar con redes CNN preentrenadas."""
        while True:
            print("\\n" + "="*60)
            print("       REDES CNN PREENTRENADAS")
            print("="*60)
            print("1. Cargar modelos CNN")
            print("2. Predicción con modelo individual")
            print("3. Comparación entre todos los modelos")
            print("4. Análisis detallado de predicción")
            print("5. Ver información de modelos")
            print("6. Guardar resultados de predicción")
            print("7. Volver al menú principal")
            
            opcion = input("\\nSeleccione una opción (1-7): ").strip()
            
            if opcion == "1":
                self.cargar_modelos_cnn()
            elif opcion == "2":
                self.prediccion_individual_cnn()
            elif opcion == "3":
                self.comparacion_modelos_cnn()
            elif opcion == "4":
                self.analisis_detallado_cnn()
            elif opcion == "5":
                self.informacion_modelos_cnn()
            elif opcion == "6":
                self.guardar_resultados_cnn()
            elif opcion == "7":
                break
            else:
                print("Opción no válida.")
                
    def cargar_modelos_cnn(self):
        """Carga los modelos CNN preentrenados."""
        print("\\n🔄 Inicializando sistema de redes CNN...")
        
        if self.redes_cnn is None:
            try:
                self.redes_cnn = RedesPreentrenadas()
                print("✅ Sistema CNN inicializado correctamente")
            except Exception as e:
                print(f"❌ Error inicializando sistema CNN: {e}")
                input("\\nPresione Enter para continuar...")
                return
        
        print("\\n" + "="*50)
        print("       CARGA DE MODELOS PREENTRENADOS")
        print("="*50)
        print("1. Cargar MobileNetV2 (ligero, rápido)")
        print("2. Cargar ResNet50 (potente, preciso)")
        print("3. Cargar VGG16 (clásico, robusto)")
        print("4. Cargar todos los modelos")
        print("5. Volver")
        
        opcion = input("\\nSeleccione una opción (1-5): ").strip()
        
        try:
            if opcion == "1":
                self.redes_cnn.cargar_modelo('mobilenet')
            elif opcion == "2":
                self.redes_cnn.cargar_modelo('resnet50')
            elif opcion == "3":
                self.redes_cnn.cargar_modelo('vgg16')
            elif opcion == "4":
                print("\\n🚀 Cargando todos los modelos...")
                print("⚠️  Este proceso puede tardar varios minutos...")
                resultados = self.redes_cnn.cargar_todos_modelos()
                
                exitosos = sum(resultados.values())
                total = len(resultados)
                print(f"\\n📊 Resumen: {exitosos}/{total} modelos cargados")
                
                if exitosos > 0:
                    print("\\n✅ Modelos listos para usar:")
                    for modelo, exito in resultados.items():
                        if exito:
                            print(f"   • {modelo.upper()}")
                else:
                    print("\\n❌ No se pudieron cargar los modelos")
                    print("   Verifique su conexión a internet y que PyTorch esté instalado")
                    
            elif opcion == "5":
                return
            else:
                print("Opción no válida.")
                
        except Exception as e:
            print(f"\\n❌ Error durante la carga: {e}")
            print("\\nPosibles soluciones:")
            print("1. Verificar conexión a internet")
            print("2. Instalar PyTorch: pip install torch torchvision")
            print("3. Verificar espacio en disco")
                
        input("\\nPresione Enter para continuar...")
        
    def prediccion_individual_cnn(self):
        """Realiza predicción con un modelo específico."""
        if not self._verificar_imagen_y_modelos():
            return
            
        print("\\n" + "="*50)
        print("     PREDICCIÓN CON MODELO INDIVIDUAL")
        print("="*50)
        
        # Mostrar modelos disponibles
        modelos_cargados = list(self.redes_cnn.modelos.keys())
        if not modelos_cargados:
            print("❌ No hay modelos cargados. Cargue al menos un modelo primero.")
            input("\\nPresione Enter para continuar...")
            return
            
        print("Modelos disponibles:")
        for i, modelo in enumerate(modelos_cargados, 1):
            print(f"{i}. {modelo.upper()}")
            
        try:
            seleccion = input(f"\\nSeleccione un modelo (1-{len(modelos_cargados)}): ").strip()
            indice = int(seleccion) - 1
            
            if 0 <= indice < len(modelos_cargados):
                modelo_elegido = modelos_cargados[indice]
                
                print(f"\\n🔍 Analizando imagen con {modelo_elegido.upper()}...")
                
                # Usar imagen actual o preprocesada
                imagen_analizar = self.imagen_preprocesada if self.imagen_preprocesada is not None else self.imagen_actual
                
                resultado = self.redes_cnn.predecir(imagen_analizar, modelo_elegido)
                
                if resultado:
                    print(f"\\n✅ Predicción completada con {modelo_elegido.upper()}")
                    
                    # Mostrar resultados
                    self.redes_cnn.visualizar_resultados(imagen_analizar, resultado)
                    
                    # Preguntar si guardar
                    guardar = input("\\n💾 ¿Desea guardar estos resultados? (s/n): ").strip().lower()
                    if guardar in ['s', 'si', 'y', 'yes']:
                        archivo = self.redes_cnn.guardar_resultados(
                            resultado, 
                            f"prediccion_{modelo_elegido}_{self.nombre_actual}"
                        )
                        if archivo:
                            print(f"✅ Resultados guardados: {archivo}")
                else:
                    print(f"❌ Error en la predicción con {modelo_elegido}")
                    
            else:
                print("Selección inválida.")
                
        except ValueError:
            print("Entrada inválida. Ingrese un número.")
        except Exception as e:
            print(f"❌ Error: {e}")
            
        input("\\nPresione Enter para continuar...")
        
    def comparacion_modelos_cnn(self):
        """Compara predicciones entre todos los modelos cargados."""
        if not self._verificar_imagen_y_modelos():
            return
            
        modelos_cargados = list(self.redes_cnn.modelos.keys())
        if len(modelos_cargados) < 2:
            print("❌ Necesita al menos 2 modelos cargados para hacer comparación.")
            print(f"   Modelos actuales: {len(modelos_cargados)}")
            input("\\nPresione Enter para continuar...")
            return
            
        print("\\n" + "="*60)
        print("     COMPARACIÓN ENTRE MODELOS CNN")
        print("="*60)
        print(f"🔍 Comparando {len(modelos_cargados)} modelos: {', '.join([m.upper() for m in modelos_cargados])}")
        print(f"📷 Imagen: {self.nombre_actual}")
        
        # Usar imagen actual o preprocesada
        imagen_analizar = self.imagen_preprocesada if self.imagen_preprocesada is not None else self.imagen_actual
        
        try:
            print("\\n🚀 Iniciando comparación completa con visualización...")
            
            # Usar el nuevo método completo que integra visualización
            comparacion = self.redes_cnn.comparar_modelos_completo(
                imagen_analizar, 
                mostrar_graficos=True
            )
            
            if comparacion:
                print("\\n✅ Comparación completa finalizada")
                
                # Preguntar si guardar resultados JSON
                guardar = input("\\n💾 ¿Desea guardar los resultados JSON? (s/n): ").strip().lower()
                if guardar in ['s', 'si', 'y', 'yes']:
                    archivo = self.redes_cnn.guardar_resultados(
                        comparacion,
                        f"comparacion_modelos_{self.nombre_actual}"
                    )
                    if archivo:
                        print(f"✅ Resultados JSON guardados: {archivo}")
            else:
                print("❌ Error en la comparación")
                
        except Exception as e:
            print(f"❌ Error durante la comparación: {e}")
            
        input("\\nPresione Enter para continuar...")
        
    def analisis_detallado_cnn(self):
        """Análisis detallado de predicciones."""
        if not self._verificar_imagen_y_modelos():
            return
            
        print("\\n" + "="*60)
        print("     ANÁLISIS DETALLADO DE PREDICCIÓN")
        print("="*60)
        
        # Información de la imagen
        imagen_analizar = self.imagen_preprocesada if self.imagen_preprocesada is not None else self.imagen_actual
        
        print(f"📷 Imagen: {self.nombre_actual}")
        print(f"📐 Dimensiones: {imagen_analizar.shape}")
        print(f"📊 Rango valores: [{np.min(imagen_analizar):.3f}, {np.max(imagen_analizar):.3f}]")
        
        # Información de preprocesamiento
        if self.imagen_preprocesada is not None:
            print("✅ Usando imagen preprocesada")
            if hasattr(self, 'info_preprocesamiento') and self.info_preprocesamiento:
                print(f"🔧 Preprocesamiento aplicado:")
                if 'normalizacion' in self.info_preprocesamiento:
                    print(f"   • Normalización: {self.info_preprocesamiento['normalizacion']}")
                if 'dimension_final' in self.info_preprocesamiento:
                    dim = self.info_preprocesamiento['dimension_final']
                    print(f"   • Redimensionado: {dim[0]}x{dim[1]}")
        else:
            print("⚠️  Usando imagen original (recomendado preprocesar)")
        
        # Realizar análisis con todos los modelos
        modelos_cargados = list(self.redes_cnn.modelos.keys())
        print(f"\\n🤖 Modelos disponibles: {len(modelos_cargados)}")
        
        if modelos_cargados:
            # Usar el método completo para análisis detallado
            comparacion = self.redes_cnn.comparar_modelos_completo(
                imagen_analizar, 
                mostrar_graficos=True
            )
            
            if comparacion:
                # Análisis de interpretación
                print("\\n" + "="*50)
                print("         INTERPRETACIÓN DE RESULTADOS")
                print("="*50)
                
                consenso = comparacion['consenso']
                
                print(f"🎯 Predicción más probable: {consenso['clase_mas_votada']}")
                print(f"🤝 Nivel de consenso: {consenso['nivel_acuerdo']*100:.1f}%")
                
                if consenso['nivel_acuerdo'] >= 0.67:
                    print("✅ ALTA CONFIANZA - Los modelos están de acuerdo")
                elif consenso['nivel_acuerdo'] >= 0.33:
                    print("⚠️  CONFIANZA MEDIA - Hay cierto desacuerdo")
                else:
                    print("❌ BAJA CONFIANZA - Los modelos no están de acuerdo")
                
                # Mostrar detalles por modelo
                print("\\n📊 Detalles por modelo:")
                for modelo, resultado in comparacion['resultados'].items():
                    pred = resultado['prediccion_principal']
                    print(f"   {modelo.upper():>12}: {pred['clase']:30} ({pred['porcentaje']:5.1f}%)")
                
                # Análisis de alternativas
                print("\\n🔍 Predicciones alternativas:")
                for modelo, resultado in comparacion['resultados'].items():
                    print(f"\\n{modelo.upper()}:")
                    for i, pred in enumerate(resultado['predicciones'][:3], 1):
                        print(f"   {i}. {pred['clase']:25} ({pred['porcentaje']:5.1f}%)")
                
                # Recomendaciones
                print("\\n💡 RECOMENDACIONES:")
                if consenso['nivel_acuerdo'] >= 0.67:
                    print("   • La predicción parece confiable")
                    print("   • Considere la clase predicha como resultado final")
                else:
                    print("   • Resultados inciertos - revisar imagen")
                    print("   • Considere aplicar preprocesamiento adicional")
                    print("   • La imagen podría contener múltiples objetos")
        
        input("\\nPresione Enter para continuar...")
        
    def informacion_modelos_cnn(self):
        """Muestra información de los modelos CNN."""
        print("\\n" + "="*60)
        print("     INFORMACIÓN DE MODELOS CNN")
        print("="*60)
        
        if self.redes_cnn is None:
            print("❌ Sistema CNN no inicializado")
            print("   Use 'Cargar modelos CNN' primero")
            input("\\nPresione Enter para continuar...")
            return
            
        info = self.redes_cnn.obtener_info_modelos()
        
        print(f"💻 Dispositivo: {info['dispositivo']}")
        print(f"🎯 Clases disponibles: {info['clases_disponibles']}")
        print(f"🔧 Modelos cargados: {len(info['modelos_cargados'])}")
        
        if info['modelos_cargados']:
            print("\\n📋 MODELOS CARGADOS:")
            for modelo in info['modelos_cargados']:
                print(f"   ✅ {modelo.upper()}")
                
            print("\\n📖 DESCRIPCIÓN DE MODELOS:")
            
            if 'mobilenet' in info['modelos_cargados']:
                print("\\n🚀 MobileNetV2:")
                print("   • Diseñado para dispositivos móviles")
                print("   • Rápido y eficiente")
                print("   • Tamaño: ~14MB")
                print("   • Uso: Aplicaciones tiempo real")
                
            if 'resnet50' in info['modelos_cargados']:
                print("\\n🏆 ResNet50:")
                print("   • Red residual de 50 capas")
                print("   • Alta precisión en ImageNet")
                print("   • Tamaño: ~98MB")
                print("   • Uso: Máxima precisión")
                
            if 'vgg16' in info['modelos_cargados']:
                print("\\n🎓 VGG16:")
                print("   • Arquitectura clásica")
                print("   • 16 capas con filtros 3x3")
                print("   • Tamaño: ~528MB") 
                print("   • Uso: Referencia estándar")
        else:
            print("\\n⚠️  No hay modelos cargados")
            print("   Use la opción 'Cargar modelos CNN'")
            
        print("\\n🔍 TRANSFORMACIONES APLICADAS:")
        print("   • Redimensionar: 256px -> Crop central 224px")
        print("   • Normalizar: ImageNet (mean=[0.485,0.456,0.406])")
        print("   • Formato: RGB Tensor")
        
        input("\\nPresione Enter para continuar...")
        
    def guardar_resultados_cnn(self):
        """Guarda resultados de predicciones CNN."""
        print("\\n" + "="*50)
        print("     GUARDAR RESULTADOS CNN")
        print("="*50)
        
        # Esta función podría expandirse para permitir
        # exportar resultados en diferentes formatos
        print("ℹ️  Los resultados se guardan automáticamente durante las predicciones")
        print("📁 Ubicación: directorio 'resultados_cnn'")
        print("📄 Formato: JSON con información detallada")
        
        # Mostrar archivos existentes si hay
        directorio_resultados = "resultados_cnn"
        if os.path.exists(directorio_resultados):
            archivos = os.listdir(directorio_resultados)
            if archivos:
                print(f"\\n📂 Archivos existentes ({len(archivos)}):")
                for archivo in sorted(archivos)[-5:]:  # Últimos 5
                    print(f"   📄 {archivo}")
                    
                if len(archivos) > 5:
                    print(f"   ... y {len(archivos)-5} más")
            else:
                print("\\n📭 No hay resultados guardados aún")
        else:
            print("\\n📭 Directorio de resultados no creado aún")
            
        input("\\nPresione Enter para continuar...")
        
    def _verificar_imagen_y_modelos(self):
        """Verifica que haya imagen cargada y sistema CNN inicializado."""
        if self.imagen_actual is None:
            print("❌ No hay imagen cargada")
            print("   Cargue una imagen primero en 'Carga y Visualización'")
            input("\\nPresione Enter para continuar...")
            return False
            
        if self.redes_cnn is None:
            print("❌ Sistema CNN no inicializado")
            print("   Use 'Cargar modelos CNN' primero")
            input("\\nPresione Enter para continuar...")
            return False
            
        return True
    
    def configurar_ruta_imagenes(self):
        """Permite configurar una nueva ruta de imágenes."""
        print("\\n" + "="*60)
        print("           CONFIGURAR RUTA DE IMÁGENES")
        print("="*60)
        print(f"Ruta actual: {self.ruta_imagenes}")
        
        nueva_ruta = input("\\nIngrese nueva ruta (Enter para cancelar): ").strip()
        if nueva_ruta and os.path.exists(nueva_ruta):
            self.ruta_imagenes = nueva_ruta
            self.gestor_imagenes = GestorImagenes(nueva_ruta)
            print(f"Nueva ruta configurada: {nueva_ruta}")
        elif nueva_ruta:
            print("La ruta especificada no existe.")
        else:
            print("Operación cancelada.")
            
        input("\\nPresione Enter para continuar...")
        
    def salir_sistema(self):
        """Maneja la salida del sistema."""
        print("\\n" + "="*60)
        print("           SALIENDO DEL SISTEMA")
        print("="*60)
        print("Gracias por usar el Sistema de Preprocesamiento")
        print("Universidad del Quindío - Visión Artificial")
        print("="*60)


def main():
    """Función principal del sistema."""
    try:
        # Obtener la ruta del script actual
        script_dir = os.path.dirname(os.path.abspath(__file__))
        images_path = os.path.join(script_dir, "images")
        
        # Verificar que el directorio de imágenes existe
        if not os.path.exists(images_path):
            print("Error: No se encontró el directorio 'images'")
            print(f"   Buscado en: {images_path}")
            print("   Asegúrese de que el directorio 'images' con las imágenes del dataset esté presente.")
            return
            
        # Cambiar al directorio del script para ejecución
        os.chdir(script_dir)
        
        # Crear y ejecutar el sistema
        sistema = SistemaVisionArtificialParcial()
        sistema.ejecutar_sistema()
        
    except KeyboardInterrupt:
        print("\\n\\nPrograma interrumpido por el usuario.")
    except Exception as e:
        print(f"\\nError crítico: {e}")
        print("   Verifique que todos los módulos estén instalados correctamente.")

if __name__ == "__main__":
    main()