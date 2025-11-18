#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Redes Convolucionales Preentrenadas
============================================

Implementa la funcionalidad para usar redes CNN preentrenadas:
- MobileNetV2: Eficiente para dispositivos móviles
- ResNet50: Red residual profunda
- VGG16: Arquitectura clásica con filtros pequeños

Universidad del Quindío - Visión Artificial
Autor: Sistema de Visión Artificial
Fecha: Noviembre 2024
"""

import numpy as np
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


class RedesPreentrenadas:
    """Clase para manejar redes convolucionales preentrenadas."""
    
    def __init__(self):
        """Inicializar el sistema de redes preentrenadas."""
        self.modelos = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nombres_clases = self._cargar_nombres_imagenet()
        
        # Transformaciones para ImageNet (estándar para modelos preentrenados)
        self.transform_imagenet = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # Media ImageNet
                std=[0.229, 0.224, 0.225]   # Desviación estándar ImageNet
            )
        ])
        
        print(f"🔧 Dispositivo: {self.device}")
        print("✅ Sistema de redes preentrenadas inicializado")
    
    def _cargar_nombres_imagenet(self):
        """Carga los nombres de las 1000 clases de ImageNet."""
        # Diccionario con algunas clases principales de ImageNet
        # En un sistema completo, esto se cargaría desde un archivo JSON
        clases_principales = {
            0: "tench", 1: "goldfish", 2: "great white shark", 3: "tiger shark",
            4: "hammerhead", 5: "electric ray", 6: "stingray", 7: "cock",
            8: "hen", 9: "ostrich", 10: "brambling", 11: "goldfinch",
            12: "house finch", 13: "junco", 14: "indigo bunting", 15: "robin",
            16: "bulbul", 17: "jay", 18: "magpie", 19: "chickadee",
            20: "water ouzel", 21: "kite", 22: "bald eagle", 23: "vulture",
            24: "great grey owl", 25: "European fire salamander", 26: "common newt",
            27: "eft", 28: "spotted salamander", 29: "axolotl", 30: "bullfrog",
            # Añadir algunas clases comunes
            281: "tabby cat", 282: "tiger cat", 283: "Persian cat", 284: "Siamese cat",
            285: "Egyptian cat", 207: "golden retriever", 208: "Labrador retriever",
            209: "cocker spaniel", 151: "Chihuahua", 152: "Japanese spaniel",
            # Objetos comunes
            924: "guacamole", 925: "consomme", 926: "hot pot", 927: "trifle",
            928: "ice cream", 929: "ice lolly", 930: "French loaf", 931: "bagel",
            # Vehículos
            817: "sports car", 511: "convertible", 609: "limousine", 627: "minivan",
            656: "minibus", 751: "racer", 717: "pickup truck", 675: "motor scooter",
            # Animales comunes
            388: "giant panda", 292: "tiger", 340: "zebra", 354: "African elephant",
            386: "African grey parrot", 395: "chimpanzee", 397: "orangutan"
        }
        
        # Generar diccionario completo con 1000 clases
        nombres_completos = {}
        for i in range(1000):
            if i in clases_principales:
                nombres_completos[i] = clases_principales[i]
            else:
                nombres_completos[i] = f"clase_{i}"
        
        return nombres_completos
    
    def cargar_modelo(self, nombre_modelo):
        """
        Carga un modelo preentrenado específico.
        
        Args:
            nombre_modelo (str): 'mobilenet', 'resnet50', o 'vgg16'
        
        Returns:
            bool: True si se cargó exitosamente
        """
        try:
            print(f"\\n🔄 Cargando modelo {nombre_modelo.upper()}...")
            
            if nombre_modelo.lower() == 'mobilenet':
                modelo = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1')
                self.modelos['mobilenet'] = modelo
                print("✅ MobileNetV2 cargado exitosamente")
                
            elif nombre_modelo.lower() == 'resnet50':
                modelo = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
                self.modelos['resnet50'] = modelo
                print("✅ ResNet50 cargado exitosamente")
                
            elif nombre_modelo.lower() == 'vgg16':
                modelo = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
                self.modelos['vgg16'] = modelo
                print("✅ VGG16 cargado exitosamente")
                
            else:
                print(f"❌ Modelo '{nombre_modelo}' no reconocido")
                return False
            
            # Configurar modelo para evaluación
            self.modelos[nombre_modelo.lower()].eval()
            self.modelos[nombre_modelo.lower()].to(self.device)
            
            return True
            
        except Exception as e:
            print(f"❌ Error cargando modelo {nombre_modelo}: {e}")
            return False
    
    def cargar_todos_modelos(self):
        """Carga todos los modelos disponibles."""
        print("\\n🚀 Cargando todos los modelos preentrenados...")
        
        modelos_disponibles = ['mobilenet', 'resnet50', 'vgg16']
        resultados = {}
        
        for modelo in modelos_disponibles:
            resultados[modelo] = self.cargar_modelo(modelo)
        
        exitosos = sum(resultados.values())
        print(f"\\n📊 Resultado: {exitosos}/{len(modelos_disponibles)} modelos cargados exitosamente")
        
        return resultados
    
    def preparar_imagen(self, imagen_np):
        """
        Prepara una imagen numpy para inferencia con modelos preentrenados.
        
        Args:
            imagen_np (numpy.ndarray): Imagen en formato numpy (RGB)
        
        Returns:
            torch.Tensor: Imagen preparada para el modelo
        """
        try:
            # Convertir numpy array a PIL Image
            if imagen_np.dtype != np.uint8:
                imagen_np = (imagen_np * 255).astype(np.uint8)
            
            imagen_pil = Image.fromarray(imagen_np)
            
            # Aplicar transformaciones de ImageNet
            imagen_tensor = self.transform_imagenet(imagen_pil)
            imagen_tensor = imagen_tensor.unsqueeze(0)  # Añadir dimensión de batch
            imagen_tensor = imagen_tensor.to(self.device)
            
            return imagen_tensor
            
        except Exception as e:
            print(f"❌ Error preparando imagen: {e}")
            return None
    
    def predecir(self, imagen_np, nombre_modelo, top_k=5):
        """
        Realiza predicción con un modelo específico.
        
        Args:
            imagen_np (numpy.ndarray): Imagen en formato numpy
            nombre_modelo (str): Nombre del modelo a usar
            top_k (int): Número de predicciones principales a retornar
        
        Returns:
            dict: Resultados de la predicción
        """
        try:
            # Verificar si el modelo está cargado
            if nombre_modelo.lower() not in self.modelos:
                print(f"❌ Modelo '{nombre_modelo}' no está cargado")
                return None
            
            # Preparar imagen
            imagen_tensor = self.preparar_imagen(imagen_np)
            if imagen_tensor is None:
                return None
            
            # Realizar predicción
            modelo = self.modelos[nombre_modelo.lower()]
            
            with torch.no_grad():
                salida = modelo(imagen_tensor)
                
                # Aplicar softmax para obtener probabilidades
                probabilidades = torch.nn.functional.softmax(salida[0], dim=0)
                
                # Obtener top-k predicciones
                valores, indices = torch.topk(probabilidades, top_k)
                
                # Preparar resultados
                predicciones = []
                for i in range(top_k):
                    indice = indices[i].item()
                    probabilidad = valores[i].item()
                    nombre_clase = self.nombres_clases.get(indice, f"Clase {indice}")
                    
                    predicciones.append({
                        'indice': indice,
                        'clase': nombre_clase,
                        'confianza': probabilidad,
                        'porcentaje': probabilidad * 100
                    })
                
                resultado = {
                    'modelo': nombre_modelo.upper(),
                    'predicciones': predicciones,
                    'prediccion_principal': predicciones[0],
                    'tiempo': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                return resultado
                
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            return None
    
    def comparar_modelos(self, imagen_np):
        """
        Compara predicciones de todos los modelos cargados.
        
        Args:
            imagen_np (numpy.ndarray): Imagen para analizar
        
        Returns:
            dict: Comparación de resultados
        """
        try:
            if not self.modelos:
                print("❌ No hay modelos cargados")
                return None
            
            print("\\n🔍 Realizando predicciones con todos los modelos...")
            
            comparacion = {
                'imagen_info': {
                    'shape': imagen_np.shape,
                    'dtype': str(imagen_np.dtype),
                    'rango': [float(np.min(imagen_np)), float(np.max(imagen_np))]
                },
                'resultados': {},
                'consenso': {},
                'tiempo_total': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Predecir con cada modelo cargado
            for nombre_modelo in self.modelos.keys():
                print(f"  📊 Analizando con {nombre_modelo.upper()}...")
                resultado = self.predecir(imagen_np, nombre_modelo, top_k=3)
                if resultado:
                    comparacion['resultados'][nombre_modelo] = resultado
            
            # Analizar consenso
            if comparacion['resultados']:
                comparacion['consenso'] = self._analizar_consenso(comparacion['resultados'])
            
            return comparacion
            
        except Exception as e:
            print(f"❌ Error en comparación: {e}")
            return None
    
    def _analizar_consenso(self, resultados):
        """Analiza el consenso entre diferentes modelos."""
        clases_encontradas = {}
        
        for modelo, resultado in resultados.items():
            clase_principal = resultado['prediccion_principal']['clase']
            if clase_principal not in clases_encontradas:
                clases_encontradas[clase_principal] = []
            clases_encontradas[clase_principal].append(modelo)
        
        # Encontrar la clase más votada
        clase_consenso = max(clases_encontradas.items(), key=lambda x: len(x[1]))
        
        return {
            'clase_mas_votada': clase_consenso[0],
            'modelos_acuerdo': clase_consenso[1],
            'nivel_acuerdo': len(clase_consenso[1]) / len(resultados),
            'todas_clases': clases_encontradas
        }
    
    def visualizar_resultados(self, imagen_np, resultados, mostrar_imagen=True):
        """
        Visualiza los resultados de predicción.
        
        Args:
            imagen_np (numpy.ndarray): Imagen original
            resultados (dict): Resultados de predicción
            mostrar_imagen (bool): Si mostrar la imagen
        """
        try:
            if mostrar_imagen:
                plt.figure(figsize=(15, 10))
                
                # Mostrar imagen
                plt.subplot(2, 2, 1)
                plt.imshow(imagen_np)
                plt.title('Imagen Analizada', fontsize=14, fontweight='bold')
                plt.axis('off')
                
                # Si es comparación de múltiples modelos
                if 'resultados' in resultados:
                    self._visualizar_comparacion_multiple(resultados)
                else:
                    self._visualizar_resultado_individual(resultados)
                
                plt.tight_layout()
                plt.show()
            
            # Imprimir resultados en texto
            self._imprimir_resultados_texto(resultados)
            
        except Exception as e:
            print(f"❌ Error visualizando resultados: {e}")
    
    def _visualizar_comparacion_multiple(self, resultados):
        """Visualiza comparación de múltiples modelos."""
        # Gráfico de barras con predicciones principales
        plt.subplot(2, 2, 2)
        modelos = list(resultados['resultados'].keys())
        confianzas = [resultados['resultados'][m]['prediccion_principal']['porcentaje'] 
                     for m in modelos]
        
        bars = plt.bar(modelos, confianzas, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        plt.title('Nivel de Confianza por Modelo', fontsize=12, fontweight='bold')
        plt.ylabel('Confianza (%)')
        plt.xticks(rotation=45)
        
        # Añadir valores en las barras
        for bar, confianza in zip(bars, confianzas):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{confianza:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Información de consenso
        plt.subplot(2, 2, 3)
        plt.axis('off')
        consenso = resultados['consenso']
        
        info_texto = f"""
📊 ANÁLISIS DE CONSENSO

🎯 Clase más votada: {consenso['clase_mas_votada']}
🤝 Modelos en acuerdo: {', '.join([m.upper() for m in consenso['modelos_acuerdo']])}
📈 Nivel de acuerdo: {consenso['nivel_acuerdo']*100:.1f}%

🔍 TODAS LAS PREDICCIONES:
"""
        for modelo, resultado in resultados['resultados'].items():
            pred = resultado['prediccion_principal']
            info_texto += f"\\n• {modelo.upper()}: {pred['clase']} ({pred['porcentaje']:.1f}%)"
        
        plt.text(0.05, 0.95, info_texto, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    def _visualizar_resultado_individual(self, resultado):
        """Visualiza resultado de un solo modelo."""
        # Top 5 predicciones
        plt.subplot(2, 2, 2)
        predicciones = resultado['predicciones'][:5]
        clases = [p['clase'][:20] for p in predicciones]  # Limitar longitud
        confianzas = [p['porcentaje'] for p in predicciones]
        
        bars = plt.barh(range(len(clases)), confianzas, color='#4ECDC4')
        plt.yticks(range(len(clases)), clases)
        plt.xlabel('Confianza (%)')
        plt.title(f'Top 5 Predicciones - {resultado["modelo"]}', fontsize=12, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Añadir valores en las barras
        for bar, confianza in zip(bars, confianzas):
            plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{confianza:.1f}%', ha='left', va='center', fontweight='bold')
    
    def _imprimir_resultados_texto(self, resultados):
        """Imprime resultados en formato texto."""
        print("\\n" + "="*60)
        print("           RESULTADOS DE PREDICCIÓN CNN")
        print("="*60)
        
        if 'resultados' in resultados:
            # Múltiples modelos
            for modelo, resultado in resultados['resultados'].items():
                print(f"\\n🤖 {modelo.upper()}:")
                pred = resultado['prediccion_principal']
                print(f"   🎯 Predicción: {pred['clase']}")
                print(f"   📊 Confianza: {pred['porcentaje']:.2f}%")
            
            # Consenso
            print(f"\\n🤝 CONSENSO:")
            consenso = resultados['consenso']
            print(f"   📈 Acuerdo: {consenso['nivel_acuerdo']*100:.1f}%")
            print(f"   🏆 Clase ganadora: {consenso['clase_mas_votada']}")
            
        else:
            # Un solo modelo
            print(f"\\n🤖 Modelo: {resultados['modelo']}")
            print(f"🎯 Predicción principal: {resultados['prediccion_principal']['clase']}")
            print(f"📊 Nivel de confianza: {resultados['prediccion_principal']['porcentaje']:.2f}%")
            
            print("\\n📋 Top 5 predicciones:")
            for i, pred in enumerate(resultados['predicciones'][:5], 1):
                print(f"   {i}. {pred['clase']}: {pred['porcentaje']:.2f}%")
    
    def guardar_resultados(self, resultados, nombre_archivo, directorio_salida="resultados_cnn"):
        """Guarda los resultados en un archivo JSON."""
        try:
            # Crear directorio si no existe
            if not os.path.exists(directorio_salida):
                os.makedirs(directorio_salida)
            
            # Generar nombre único
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archivo_completo = os.path.join(directorio_salida, f"{nombre_archivo}_{timestamp}.json")
            
            # Guardar resultados
            with open(archivo_completo, 'w', encoding='utf-8') as f:
                json.dump(resultados, f, indent=2, ensure_ascii=False)
            
            print(f"\\n💾 Resultados guardados en: {archivo_completo}")
            return archivo_completo
            
        except Exception as e:
            print(f"❌ Error guardando resultados: {e}")
            return None
    
    def obtener_info_modelos(self):
        """Retorna información sobre los modelos cargados."""
        info = {
            'modelos_cargados': list(self.modelos.keys()),
            'dispositivo': str(self.device),
            'transformaciones': str(self.transform_imagenet),
            'clases_disponibles': len(self.nombres_clases)
        }
        return info

    def comparar_modelos_completo(self, imagen_np, mostrar_graficos=True, guardar_imagen=None):
        """
        Método completo que realiza la comparación de modelos CNN con visualización integrada.
        
        Args:
            imagen_np (numpy.ndarray): Imagen para analizar
            mostrar_graficos (bool): Si mostrar los gráficos de estadísticas
            guardar_imagen (str): Nombre del archivo para guardar la visualización (opcional)
        
        Returns:
            dict: Resultados completos de la comparación
        """
        try:
            if not self.modelos:
                print("❌ No hay modelos cargados")
                return None

            print("\n🔍 Realizando comparación completa de modelos CNN...")
            print("=" * 60)
            
            # Realizar comparación básica
            comparacion = self.comparar_modelos(imagen_np)
            
            if not comparacion or not comparacion.get('resultados'):
                print("❌ No se pudieron obtener resultados de comparación")
                return None
            
            # Mostrar resultados en texto
            self._imprimir_resultados_texto(comparacion)
            
            # Generar visualización inmediata
            if mostrar_graficos:
                print("\n📊 Generando visualización de estadísticas...")
                figuras = self._generar_graficos_estadisticas(imagen_np, comparacion)
                
                # Mostrar todas las figuras
                if figuras:
                    print(f"✅ Se generaron {len(figuras)} visualizaciones")
                    
                    # Opción de guardar
                    if guardar_imagen:
                        for i, fig in enumerate(figuras):
                            nombre_figura = f"{guardar_imagen}_parte_{i+1}"
                            self._guardar_visualizacion(fig, nombre_figura)
                    else:
                        # Preguntar al usuario si quiere guardar
                        respuesta = input("\n💾 ¿Desea guardar las visualizaciones? (s/n): ").lower().strip()
                        if respuesta in ['s', 'si', 'sí', 'y', 'yes']:
                            nombre = input("📝 Ingrese nombre base para las imágenes (sin extensión): ").strip()
                            if nombre:
                                for i, fig in enumerate(figuras):
                                    nombre_figura = f"{nombre}_parte_{i+1}"
                                    self._guardar_visualizacion(fig, nombre_figura)
                    
                    plt.show()
            
            return comparacion
            
        except Exception as e:
            print(f"❌ Error en comparación completa: {e}")
            return None

    def _generar_graficos_estadisticas(self, imagen_np, resultados):
        """
        Genera gráficos estadísticos mejorados con mejor legibilidad.
        
        Args:
            imagen_np (numpy.ndarray): Imagen analizada
            resultados (dict): Resultados de la comparación
        
        Returns:
            list: Lista de figuras generadas
        """
        try:
            # Configurar estilo
            plt.style.use('default')
            sns.set_palette("husl")
            
            figuras = []
            
            # FIGURA 1: Comparación principal de modelos
            print("📊 Generando gráfico principal de comparación...")
            fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
            fig1.suptitle('🔍 COMPARACIÓN PRINCIPAL - MODELOS CNN', 
                        fontsize=16, fontweight='bold', y=0.96)
            
            # Imagen original
            axes1[0, 0].imshow(imagen_np)
            axes1[0, 0].set_title('📷 Imagen Analizada', fontweight='bold', fontsize=14)
            axes1[0, 0].axis('off')
            
            # Gráfico de barras de confianza mejorado
            self._crear_grafico_confianza(axes1[0, 1], resultados)
            
            # Top predicciones mejorado
            self._crear_grafico_top_predicciones(axes1[1, 0], resultados)
            
            # Panel de información completo
            self._crear_panel_informacion_completo(axes1[1, 1], resultados)
            
            plt.tight_layout()
            figuras.append(fig1)
            
            # FIGURA 2: Análisis detallado por modelo
            print("📈 Generando análisis detallado por modelo...")
            fig2 = self._crear_figura_detallada_modelos(imagen_np, resultados)
            if fig2:
                figuras.append(fig2)
            
            # FIGURA 3: Gráficos de consenso y estadísticas
            print("🎯 Generando análisis de consenso...")
            fig3 = self._crear_figura_consenso(resultados)
            if fig3:
                figuras.append(fig3)
            
            return figuras
            
        except Exception as e:
            print(f"❌ Error generando gráficos: {e}")
            return []

    def _crear_figura_detallada_modelos(self, imagen_np, resultados):
        """Crea figura con análisis detallado de cada modelo."""
        try:
            num_modelos = len(resultados['resultados'])
            fig, axes = plt.subplots(2, num_modelos, figsize=(6 * num_modelos, 12))
            
            if num_modelos == 1:
                axes = axes.reshape(2, 1)
            
            fig.suptitle('📊 ANÁLISIS DETALLADO POR MODELO CNN', 
                        fontsize=16, fontweight='bold', y=0.96)
            
            for i, (modelo, resultado) in enumerate(resultados['resultados'].items()):
                # Top 5 predicciones para cada modelo
                predicciones = resultado['predicciones'][:5]
                clases = [p['clase'][:15] + "..." if len(p['clase']) > 15 else p['clase'] 
                         for p in predicciones]
                confianzas = [p['porcentaje'] for p in predicciones]
                
                # Gráfico de barras horizontales
                colors = plt.cm.Set3(np.linspace(0, 1, 5))
                bars = axes[0, i].barh(range(len(clases)), confianzas, color=colors)
                axes[0, i].set_yticks(range(len(clases)))
                axes[0, i].set_yticklabels(clases, fontsize=10)
                axes[0, i].set_xlabel('Confianza (%)', fontsize=11, fontweight='bold')
                axes[0, i].set_title(f'🔹 {modelo.upper()}\nTop 5 Predicciones', 
                                   fontweight='bold', fontsize=12)
                axes[0, i].set_xlim(0, max(confianzas) * 1.2)
                axes[0, i].grid(True, alpha=0.3, axis='x')
                axes[0, i].invert_yaxis()
                
                # Añadir valores en las barras
                for bar, confianza in zip(bars, confianzas):
                    axes[0, i].text(bar.get_width() + max(confianzas)*0.01, 
                                   bar.get_y() + bar.get_height()/2,
                                   f'{confianza:.1f}%', 
                                   ha='left', va='center', fontweight='bold', fontsize=9)
                
                # Gráfico de distribución de confianzas
                axes[1, i].hist(confianzas, bins=5, color=colors[0], alpha=0.7, edgecolor='black')
                axes[1, i].set_xlabel('Confianza (%)', fontsize=11, fontweight='bold')
                axes[1, i].set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
                axes[1, i].set_title(f'📈 Distribución de Confianza\n{modelo.upper()}', 
                                   fontweight='bold', fontsize=12)
                axes[1, i].grid(True, alpha=0.3)
                
                # Estadísticas del modelo
                media = np.mean(confianzas)
                std = np.std(confianzas)
                axes[1, i].axvline(media, color='red', linestyle='--', linewidth=2, 
                                  label=f'Media: {media:.1f}%')
                axes[1, i].legend(fontsize=10)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"❌ Error creando figura detallada: {e}")
            return None

    def _crear_figura_consenso(self, resultados):
        """Crea figura con análisis de consenso y estadísticas globales."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('🎯 ANÁLISIS DE CONSENSO Y ESTADÍSTICAS GLOBALES', 
                        fontsize=16, fontweight='bold', y=0.96)
            
            # Gráfico de consenso circular mejorado
            self._crear_grafico_consenso_mejorado(axes[0, 0], resultados)
            
            # Comparación de confianzas entre modelos
            self._crear_grafico_comparacion_confianzas(axes[0, 1], resultados)
            
            # Matriz de similitud entre predicciones
            self._crear_matriz_similitud(axes[1, 0], resultados)
            
            # Estadísticas globales
            self._crear_panel_estadisticas_globales(axes[1, 1], resultados)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"❌ Error creando figura de consenso: {e}")
            return None
            axes[0, 0].set_title('📷 Imagen Analizada', fontweight='bold', fontsize=12)
            axes[0, 0].axis('off')
            
            # 2. Gráfico de barras de confianza
            self._crear_grafico_confianza(axes[0, 1], resultados)
            
            # 3. Gráfico de consenso
            self._crear_grafico_consenso(axes[0, 2], resultados)
            
            # 4. Top predicciones por modelo
            self._crear_grafico_top_predicciones(axes[1, 0], resultados)
            
            # 5. Distribución de confianza
            self._crear_grafico_distribucion(axes[1, 1], resultados)
            
            # 6. Panel de información
            self._crear_panel_informacion(axes[1, 2], resultados)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"❌ Error generando gráficos: {e}")
            return None

    def _crear_grafico_confianza(self, ax, resultados):
        """Crea gráfico de barras con niveles de confianza mejorado."""
        modelos = list(resultados['resultados'].keys())
        confianzas = [resultados['resultados'][m]['prediccion_principal']['porcentaje'] 
                     for m in modelos]
        clases = [resultados['resultados'][m]['prediccion_principal']['clase'][:15] + "..." 
                 if len(resultados['resultados'][m]['prediccion_principal']['clase']) > 15
                 else resultados['resultados'][m]['prediccion_principal']['clase']
                 for m in modelos]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3', '#FEC8D8']
        bars = ax.bar(range(len(modelos)), confianzas, color=colors[:len(modelos)])
        
        ax.set_title('📊 Nivel de Confianza por Modelo', fontweight='bold', fontsize=12, pad=20)
        ax.set_ylabel('Confianza (%)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Modelos CNN', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 105)  # Más espacio en la parte superior
        ax.set_xticks(range(len(modelos)))
        ax.set_xticklabels(modelos, fontsize=10, fontweight='bold')
        
        # Añadir grilla para mejor legibilidad
        ax.grid(True, alpha=0.3, axis='y')
        
        # Añadir valores en las barras con mejor posicionamiento
        for i, (bar, confianza, clase) in enumerate(zip(bars, confianzas, clases)):
            # Valor de confianza
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{confianza:.1f}%', ha='center', va='bottom', 
                   fontweight='bold', fontsize=10)
            # Clase predicha en la parte inferior
            ax.text(bar.get_x() + bar.get_width()/2, 2,
                   clase, ha='center', va='bottom', 
                   fontsize=8, rotation=0, wrap=True)
        
        ax.grid(True, alpha=0.3)

    def _crear_grafico_consenso(self, ax, resultados):
        """Crea gráfico circular del consenso."""
        consenso = resultados['consenso']
        nivel_acuerdo = consenso['nivel_acuerdo']
        
        # Datos para el gráfico circular
        sizes = [nivel_acuerdo * 100, (1 - nivel_acuerdo) * 100]
        labels = ['En Acuerdo', 'En Desacuerdo']
        colors = ['#2ECC71', '#E74C3C']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                         startangle=90, textprops={'fontweight': 'bold'})
        
        ax.set_title('🤝 Nivel de Consenso', fontweight='bold', fontsize=11)
        
        # Añadir información adicional
        clase_ganadora = consenso['clase_mas_votada']
        ax.text(0, -1.3, f'🏆 Clase Ganadora: {clase_ganadora}', 
               ha='center', fontweight='bold', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue'))

    def _crear_grafico_top_predicciones(self, ax, resultados):
        """Crea gráfico de barras horizontales con top predicciones mejorado."""
        # Obtener todas las predicciones principales
        datos = []
        for modelo, resultado in resultados['resultados'].items():
            pred = resultado['prediccion_principal']
            # Acortar nombre de clase para mejor visualización
            clase_corta = pred['clase'][:20] + "..." if len(pred['clase']) > 20 else pred['clase']
            datos.append({
                'modelo': modelo.upper(),
                'clase': clase_corta,
                'clase_completa': pred['clase'],
                'confianza': pred['porcentaje']
            })
        
        y_pos = np.arange(len(datos))
        confianzas = [d['confianza'] for d in datos]
        modelos = [d['modelo'] for d in datos]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3', '#FEC8D8']
        bars = ax.barh(y_pos, confianzas, color=colors[:len(datos)], height=0.6)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(modelos, fontsize=11, fontweight='bold')
        ax.set_xlabel('Confianza (%)', fontsize=11, fontweight='bold')
        ax.set_title('🎯 Predicciones Principales por Modelo', fontweight='bold', fontsize=12, pad=15)
        ax.set_xlim(0, 105)
        
        # Añadir grilla
        ax.grid(True, alpha=0.3, axis='x')
        
        # Añadir nombres de clases y valores con mejor formato
        for i, (bar, dato) in enumerate(zip(bars, datos)):
            # Valor de confianza dentro de la barra si es lo suficientemente ancha
            if dato['confianza'] > 15:
                ax.text(bar.get_width()/2, bar.get_y() + bar.get_height()/2,
                       f'{dato["confianza"]:.1f}%', 
                       ha='center', va='center', fontweight='bold', color='white', fontsize=10)
            else:
                ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                       f'{dato["confianza"]:.1f}%', 
                       ha='left', va='center', fontweight='bold', fontsize=10)
            
            # Nombre de clase abajo de la barra
            ax.text(2, bar.get_y() + bar.get_height()/2 - 0.15,
                   dato['clase'], 
                   ha='left', va='center', fontsize=9, style='italic')

        # Establecer límites más amplios para las etiquetas
        ax.set_ylim(-0.5, len(datos) - 0.5)

    def _crear_grafico_distribucion(self, ax, resultados):
        """Crea gráfico de distribución de confianzas."""
        todas_confianzas = []
        modelos_labels = []
        
        for modelo, resultado in resultados['resultados'].items():
            for pred in resultado['predicciones'][:3]:  # Top 3
                todas_confianzas.append(pred['porcentaje'])
                modelos_labels.append(modelo.upper())
        
        # Crear boxplot
        datos_por_modelo = []
        modelos = list(resultados['resultados'].keys())
        
        for modelo in modelos:
            confianzas_modelo = [p['porcentaje'] for p in 
                               resultados['resultados'][modelo]['predicciones'][:3]]
            datos_por_modelo.append(confianzas_modelo)
        
        bp = ax.boxplot(datos_por_modelo, labels=[m.upper() for m in modelos], 
                       patch_artist=True)
        
        # Colorear cajas
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title('📈 Distribución de Confianzas', fontweight='bold', fontsize=11)
        ax.set_ylabel('Confianza (%)')
        ax.grid(True, alpha=0.3)

    def _crear_panel_informacion(self, ax, resultados):
        """Crea panel de información textual."""
        ax.axis('off')
        
        # Preparar texto informativo
        consenso = resultados['consenso']
        timestamp = resultados['tiempo_total']
        
        info_texto = f"""
📊 RESUMEN EJECUTIVO
{'=' * 30}

🕐 Análisis realizado: {timestamp}
🤖 Modelos evaluados: {len(resultados['resultados'])}

🎯 CONSENSO:
   • Clase ganadora: {consenso['clase_mas_votada']}
   • Nivel de acuerdo: {consenso['nivel_acuerdo']*100:.1f}%
   • Modelos en acuerdo: {len(consenso['modelos_acuerdo'])}

📈 PREDICCIONES INDIVIDUALES:
"""
        
        for i, (modelo, resultado) in enumerate(resultados['resultados'].items(), 1):
            pred = resultado['prediccion_principal']
            info_texto += f"\\n{i}. {modelo.upper()}:"
            info_texto += f"\\n   🎯 {pred['clase'][:30]}"
            info_texto += f"\\n   📊 {pred['porcentaje']:.2f}%\\n"
        
        # Añadir estadísticas adicionales
        confianzas = [r['prediccion_principal']['porcentaje'] 
                     for r in resultados['resultados'].values()]
        info_texto += f"""
📊 ESTADÍSTICAS:
   • Confianza promedio: {np.mean(confianzas):.2f}%
   • Confianza máxima: {np.max(confianzas):.2f}%
   • Confianza mínima: {np.min(confianzas):.2f}%
   • Desviación estándar: {np.std(confianzas):.2f}%
"""
        
        ax.text(0.05, 0.95, info_texto, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F8FF', alpha=0.8))

    def _guardar_visualizacion(self, fig, nombre_base):
        """Guarda la visualización en un archivo."""
        try:
            # Crear directorio si no existe
            directorio = "resultados_cnn/visualizaciones"
            if not os.path.exists(directorio):
                os.makedirs(directorio)
            
            # Generar nombre completo con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_completo = f"{nombre_base}_comparacion_{timestamp}.png"
            ruta_completa = os.path.join(directorio, nombre_completo)
            
            # Guardar con alta calidad
            fig.savefig(ruta_completa, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            
            print(f"\\n💾 Visualización guardada en: {ruta_completa}")
            return ruta_completa
            
        except Exception as e:
            print(f"❌ Error guardando visualización: {e}")
            return None

    def _crear_panel_informacion_completo(self, ax, resultados):
        """Crea panel de información completo mejorado."""
        ax.axis('off')
        consenso = resultados['consenso']
        
        info_texto = f"""
🎯 RESUMEN DE ANÁLISIS CNN

📊 CONSENSO ENTRE MODELOS:
   • Clase más votada: {consenso['clase_mas_votada'][:30]}
   • Modelos de acuerdo: {', '.join([m.upper() for m in consenso['modelos_acuerdo']])}
   • Nivel de consenso: {consenso['nivel_acuerdo']*100:.1f}%

🔍 PREDICCIONES DETALLADAS:
"""
        
        for modelo, resultado in resultados['resultados'].items():
            pred = resultado['prediccion_principal']
            info_texto += f"\\n   • {modelo.upper():12s}: {pred['clase'][:25]:<25} ({pred['porcentaje']:5.1f}%)"
        
        # Calcular estadísticas
        confianzas = [resultados['resultados'][m]['prediccion_principal']['porcentaje'] 
                     for m in resultados['resultados']]
        
        info_texto += f"""

📈 ESTADÍSTICAS GENERALES:
   • Confianza promedio: {np.mean(confianzas):.1f}%
   • Confianza máxima:   {np.max(confianzas):.1f}%
   • Confianza mínima:   {np.min(confianzas):.1f}%
   • Desviación estándar: {np.std(confianzas):.1f}%
   • Rango de confianza: {np.max(confianzas) - np.min(confianzas):.1f}%
"""
        
        ax.text(0.05, 0.95, info_texto, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox={'boxstyle': 'round,pad=0.5', 'facecolor': '#F0F8FF', 'alpha': 0.9})

    def _crear_grafico_consenso_mejorado(self, ax, resultados):
        """Crea gráfico de consenso circular mejorado."""
        consenso = resultados['consenso']
        
        # Datos para el gráfico circular
        labels = list(consenso['todas_clases'].keys())[:5]  # Top 5
        sizes = [len(consenso['todas_clases'][label]) for label in labels]
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        # Crear gráfico de dona
        _, _, _ = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                        startangle=90, wedgeprops={'edgecolor': 'black', 'linewidth': 2})
        
        ax.set_title('🎯 Distribución de Predicciones\\nEntre Modelos', 
                    fontweight='bold', fontsize=12)
        
        # Añadir información central
        circle = plt.Circle((0,0), 0.70, fc='white', ec='black', linewidth=2)
        ax.add_artist(circle)
        
        ax.text(0, 0, f'{consenso["nivel_acuerdo"]*100:.1f}%\\nConsenso',
               ha='center', va='center', fontweight='bold', fontsize=12,
               bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'lightblue'})

    def _crear_grafico_comparacion_confianzas(self, ax, resultados):
        """Crea gráfico de comparación de confianzas entre modelos."""
        modelos = list(resultados['resultados'].keys())
        
        # Obtener top 3 predicciones de cada modelo
        datos_plot = []
        labels_plot = []
        
        for modelo in modelos:
            predicciones = resultados['resultados'][modelo]['predicciones'][:3]
            confianzas = [p['porcentaje'] for p in predicciones]
            datos_plot.extend(confianzas)
            labels_plot.extend([f"{modelo}\\nTop{i+1}" for i in range(len(confianzas))])
        
        # Crear gráfico de violin plot
        positions = range(len(datos_plot))
        colors = ['#FF6B6B', '#FF6B6B', '#FF6B6B', '#4ECDC4', '#4ECDC4', '#4ECDC4', 
                 '#45B7D1', '#45B7D1', '#45B7D1']
        
        bars = ax.bar(positions, datos_plot, color=colors[:len(datos_plot)])
        ax.set_xticks(positions)
        ax.set_xticklabels(labels_plot, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Confianza (%)', fontweight='bold')
        ax.set_title('📊 Comparación de Top 3 Predicciones\\nPor Modelo', 
                    fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Añadir valores en las barras
        for bar, valor in zip(bars, datos_plot):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{valor:.1f}%', ha='center', va='bottom', fontsize=8)

    def _crear_matriz_similitud(self, ax, resultados):
        """Crea matriz de similitud entre predicciones de modelos."""
        modelos = list(resultados['resultados'].keys())
        n_modelos = len(modelos)
        
        # Crear matriz de similitud
        similitud = np.ones((n_modelos, n_modelos))
        
        for i, modelo1 in enumerate(modelos):
            for j, modelo2 in enumerate(modelos):
                if i != j:
                    pred1 = resultados['resultados'][modelo1]['prediccion_principal']['clase']
                    pred2 = resultados['resultados'][modelo2]['prediccion_principal']['clase']
                    # Similitud simple: 1 si es la misma clase, 0 si no
                    similitud[i][j] = 1.0 if pred1 == pred2 else 0.0
        
        # Crear heatmap
        im = ax.imshow(similitud, cmap='RdYlBu', aspect='auto', vmin=0, vmax=1)
        
        # Configurar etiquetas
        ax.set_xticks(range(n_modelos))
        ax.set_yticks(range(n_modelos))
        ax.set_xticklabels(modelos, fontweight='bold')
        ax.set_yticklabels(modelos, fontweight='bold')
        ax.set_title('🔄 Matriz de Similitud\\nEntre Predicciones', 
                    fontweight='bold', fontsize=12)
        
        # Añadir valores en las celdas
        for i in range(n_modelos):
            for j in range(n_modelos):
                text = ax.text(j, i, f'{similitud[i, j]:.1f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        # Añadir colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)

    def _crear_panel_estadisticas_globales(self, ax, resultados):
        """Crea panel con estadísticas globales del análisis."""
        ax.axis('off')
        
        # Calcular estadísticas avanzadas
        confianzas = [resultados['resultados'][m]['prediccion_principal']['porcentaje'] 
                     for m in resultados['resultados']]
        
        media = np.mean(confianzas)
        mediana = np.median(confianzas)
        std = np.std(confianzas)
        rango = np.max(confianzas) - np.min(confianzas)
        
        # Contar clases únicas
        clases_predichas = [resultados['resultados'][m]['prediccion_principal']['clase'] 
                          for m in resultados['resultados']]
        clases_unicas = len(set(clases_predichas))
        
        info_estadisticas = f"""
📈 ESTADÍSTICAS GLOBALES DEL ANÁLISIS

🔢 MÉTRICAS DE CONFIANZA:
   • Media:           {media:.2f}%
   • Mediana:         {mediana:.2f}%
   • Desv. Estándar:  {std:.2f}%
   • Rango:           {rango:.2f}%
   • Coef. Variación: {(std/media)*100:.1f}%

🎯 DIVERSIDAD DE PREDICCIONES:
   • Modelos analizados:    {len(resultados['resultados'])}
   • Clases únicas:         {clases_unicas}
   • Nivel de consenso:     {resultados['consenso']['nivel_acuerdo']*100:.1f}%
   • Acuerdo alto:          {'Sí' if resultados['consenso']['nivel_acuerdo'] > 0.66 else 'No'}

📊 EVALUACIÓN GENERAL:
   • Confianza general:     {'Alta' if media > 70 else 'Media' if media > 40 else 'Baja'}
   • Consistencia:          {'Alta' if std < 15 else 'Media' if std < 30 else 'Baja'}
   • Recomendación:         {'Predicción confiable' if media > 60 and std < 20 else 'Revisar resultados'}
"""
        
        ax.text(0.05, 0.95, info_estadisticas, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox={'boxstyle': 'round,pad=0.5', 'facecolor': '#E8F8F5', 'alpha': 0.9})