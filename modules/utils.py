#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Utilidades
====================

Funciones y clases de utilidad para el sistema de análisis
de tráfico vehicular.

Funcionalidades:
- Gestión de imágenes
- Gestión de resultados
- Exportación de datos
- Visualización auxiliar
"""

import os
import cv2
import numpy as np
import json
import csv
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt


class ImageUtils:
    """Utilidades para manejo de imágenes."""
    
    @staticmethod
    def load_image(image_path):
        """
        Carga una imagen desde archivo.
        
        Args:
            image_path (str): Ruta de la imagen
            
        Returns:
            np.ndarray: Imagen cargada o None si hay error
        """
        if not os.path.exists(image_path):
            print(f"❌ Error: No se encuentra el archivo {image_path}")
            return None
        
        imagen = cv2.imread(image_path)
        if imagen is None:
            print(f"❌ Error: No se pudo cargar la imagen {image_path}")
            return None
        
        return imagen
    
    @staticmethod
    def save_image(imagen, output_path, crear_directorio=True):
        """
        Guarda una imagen en archivo.
        
        Args:
            imagen (np.ndarray): Imagen a guardar
            output_path (str): Ruta de salida
            crear_directorio (bool): Crear directorio si no existe
            
        Returns:
            bool: True si se guardó exitosamente
        """
        if crear_directorio:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
        
        try:
            cv2.imwrite(output_path, imagen)
            return True
        except Exception as e:
            print(f"❌ Error al guardar imagen: {str(e)}")
            return False
    
    @staticmethod
    def get_image_info(image_path):
        """
        Obtiene información de una imagen sin cargarla completamente.
        
        Args:
            image_path (str): Ruta de la imagen
            
        Returns:
            dict: Información de la imagen
        """
        if not os.path.exists(image_path):
            return None
        
        try:
            # Leer solo cabecera
            imagen = cv2.imread(image_path)
            if imagen is None:
                return None
            
            h, w = imagen.shape[:2]
            channels = imagen.shape[2] if len(imagen.shape) == 3 else 1
            size_bytes = os.path.getsize(image_path)
            size_mb = size_bytes / (1024 * 1024)
            
            return {
                'ruta': image_path,
                'nombre': os.path.basename(image_path),
                'ancho': w,
                'alto': h,
                'canales': channels,
                'tamaño_bytes': size_bytes,
                'tamaño_mb': size_mb,
                'extension': os.path.splitext(image_path)[1]
            }
        except Exception as e:
            print(f"Error al obtener info: {str(e)}")
            return None
    
    @staticmethod
    def list_images_in_folder(folder_path, extensions=None):
        """
        Lista todas las imágenes en una carpeta.
        
        Args:
            folder_path (str): Ruta de la carpeta
            extensions (list): Lista de extensiones permitidas
            
        Returns:
            list: Lista de rutas de imágenes
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
        
        if not os.path.exists(folder_path):
            print(f"❌ La carpeta {folder_path} no existe")
            return []
        
        images = []
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in extensions):
                images.append(os.path.join(folder_path, file))
        
        return sorted(images)
    
    @staticmethod
    def create_thumbnail(imagen, max_size=200):
        """
        Crea una miniatura de la imagen.
        
        Args:
            imagen (np.ndarray): Imagen original
            max_size (int): Tamaño máximo del lado más largo
            
        Returns:
            np.ndarray: Miniatura
        """
        h, w = imagen.shape[:2]
        if h > w:
            new_h = max_size
            new_w = int(w * (max_size / h))
        else:
            new_w = max_size
            new_h = int(h * (max_size / w))
        
        return cv2.resize(imagen, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def draw_bounding_boxes(imagen, boxes, labels=None, colors=None, thickness=2):
        """
        Dibuja bounding boxes en una imagen.
        
        Args:
            imagen (np.ndarray): Imagen
            boxes (list): Lista de boxes [(x1, y1, x2, y2), ...]
            labels (list): Etiquetas opcionales
            colors (list): Colores para cada box
            thickness (int): Grosor de las líneas
            
        Returns:
            np.ndarray: Imagen con boxes dibujados
        """
        resultado = imagen.copy()
        
        if colors is None:
            colors = [(0, 255, 0)] * len(boxes)  # Verde por defecto
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            color = colors[i % len(colors)]
            
            cv2.rectangle(resultado, (int(x1), int(y1)), (int(x2), int(y2)), 
                         color, thickness)
            
            if labels and i < len(labels):
                label = labels[i]
                # Fondo para el texto
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(resultado, 
                            (int(x1), int(y1) - text_height - 5),
                            (int(x1) + text_width, int(y1)),
                            color, -1)
                cv2.putText(resultado, label, (int(x1), int(y1) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return resultado


class ResultsManager:
    """Gestor de resultados de análisis."""
    
    def __init__(self, output_dir="./resultados"):
        """
        Inicializar el gestor de resultados.
        
        Args:
            output_dir (str): Directorio de salida
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def save_results_json(self, data, filename=None):
        """
        Guarda resultados en formato JSON.
        
        Args:
            data (dict): Datos a guardar
            filename (str): Nombre del archivo (opcional)
            
        Returns:
            str: Ruta del archivo guardado
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"resultados_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convertir numpy arrays a listas para JSON
        data_serializable = self._make_json_serializable(data)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data_serializable, f, indent=2, ensure_ascii=False)
            print(f"✅ Resultados guardados en: {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ Error al guardar JSON: {str(e)}")
            return None
    
    def save_results_csv(self, data, filename=None):
        """
        Guarda resultados en formato CSV.
        
        Args:
            data (list): Lista de diccionarios con datos
            filename (str): Nombre del archivo
            
        Returns:
            str: Ruta del archivo guardado
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"resultados_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        if not data or not isinstance(data, list):
            print("❌ Datos inválidos para CSV")
            return None
        
        try:
            # Obtener todas las claves únicas
            all_keys = set()
            for item in data:
                if isinstance(item, dict):
                    all_keys.update(item.keys())
            
            fieldnames = sorted(all_keys)
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for item in data:
                    if isinstance(item, dict):
                        # Convertir valores complejos a strings
                        row = {}
                        for key in fieldnames:
                            value = item.get(key, '')
                            if isinstance(value, (list, dict, np.ndarray)):
                                row[key] = str(value)
                            else:
                                row[key] = value
                        writer.writerow(row)
            
            print(f"✅ Resultados guardados en: {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ Error al guardar CSV: {str(e)}")
            return None
    
    def save_results_txt(self, text, filename=None):
        """
        Guarda reporte de texto.
        
        Args:
            text (str): Texto a guardar
            filename (str): Nombre del archivo
            
        Returns:
            str: Ruta del archivo guardado
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reporte_{timestamp}.txt"
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"✅ Reporte guardado en: {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ Error al guardar TXT: {str(e)}")
            return None
    
    def load_results_json(self, filepath):
        """
        Carga resultados desde JSON.
        
        Args:
            filepath (str): Ruta del archivo
            
        Returns:
            dict: Datos cargados
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error al cargar JSON: {str(e)}")
            return None
    
    def _make_json_serializable(self, obj):
        """
        Convierte objetos no serializables a formatos compatibles con JSON.
        
        Args:
            obj: Objeto a convertir
            
        Returns:
            Objeto serializable
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) 
                    for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, cv2.KeyPoint):
            return {
                'pt': obj.pt,
                'size': obj.size,
                'angle': obj.angle,
                'response': obj.response
            }
        else:
            return obj
    
    def create_summary_report(self, resultados_list, output_filename=None):
        """
        Crea un reporte resumen de múltiples análisis.
        
        Args:
            resultados_list (list): Lista de resultados
            output_filename (str): Nombre del archivo de salida
            
        Returns:
            str: Ruta del reporte generado
        """
        if not resultados_list:
            print("No hay resultados para resumir")
            return None
        
        # Generar reporte de texto
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("REPORTE RESUMEN DE ANÁLISIS DE TRÁFICO VEHICULAR")
        report_lines.append("="*80)
        report_lines.append(f"\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total de análisis: {len(resultados_list)}\n")
        
        for i, resultado in enumerate(resultados_list, 1):
            report_lines.append(f"\n--- Análisis #{i} ---")
            
            if isinstance(resultado, dict):
                for key, value in resultado.items():
                    if isinstance(value, (int, float, str, bool)):
                        report_lines.append(f"{key}: {value}")
                    elif isinstance(value, dict):
                        report_lines.append(f"{key}:")
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, (int, float, str, bool)):
                                report_lines.append(f"  - {subkey}: {subvalue}")
        
        report_lines.append("\n" + "="*80)
        
        report_text = "\n".join(report_lines)
        
        return self.save_results_txt(report_text, output_filename)


def format_time(seconds):
    """
    Formatea tiempo en segundos a formato legible.
    
    Args:
        seconds (float): Tiempo en segundos
        
    Returns:
        str: Tiempo formateado
    """
    if seconds < 1:
        return f"{seconds*1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.2f}s"


def create_progress_bar(current, total, bar_length=50, prefix='Progreso'):
    """
    Crea una barra de progreso en consola.
    
    Args:
        current (int): Valor actual
        total (int): Valor total
        bar_length (int): Longitud de la barra
        prefix (str): Prefijo del texto
    """
    percent = current / total
    filled_length = int(bar_length * percent)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    print(f'\r{prefix}: |{bar}| {percent*100:.1f}% ({current}/{total})', end='', flush=True)
    
    if current == total:
        print()  # Nueva línea al terminar


def visualize_comparison(results_dict, save_path=None):
    """
    Visualiza comparación de resultados.
    
    Args:
        results_dict (dict): Diccionario con resultados
        save_path (str): Ruta para guardar la visualización
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = list(results_dict.keys())
    values = list(results_dict.values())
    
    ax.bar(labels, values, color='steelblue')
    ax.set_xlabel('Métodos')
    ax.set_ylabel('Valores')
    ax.set_title('Comparación de Resultados')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualización guardada en: {save_path}")
    
    plt.show()


def get_timestamp():
    """
    Obtiene timestamp actual en formato string.
    
    Returns:
        str: Timestamp
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(directory):
    """
    Asegura que un directorio existe, creándolo si es necesario.
    
    Args:
        directory (str): Ruta del directorio
        
    Returns:
        bool: True si existe o se creó exitosamente
    """
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        print(f"❌ Error al crear directorio {directory}: {str(e)}")
        return False


def save_image_batch(imagen, output_path, nombre_tecnica, timestamp=None):
    """
    Guarda una imagen procesada en modo lotes con nomenclatura estándar.
    
    Args:
        imagen (np.ndarray): Imagen a guardar
        output_path (str): Directorio de salida
        nombre_tecnica (str): Nombre de la técnica aplicada
        timestamp (str): Timestamp opcional
        
    Returns:
        str: Ruta del archivo guardado o None si hay error
    """
    if timestamp is None:
        timestamp = get_timestamp()
    
    try:
        # Crear directorio si no existe
        ensure_dir(output_path)
        
        # Generar nombre de archivo
        nombre_archivo = f"{nombre_tecnica.lower().replace(' ', '_')}_{timestamp}.jpg"
        ruta_completa = os.path.join(output_path, nombre_archivo)
        
        # Guardar imagen
        cv2.imwrite(ruta_completa, imagen)
        return ruta_completa
        
    except Exception as e:
        print(f"❌ Error al guardar imagen en lotes: {str(e)}")
        return None


def create_batch_report(resultados_lotes, output_dir, nombre_reporte="batch_report"):
    """
    Crea un reporte consolidado de procesamiento por lotes.
    
    Args:
        resultados_lotes (list): Lista de resultados de procesamiento
        output_dir (str): Directorio de salida
        nombre_reporte (str): Nombre base del reporte
        
    Returns:
        dict: Rutas de los archivos generados
    """
    timestamp = get_timestamp()
    archivos_generados = {}
    
    try:
        ensure_dir(output_dir)
        
        # Reporte JSON
        json_path = os.path.join(output_dir, f"{nombre_reporte}_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(resultados_lotes, f, indent=2, ensure_ascii=False, default=str)
        archivos_generados['json'] = json_path
        
        # Reporte de texto
        txt_path = os.path.join(output_dir, f"{nombre_reporte}_{timestamp}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("REPORTE DE PROCESAMIENTO POR LOTES\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total de imágenes procesadas: {len(resultados_lotes)}\n\n")
            
            for i, resultado in enumerate(resultados_lotes, 1):
                f.write(f"Imagen {i}: {resultado.get('nombre', 'Desconocida')}\n")
                f.write(f"  Estado: {'✓ Exitoso' if resultado.get('exitoso', False) else '❌ Error'}\n")
                if 'metodos' in resultado:
                    f.write(f"  Métodos aplicados: {', '.join(resultado['metodos'])}\n")
                if 'errores' in resultado and resultado['errores']:
                    f.write(f"  Errores: {len(resultado['errores'])}\n")
                f.write("\n")
        
        archivos_generados['txt'] = txt_path
        
        print(f"📋 Reporte de lotes generado:")
        print(f"  JSON: {json_path}")
        print(f"  TXT: {txt_path}")
        
    except Exception as e:
        print(f"❌ Error generando reporte de lotes: {str(e)}")
    
    return archivos_generados


class BatchProcessor:
    """Procesador específico para operaciones por lotes sin previsualización."""
    
    def __init__(self, output_dir="./resultados_lotes"):
        """
        Inicializar procesador por lotes.
        
        Args:
            output_dir (str): Directorio de salida
        """
        self.output_dir = output_dir
        ensure_dir(output_dir)
        self.processed_count = 0
        self.error_count = 0
        self.start_time = None
    
    def start_batch(self):
        """Inicia el cronometraje del lote."""
        self.start_time = datetime.now()
        self.processed_count = 0
        self.error_count = 0
        print("🚀 Iniciando procesamiento por lotes...")
    
    def process_image(self, imagen, metodos, nombre_imagen):
        """
        Procesa una imagen con los métodos especificados.
        
        Args:
            imagen: Imagen a procesar
            metodos: Lista de métodos a aplicar
            nombre_imagen: Nombre de la imagen
            
        Returns:
            dict: Resultado del procesamiento
        """
        resultado = {
            'nombre': nombre_imagen,
            'exitoso': False,
            'metodos': [],
            'errores': [],
            'archivos_generados': []
        }
        
        try:
            print(f"  📷 Procesando: {nombre_imagen}")
            
            imagen_trabajo = imagen.copy()
            
            for metodo in metodos:
                try:
                    # Aquí iría la lógica específica de cada método
                    # Esto es un placeholder que debe ser implementado según necesidades
                    print(f"    → {metodo}")
                    
                    # Simular procesamiento exitoso
                    resultado['metodos'].append(metodo)
                    
                    # Guardar resultado
                    timestamp = get_timestamp()
                    archivo_guardado = save_image_batch(
                        imagen_trabajo, 
                        self.output_dir, 
                        f"{nombre_imagen}_{metodo}",
                        timestamp
                    )
                    
                    if archivo_guardado:
                        resultado['archivos_generados'].append(archivo_guardado)
                    
                except Exception as e:
                    error_msg = f"Error en {metodo}: {str(e)}"
                    resultado['errores'].append(error_msg)
                    print(f"    ❌ {error_msg}")
            
            if len(resultado['metodos']) > 0:
                resultado['exitoso'] = True
                self.processed_count += 1
                print(f"    ✓ Completado ({len(resultado['metodos'])} métodos)")
            else:
                self.error_count += 1
                print(f"    ❌ Sin métodos exitosos")
                
        except Exception as e:
            resultado['errores'].append(f"Error general: {str(e)}")
            self.error_count += 1
            print(f"    ❌ Error procesando imagen: {str(e)}")
        
        return resultado
    
    def finish_batch(self):
        """Finaliza el lote y muestra estadísticas."""
        if self.start_time:
            tiempo_total = (datetime.now() - self.start_time).total_seconds()
            print(f"\n✅ Procesamiento por lotes completado")
            print(f"   📊 Estadísticas:")
            print(f"   • Imágenes procesadas: {self.processed_count}")
            print(f"   • Imágenes con error: {self.error_count}")
            print(f"   • Tiempo total: {tiempo_total:.2f} segundos")
            if self.processed_count > 0:
                print(f"   • Tiempo promedio: {tiempo_total/self.processed_count:.2f}s por imagen")
            print(f"   📁 Resultados en: {self.output_dir}")
        else:
            print("⚠️ Lote no iniciado correctamente")
