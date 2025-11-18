#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrenador YOLO para Detección de Sombreros
===========================================

Módulo completo para entrenar modelos YOLOv8 personalizados
para detectar diferentes tipos de sombreros.

Universidad del Quindío - Visión Artificial
Fecha: Noviembre 2025
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd

class EntrenadorYOLOSombreros:
    """
    Clase para entrenar modelos YOLO personalizados para detectar sombreros.
    """
    
    def __init__(self):
        """Inicializa el entrenador."""
        self.model = None
        self.results = None
        self.config = {
            'epochs': 100,
            'batch_size': 16,
            'img_size': 640,
            'learning_rate': 0.01,
            'patience': 50,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        print("=" * 60)
        print("ENTRENADOR YOLO PARA DETECCIÓN DE SOMBREROS")
        print("=" * 60)
        print(f"Dispositivo: {self.config['device']}")
        if self.config['device'] == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Versión: {torch.version.cuda}")
        print("=" * 60)
    
    def validar_dataset(self, dataset_yaml):
        """
        Valida que el dataset esté correctamente formateado.
        
        Args:
            dataset_yaml: Ruta al archivo data.yaml
            
        Returns:
            bool: True si es válido
        """
        print("\n🔍 VALIDANDO DATASET...")
        print("-" * 60)
        
        try:
            # Cargar configuración
            with open(dataset_yaml, 'r') as f:
                data = yaml.safe_load(f)
            
            # Verificar campos requeridos
            required_fields = ['train', 'val', 'nc', 'names']
            for field in required_fields:
                if field not in data:
                    print(f"❌ Campo requerido '{field}' no encontrado en {dataset_yaml}")
                    return False
            
            print(f"✓ Archivo YAML válido")
            print(f"  Clases: {data['nc']}")
            print(f"  Nombres: {', '.join(data['names'].values() if isinstance(data['names'], dict) else data['names'])}")
            
            # Verificar rutas
            base_path = Path(dataset_yaml).parent
            
            for split in ['train', 'val']:
                if split in data:
                    images_path = base_path / data[split]
                    labels_path = images_path.parent.parent / 'labels' / images_path.name
                    
                    if not images_path.exists():
                        print(f"❌ Ruta de imágenes no existe: {images_path}")
                        return False
                    
                    # Contar imágenes
                    images = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
                    num_images = len(images)
                    
                    # Contar labels
                    if labels_path.exists():
                        labels = list(labels_path.glob('*.txt'))
                        num_labels = len(labels)
                    else:
                        num_labels = 0
                    
                    print(f"\n  {split.upper()}:")
                    print(f"    📂 Imágenes: {num_images}")
                    print(f"    📄 Labels: {num_labels}")
                    
                    if num_images == 0:
                        print(f"    ⚠️  No se encontraron imágenes en {images_path}")
                    
                    if num_images > 0 and num_labels == 0:
                        print(f"    ⚠️  No se encontraron labels en {labels_path}")
                    
                    # Validar algunas anotaciones
                    if num_labels > 0:
                        sample_labels = labels[:min(10, len(labels))]
                        valid_labels = 0
                        
                        for label_file in sample_labels:
                            try:
                                with open(label_file, 'r') as f:
                                    lines = f.readlines()
                                    for line in lines:
                                        parts = line.strip().split()
                                        if len(parts) == 5:
                                            cls, x, y, w, h = map(float, parts)
                                            if 0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
                                                valid_labels += 1
                            except Exception as e:
                                print(f"    ⚠️  Error leyendo {label_file.name}: {e}")
                        
                        print(f"    ✓ Labels válidos (muestra): {valid_labels}/{len(sample_labels) * 2}")
            
            print("\n" + "=" * 60)
            print("✅ DATASET VÁLIDO")
            print("=" * 60)
            return True
            
        except Exception as e:
            print(f"\n❌ Error validando dataset: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def entrenar(self, dataset_yaml, epochs=None, batch_size=None, img_size=None,
                 model_size='n', resume=None, device=None):
        """
        Entrena el modelo YOLO.
        
        Args:
            dataset_yaml: Ruta al archivo data.yaml
            epochs: Número de épocas
            batch_size: Tamaño de batch
            img_size: Tamaño de imagen
            model_size: Tamaño del modelo ('n', 's', 'm', 'l', 'x')
            resume: Ruta al checkpoint para continuar
            device: Dispositivo ('cuda', 'cpu', o número de GPU)
        """
        print("\n🏋️  INICIANDO ENTRENAMIENTO...")
        print("=" * 60)
        
        # Actualizar configuración
        if epochs:
            self.config['epochs'] = epochs
        if batch_size:
            self.config['batch_size'] = batch_size
        if img_size:
            self.config['img_size'] = img_size
        if device:
            self.config['device'] = device
        
        # Mostrar configuración
        print("📋 Configuración:")
        print(f"  Modelo base: YOLOv8{model_size}")
        print(f"  Épocas: {self.config['epochs']}")
        print(f"  Batch size: {self.config['batch_size']}")
        print(f"  Tamaño imagen: {self.config['img_size']}")
        print(f"  Learning rate: {self.config['learning_rate']}")
        print(f"  Patience: {self.config['patience']}")
        print(f"  Dispositivo: {self.config['device']}")
        print("=" * 60)
        
        try:
            # Cargar modelo preentrenado o reanudar
            if resume:
                print(f"\n📂 Reanudando desde: {resume}")
                self.model = YOLO(resume)
            else:
                model_name = f'yolov8{model_size}.pt'
                print(f"\n📥 Cargando modelo base: {model_name}")
                self.model = YOLO(model_name)
            
            # Iniciar entrenamiento
            print("\n🚀 Iniciando entrenamiento...")
            print("=" * 60)
            
            self.results = self.model.train(
                data=dataset_yaml,
                epochs=self.config['epochs'],
                batch=self.config['batch_size'],
                imgsz=self.config['img_size'],
                lr0=self.config['learning_rate'],
                patience=self.config['patience'],
                device=self.config['device'],
                project='runs/detect',
                name='train',
                exist_ok=False,
                pretrained=True,
                optimizer='auto',
                verbose=True,
                seed=42,
                deterministic=True,
                single_cls=False,
                rect=False,
                cos_lr=False,
                close_mosaic=10,
                resume=resume is not None,
                amp=True,  # Automatic Mixed Precision
                fraction=1.0,
                profile=False,
                overlap_mask=True,
                mask_ratio=4,
                dropout=0.0,
                val=True,
                split='val',
                save=True,
                save_period=-1,
                cache=False,
                workers=8,
                plots=True
            )
            
            print("\n" + "=" * 60)
            print("✅ ENTRENAMIENTO COMPLETADO")
            print("=" * 60)
            
            # Mostrar resultados
            self.mostrar_resultados()
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error durante el entrenamiento: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def evaluar(self, weights_path, dataset_yaml):
        """
        Evalúa el modelo en el conjunto de test.
        
        Args:
            weights_path: Ruta al modelo entrenado (.pt)
            dataset_yaml: Ruta al archivo data.yaml
        """
        print("\n📊 EVALUANDO MODELO...")
        print("=" * 60)
        
        try:
            # Cargar modelo
            self.model = YOLO(weights_path)
            
            # Evaluar
            results = self.model.val(
                data=dataset_yaml,
                split='test',
                batch=1,
                imgsz=self.config['img_size'],
                device=self.config['device'],
                plots=True,
                save_json=True,
                save_hybrid=False,
                conf=0.001,
                iou=0.6,
                max_det=300,
                half=False,
                dnn=False,
                verbose=True
            )
            
            print("\n📈 Resultados de Evaluación:")
            print(f"  mAP@0.5: {results.box.map50:.4f}")
            print(f"  mAP@0.5:0.95: {results.box.map:.4f}")
            print(f"  Precision: {results.box.mp:.4f}")
            print(f"  Recall: {results.box.mr:.4f}")
            
            print("\n✅ Evaluación completada")
            return results
            
        except Exception as e:
            print(f"\n❌ Error en evaluación: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predecir(self, weights_path, source, conf_threshold=0.25, save=True):
        """
        Realiza predicciones con el modelo entrenado.
        
        Args:
            weights_path: Ruta al modelo (.pt)
            source: Imagen, video, carpeta, o 0 para webcam
            conf_threshold: Umbral de confianza
            save: Si guardar resultados
        """
        print("\n🔍 REALIZANDO PREDICCIONES...")
        print("=" * 60)
        
        try:
            # Cargar modelo
            self.model = YOLO(weights_path)
            
            # Predecir
            results = self.model.predict(
                source=source,
                conf=conf_threshold,
                imgsz=self.config['img_size'],
                device=self.config['device'],
                save=save,
                save_txt=True,
                save_conf=True,
                show=False,
                stream=False,
                verbose=True,
                visualize=False,
                augment=False,
                agnostic_nms=False,
                classes=None,
                retina_masks=False,
                boxes=True
            )
            
            print(f"\n✅ Predicción completada")
            print(f"  Resultados guardados en: runs/detect/predict")
            
            return results
            
        except Exception as e:
            print(f"\n❌ Error en predicción: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def mostrar_resultados(self):
        """Muestra un resumen de los resultados del entrenamiento."""
        print("\n📊 RESUMEN DE RESULTADOS:")
        print("=" * 60)
        
        try:
            # Buscar directorio de resultados más reciente
            runs_dir = Path('runs/detect')
            if not runs_dir.exists():
                print("⚠️  No se encontró directorio de resultados")
                return
            
            # Obtener último directorio train
            train_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and 'train' in d.name])
            if not train_dirs:
                print("⚠️  No se encontraron directorios de entrenamiento")
                return
            
            latest_dir = train_dirs[-1]
            print(f"📂 Directorio: {latest_dir}")
            
            # Leer results.csv
            results_csv = latest_dir / 'results.csv'
            if results_csv.exists():
                df = pd.read_csv(results_csv)
                df.columns = df.columns.str.strip()
                
                # Última época
                last_row = df.iloc[-1]
                
                print(f"\n🎯 Última Época ({len(df)}/{self.config['epochs']}):")
                print(f"  mAP@0.5: {last_row['metrics/mAP50(B)']:.4f}")
                print(f"  mAP@0.5:0.95: {last_row['metrics/mAP50-95(B)']:.4f}")
                print(f"  Precision: {last_row['metrics/precision(B)']:.4f}")
                print(f"  Recall: {last_row['metrics/recall(B)']:.4f}")
                
                # Mejor época
                best_map50 = df['metrics/mAP50(B)'].max()
                best_epoch = df['metrics/mAP50(B)'].idxmax() + 1
                print(f"\n🏆 Mejor Época ({best_epoch}):")
                print(f"  mAP@0.5: {best_map50:.4f}")
            
            # Archivos generados
            print(f"\n📁 Archivos Generados:")
            weights_dir = latest_dir / 'weights'
            if weights_dir.exists():
                print(f"  ✓ {weights_dir / 'best.pt'} (Mejor modelo)")
                print(f"  ✓ {weights_dir / 'last.pt'} (Último checkpoint)")
            
            plots = ['confusion_matrix.png', 'F1_curve.png', 'P_curve.png', 
                    'R_curve.png', 'PR_curve.png', 'results.png']
            for plot in plots:
                plot_path = latest_dir / plot
                if plot_path.exists():
                    print(f"  ✓ {plot}")
            
            print("\n" + "=" * 60)
            print("💡 Siguiente paso:")
            print(f"   Usar modelo: {weights_dir / 'best.pt'}")
            print("=" * 60)
            
        except Exception as e:
            print(f"⚠️  Error mostrando resultados: {e}")
    
    def exportar_modelo(self, weights_path, formato='onnx'):
        """
        Exporta el modelo a diferentes formatos.
        
        Args:
            weights_path: Ruta al modelo (.pt)
            formato: Formato de exportación ('onnx', 'torchscript', 'coreml', etc.)
        """
        print(f"\n📤 EXPORTANDO MODELO A {formato.upper()}...")
        print("=" * 60)
        
        try:
            self.model = YOLO(weights_path)
            
            export_path = self.model.export(
                format=formato,
                imgsz=self.config['img_size'],
                keras=False,
                optimize=False,
                half=False,
                int8=False,
                dynamic=False,
                simplify=False,
                opset=None,
                workspace=4,
                nms=False
            )
            
            print(f"\n✅ Modelo exportado exitosamente")
            print(f"  Ruta: {export_path}")
            return export_path
            
        except Exception as e:
            print(f"\n❌ Error exportando modelo: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Entrenador YOLO para Detección de Sombreros',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Validar dataset
  python entrenador_yolo_sombreros.py --mode validate --dataset data.yaml

  # Entrenar modelo
  python entrenador_yolo_sombreros.py --mode train --dataset data.yaml --epochs 100

  # Evaluar modelo
  python entrenador_yolo_sombreros.py --mode test --weights best.pt --dataset data.yaml

  # Predecir en imagen
  python entrenador_yolo_sombreros.py --mode predict --weights best.pt --source image.jpg

  # Predecir en video
  python entrenador_yolo_sombreros.py --mode predict --weights best.pt --source video.mp4

  # Predecir en webcam
  python entrenador_yolo_sombreros.py --mode predict --weights best.pt --source 0
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['validate', 'train', 'test', 'predict', 'export'],
                       help='Modo de operación')
    
    parser.add_argument('--dataset', type=str,
                       help='Ruta al archivo data.yaml')
    
    parser.add_argument('--weights', type=str,
                       help='Ruta al archivo de pesos (.pt)')
    
    parser.add_argument('--source', type=str,
                       help='Fuente para predicción (imagen, video, carpeta, o 0 para webcam)')
    
    parser.add_argument('--epochs', type=int, default=100,
                       help='Número de épocas (default: 100)')
    
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Tamaño de batch (default: 16)')
    
    parser.add_argument('--img-size', type=int, default=640,
                       help='Tamaño de imagen (default: 640)')
    
    parser.add_argument('--model-size', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Tamaño del modelo: n(nano), s(small), m(medium), l(large), x(xlarge)')
    
    parser.add_argument('--device', type=str, default=None,
                       help='Dispositivo (cuda, cpu, 0, 1, etc.)')
    
    parser.add_argument('--resume', type=str,
                       help='Ruta al checkpoint para continuar entrenamiento')
    
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Umbral de confianza para predicción (default: 0.25)')
    
    parser.add_argument('--export-format', type=str, default='onnx',
                       choices=['onnx', 'torchscript', 'coreml', 'engine', 'saved_model'],
                       help='Formato de exportación')
    
    args = parser.parse_args()
    
    # Crear entrenador
    entrenador = EntrenadorYOLOSombreros()
    
    # Ejecutar modo seleccionado
    if args.mode == 'validate':
        if not args.dataset:
            print("❌ Error: --dataset es requerido para validación")
            sys.exit(1)
        
        if entrenador.validar_dataset(args.dataset):
            print("\n✅ Dataset válido y listo para entrenamiento")
            sys.exit(0)
        else:
            print("\n❌ Dataset inválido, revisar errores")
            sys.exit(1)
    
    elif args.mode == 'train':
        if not args.dataset:
            print("❌ Error: --dataset es requerido para entrenamiento")
            sys.exit(1)
        
        # Validar dataset primero
        if not entrenador.validar_dataset(args.dataset):
            print("\n❌ Dataset inválido, corregir antes de entrenar")
            sys.exit(1)
        
        # Entrenar
        if entrenador.entrenar(
            dataset_yaml=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            model_size=args.model_size,
            resume=args.resume,
            device=args.device
        ):
            print("\n✅ Entrenamiento exitoso")
            sys.exit(0)
        else:
            print("\n❌ Error en entrenamiento")
            sys.exit(1)
    
    elif args.mode == 'test':
        if not args.weights:
            print("❌ Error: --weights es requerido para evaluación")
            sys.exit(1)
        if not args.dataset:
            print("❌ Error: --dataset es requerido para evaluación")
            sys.exit(1)
        
        if entrenador.evaluar(args.weights, args.dataset):
            print("\n✅ Evaluación completada")
            sys.exit(0)
        else:
            print("\n❌ Error en evaluación")
            sys.exit(1)
    
    elif args.mode == 'predict':
        if not args.weights:
            print("❌ Error: --weights es requerido para predicción")
            sys.exit(1)
        if not args.source:
            print("❌ Error: --source es requerido para predicción")
            sys.exit(1)
        
        if entrenador.predecir(args.weights, args.source, args.conf):
            print("\n✅ Predicción completada")
            sys.exit(0)
        else:
            print("\n❌ Error en predicción")
            sys.exit(1)
    
    elif args.mode == 'export':
        if not args.weights:
            print("❌ Error: --weights es requerido para exportación")
            sys.exit(1)
        
        if entrenador.exportar_modelo(args.weights, args.export_format):
            print("\n✅ Exportación completada")
            sys.exit(0)
        else:
            print("\n❌ Error en exportación")
            sys.exit(1)

if __name__ == '__main__':
    main()
