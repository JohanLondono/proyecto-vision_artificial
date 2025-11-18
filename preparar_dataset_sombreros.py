#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preparador de Dataset para YOLO
================================

Reorganiza datasets descargados de Open Images al formato YOLO estándar:
- De estructura por clase a estructura train/val/test
- Valida anotaciones
- Genera data.yaml automáticamente
- Split automático (70/20/10)

Universidad del Quindío - Visión Artificial
"""

import os
import shutil
import yaml
import random
from pathlib import Path
from collections import defaultdict
import argparse

class PreparadorDatasetYOLO:
    """
    Reorganiza datasets de Open Images a formato YOLO estándar.
    """
    
    def __init__(self, input_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """
        Inicializa el preparador.
        
        Args:
            input_dir: Carpeta con estructura Open Images (por clase)
            output_dir: Carpeta de salida en formato YOLO
            train_ratio: Proporción para entrenamiento
            val_ratio: Proporción para validación
            test_ratio: Proporción para test
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Mapeo de clases Open Images a IDs YOLO
        self.class_mapping = {}
        self.class_names = []
        
        print("=" * 60)
        print("PREPARADOR DE DATASET YOLO")
        print("=" * 60)
        print(f"📂 Entrada: {self.input_dir}")
        print(f"📂 Salida: {self.output_dir}")
        print(f"📊 Split: {train_ratio*100:.0f}% train / {val_ratio*100:.0f}% val / {test_ratio*100:.0f}% test")
        print("=" * 60)
    
    def detectar_estructura(self):
        """
        Detecta si es estructura Open Images o personalizada.
        
        Returns:
            str: 'openimages' o 'custom'
        """
        print("\n🔍 Detectando estructura del dataset...")
        
        # Buscar carpetas con imágenes
        subdirs = [d for d in self.input_dir.iterdir() if d.is_dir()]
        
        # Verificar estructura Open Images (carpetas por clase con /images y /darknet)
        is_openimages = False
        for subdir in subdirs:
            images_dir = subdir / 'images'
            darknet_dir = subdir / 'darknet'
            if images_dir.exists() and darknet_dir.exists():
                is_openimages = True
                break
        
        if is_openimages:
            print("  ✓ Estructura detectada: Open Images")
            return 'openimages'
        else:
            print("  ✓ Estructura detectada: Personalizada")
            return 'custom'
    
    def reorganizar_openimages(self):
        """
        Reorganiza estructura Open Images a formato YOLO.
        
        Open Images:
            dataset/
                cowboy hat/
                    images/
                    darknet/
                fedora/
                    images/
                    darknet/
        
        YOLO:
            dataset/
                images/
                    train/
                    val/
                    test/
                labels/
                    train/
                    val/
                    test/
        """
        print("\n🔄 Reorganizando dataset de Open Images a YOLO...")
        
        # Crear estructura de salida
        for split in ['train', 'val', 'test']:
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Recolectar todas las clases
        class_dirs = [d for d in self.input_dir.iterdir() 
                     if d.is_dir() and (d / 'images').exists()]
        
        # Leer archivo de nombres de clase si existe
        names_file = self.input_dir / 'darknet_obj_names.txt'
        if names_file.exists():
            with open(names_file, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            print(f"\n📋 Clases encontradas: {len(self.class_names)}")
        else:
            # Usar nombres de carpetas
            self.class_names = [d.name for d in class_dirs]
            print(f"\n📋 Clases detectadas: {len(self.class_names)}")
        
        for i, name in enumerate(self.class_names):
            print(f"  {i}: {name}")
        
        # Procesar cada clase
        all_files = []
        stats = defaultdict(int)
        
        print("\n📦 Procesando clases...")
        for class_dir in class_dirs:
            class_name = class_dir.name
            images_dir = class_dir / 'images'
            labels_dir = class_dir / 'darknet'
            
            if not images_dir.exists() or not labels_dir.exists():
                print(f"  ⚠️  Saltando {class_name}: estructura incompleta")
                continue
            
            # Recolectar pares imagen-label
            images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
            
            for img_path in images:
                label_path = labels_dir / f"{img_path.stem}.txt"
                
                if label_path.exists():
                    all_files.append({
                        'image': img_path,
                        'label': label_path,
                        'class': class_name
                    })
                    stats[class_name] += 1
            
            print(f"  ✓ {class_name}: {stats[class_name]} imágenes")
        
        # Mezclar aleatoriamente
        random.seed(42)
        random.shuffle(all_files)
        
        # Split dataset
        total = len(all_files)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)
        
        splits = {
            'train': all_files[:train_end],
            'val': all_files[train_end:val_end],
            'test': all_files[val_end:]
        }
        
        print(f"\n📊 Distribución:")
        print(f"  Train: {len(splits['train'])} imágenes")
        print(f"  Val: {len(splits['val'])} imágenes")
        print(f"  Test: {len(splits['test'])} imágenes")
        print(f"  Total: {total} imágenes")
        
        # Copiar archivos a estructura YOLO
        print("\n📋 Copiando archivos...")
        copied = {'train': 0, 'val': 0, 'test': 0}
        
        for split, files in splits.items():
            for item in files:
                # Copiar imagen
                dst_img = self.output_dir / 'images' / split / item['image'].name
                shutil.copy2(item['image'], dst_img)
                
                # Copiar label
                dst_label = self.output_dir / 'labels' / split / item['label'].name
                shutil.copy2(item['label'], dst_label)
                
                copied[split] += 1
            
            print(f"  ✓ {split}: {copied[split]} archivos copiados")
        
        # Generar data.yaml
        self.generar_data_yaml()
        
        print("\n" + "=" * 60)
        print("✅ REORGANIZACIÓN COMPLETADA")
        print("=" * 60)
        print(f"📂 Dataset YOLO listo en: {self.output_dir}")
        print(f"📄 Configuración: {self.output_dir / 'data.yaml'}")
        
        return True
    
    def generar_data_yaml(self):
        """
        Genera archivo data.yaml para YOLO.
        """
        print("\n📝 Generando data.yaml...")
        
        # Rutas relativas
        yaml_content = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.class_names),
            'names': {i: name for i, name in enumerate(self.class_names)}
        }
        
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
        
        print(f"  ✓ data.yaml generado")
        
        # Mostrar contenido
        print("\n📄 Contenido de data.yaml:")
        print("-" * 60)
        with open(yaml_path, 'r') as f:
            print(f.read())
        print("-" * 60)
    
    def validar_dataset(self):
        """
        Valida que el dataset esté correctamente formateado.
        """
        print("\n🔍 VALIDANDO DATASET...")
        print("=" * 60)
        
        errors = []
        warnings = []
        
        # Verificar estructura
        required_dirs = [
            'images/train', 'images/val', 'images/test',
            'labels/train', 'labels/val', 'labels/test'
        ]
        
        for dir_path in required_dirs:
            full_path = self.output_dir / dir_path
            if not full_path.exists():
                errors.append(f"Carpeta faltante: {dir_path}")
        
        # Verificar data.yaml
        yaml_path = self.output_dir / 'data.yaml'
        if not yaml_path.exists():
            errors.append("Archivo data.yaml faltante")
        else:
            try:
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    
                if 'nc' not in data or 'names' not in data:
                    errors.append("data.yaml incompleto (falta nc o names)")
            except Exception as e:
                errors.append(f"Error leyendo data.yaml: {e}")
        
        # Verificar correspondencia imagen-label
        for split in ['train', 'val', 'test']:
            images_dir = self.output_dir / 'images' / split
            labels_dir = self.output_dir / 'labels' / split
            
            if images_dir.exists():
                images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
                
                for img in images:
                    label = labels_dir / f"{img.stem}.txt"
                    if not label.exists():
                        warnings.append(f"{split}: {img.name} sin label")
                
                print(f"\n📂 {split.upper()}:")
                print(f"  Imágenes: {len(images)}")
                
                if labels_dir.exists():
                    labels = list(labels_dir.glob('*.txt'))
                    print(f"  Labels: {len(labels)}")
                    
                    # Validar formato de algunas labels
                    sample_labels = labels[:min(5, len(labels))]
                    for label_file in sample_labels:
                        try:
                            with open(label_file, 'r') as f:
                                lines = f.readlines()
                                for line in lines:
                                    parts = line.strip().split()
                                    if len(parts) != 5:
                                        errors.append(f"{label_file.name}: formato inválido (esperado 5 valores)")
                                        break
                                    
                                    # Verificar rangos
                                    cls, x, y, w, h = map(float, parts)
                                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                                        errors.append(f"{label_file.name}: coordenadas fuera de rango [0-1]")
                                        break
                        except Exception as e:
                            errors.append(f"{label_file.name}: error de lectura ({e})")
        
        # Resumen
        print("\n" + "=" * 60)
        if errors:
            print("❌ ERRORES ENCONTRADOS:")
            for error in errors:
                print(f"  • {error}")
        
        if warnings:
            print("\n⚠️  ADVERTENCIAS:")
            for warning in warnings[:10]:  # Limitar a 10
                print(f"  • {warning}")
            if len(warnings) > 10:
                print(f"  ... y {len(warnings) - 10} más")
        
        if not errors and not warnings:
            print("✅ DATASET VÁLIDO")
            print("  Sin errores ni advertencias")
        
        print("=" * 60)
        
        return len(errors) == 0
    
    def generar_estadisticas(self):
        """
        Genera estadísticas del dataset.
        """
        print("\n📊 ESTADÍSTICAS DEL DATASET")
        print("=" * 60)
        
        yaml_path = self.output_dir / 'data.yaml'
        if not yaml_path.exists():
            print("❌ data.yaml no encontrado")
            return
        
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        class_names = data.get('names', {})
        class_counts = {split: defaultdict(int) for split in ['train', 'val', 'test']}
        
        # Contar instancias por clase
        for split in ['train', 'val', 'test']:
            labels_dir = self.output_dir / 'labels' / split
            
            if labels_dir.exists():
                for label_file in labels_dir.glob('*.txt'):
                    try:
                        with open(label_file, 'r') as f:
                            for line in f:
                                cls = int(line.split()[0])
                                class_name = class_names.get(cls, f'clase_{cls}')
                                class_counts[split][class_name] += 1
                    except:
                        continue
        
        # Mostrar tabla
        print("\n📋 Distribución por Clase:\n")
        print(f"{'Clase':<20} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
        print("-" * 60)
        
        for cls_id, cls_name in sorted(class_names.items()):
            train_count = class_counts['train'][cls_name]
            val_count = class_counts['val'][cls_name]
            test_count = class_counts['test'][cls_name]
            total = train_count + val_count + test_count
            
            print(f"{cls_name:<20} {train_count:<10} {val_count:<10} {test_count:<10} {total:<10}")
        
        # Totales
        total_train = sum(class_counts['train'].values())
        total_val = sum(class_counts['val'].values())
        total_test = sum(class_counts['test'].values())
        total_all = total_train + total_val + total_test
        
        print("-" * 60)
        print(f"{'TOTAL':<20} {total_train:<10} {total_val:<10} {total_test:<10} {total_all:<10}")
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description='Preparador de Dataset YOLO desde Open Images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Reorganizar dataset de Open Images
  python preparar_dataset_sombreros.py --input ./dataset_sombreros --output ./dataset_yolo

  # Con split personalizado (80/15/5)
  python preparar_dataset_sombreros.py --input ./dataset_sombreros --output ./dataset_yolo --split 0.8 0.15 0.05

  # Solo validar dataset existente
  python preparar_dataset_sombreros.py --input ./dataset_yolo --validate-only

  # Mostrar estadísticas
  python preparar_dataset_sombreros.py --input ./dataset_yolo --stats-only
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Carpeta de entrada (Open Images o YOLO)')
    
    parser.add_argument('--output', type=str,
                       help='Carpeta de salida (formato YOLO)')
    
    parser.add_argument('--split', type=float, nargs=3, default=[0.7, 0.2, 0.1],
                       metavar=('TRAIN', 'VAL', 'TEST'),
                       help='Proporción train/val/test (default: 0.7 0.2 0.1)')
    
    parser.add_argument('--validate-only', action='store_true',
                       help='Solo validar dataset sin reorganizar')
    
    parser.add_argument('--stats-only', action='store_true',
                       help='Solo mostrar estadísticas')
    
    args = parser.parse_args()
    
    # Validar split
    if sum(args.split) != 1.0:
        print(f"❌ Error: Split debe sumar 1.0 (actual: {sum(args.split)})")
        return
    
    # Determinar output
    if not args.output:
        if args.validate_only or args.stats_only:
            args.output = args.input
        else:
            args.output = str(Path(args.input).parent / f"{Path(args.input).name}_yolo")
    
    # Crear preparador
    preparador = PreparadorDatasetYOLO(
        input_dir=args.input,
        output_dir=args.output,
        train_ratio=args.split[0],
        val_ratio=args.split[1],
        test_ratio=args.split[2]
    )
    
    # Ejecutar según modo
    if args.stats_only:
        preparador.generar_estadisticas()
    
    elif args.validate_only:
        preparador.validar_dataset()
    
    else:
        # Reorganizar
        estructura = preparador.detectar_estructura()
        
        if estructura == 'openimages':
            preparador.reorganizar_openimages()
        else:
            print("❌ Estructura no reconocida")
            print("💡 Asegúrate de que la carpeta contenga subcarpetas por clase con /images y /darknet")
            return
        
        # Validar resultado
        preparador.validar_dataset()
        
        # Mostrar estadísticas
        preparador.generar_estadisticas()

if __name__ == '__main__':
    main()
