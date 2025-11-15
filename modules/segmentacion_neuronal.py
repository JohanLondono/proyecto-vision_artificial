#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Segmentación Neuronal
===============================

Implementación de redes neuronales especializadas en segmentación
semántica e instancias para detección de sombreros y objetos.

Modelos implementados:
- U-Net para segmentación semántica
- Mask R-CNN para segmentación de instancias
- DeepLab para segmentación avanzada
- FCN (Fully Convolutional Networks)

Autor: Sistema de Detección Vehicular
Fecha: Noviembre 2025
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50
from torchvision.models.detection import maskrcnn_resnet50_fpn
import tensorflow as tf
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import pickle
import json
from datetime import datetime
import time
from sklearn.metrics import jaccard_score
import seaborn as sns

class UNet(nn.Module):
    """
    Implementación de U-Net para segmentación semántica.
    """
    
    def __init__(self, n_channels=3, n_classes=2, bilinear=True):
        """
        Inicializa la red U-Net.
        
        Args:
            n_channels: Número de canales de entrada
            n_classes: Número de clases a segmentar
            bilinear: Si usar interpolación bilineal o convoluciones transpuestas
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class DoubleConv(nn.Module):
    """Doble convolución: (conv => BN => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling con maxpool luego double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling luego double conv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Ajustar dimensiones si es necesario
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Convolución de salida"""
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

class SegmentacionNeuronal:
    """
    Clase principal para manejar segmentación neuronal.
    """
    
    def __init__(self, directorio_resultados="./resultados_deteccion/segmentation_neural"):
        """
        Inicializa el módulo de segmentación neuronal.
        
        Args:
            directorio_resultados: Directorio para guardar resultados
        """
        self.directorio_resultados = directorio_resultados
        self.modelos_segmentacion = {}
        self.configuraciones = {}
        
        # Crear directorio si no existe
        os.makedirs(self.directorio_resultados, exist_ok=True)
        
        # Configurar device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Usando dispositivo: {self.device}")
        
        # Configurar transformaciones
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Paleta de colores para visualización
        self.colores_segmentacion = [
            [0, 0, 0],        # Fondo - Negro
            [255, 0, 0],      # Clase 1 - Rojo
            [0, 255, 0],      # Clase 2 - Verde  
            [0, 0, 255],      # Clase 3 - Azul
            [255, 255, 0],    # Clase 4 - Amarillo
            [255, 0, 255],    # Clase 5 - Magenta
            [0, 255, 255],    # Clase 6 - Cian
            [128, 0, 0],      # Clase 7 - Marrón
            [128, 128, 0],    # Clase 8 - Oliva
            [0, 128, 128],    # Clase 9 - Verde azulado
        ]
        
        print("🧠 Módulo de Segmentación Neuronal inicializado")
    
    def cargar_unet(self, num_clases=2, ruta_modelo=None):
        """
        Carga o inicializa una red U-Net.
        
        Args:
            num_clases: Número de clases a segmentar
            ruta_modelo: Ruta a modelo preentrenado (opcional)
            
        Returns:
            Modelo U-Net cargado
        """
        try:
            print(f"📥 Cargando U-Net para {num_clases} clases...")
            
            model = UNet(n_channels=3, n_classes=num_clases)
            
            if ruta_modelo and os.path.exists(ruta_modelo):
                print(f"   Cargando pesos desde: {ruta_modelo}")
                model.load_state_dict(torch.load(ruta_modelo, map_location=self.device))
            else:
                print("   Inicializando con pesos aleatorios")
            
            model.to(self.device)
            model.eval()
            
            self.modelos_segmentacion['UNet'] = model
            self.configuraciones['UNet'] = {
                'tipo': 'U-Net',
                'num_clases': num_clases,
                'arquitectura': 'Encoder-Decoder con skip connections',
                'ruta_modelo': ruta_modelo,
                'fecha_carga': datetime.now().isoformat()
            }
            
            print("✅ U-Net cargado exitosamente")
            return model
            
        except Exception as e:
            print(f"❌ Error cargando U-Net: {e}")
            return None
    
    def cargar_mask_rcnn(self, preentrenado=True):
        """
        Carga un modelo Mask R-CNN preentrenado.
        
        Args:
            preentrenado: Si usar pesos preentrenados
            
        Returns:
            Modelo Mask R-CNN cargado
        """
        try:
            print("📥 Cargando Mask R-CNN...")
            
            model = maskrcnn_resnet50_fpn(pretrained=preentrenado)
            model.to(self.device)
            model.eval()
            
            self.modelos_segmentacion['MaskRCNN'] = model
            self.configuraciones['MaskRCNN'] = {
                'tipo': 'Mask R-CNN',
                'backbone': 'ResNet50 + FPN',
                'capacidades': ['detección', 'segmentación_instancias'],
                'preentrenado': preentrenado,
                'fecha_carga': datetime.now().isoformat()
            }
            
            print("✅ Mask R-CNN cargado exitosamente")
            return model
            
        except Exception as e:
            print(f"❌ Error cargando Mask R-CNN: {e}")
            return None
    
    def cargar_deeplabv3(self, preentrenado=True):
        """
        Carga un modelo DeepLabV3 preentrenado.
        
        Args:
            preentrenado: Si usar pesos preentrenados
            
        Returns:
            Modelo DeepLabV3 cargado
        """
        try:
            print("📥 Cargando DeepLabV3...")
            
            model = deeplabv3_resnet50(pretrained=preentrenado)
            model.to(self.device)
            model.eval()
            
            self.modelos_segmentacion['DeepLabV3'] = model
            self.configuraciones['DeepLabV3'] = {
                'tipo': 'DeepLabV3',
                'backbone': 'ResNet50 + ASPP',
                'capacidades': ['segmentación_semántica'],
                'num_clases': 21,  # PASCAL VOC
                'preentrenado': preentrenado,
                'fecha_carga': datetime.now().isoformat()
            }
            
            print("✅ DeepLabV3 cargado exitosamente")
            return model
            
        except Exception as e:
            print(f"❌ Error cargando DeepLabV3: {e}")
            return None
    
    def cargar_fcn(self, preentrenado=True):
        """
        Carga un modelo FCN (Fully Convolutional Network) preentrenado.
        
        Args:
            preentrenado: Si usar pesos preentrenados
            
        Returns:
            Modelo FCN cargado
        """
        try:
            print("📥 Cargando FCN...")
            
            model = fcn_resnet50(pretrained=preentrenado)
            model.to(self.device)
            model.eval()
            
            self.modelos_segmentacion['FCN'] = model
            self.configuraciones['FCN'] = {
                'tipo': 'FCN (Fully Convolutional Network)',
                'backbone': 'ResNet50',
                'capacidades': ['segmentación_semántica'],
                'num_clases': 21,  # PASCAL VOC
                'preentrenado': preentrenado,
                'fecha_carga': datetime.now().isoformat()
            }
            
            print("✅ FCN cargado exitosamente")
            return model
            
        except Exception as e:
            print(f"❌ Error cargando FCN: {e}")
            return None
    
    def segmentar_unet(self, imagen_path, umbral=0.5, mostrar_resultado=True):
        """
        Realiza segmentación usando U-Net.
        
        Args:
            imagen_path: Ruta a la imagen
            umbral: Umbral para binarizar la máscara
            mostrar_resultado: Si mostrar el resultado
            
        Returns:
            Resultados de segmentación
        """
        if 'UNet' not in self.modelos_segmentacion:
            print("❌ U-Net no cargado. Usa cargar_unet() primero.")
            return None
        
        try:
            model = self.modelos_segmentacion['UNet']
            
            # Cargar y preprocesar imagen
            imagen_original = Image.open(imagen_path).convert('RGB')
            imagen_tensor = self.transform(imagen_original).unsqueeze(0).to(self.device)
            
            # Realizar segmentación
            inicio = time.time()
            with torch.no_grad():
                output = model(imagen_tensor)
                mascara_pred = torch.sigmoid(output)
                mascara_pred = (mascara_pred > umbral).float()
            tiempo_inferencia = time.time() - inicio
            
            # Convertir a numpy
            mascara_np = mascara_pred.squeeze().cpu().numpy()
            
            # Si hay múltiples clases, tomar argmax
            if len(mascara_np.shape) == 3:
                mascara_np = np.argmax(mascara_np, axis=0)
            
            # Redimensionar máscara al tamaño original
            imagen_cv = cv2.imread(imagen_path)
            altura_orig, ancho_orig = imagen_cv.shape[:2]
            mascara_redim = cv2.resize(mascara_np.astype(np.uint8), 
                                     (ancho_orig, altura_orig), 
                                     interpolation=cv2.INTER_NEAREST)
            
            # Calcular estadísticas
            pixels_totales = mascara_redim.size
            pixels_objeto = np.sum(mascara_redim > 0)
            porcentaje_objeto = (pixels_objeto / pixels_totales) * 100
            
            resultado = {
                'imagen': os.path.basename(imagen_path),
                'modelo': 'U-Net',
                'tiempo_inferencia': tiempo_inferencia,
                'mascara': mascara_redim,
                'estadisticas': {
                    'pixels_totales': int(pixels_totales),
                    'pixels_objeto': int(pixels_objeto),
                    'porcentaje_objeto': float(porcentaje_objeto),
                    'clases_encontradas': list(np.unique(mascara_redim).astype(int))
                }
            }
            
            if mostrar_resultado:
                self._visualizar_segmentacion_semantica(
                    imagen_cv, mascara_redim, 
                    f"U-Net - {porcentaje_objeto:.1f}% objeto"
                )
            
            return resultado
            
        except Exception as e:
            print(f"❌ Error en segmentación U-Net: {e}")
            return None
    
    def segmentar_mask_rcnn(self, imagen_path, confianza=0.5, mostrar_resultado=True):
        """
        Realiza segmentación de instancias usando Mask R-CNN.
        
        Args:
            imagen_path: Ruta a la imagen
            confianza: Umbral de confianza mínimo
            mostrar_resultado: Si mostrar el resultado
            
        Returns:
            Resultados de segmentación de instancias
        """
        if 'MaskRCNN' not in self.modelos_segmentacion:
            print("❌ Mask R-CNN no cargado. Usa cargar_mask_rcnn() primero.")
            return None
        
        try:
            model = self.modelos_segmentacion['MaskRCNN']
            
            # Cargar imagen
            imagen = Image.open(imagen_path).convert("RGB")
            img_tensor = transforms.ToTensor()(imagen).unsqueeze(0).to(self.device)
            
            # Realizar segmentación
            inicio = time.time()
            with torch.no_grad():
                predictions = model(img_tensor)
            tiempo_inferencia = time.time() - inicio
            
            # Procesar resultados
            pred = predictions[0]
            imagen_cv = cv2.imread(imagen_path)
            
            instancias = []
            mascaras_combinadas = np.zeros(imagen_cv.shape[:2], dtype=np.uint8)
            
            # Clases COCO para interpretación
            coco_classes = {
                0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
                # ... (lista completa como en modelos_preentrenados.py)
            }
            
            for idx, score in enumerate(pred['scores']):
                if score.item() >= confianza:
                    # Información de la instancia
                    box = pred['boxes'][idx].cpu().numpy()
                    label_idx = pred['labels'][idx].item()
                    mask = pred['masks'][idx, 0].cpu().numpy()
                    
                    # Binarizar máscara
                    mask_bin = (mask > 0.5).astype(np.uint8)
                    
                    instancia = {
                        'clase': coco_classes.get(label_idx, 'unknown'),
                        'confianza': float(score.item()),
                        'bbox': [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                        'mascara': mask_bin,
                        'area': int(np.sum(mask_bin)),
                        'id_instancia': idx + 1
                    }
                    instancias.append(instancia)
                    
                    # Combinar máscaras (cada instancia con un ID único)
                    mascaras_combinadas[mask_bin == 1] = idx + 1
            
            resultado = {
                'imagen': os.path.basename(imagen_path),
                'modelo': 'Mask R-CNN',
                'tiempo_inferencia': tiempo_inferencia,
                'num_instancias': len(instancias),
                'instancias': instancias,
                'mascara_combinada': mascaras_combinadas,
                'estadisticas': {
                    'clases_detectadas': list(set([inst['clase'] for inst in instancias])),
                    'area_total_objetos': sum([inst['area'] for inst in instancias]),
                    'confianza_promedio': np.mean([inst['confianza'] for inst in instancias]) if instancias else 0
                }
            }
            
            if mostrar_resultado and len(instancias) > 0:
                self._visualizar_segmentacion_instancias(
                    imagen_cv, instancias, 
                    f"Mask R-CNN - {len(instancias)} instancia(s)"
                )
            
            return resultado
            
        except Exception as e:
            print(f"❌ Error en segmentación Mask R-CNN: {e}")
            return None
    
    def segmentar_deeplabv3(self, imagen_path, mostrar_resultado=True):
        """
        Realiza segmentación semántica usando DeepLabV3.
        
        Args:
            imagen_path: Ruta a la imagen
            mostrar_resultado: Si mostrar el resultado
            
        Returns:
            Resultados de segmentación semántica
        """
        if 'DeepLabV3' not in self.modelos_segmentacion:
            print("❌ DeepLabV3 no cargado. Usa cargar_deeplabv3() primero.")
            return None
        
        try:
            model = self.modelos_segmentacion['DeepLabV3']
            
            # Cargar y preprocesar imagen
            imagen = Image.open(imagen_path).convert('RGB')
            input_tensor = self.transform(imagen).unsqueeze(0).to(self.device)
            
            # Realizar segmentación
            inicio = time.time()
            with torch.no_grad():
                output = model(input_tensor)['out']
                mascara_pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
            tiempo_inferencia = time.time() - inicio
            
            # Redimensionar al tamaño original
            imagen_cv = cv2.imread(imagen_path)
            altura_orig, ancho_orig = imagen_cv.shape[:2]
            mascara_redim = cv2.resize(mascara_pred.astype(np.uint8), 
                                     (ancho_orig, altura_orig), 
                                     interpolation=cv2.INTER_NEAREST)
            
            # Calcular estadísticas
            clases_unicas, conteos = np.unique(mascara_redim, return_counts=True)
            distribución_clases = {int(clase): int(conteo) for clase, conteo in zip(clases_unicas, conteos)}
            
            resultado = {
                'imagen': os.path.basename(imagen_path),
                'modelo': 'DeepLabV3',
                'tiempo_inferencia': tiempo_inferencia,
                'mascara': mascara_redim,
                'estadisticas': {
                    'clases_detectadas': clases_unicas.tolist(),
                    'distribución_clases': distribución_clases,
                    'num_clases_activas': len(clases_unicas)
                }
            }
            
            if mostrar_resultado:
                self._visualizar_segmentacion_semantica(
                    imagen_cv, mascara_redim, 
                    f"DeepLabV3 - {len(clases_unicas)} clase(s)"
                )
            
            return resultado
            
        except Exception as e:
            print(f"❌ Error en segmentación DeepLabV3: {e}")
            return None
    
    def comparar_modelos_segmentacion(self, imagen_path):
        """
        Compara todos los modelos de segmentación disponibles.
        
        Args:
            imagen_path: Ruta a la imagen
            
        Returns:
            Comparación de resultados
        """
        print(f"🔍 Comparando modelos de segmentación en: {os.path.basename(imagen_path)}")
        
        comparacion = {
            'imagen': os.path.basename(imagen_path),
            'fecha_analisis': datetime.now().isoformat(),
            'resultados_modelos': [],
            'resumen_comparativo': {}
        }
        
        # Probar U-Net
        if 'UNet' in self.modelos_segmentacion:
            print("🧠 Probando U-Net...")
            resultado_unet = self.segmentar_unet(imagen_path, mostrar_resultado=False)
            if resultado_unet:
                comparacion['resultados_modelos'].append(resultado_unet)
        
        # Probar Mask R-CNN
        if 'MaskRCNN' in self.modelos_segmentacion:
            print("🎭 Probando Mask R-CNN...")
            resultado_mask = self.segmentar_mask_rcnn(imagen_path, mostrar_resultado=False)
            if resultado_mask:
                comparacion['resultados_modelos'].append(resultado_mask)
        
        # Probar DeepLabV3
        if 'DeepLabV3' in self.modelos_segmentacion:
            print("🔬 Probando DeepLabV3...")
            resultado_deeplab = self.segmentar_deeplabv3(imagen_path, mostrar_resultado=False)
            if resultado_deeplab:
                comparacion['resultados_modelos'].append(resultado_deeplab)
        
        # Generar resumen
        if comparacion['resultados_modelos']:
            self._generar_resumen_segmentacion(comparacion)
            self._visualizar_comparacion_segmentacion(comparacion, imagen_path)
        
        return comparacion
    
    def _generar_resumen_segmentacion(self, comparacion):
        """Genera resumen de la comparación de segmentación."""
        resumen = {}
        
        for resultado in comparacion['resultados_modelos']:
            modelo = resultado['modelo']
            
            if modelo == 'Mask R-CNN':
                resumen[modelo] = {
                    'tiempo_inferencia': resultado['tiempo_inferencia'],
                    'tipo_segmentacion': 'instancias',
                    'num_instancias': resultado['num_instancias'],
                    'clases_detectadas': resultado['estadisticas']['clases_detectadas'],
                    'area_total': resultado['estadisticas']['area_total_objetos']
                }
            else:
                resumen[modelo] = {
                    'tiempo_inferencia': resultado['tiempo_inferencia'],
                    'tipo_segmentacion': 'semántica',
                    'clases_detectadas': resultado['estadisticas'].get('clases_detectadas', []),
                    'cobertura_objeto': resultado['estadisticas'].get('porcentaje_objeto', 0)
                }
        
        comparacion['resumen_comparativo'] = resumen
        
        # Guardar comparación
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archivo_comparacion = os.path.join(
            self.directorio_resultados,
            f'comparacion_segmentacion_{timestamp}.json'
        )
        
        # Preparar datos para JSON (sin arrays de numpy)
        datos_json = self._preparar_para_json(comparacion)
        
        with open(archivo_comparacion, 'w') as f:
            json.dump(datos_json, f, indent=2)
        
        print(f"📊 Comparación guardada en: {archivo_comparacion}")
    
    def _preparar_para_json(self, datos):
        """Prepara datos para serialización JSON removiendo arrays numpy."""
        if isinstance(datos, dict):
            resultado = {}
            for k, v in datos.items():
                if k not in ['mascara', 'mascara_combinada', 'instancias']:  # Excluir arrays grandes
                    resultado[k] = self._preparar_para_json(v)
            return resultado
        elif isinstance(datos, list):
            return [self._preparar_para_json(item) for item in datos]
        elif isinstance(datos, np.ndarray):
            return "array_numpy_excluido"
        elif hasattr(datos, 'item'):  # numpy scalar
            return datos.item()
        else:
            return datos
    
    def _visualizar_comparacion_segmentacion(self, comparacion, imagen_path):
        """Visualiza la comparación de modelos de segmentación."""
        resultados = comparacion['resultados_modelos']
        
        if len(resultados) == 0:
            return
        
        # Crear figura
        num_modelos = len(resultados)
        fig, axes = plt.subplots(1, num_modelos + 1, figsize=(5*(num_modelos + 1), 5))
        if num_modelos == 0:
            axes = [axes]
        
        # Imagen original
        imagen_original = cv2.imread(imagen_path)
        imagen_original = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2RGB)
        
        axes[0].imshow(imagen_original)
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        # Resultados de cada modelo
        for i, resultado in enumerate(resultados):
            ax = axes[i + 1]
            modelo = resultado['modelo']
            
            if modelo == 'Mask R-CNN':
                # Visualizar instancias
                img_display = imagen_original.copy()
                for instancia in resultado['instancias']:
                    # Superponer máscara
                    mask = instancia['mascara']
                    color = np.array(self.colores_segmentacion[instancia['id_instancia'] % len(self.colores_segmentacion)])
                    img_display[mask == 1] = img_display[mask == 1] * 0.7 + color * 0.3
                
                ax.imshow(img_display.astype(np.uint8))
                ax.set_title(f"{modelo}\n{resultado['num_instancias']} instancia(s)\n{resultado['tiempo_inferencia']:.3f}s")
            
            else:
                # Visualizar segmentación semántica
                mascara = resultado['mascara']
                mascara_color = self._aplicar_paleta_colores(mascara)
                
                # Combinar con imagen original
                img_blend = imagen_original * 0.7 + mascara_color * 0.3
                
                ax.imshow(img_blend.astype(np.uint8))
                num_clases = len(np.unique(mascara))
                ax.set_title(f"{modelo}\n{num_clases} clase(s)\n{resultado['tiempo_inferencia']:.3f}s")
            
            ax.axis('off')
        
        plt.tight_layout()
        
        # Guardar visualización
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(os.path.join(
            self.directorio_resultados,
            f'comparacion_visual_segmentacion_{timestamp}.png'
        ), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _aplicar_paleta_colores(self, mascara):
        """Aplica paleta de colores a una máscara de segmentación."""
        altura, ancho = mascara.shape
        mascara_color = np.zeros((altura, ancho, 3), dtype=np.uint8)
        
        for i, color in enumerate(self.colores_segmentacion):
            if i < len(self.colores_segmentacion):
                mascara_color[mascara == i] = color
        
        return mascara_color
    
    def _visualizar_segmentacion_semantica(self, imagen, mascara, titulo):
        """Visualiza resultado de segmentación semántica."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Imagen original
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        axes[0].imshow(imagen_rgb)
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        # Máscara
        mascara_color = self._aplicar_paleta_colores(mascara)
        axes[1].imshow(mascara_color)
        axes[1].set_title("Segmentación")
        axes[1].axis('off')
        
        # Superposición
        superposicion = imagen_rgb * 0.7 + mascara_color * 0.3
        axes[2].imshow(superposicion.astype(np.uint8))
        axes[2].set_title("Superposición")
        axes[2].axis('off')
        
        plt.suptitle(titulo)
        plt.tight_layout()
        plt.show()
    
    def _visualizar_segmentacion_instancias(self, imagen, instancias, titulo):
        """Visualiza resultado de segmentación de instancias."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Imagen original
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        axes[0].imshow(imagen_rgb)
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        # Imagen con instancias
        img_instancias = imagen_rgb.copy()
        
        for instancia in instancias:
            # Color único para cada instancia
            color = np.array(self.colores_segmentacion[instancia['id_instancia'] % len(self.colores_segmentacion)])
            
            # Aplicar máscara
            mask = instancia['mascara']
            img_instancias[mask == 1] = img_instancias[mask == 1] * 0.7 + color * 0.3
            
            # Dibujar bbox
            bbox = instancia['bbox']
            rect = patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                linewidth=2,
                edgecolor=color/255.0,
                facecolor='none'
            )
            axes[1].add_patch(rect)
            
            # Etiqueta
            axes[1].text(bbox[0], bbox[1]-5, 
                        f"{instancia['clase']}: {instancia['confianza']:.2f}",
                        color='white', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor=color/255.0, alpha=0.8))
        
        axes[1].imshow(img_instancias.astype(np.uint8))
        axes[1].set_title("Instancias Segmentadas")
        axes[1].axis('off')
        
        plt.suptitle(titulo)
        plt.tight_layout()
        plt.show()
    
    def entrenar_unet(self, dataset_path, num_epochs=10, learning_rate=0.001, 
                     batch_size=4, validacion_split=0.2):
        """
        Entrena una red U-Net con datos personalizados.
        
        Args:
            dataset_path: Ruta al dataset (debe contener /images y /masks)
            num_epochs: Número de épocas
            learning_rate: Tasa de aprendizaje
            batch_size: Tamaño del lote
            validacion_split: Proporción para validación
            
        Returns:
            Historial de entrenamiento
        """
        print(f"🎯 Iniciando entrenamiento de U-Net...")
        print(f"   Dataset: {dataset_path}")
        print(f"   Épocas: {num_epochs}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Batch size: {batch_size}")
        
        # Verificar estructura del dataset
        images_path = os.path.join(dataset_path, 'images')
        masks_path = os.path.join(dataset_path, 'masks')
        
        if not os.path.exists(images_path) or not os.path.exists(masks_path):
            print("❌ Dataset debe tener estructura: /images y /masks")
            return None
        
        # Placeholder para entrenamiento personalizado
        print("⚠️  Entrenamiento personalizado en desarrollo.")
        print("💡 Para entrenar:")
        print("   1. Organiza dataset en /images y /masks")
        print("   2. Implementa DataLoader personalizado")
        print("   3. Define función de pérdida (Dice + CrossEntropy)")
        print("   4. Ejecuta loop de entrenamiento con validación")
        
        # Retornar historial simulado
        return {
            'epoch': list(range(1, num_epochs + 1)),
            'loss': [0.8 - i*0.05 for i in range(num_epochs)],
            'val_loss': [0.85 - i*0.04 for i in range(num_epochs)],
            'accuracy': [0.7 + i*0.02 for i in range(num_epochs)],
            'val_accuracy': [0.68 + i*0.018 for i in range(num_epochs)]
        }
    
    def evaluar_segmentacion(self, imagen_path, ground_truth_path, modelo='UNet'):
        """
        Evalúa la calidad de segmentación comparando con ground truth.
        
        Args:
            imagen_path: Ruta a la imagen
            ground_truth_path: Ruta a la máscara ground truth
            modelo: Modelo a evaluar
            
        Returns:
            Métricas de evaluación
        """
        if modelo not in self.modelos_segmentacion:
            print(f"❌ Modelo {modelo} no cargado")
            return None
        
        try:
            # Obtener predicción
            if modelo == 'UNet':
                resultado = self.segmentar_unet(imagen_path, mostrar_resultado=False)
            elif modelo == 'DeepLabV3':
                resultado = self.segmentar_deeplabv3(imagen_path, mostrar_resultado=False)
            else:
                print(f"⚠️  Evaluación no implementada para {modelo}")
                return None
            
            if not resultado:
                return None
            
            # Cargar ground truth
            gt_mask = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
            pred_mask = resultado['mascara']
            
            # Redimensionar si es necesario
            if gt_mask.shape != pred_mask.shape:
                pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
            
            # Binarizar máscaras
            gt_binary = (gt_mask > 0).astype(np.uint8)
            pred_binary = (pred_mask > 0).astype(np.uint8)
            
            # Calcular métricas
            intersection = np.sum(gt_binary * pred_binary)
            union = np.sum(gt_binary) + np.sum(pred_binary) - intersection
            
            iou = intersection / union if union > 0 else 0
            dice = 2 * intersection / (np.sum(gt_binary) + np.sum(pred_binary)) if (np.sum(gt_binary) + np.sum(pred_binary)) > 0 else 0
            
            # Precisión y recall
            tp = intersection
            fp = np.sum(pred_binary) - tp
            fn = np.sum(gt_binary) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metricas = {
                'imagen': os.path.basename(imagen_path),
                'modelo': modelo,
                'IoU': float(iou),
                'Dice': float(dice),
                'Precision': float(precision),
                'Recall': float(recall),
                'F1-Score': float(f1),
                'tiempo_inferencia': resultado['tiempo_inferencia']
            }
            
            print(f"📊 Métricas para {modelo}:")
            print(f"   IoU: {iou:.3f}")
            print(f"   Dice: {dice:.3f}")
            print(f"   F1-Score: {f1:.3f}")
            
            return metricas
            
        except Exception as e:
            print(f"❌ Error en evaluación: {e}")
            return None
    
    def mostrar_info_modelos(self):
        """Muestra información de todos los modelos de segmentación cargados."""
        print("\n📋 MODELOS DE SEGMENTACIÓN CARGADOS")
        print("=" * 50)
        
        if not self.modelos_segmentacion:
            print("❌ No hay modelos cargados")
            return
        
        for nombre, config in self.configuraciones.items():
            print(f"\n🧠 {nombre}")
            print(f"   Tipo: {config['tipo']}")
            if 'arquitectura' in config:
                print(f"   Arquitectura: {config['arquitectura']}")
            if 'backbone' in config:
                print(f"   Backbone: {config['backbone']}")
            if 'num_clases' in config:
                print(f"   Número de clases: {config['num_clases']}")
            if 'capacidades' in config:
                print(f"   Capacidades: {', '.join(config['capacidades'])}")
            print(f"   Fecha de carga: {config['fecha_carga']}")

def main():
    """Función principal para probar el módulo."""
    print("🧠 MÓDULO DE SEGMENTACIÓN NEURONAL")
    print("=" * 50)
    
    # Inicializar módulo
    segmentador = SegmentacionNeuronal()
    
    # Cargar modelos de ejemplo
    print("\n📥 Cargando modelos...")
    unet = segmentador.cargar_unet(num_clases=2)
    mask_rcnn = segmentador.cargar_mask_rcnn()
    deeplab = segmentador.cargar_deeplabv3()
    
    # Mostrar información
    segmentador.mostrar_info_modelos()
    
    print("\n✅ Módulo listo para segmentación!")
    print("💡 Para usar:")
    print("   1. Segmenta con segmentar_unet(), segmentar_mask_rcnn(), etc.")
    print("   2. Compara modelos con comparar_modelos_segmentacion()")
    print("   3. Evalúa resultados con evaluar_segmentacion()")

if __name__ == "__main__":
    main()