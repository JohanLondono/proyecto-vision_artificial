#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Entrenamiento para Detección de Sombreros
===================================================

Módulo especializado para entrenamiento de modelos de detección de sombreros
con diferentes arquitecturas y técnicas avanzadas.

Autor: Sistema de Detección Vehicular
Fecha: Noviembre 2025
"""

import os
import numpy as np
import cv2
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Configuración silenciosa
try:
    from utils.tensorflow_quiet_config import configure_libraries
    configure_libraries()
except ImportError:
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, optimizers, callbacks
    TENSORFLOW_DISPONIBLE = True
except ImportError:
    print("⚠️  TensorFlow no disponible - funcionalidad limitada")
    TENSORFLOW_DISPONIBLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    from torchvision import models
    PYTORCH_DISPONIBLE = True
except ImportError:
    print("⚠️  PyTorch no disponible - funcionalidad limitada")
    PYTORCH_DISPONIBLE = False

class EntrenadorSombreros:
    """
    Clase principal para entrenamiento de modelos de detección de sombreros.
    """
    
    def __init__(self, directorio_datos=None, directorio_salida="modelos_entrenados"):
        """
        Inicializa el entrenador.
        
        Args:
            directorio_datos (str): Directorio con los datos de entrenamiento
            directorio_salida (str): Directorio donde guardar modelos entrenados
        """
        self.directorio_datos = directorio_datos
        self.directorio_salida = directorio_salida
        self.configuracion_entrenamiento = {
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
            'validacion_split': 0.2,
            'imagen_size': (224, 224),
            'data_augmentation': True,
            'early_stopping': True,
            'patience': 10
        }
        
        # Crear directorio de salida
        os.makedirs(self.directorio_salida, exist_ok=True)
        
        # Estadísticas de entrenamiento
        self.historial_entrenamiento = {}
        
    def configurar_datos(self, directorio_datos):
        """
        Configura el directorio de datos y verifica su estructura.
        """
        self.directorio_datos = directorio_datos
        
        if not os.path.exists(directorio_datos):
            raise ValueError(f"Directorio de datos no existe: {directorio_datos}")
        
        # Verificar estructura esperada
        subdirs_esperados = ['train', 'validation', 'test']
        categorias_esperadas = ['con_sombrero', 'sin_sombrero']
        
        for subdir in subdirs_esperados:
            path_subdir = os.path.join(directorio_datos, subdir)
            if not os.path.exists(path_subdir):
                raise ValueError(f"Falta directorio: {path_subdir}")
            
            for categoria in categorias_esperadas:
                path_categoria = os.path.join(path_subdir, categoria)
                if not os.path.exists(path_categoria):
                    raise ValueError(f"Falta categoría: {path_categoria}")
        
        print(f"✅ Estructura de datos verificada: {directorio_datos}")
        return True
    
    def analizar_dataset(self):
        """
        Analiza el dataset y muestra estadísticas.
        """
        if not self.directorio_datos:
            print("❌ Configure el directorio de datos primero")
            return None
        
        print(f"\n📊 ANÁLISIS DEL DATASET")
        print("=" * 30)
        
        estadisticas = {
            'train': {'con_sombrero': 0, 'sin_sombrero': 0},
            'validation': {'con_sombrero': 0, 'sin_sombrero': 0},
            'test': {'con_sombrero': 0, 'sin_sombrero': 0}
        }
        
        formatos_encontrados = set()
        tamaños_imagenes = []
        
        for split in ['train', 'validation', 'test']:
            for categoria in ['con_sombrero', 'sin_sombrero']:
                path_categoria = os.path.join(self.directorio_datos, split, categoria)
                
                if os.path.exists(path_categoria):
                    archivos = [f for f in os.listdir(path_categoria) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                    
                    estadisticas[split][categoria] = len(archivos)
                    
                    # Analizar algunas imágenes para obtener información
                    for i, archivo in enumerate(archivos[:10]):  # Solo primeras 10
                        try:
                            img_path = os.path.join(path_categoria, archivo)
                            img = cv2.imread(img_path)
                            if img is not None:
                                h, w = img.shape[:2]
                                tamaños_imagenes.append((w, h))
                                
                                ext = os.path.splitext(archivo)[1].lower()
                                formatos_encontrados.add(ext)
                        except:
                            continue
        
        # Mostrar estadísticas
        total_imagenes = 0
        for split in estadisticas:
            print(f"\n{split.upper()}:")
            con_sombrero = estadisticas[split]['con_sombrero']
            sin_sombrero = estadisticas[split]['sin_sombrero']
            total_split = con_sombrero + sin_sombrero
            total_imagenes += total_split
            
            print(f"  Con sombrero: {con_sombrero}")
            print(f"  Sin sombrero: {sin_sombrero}")
            print(f"  Total: {total_split}")
            
            if total_split > 0:
                balance = min(con_sombrero, sin_sombrero) / max(con_sombrero, sin_sombrero)
                print(f"  Balance: {balance:.2f}")
        
        print(f"\n📋 RESUMEN GENERAL:")
        print(f"  Total de imágenes: {total_imagenes}")
        print(f"  Formatos encontrados: {', '.join(formatos_encontrados)}")
        
        if tamaños_imagenes:
            widths = [w for w, h in tamaños_imagenes]
            heights = [h for w, h in tamaños_imagenes]
            
            print(f"  Tamaño promedio: {np.mean(widths):.0f}x{np.mean(heights):.0f}")
            print(f"  Tamaño min: {min(widths)}x{min(heights)}")
            print(f"  Tamaño max: {max(widths)}x{max(heights)}")
        
        return estadisticas
    
    def entrenar_modelo_tensorflow(self, arquitectura='cnn_simple'):
        """
        Entrena un modelo usando TensorFlow/Keras.
        """
        if not TENSORFLOW_DISPONIBLE:
            print("❌ TensorFlow no está disponible")
            return None
        
        print(f"\n🧠 ENTRENAMIENTO CON TENSORFLOW")
        print(f"Arquitectura: {arquitectura}")
        print("=" * 40)
        
        # Preparar datos
        train_generator, val_generator = self._preparar_datos_tensorflow()
        
        if train_generator is None:
            return None
        
        # Crear modelo según arquitectura
        modelo = self._crear_modelo_tensorflow(arquitectura)
        
        if modelo is None:
            return None
        
        # Configurar callbacks
        callbacks_list = self._configurar_callbacks_tensorflow(arquitectura)
        
        # Entrenar
        print("🚀 Iniciando entrenamiento...")
        inicio = time.time()
        
        try:
            historial = modelo.fit(
                train_generator,
                epochs=self.configuracion_entrenamiento['epochs'],
                validation_data=val_generator,
                callbacks=callbacks_list,
                verbose=1
            )
            
            tiempo_entrenamiento = time.time() - inicio
            print(f"⏱️  Tiempo de entrenamiento: {tiempo_entrenamiento:.2f} segundos")
            
            # Guardar modelo
            modelo_path = os.path.join(self.directorio_salida, f"modelo_tensorflow_{arquitectura}.h5")
            modelo.save(modelo_path)
            print(f"💾 Modelo guardado en: {modelo_path}")
            
            # Evaluar en conjunto de prueba
            self._evaluar_modelo_tensorflow(modelo, arquitectura)
            
            # Guardar historial
            self.historial_entrenamiento[f"tensorflow_{arquitectura}"] = {
                'historial': historial.history,
                'tiempo_entrenamiento': tiempo_entrenamiento,
                'fecha': datetime.now().isoformat()
            }
            
            return modelo
            
        except Exception as e:
            print(f"❌ Error durante el entrenamiento: {e}")
            return None
    
    def _preparar_datos_tensorflow(self):
        """Prepara los generadores de datos para TensorFlow."""
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        # Configurar data augmentation
        if self.configuracion_entrenamiento['data_augmentation']:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Crear generadores
        try:
            train_generator = train_datagen.flow_from_directory(
                os.path.join(self.directorio_datos, 'train'),
                target_size=self.configuracion_entrenamiento['imagen_size'],
                batch_size=self.configuracion_entrenamiento['batch_size'],
                class_mode='binary'
            )
            
            val_generator = val_datagen.flow_from_directory(
                os.path.join(self.directorio_datos, 'validation'),
                target_size=self.configuracion_entrenamiento['imagen_size'],
                batch_size=self.configuracion_entrenamiento['batch_size'],
                class_mode='binary'
            )
            
            print(f"✅ Generadores creados:")
            print(f"   Entrenamiento: {train_generator.samples} imágenes")
            print(f"   Validación: {val_generator.samples} imágenes")
            print(f"   Clases: {train_generator.class_indices}")
            
            return train_generator, val_generator
            
        except Exception as e:
            print(f"❌ Error preparando datos: {e}")
            return None, None
    
    def _crear_modelo_tensorflow(self, arquitectura):
        """Crea el modelo de TensorFlow según la arquitectura especificada."""
        img_height, img_width = self.configuracion_entrenamiento['imagen_size']
        
        if arquitectura == 'cnn_simple':
            modelo = keras.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(512, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
            
        elif arquitectura == 'transfer_learning':
            # Usar modelo preentrenado
            base_model = keras.applications.VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=(img_height, img_width, 3)
            )
            
            base_model.trainable = False  # Congelar capas base
            
            modelo = keras.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.2),
                layers.Dense(128, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
            
        elif arquitectura == 'resnet_custom':
            # ResNet personalizada simple
            inputs = keras.Input(shape=(img_height, img_width, 3))
            
            x = layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
            
            # Bloques residuales
            for filters in [64, 128, 256]:
                x = self._bloque_residual(x, filters)
                x = self._bloque_residual(x, filters)
            
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
            outputs = layers.Dense(1, activation='sigmoid')(x)
            
            modelo = keras.Model(inputs, outputs)
            
        else:
            print(f"❌ Arquitectura no reconocida: {arquitectura}")
            return None
        
        # Compilar modelo
        modelo.compile(
            optimizer=optimizers.Adam(learning_rate=self.configuracion_entrenamiento['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"📋 Modelo {arquitectura} creado:")
        modelo.summary()
        
        return modelo
    
    def _bloque_residual(self, x, filters):
        """Crea un bloque residual para ResNet."""
        shortcut = x
        
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Ajustar shortcut si es necesario
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), padding='same')(shortcut)
        
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        
        return x
    
    def _configurar_callbacks_tensorflow(self, arquitectura):
        """Configura callbacks para el entrenamiento."""
        callbacks_list = []
        
        # Early stopping
        if self.configuracion_entrenamiento['early_stopping']:
            early_stopping = callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=self.configuracion_entrenamiento['patience'],
                restore_best_weights=True,
                verbose=1
            )
            callbacks_list.append(early_stopping)
        
        # Model checkpoint
        checkpoint_path = os.path.join(self.directorio_salida, f"checkpoint_{arquitectura}.h5")
        checkpoint = callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        callbacks_list.append(checkpoint)
        
        # Reduce learning rate
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001,
            verbose=1
        )
        callbacks_list.append(reduce_lr)
        
        return callbacks_list
    
    def _evaluar_modelo_tensorflow(self, modelo, arquitectura):
        """Evalúa el modelo en el conjunto de prueba."""
        print(f"\n📊 EVALUACIÓN DEL MODELO")
        print("-" * 25)
        
        try:
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            
            test_datagen = ImageDataGenerator(rescale=1./255)
            test_generator = test_datagen.flow_from_directory(
                os.path.join(self.directorio_datos, 'test'),
                target_size=self.configuracion_entrenamiento['imagen_size'],
                batch_size=1,
                class_mode='binary',
                shuffle=False
            )
            
            # Predicciones
            predicciones = modelo.predict(test_generator)
            predicciones_binarias = (predicciones > 0.5).astype(int)
            
            # Etiquetas reales
            etiquetas_reales = test_generator.classes
            
            # Métricas
            accuracy = accuracy_score(etiquetas_reales, predicciones_binarias)
            print(f"✅ Accuracy en test: {accuracy:.4f}")
            
            # Matriz de confusión
            cm = confusion_matrix(etiquetas_reales, predicciones_binarias)
            print(f"📊 Matriz de confusión:")
            print(cm)
            
            # Reporte de clasificación
            nombres_clases = ['sin_sombrero', 'con_sombrero']
            reporte = classification_report(etiquetas_reales, predicciones_binarias, 
                                          target_names=nombres_clases)
            print(f"📋 Reporte de clasificación:")
            print(reporte)
            
            # Guardar visualización
            self._guardar_matriz_confusion(cm, nombres_clases, arquitectura)
            
            return accuracy
            
        except Exception as e:
            print(f"❌ Error en evaluación: {e}")
            return None
    
    def _guardar_matriz_confusion(self, cm, nombres_clases, arquitectura):
        """Guarda visualización de matriz de confusión."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=nombres_clases, yticklabels=nombres_clases)
        plt.title(f'Matriz de Confusión - {arquitectura}')
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Predicción')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.directorio_salida, f"confusion_matrix_{arquitectura}_{timestamp}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Matriz de confusión guardada: {filename}")
    
    def entrenar_modelo_pytorch(self, arquitectura='resnet18'):
        """
        Entrena un modelo usando PyTorch.
        """
        if not PYTORCH_DISPONIBLE:
            print("❌ PyTorch no está disponible")
            return None
        
        print(f"\n🔥 ENTRENAMIENTO CON PYTORCH")
        print(f"Arquitectura: {arquitectura}")
        print("=" * 40)
        
        # Configurar device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Dispositivo: {device}")
        
        # Preparar datos
        train_loader, val_loader, test_loader = self._preparar_datos_pytorch()
        
        if train_loader is None:
            return None
        
        # Crear modelo
        modelo = self._crear_modelo_pytorch(arquitectura).to(device)
        
        # Configurar optimizador y pérdida
        criterion = nn.BCELoss()
        optimizer = optim.Adam(modelo.parameters(), 
                              lr=self.configuracion_entrenamiento['learning_rate'])
        
        # Entrenar
        print("🚀 Iniciando entrenamiento...")
        inicio = time.time()
        
        historial = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        mejor_acc = 0.0
        
        try:
            for epoch in range(self.configuracion_entrenamiento['epochs']):
                # Entrenamiento
                modelo.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device).float()
                    
                    optimizer.zero_grad()
                    outputs = modelo(images).squeeze()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                
                # Validación
                modelo.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device).float()
                        outputs = modelo(images).squeeze()
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        predicted = (outputs > 0.5).float()
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                # Calcular métricas
                train_acc = train_correct / train_total
                val_acc = val_correct / val_total
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                
                # Guardar historial
                historial['train_loss'].append(avg_train_loss)
                historial['train_acc'].append(train_acc)
                historial['val_loss'].append(avg_val_loss)
                historial['val_acc'].append(val_acc)
                
                # Imprimir progreso
                if epoch % 10 == 0 or epoch == self.configuracion_entrenamiento['epochs'] - 1:
                    print(f"Época [{epoch+1}/{self.configuracion_entrenamiento['epochs']}]")
                    print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
                    print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                # Guardar mejor modelo
                if val_acc > mejor_acc:
                    mejor_acc = val_acc
                    modelo_path = os.path.join(self.directorio_salida, f"mejor_modelo_pytorch_{arquitectura}.pth")
                    torch.save(modelo.state_dict(), modelo_path)
            
            tiempo_entrenamiento = time.time() - inicio
            print(f"⏱️  Tiempo de entrenamiento: {tiempo_entrenamiento:.2f} segundos")
            print(f"🎯 Mejor accuracy de validación: {mejor_acc:.4f}")
            
            # Evaluar en test
            test_acc = self._evaluar_modelo_pytorch(modelo, test_loader, device)
            
            # Guardar historial
            self.historial_entrenamiento[f"pytorch_{arquitectura}"] = {
                'historial': historial,
                'mejor_accuracy': mejor_acc,
                'test_accuracy': test_acc,
                'tiempo_entrenamiento': tiempo_entrenamiento,
                'fecha': datetime.now().isoformat()
            }
            
            return modelo
            
        except Exception as e:
            print(f"❌ Error durante el entrenamiento: {e}")
            return None
    
    def _preparar_datos_pytorch(self):
        """Prepara los DataLoaders para PyTorch."""
        from torchvision.datasets import ImageFolder
        
        # Transformaciones
        if self.configuracion_entrenamiento['data_augmentation']:
            train_transform = transforms.Compose([
                transforms.Resize(self.configuracion_entrenamiento['imagen_size']),
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(self.configuracion_entrenamiento['imagen_size']),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        val_transform = transforms.Compose([
            transforms.Resize(self.configuracion_entrenamiento['imagen_size']),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        try:
            # Crear datasets
            train_dataset = ImageFolder(
                os.path.join(self.directorio_datos, 'train'),
                transform=train_transform
            )
            
            val_dataset = ImageFolder(
                os.path.join(self.directorio_datos, 'validation'),
                transform=val_transform
            )
            
            test_dataset = ImageFolder(
                os.path.join(self.directorio_datos, 'test'),
                transform=val_transform
            )
            
            # Crear DataLoaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.configuracion_entrenamiento['batch_size'],
                shuffle=True,
                num_workers=0  # Para Windows
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.configuracion_entrenamiento['batch_size'],
                shuffle=False,
                num_workers=0
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0
            )
            
            print(f"✅ DataLoaders creados:")
            print(f"   Entrenamiento: {len(train_dataset)} imágenes")
            print(f"   Validación: {len(val_dataset)} imágenes")
            print(f"   Test: {len(test_dataset)} imágenes")
            print(f"   Clases: {train_dataset.classes}")
            
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            print(f"❌ Error preparando datos PyTorch: {e}")
            return None, None, None
    
    def _crear_modelo_pytorch(self, arquitectura):
        """Crea modelo de PyTorch según arquitectura."""
        if arquitectura == 'resnet18':
            modelo = models.resnet18(pretrained=True)
            modelo.fc = nn.Linear(modelo.fc.in_features, 1)
            modelo.fc = nn.Sequential(
                modelo.fc,
                nn.Sigmoid()
            )
            
        elif arquitectura == 'cnn_custom':
            class CNNCustom(nn.Module):
                def __init__(self):
                    super(CNNCustom, self).__init__()
                    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                    self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                    self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
                    
                    self.pool = nn.MaxPool2d(2, 2)
                    self.dropout = nn.Dropout(0.5)
                    
                    # Calcular tamaño después de convoluciones
                    # 224 -> 112 -> 56 -> 28 -> 14
                    self.fc1 = nn.Linear(256 * 14 * 14, 512)
                    self.fc2 = nn.Linear(512, 1)
                    
                def forward(self, x):
                    x = self.pool(torch.relu(self.conv1(x)))
                    x = self.pool(torch.relu(self.conv2(x)))
                    x = self.pool(torch.relu(self.conv3(x)))
                    x = self.pool(torch.relu(self.conv4(x)))
                    
                    x = x.view(-1, 256 * 14 * 14)
                    x = self.dropout(torch.relu(self.fc1(x)))
                    x = torch.sigmoid(self.fc2(x))
                    
                    return x
            
            modelo = CNNCustom()
            
        else:
            print(f"❌ Arquitectura PyTorch no reconocida: {arquitectura}")
            return None
        
        print(f"📋 Modelo PyTorch {arquitectura} creado")
        return modelo
    
    def _evaluar_modelo_pytorch(self, modelo, test_loader, device):
        """Evalúa modelo PyTorch en conjunto de prueba."""
        modelo.eval()
        correct = 0
        total = 0
        
        predicciones = []
        etiquetas_reales = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs = modelo(images).squeeze()
                predicted = (outputs > 0.5).float()
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                predicciones.extend(predicted.cpu().numpy())
                etiquetas_reales.extend(labels.cpu().numpy())
        
        accuracy = correct / total
        print(f"✅ Test Accuracy (PyTorch): {accuracy:.4f}")
        
        return accuracy
    
    def generar_reporte_completo(self):
        """Genera un reporte completo del entrenamiento."""
        print(f"\n📊 REPORTE COMPLETO DE ENTRENAMIENTO")
        print("=" * 45)
        
        if not self.historial_entrenamiento:
            print("❌ No hay entrenamientos registrados")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reporte_path = os.path.join(self.directorio_salida, f"reporte_entrenamiento_{timestamp}.json")
        
        # Preparar datos del reporte
        reporte = {
            'configuracion': self.configuracion_entrenamiento,
            'directorio_datos': self.directorio_datos,
            'fecha_reporte': datetime.now().isoformat(),
            'entrenamientos': self.historial_entrenamiento
        }
        
        # Guardar reporte JSON
        with open(reporte_path, 'w', encoding='utf-8') as f:
            json.dump(reporte, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Reporte guardado en: {reporte_path}")
        
        # Mostrar resumen
        print(f"\n📋 RESUMEN DE ENTRENAMIENTOS:")
        for nombre, datos in self.historial_entrenamiento.items():
            print(f"\n🎯 {nombre}:")
            print(f"   Tiempo: {datos['tiempo_entrenamiento']:.2f} segundos")
            print(f"   Fecha: {datos['fecha']}")
            
            if 'mejor_accuracy' in datos:
                print(f"   Mejor accuracy: {datos['mejor_accuracy']:.4f}")
            if 'test_accuracy' in datos:
                print(f"   Test accuracy: {datos['test_accuracy']:.4f}")

def main():
    """Función de prueba."""
    print("🎩 MÓDULO DE ENTRENAMIENTO - DETECCIÓN DE SOMBREROS")
    print("Universidad del Quindío - Visión Artificial")
    print("=" * 55)
    
    # Ejemplo de uso
    entrenador = EntrenadorSombreros()
    
    # Si hay datos de ejemplo, analizarlos
    if os.path.exists("datos_sombreros"):
        entrenador.configurar_datos("datos_sombreros")
        entrenador.analizar_dataset()
    else:
        print("⚠️  No se encontraron datos de entrenamiento")
        print("💡 Ejecute el sistema principal para crear estructura de datos")

if __name__ == "__main__":
    main()