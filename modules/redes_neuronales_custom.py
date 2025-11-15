#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Redes Neuronales Entrenadas desde Cero
=================================================

Implementación de redes neuronales clásicas (AlexNet, VGG, ResNet) 
entrenadas desde cero para detección de sombreros.

Redes implementadas:
- AlexNet
- VGG16/VGG19
- ResNet50/ResNet101

Autor: Sistema de Detección Vehicular
Fecha: Noviembre 2025
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, ResNet101
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import json

class RedesNeuronalesCustom:
    """
    Clase para implementar y entrenar redes neuronales desde cero
    para la detección de sombreros.
    """
    
    def __init__(self, directorio_resultados="./resultados_deteccion/neural_networks"):
        """
        Inicializa el módulo de redes neuronales.
        
        Args:
            directorio_resultados: Directorio para guardar modelos y resultados
        """
        self.directorio_resultados = directorio_resultados
        self.modelos_disponibles = {}
        self.historial_entrenamiento = {}
        
        # Crear directorio si no existe
        os.makedirs(self.directorio_resultados, exist_ok=True)
        
        # Parámetros por defecto
        self.input_shape = (224, 224, 3)
        self.num_classes = 2  # Con sombrero / Sin sombrero
        self.batch_size = 32
        self.epochs = 50
        
        print("🧠 Módulo de Redes Neuronales inicializado")
    
    def crear_alexnet(self, input_shape=(224, 224, 3), num_classes=2):
        """
        Crea la arquitectura AlexNet desde cero.
        
        Args:
            input_shape: Forma de entrada de la imagen
            num_classes: Número de clases a clasificar
            
        Returns:
            Modelo AlexNet compilado
        """
        model = models.Sequential([
            # Primera capa convolucional
            layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', 
                         input_shape=input_shape, name='conv1'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            
            # Segunda capa convolucional
            layers.Conv2D(256, (5, 5), padding='same', activation='relu', name='conv2'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            
            # Tercera capa convolucional
            layers.Conv2D(384, (3, 3), padding='same', activation='relu', name='conv3'),
            layers.BatchNormalization(),
            
            # Cuarta capa convolucional
            layers.Conv2D(384, (3, 3), padding='same', activation='relu', name='conv4'),
            layers.BatchNormalization(),
            
            # Quinta capa convolucional
            layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv5'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            
            # Capas completamente conectadas
            layers.Flatten(),
            layers.Dense(4096, activation='relu', name='fc1'),
            layers.Dropout(0.5),
            layers.Dense(4096, activation='relu', name='fc2'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax', name='predictions')
        ])
        
        # Compilar modelo
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.modelos_disponibles['AlexNet'] = model
        print("✅ AlexNet creado exitosamente")
        return model
    
    def crear_vgg16(self, input_shape=(224, 224, 3), num_classes=2):
        """
        Crea la arquitectura VGG16 desde cero.
        
        Args:
            input_shape: Forma de entrada de la imagen
            num_classes: Número de clases a clasificar
            
        Returns:
            Modelo VGG16 compilado
        """
        model = models.Sequential([
            # Bloque 1
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', 
                         input_shape=input_shape, name='block1_conv1'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'),
            layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'),
            
            # Bloque 2
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'),
            layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'),
            
            # Bloque 3
            layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'),
            layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'),
            
            # Bloque 4
            layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'),
            layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'),
            
            # Bloque 5
            layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'),
            layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'),
            
            # Clasificador
            layers.Flatten(name='flatten'),
            layers.Dense(4096, activation='relu', name='fc1'),
            layers.Dropout(0.5),
            layers.Dense(4096, activation='relu', name='fc2'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax', name='predictions')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.modelos_disponibles['VGG16'] = model
        print("✅ VGG16 creado exitosamente")
        return model
    
    def crear_vgg19(self, input_shape=(224, 224, 3), num_classes=2):
        """
        Crea la arquitectura VGG19 desde cero.
        Similar a VGG16 pero con más capas convolucionales.
        """
        model = models.Sequential([
            # Bloque 1
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', 
                         input_shape=input_shape, name='block1_conv1'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'),
            layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'),
            
            # Bloque 2
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'),
            layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'),
            
            # Bloque 3
            layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4'),  # Extra en VGG19
            layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'),
            
            # Bloque 4
            layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4'),  # Extra en VGG19
            layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'),
            
            # Bloque 5
            layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4'),  # Extra en VGG19
            layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'),
            
            # Clasificador
            layers.Flatten(name='flatten'),
            layers.Dense(4096, activation='relu', name='fc1'),
            layers.Dropout(0.5),
            layers.Dense(4096, activation='relu', name='fc2'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax', name='predictions')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.modelos_disponibles['VGG19'] = model
        print("✅ VGG19 creado exitosamente")
        return model
    
    def crear_resnet50_custom(self, input_shape=(224, 224, 3), num_classes=2):
        """
        Crea una versión simplificada de ResNet50 desde cero.
        """
        def residual_block(x, filters, stride=1):
            """Bloque residual básico"""
            shortcut = x
            
            # Primera convolución
            x = layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            
            # Segunda convolución
            x = layers.Conv2D(filters, (3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            
            # Tercera convolución
            x = layers.Conv2D(filters * 4, (1, 1), padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            # Ajustar shortcut si es necesario
            if stride != 1 or shortcut.shape[-1] != filters * 4:
                shortcut = layers.Conv2D(filters * 4, (1, 1), strides=stride, padding='same')(shortcut)
                shortcut = layers.BatchNormalization()(shortcut)
            
            # Suma residual
            x = layers.Add()([x, shortcut])
            x = layers.ReLU()(x)
            
            return x
        
        # Entrada
        inputs = layers.Input(shape=input_shape)
        
        # Capa inicial
        x = layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        
        # Bloques residuales
        x = residual_block(x, 64)
        x = residual_block(x, 64)
        x = residual_block(x, 64)
        
        x = residual_block(x, 128, stride=2)
        x = residual_block(x, 128)
        x = residual_block(x, 128)
        x = residual_block(x, 128)
        
        x = residual_block(x, 256, stride=2)
        x = residual_block(x, 256)
        x = residual_block(x, 256)
        x = residual_block(x, 256)
        x = residual_block(x, 256)
        x = residual_block(x, 256)
        
        x = residual_block(x, 512, stride=2)
        x = residual_block(x, 512)
        x = residual_block(x, 512)
        
        # Clasificador
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
        
        model = models.Model(inputs, x, name='ResNet50_Custom')
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.modelos_disponibles['ResNet50_Custom'] = model
        print("✅ ResNet50 Custom creado exitosamente")
        return model
    
    def preparar_datos_entrenamiento(self, directorio_imagenes, test_size=0.2, val_size=0.1):
        """
        Prepara los datos para el entrenamiento.
        
        Args:
            directorio_imagenes: Directorio con las imágenes organizadas por clase
            test_size: Proporción del conjunto de prueba
            val_size: Proporción del conjunto de validación
            
        Returns:
            Generadores de datos para entrenamiento, validación y prueba
        """
        # Generador de datos con aumento
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
        
        validation_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Generadores
        train_generator = train_datagen.flow_from_directory(
            os.path.join(directorio_imagenes, 'train'),
            target_size=self.input_shape[:2],
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            os.path.join(directorio_imagenes, 'validation'),
            target_size=self.input_shape[:2],
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        
        test_generator = test_datagen.flow_from_directory(
            os.path.join(directorio_imagenes, 'test'),
            target_size=self.input_shape[:2],
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, validation_generator, test_generator
    
    def entrenar_modelo(self, modelo, train_generator, validation_generator, 
                       nombre_modelo="modelo", epochs=None):
        """
        Entrena un modelo con los datos proporcionados.
        
        Args:
            modelo: Modelo a entrenar
            train_generator: Generador de datos de entrenamiento
            validation_generator: Generador de datos de validación
            nombre_modelo: Nombre para guardar el modelo
            epochs: Número de épocas (usa self.epochs si es None)
            
        Returns:
            Historial del entrenamiento
        """
        if epochs is None:
            epochs = self.epochs
        
        # Callbacks
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        checkpoint_path = os.path.join(
            self.directorio_resultados, 
            f'{nombre_modelo}_best_{timestamp}.h5'
        )
        
        callbacks_list = [
            callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        print(f"🚀 Iniciando entrenamiento de {nombre_modelo}...")
        
        # Entrenar modelo
        history = modelo.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Guardar historial
        self.historial_entrenamiento[nombre_modelo] = history.history
        
        # Guardar modelo final
        modelo_final_path = os.path.join(
            self.directorio_resultados,
            f'{nombre_modelo}_final_{timestamp}.h5'
        )
        modelo.save(modelo_final_path)
        
        print(f"✅ Entrenamiento completado!")
        print(f"📁 Modelo guardado en: {modelo_final_path}")
        
        return history
    
    def evaluar_modelo(self, modelo, test_generator, nombre_modelo="modelo"):
        """
        Evalúa un modelo entrenado.
        
        Args:
            modelo: Modelo entrenado
            test_generator: Generador de datos de prueba
            nombre_modelo: Nombre del modelo para el reporte
            
        Returns:
            Métricas de evaluación
        """
        print(f"📊 Evaluando modelo {nombre_modelo}...")
        
        # Predicciones
        test_generator.reset()
        predictions = modelo.predict(test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Etiquetas verdaderas
        true_classes = test_generator.classes
        class_labels = list(test_generator.class_indices.keys())
        
        # Métricas
        test_loss, test_accuracy = modelo.evaluate(test_generator)
        
        # Reporte de clasificación
        classification_rep = classification_report(
            true_classes, predicted_classes, 
            target_names=class_labels,
            output_dict=True
        )
        
        # Matriz de confusión
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Guardar resultados
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        resultados = {
            'modelo': nombre_modelo,
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'classification_report': classification_rep,
            'confusion_matrix': cm.tolist(),
            'class_labels': class_labels,
            'fecha_evaluacion': timestamp
        }
        
        # Guardar JSON con resultados
        resultados_path = os.path.join(
            self.directorio_resultados,
            f'evaluacion_{nombre_modelo}_{timestamp}.json'
        )
        
        with open(resultados_path, 'w') as f:
            json.dump(resultados, f, indent=2)
        
        # Visualizar matriz de confusión
        self.plot_confusion_matrix(cm, class_labels, nombre_modelo)
        
        print(f"✅ Evaluación completada!")
        print(f"📈 Precisión: {test_accuracy:.4f}")
        print(f"📁 Resultados guardados en: {resultados_path}")
        
        return resultados
    
    def plot_confusion_matrix(self, cm, class_labels, nombre_modelo):
        """
        Visualiza la matriz de confusión.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_labels, yticklabels=class_labels)
        plt.title(f'Matriz de Confusión - {nombre_modelo}')
        plt.xlabel('Predicción')
        plt.ylabel('Etiqueta Verdadera')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(os.path.join(
            self.directorio_resultados,
            f'confusion_matrix_{nombre_modelo}_{timestamp}.png'
        ), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self, history, nombre_modelo):
        """
        Visualiza el historial de entrenamiento.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Precisión
        ax1.plot(history.history['accuracy'], label='Entrenamiento')
        ax1.plot(history.history['val_accuracy'], label='Validación')
        ax1.set_title(f'Precisión del Modelo - {nombre_modelo}')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Precisión')
        ax1.legend()
        ax1.grid(True)
        
        # Pérdida
        ax2.plot(history.history['loss'], label='Entrenamiento')
        ax2.plot(history.history['val_loss'], label='Validación')
        ax2.set_title(f'Pérdida del Modelo - {nombre_modelo}')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Pérdida')
        ax2.legend()
        ax2.grid(True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(os.path.join(
            self.directorio_resultados,
            f'training_history_{nombre_modelo}_{timestamp}.png'
        ), dpi=300, bbox_inches='tight')
        plt.show()
    
    def cargar_modelo(self, ruta_modelo):
        """
        Carga un modelo previamente entrenado.
        
        Args:
            ruta_modelo: Ruta al archivo del modelo
            
        Returns:
            Modelo cargado
        """
        try:
            modelo = tf.keras.models.load_model(ruta_modelo)
            print(f"✅ Modelo cargado desde: {ruta_modelo}")
            return modelo
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return None
    
    def predecir_imagen(self, modelo, imagen_path, umbral_confianza=0.5):
        """
        Realiza predicción en una imagen individual.
        
        Args:
            modelo: Modelo entrenado
            imagen_path: Ruta a la imagen
            umbral_confianza: Umbral mínimo de confianza
            
        Returns:
            Resultado de la predicción
        """
        # Cargar y preprocesar imagen
        imagen = cv2.imread(imagen_path)
        if imagen is None:
            print(f"❌ No se pudo cargar la imagen: {imagen_path}")
            return None
        
        # Redimensionar a la entrada del modelo
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        imagen_resized = cv2.resize(imagen_rgb, self.input_shape[:2])
        imagen_norm = imagen_resized / 255.0
        imagen_batch = np.expand_dims(imagen_norm, axis=0)
        
        # Predicción
        prediccion = modelo.predict(imagen_batch)
        clase_predicha = np.argmax(prediccion[0])
        confianza = float(np.max(prediccion[0]))
        
        # Interpretar resultado
        clases = ['Sin Sombrero', 'Con Sombrero']  # Ajustar según tus clases
        
        resultado = {
            'imagen': os.path.basename(imagen_path),
            'clase_predicha': clases[clase_predicha],
            'confianza': confianza,
            'es_confiable': confianza >= umbral_confianza,
            'probabilidades': {
                clases[i]: float(prediccion[0][i]) 
                for i in range(len(clases))
            }
        }
        
        return resultado
    
    def mostrar_arquitectura(self, modelo, nombre_modelo):
        """
        Muestra la arquitectura del modelo.
        """
        print(f"\n🏗️  Arquitectura de {nombre_modelo}")
        print("=" * 50)
        modelo.summary()
        
        # Guardar diagrama si es posible
        try:
            tf.keras.utils.plot_model(
                modelo, 
                to_file=os.path.join(
                    self.directorio_resultados,
                    f'arquitectura_{nombre_modelo}.png'
                ),
                show_shapes=True,
                show_layer_names=True
            )
            print(f"📊 Diagrama guardado en arquitectura_{nombre_modelo}.png")
        except:
            print("⚠️  No se pudo generar el diagrama de arquitectura")
    
    def cargar_alexnet(self, ruta_modelo=None):
        """
        Carga o crea el modelo AlexNet.
        
        Args:
            ruta_modelo: Ruta al modelo preentrenado (opcional)
            
        Returns:
            Modelo AlexNet
        """
        try:
            if ruta_modelo and os.path.exists(ruta_modelo):
                modelo = self.cargar_modelo(ruta_modelo)
                if modelo:
                    self.modelos_disponibles['alexnet'] = modelo
                    print("✅ AlexNet cargado desde archivo")
                    return modelo
            
            # Crear nuevo modelo si no hay archivo o falla la carga
            modelo = self.crear_alexnet(self.input_shape, self.num_classes)
            self.modelos_disponibles['alexnet'] = modelo
            print("✅ AlexNet creado (sin entrenar)")
            return modelo
            
        except Exception as e:
            print(f"❌ Error cargando AlexNet: {e}")
            return None
    
    def cargar_vgg16(self, ruta_modelo=None):
        """
        Carga o crea el modelo VGG16.
        
        Args:
            ruta_modelo: Ruta al modelo preentrenado (opcional)
            
        Returns:
            Modelo VGG16
        """
        try:
            if ruta_modelo and os.path.exists(ruta_modelo):
                modelo = self.cargar_modelo(ruta_modelo)
                if modelo:
                    self.modelos_disponibles['vgg16'] = modelo
                    print("✅ VGG16 cargado desde archivo")
                    return modelo
            
            # Crear nuevo modelo si no hay archivo o falla la carga
            modelo = self.crear_vgg16(self.input_shape, self.num_classes)
            self.modelos_disponibles['vgg16'] = modelo
            print("✅ VGG16 creado (sin entrenar)")
            return modelo
            
        except Exception as e:
            print(f"❌ Error cargando VGG16: {e}")
            return None
    
    def cargar_vgg19(self, ruta_modelo=None):
        """
        Carga o crea el modelo VGG19.
        
        Args:
            ruta_modelo: Ruta al modelo preentrenado (opcional)
            
        Returns:
            Modelo VGG19
        """
        try:
            if ruta_modelo and os.path.exists(ruta_modelo):
                modelo = self.cargar_modelo(ruta_modelo)
                if modelo:
                    self.modelos_disponibles['vgg19'] = modelo
                    print("✅ VGG19 cargado desde archivo")
                    return modelo
            
            # Crear nuevo modelo si no hay archivo o falla la carga
            modelo = self.crear_vgg19(self.input_shape, self.num_classes)
            self.modelos_disponibles['vgg19'] = modelo
            print("✅ VGG19 creado (sin entrenar)")
            return modelo
            
        except Exception as e:
            print(f"❌ Error cargando VGG19: {e}")
            return None
    
    def cargar_resnet50(self, ruta_modelo=None):
        """
        Carga o crea el modelo ResNet50.
        
        Args:
            ruta_modelo: Ruta al modelo preentrenado (opcional)
            
        Returns:
            Modelo ResNet50
        """
        try:
            if ruta_modelo and os.path.exists(ruta_modelo):
                modelo = self.cargar_modelo(ruta_modelo)
                if modelo:
                    self.modelos_disponibles['resnet50'] = modelo
                    print("✅ ResNet50 cargado desde archivo")
                    return modelo
            
            # Crear nuevo modelo si no hay archivo o falla la carga
            modelo = self.crear_resnet50_custom(self.input_shape, self.num_classes)
            self.modelos_disponibles['resnet50'] = modelo
            print("✅ ResNet50 creado (sin entrenar)")
            return modelo
            
        except Exception as e:
            print(f"❌ Error cargando ResNet50: {e}")
            return None
    
    def obtener_modelo_disponible(self, nombre_modelo):
        """
        Obtiene un modelo previamente cargado.
        
        Args:
            nombre_modelo: Nombre del modelo ('alexnet', 'vgg16', 'vgg19', 'resnet50')
            
        Returns:
            Modelo si está disponible, None si no
        """
        return self.modelos_disponibles.get(nombre_modelo.lower())
    
    def listar_modelos_disponibles(self):
        """
        Lista todos los modelos que están cargados.
        
        Returns:
            Lista de nombres de modelos disponibles
        """
        return list(self.modelos_disponibles.keys())

def crear_estructura_datos_ejemplo():
    """
    Función auxiliar para crear estructura de directorios de ejemplo
    para el entrenamiento de detección de sombreros.
    """
    directorios = [
        "datos_sombreros/train/con_sombrero",
        "datos_sombreros/train/sin_sombrero",
        "datos_sombreros/validation/con_sombrero", 
        "datos_sombreros/validation/sin_sombrero",
        "datos_sombreros/test/con_sombrero",
        "datos_sombreros/test/sin_sombrero"
    ]
    
    for directorio in directorios:
        os.makedirs(directorio, exist_ok=True)
    
    print("📁 Estructura de directorios creada:")
    for directorio in directorios:
        print(f"   - {directorio}")
    
    print("\n💡 Instrucciones:")
    print("1. Coloca imágenes con sombreros en las carpetas 'con_sombrero'")
    print("2. Coloca imágenes sin sombreros en las carpetas 'sin_sombrero'")
    print("3. Distribuye las imágenes: 70% train, 20% test, 10% validation")

def main():
    """Función principal para probar el módulo."""
    print("🧠 MÓDULO DE REDES NEURONALES CUSTOM")
    print("=" * 50)
    
    # Inicializar módulo
    redes = RedesNeuronalesCustom()
    
    # Crear ejemplos de arquitecturas
    print("\n🏗️  Creando arquitecturas de ejemplo...")
    
    alexnet = redes.crear_alexnet()
    redes.mostrar_arquitectura(alexnet, "AlexNet")
    
    vgg16 = redes.crear_vgg16()
    redes.mostrar_arquitectura(vgg16, "VGG16")
    
    resnet = redes.crear_resnet50_custom()
    redes.mostrar_arquitectura(resnet, "ResNet50_Custom")
    
    print("\n✅ Módulo listo para el entrenamiento!")
    print("📚 Para entrenar:")
    print("   1. Prepara tus datos con crear_estructura_datos_ejemplo()")
    print("   2. Usa preparar_datos_entrenamiento()")
    print("   3. Llama entrenar_modelo()")

if __name__ == "__main__":
    main()