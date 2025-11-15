#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuración de TensorFlow - Eliminación de Warnings
====================================================

Script para configurar TensorFlow y eliminar mensajes informativos
que pueden confundirse con errores.

Uso:
    python utils/config_tensorflow.py

O importar en otros scripts:
    from utils.config_tensorflow import configurar_tensorflow
    configurar_tensorflow()

Autor: Sistema de Detección Vehicular
Fecha: Noviembre 2025
"""

import os
import sys
import warnings

def configurar_tensorflow():
    """Configura TensorFlow para eliminar mensajes informativos."""
    
    print("🔧 Configurando TensorFlow...")
    
    # Configurar variables de entorno ANTES de importar TensorFlow
    configuraciones = {
        # Eliminar mensajes oneDNN
        'TF_ENABLE_ONEDNN_OPTS': '0',
        
        # Controlar nivel de logging (0=INFO, 1=WARN, 2=ERROR, 3=FATAL)
        'TF_CPP_MIN_LOG_LEVEL': '2',
        
        # Deshabilitar warnings adicionales
        'TF_DISABLE_DEPRECATED_WARNING': '1',
        
        # Configurar para CPU solamente (evita mensajes de GPU)
        'CUDA_VISIBLE_DEVICES': '-1'
    }
    
    for var, valor in configuraciones.items():
        os.environ[var] = valor
        print(f"   ✅ {var} = {valor}")
    
    # Configurar warnings de Python
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    # Intentar importar y configurar TensorFlow
    try:
        import tensorflow as tf
        
        # Configurar nivel de logging de TensorFlow
        tf.get_logger().setLevel('ERROR')
        
        # Deshabilitar warnings específicos
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        
        print(f"✅ TensorFlow {tf.__version__} configurado correctamente")
        print("   📝 Mensajes informativos eliminados")
        
        return True
        
    except ImportError:
        print("⚠️  TensorFlow no está instalado")
        return False
    
    except Exception as e:
        print(f"❌ Error configurando TensorFlow: {e}")
        return False

def test_configuracion():
    """Prueba la configuración de TensorFlow."""
    print("\n🧪 PROBANDO CONFIGURACIÓN")
    print("=" * 27)
    
    try:
        import tensorflow as tf
        
        print(f"📊 TensorFlow version: {tf.__version__}")
        print(f"🖥️  Dispositivos disponibles: {len(tf.config.list_physical_devices())}")
        
        # Test básico sin mensajes
        print("🔍 Ejecutando test básico...")
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = tf.add(a, b)
        
        print(f"✅ Test exitoso: [1,2,3] + [4,5,6] = {c.numpy()}")
        
        # Información sobre oneDNN
        print(f"\n💡 Información:")
        print(f"   • oneDNN está deshabilitado (TF_ENABLE_ONEDNN_OPTS=0)")
        print(f"   • Esto elimina los mensajes informativos")
        print(f"   • El rendimiento puede ser ligeramente menor")
        print(f"   • Los resultados serán más consistentes")
        
    except Exception as e:
        print(f"❌ Error en test: {e}")

def configuracion_alternativa():
    """Configuración alternativa si se prefiere mantener oneDNN."""
    print(f"\n🔄 CONFIGURACIÓN ALTERNATIVA")
    print("=" * 30)
    print("Si prefiere mantener oneDNN para mejor rendimiento:")
    print("(Los mensajes aparecerán pero el sistema funcionará igual)")
    
    configuraciones_alt = {
        'TF_CPP_MIN_LOG_LEVEL': '1',  # Solo errores y warnings críticos
        'TF_ENABLE_ONEDNN_OPTS': '1', # Mantener oneDNN
    }
    
    for var, valor in configuraciones_alt.items():
        print(f"   export {var}={valor}  # Linux/Mac")
        print(f"   set {var}={valor}     # Windows CMD")
        print(f"   $env:{var}='{valor}'  # Windows PowerShell")

def mostrar_informacion():
    """Muestra información detallada sobre los mensajes."""
    print(f"\n📖 INFORMACIÓN SOBRE LOS MENSAJES")
    print("=" * 35)
    
    print(f"🔍 ¿Qué es oneDNN?")
    print("   • Intel Deep Neural Network Library")
    print("   • Optimiza operaciones matemáticas")
    print("   • Mejora el rendimiento en CPUs Intel")
    print("   • Es completamente normal y beneficioso")
    
    print(f"\n📊 Diferencias numéricas:")
    print("   • Muy pequeñas (orden de 1e-7 o menor)")
    print("   • Debido a diferentes órdenes de cálculo")
    print("   • No afectan la funcionalidad del sistema")
    print("   • Solo importantes en investigación científica muy precisa")
    
    print(f"\n✅ Conclusión:")
    print("   • Los mensajes NO son errores")
    print("   • El sistema funciona correctamente")
    print("   • Se pueden eliminar si molestan")

def aplicar_configuracion_permanente():
    """Crea un archivo de configuración permanente."""
    print(f"\n💾 CONFIGURACIÓN PERMANENTE")
    print("=" * 28)
    
    config_content = '''# Configuración de TensorFlow para el proyecto
# Elimina mensajes informativos que pueden confundir

import os
import warnings

# Configurar antes de importar TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_DISABLE_DEPRECATED_WARNING'] = '1'

# Configurar warnings de Python
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def setup_tensorflow():
    """Configura TensorFlow silenciosamente."""
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        return tf
    except ImportError:
        return None
'''
    
    try:
        with open('tensorflow_config.py', 'w', encoding='utf-8') as f:
            f.write(config_content)
        print("✅ Archivo 'tensorflow_config.py' creado")
        print("   Para usar: import tensorflow_config; tf = tensorflow_config.setup_tensorflow()")
        
    except Exception as e:
        print(f"❌ Error creando archivo: {e}")

if __name__ == "__main__":
    print("🔧 CONFIGURADOR DE TENSORFLOW")
    print("Universidad del Quindío - Visión Artificial")
    print("=" * 45)
    
    # Mostrar información
    mostrar_informacion()
    
    # Configurar TensorFlow
    configurar_tensorflow()
    
    # Probar configuración
    test_configuracion()
    
    # Mostrar alternativas
    configuracion_alternativa()
    
    # Crear configuración permanente
    aplicar_configuracion_permanente()
    
    print(f"\n🎯 RECOMENDACIÓN FINAL:")
    print("=" * 20)
    print("✅ Los mensajes son normales y no son errores")
    print("✅ El sistema funciona correctamente")
    print("💡 Use la configuración si prefiere no verlos")
    print("🚀 Continúe usando el sistema normalmente")