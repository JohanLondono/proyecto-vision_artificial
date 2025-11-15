#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuración Global del Sistema
===============================

Configuración única para todo el proyecto que elimina
mensajes informativos de TensorFlow y PyTorch.

Uso:
    # Al inicio de cualquier script:
    from utils import tensorflow_quiet_config
    
    # O explícitamente:
    from utils.tensorflow_quiet_config import configure_libraries

Autor: Sistema de Detección Vehicular  
Fecha: Noviembre 2025
"""

import os
import warnings

def configure_tensorflow():
    """Configura TensorFlow para ejecutarse silenciosamente."""
    # Configurar variables de entorno
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_DISABLE_DEPRECATED_WARNING'] = '1'
    
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        return True
    except ImportError:
        return False

def configure_pytorch():
    """Configura PyTorch para ejecutarse silenciosamente."""
    try:
        import torch
        # Deshabilitar warnings de PyTorch
        torch.set_warn_always(False)
        return True
    except ImportError:
        return False

def configure_warnings():
    """Configura warnings de Python."""
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

def configure_libraries():
    """Configura todas las librerías para ejecución silenciosa."""
    configure_warnings()
    tf_ok = configure_tensorflow()
    torch_ok = configure_pytorch()
    
    return {
        'tensorflow': tf_ok,
        'pytorch': torch_ok,
        'configured': True
    }

# Configuración automática al importar
configure_libraries()

# Mensaje de confirmación (solo una vez)
if not hasattr(configure_libraries, '_already_configured'):
    print("Configuracion silenciosa activada para ML libraries")
    configure_libraries._already_configured = True