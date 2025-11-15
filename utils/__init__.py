"""
Utilidades del Sistema de Detección Vehicular
============================================

Módulo de utilidades que incluye:
- image_utils: Utilidades para manejo de imágenes
- consolidador_rapido: Consolidación rápida de descriptores
- consolidador_descriptores: Consolidación completa de descriptores  
- config_tensorflow: Configuración de TensorFlow
- tensorflow_quiet_config: Configuración silenciosa de ML libraries

Autor: Sistema de Detección Vehicular
Fecha: Noviembre 2025
"""

# Importar módulos principales
try:
    from .image_utils import *
except ImportError:
    pass

try:
    from .consolidador_rapido import ConsolidadorRapido
except ImportError:
    pass

try:
    from .consolidador_descriptores import *
except ImportError:
    pass

try:
    from .config_tensorflow import configurar_tensorflow, test_configuracion
except ImportError:
    pass

try:
    from .tensorflow_quiet_config import configure_libraries
except ImportError:
    pass

# Configuración automática silenciosa
try:
    from .tensorflow_quiet_config import configure_libraries
    configure_libraries()
except ImportError:
    pass