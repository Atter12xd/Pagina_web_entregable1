import sys
import io

# Configura la salida estándar y de errores a UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

import os
import sys
if sys.platform == "win32":
    os.system('chcp 65001')  # Cambia la codificación a UTF-8

