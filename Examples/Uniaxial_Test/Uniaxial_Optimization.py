import sys
import os

# Agregar el directorio 'src' al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))


from Elements import *
from Material import *
from Functions import *
from Solvers import *
