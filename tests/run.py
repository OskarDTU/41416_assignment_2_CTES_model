import sys
import os

# Tilføjer projektets rod-mappe til Python-stien
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.Part_1 import simulate