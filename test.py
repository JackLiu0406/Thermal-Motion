import os
import matplotlib.pyplot as plt
from scipy.stats import linregress, rayleigh
import numpy as np
def load_data_as_tuples(folder_path):
    """Load all .txt files in the specified folder into a dictionary with arrays of (x, y) tuples."""
    data_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            key = os.path.splitext(filename)[0]
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()[2:]  # Skip the first two header lines
                data = [(float(line.split()[0]), float(line.split()[1])) for line in lines]
            data_dict[key] = data
    return data_dict

folder_path = '/Users/dingshengliu/Desktop/Thermal Motion Lab/Thermal Motion Lab/Data'
data_dict = load_data_as_tuples(folder_path)
