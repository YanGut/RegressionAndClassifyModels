import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('aerogerador.dat')

print(f"{data.head()}")

x1 = np.array(data['0\t0'])

# print(f"{x1}")