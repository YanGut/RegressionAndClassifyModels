import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('aerogerador.dat', sep='\t')
df.columns = ['velocidade do vento', 'potência gerada']

velocidade_do_vento_array = df['velocidade do vento'].values
potencia_gerada_array = df['potência gerada'].values

plt.scatter(velocidade_do_vento_array, potencia_gerada_array, color='purple')
# plt.plot(velocidade_do_vento_array, potencia_gerada_array, color='blue')
# plt.show()

velocidade_do_vento_array.shape = (len(velocidade_do_vento_array), 1)
potencia_gerada_array.shape = (len(potencia_gerada_array), 1)

velocidade_do_vento_matriz = np.concatenate(
    (
        np.ones((len(velocidade_do_vento_array), 1)), 
        velocidade_do_vento_array
    ), 
    axis=1
)
potencia_gerada_matriz = potencia_gerada_array 

b_hat_tradicional = np.linalg.pinv(velocidade_do_vento_matriz.T @ velocidade_do_vento_matriz) @ velocidade_do_vento_matriz.T @ potencia_gerada_matriz

x_axis = np.linspace(0, 15, 15).reshape(15, 1)
x_axis_matriz = np.concatenate(
    (
        np.ones((15, 1)),
        x_axis
    ),
    axis=1
)
y_hat_tradicional = x_axis_matriz @ b_hat_tradicional

plt.plot(x_axis, y_hat_tradicional, color='blue')
plt.show()
