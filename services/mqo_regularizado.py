import numpy as np
from numpy.typing import NDArray

def fit_mqo_regularizado(X: NDArray[np.float64], y: NDArray[np.float64], lamb: float) -> NDArray[np.float64]:
    """
    Calcula os coeficientes do MQO regularizado com Tikhonov.
    
    Args:
        X (NDArray[np.float64]): Matriz de entrada com as variáveis independentes.
        y (NDArray[np.float64]): Matriz de saída com as variáveis dependentes.
        lamb (float): Hiperparâmetro lambda para regularização.

    Returns:
        NDArray[np.float64]: Vetor de coeficientes estimados com regularização.
    """
    identity: NDArray[np.float64] = np.eye(X.shape[1])
    return np.linalg.inv(X.T @ X + lamb * identity) @ X.T @ y