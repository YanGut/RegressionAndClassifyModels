import numpy as np
from numpy.typing import NDArray

def fit_mqo_tradicional(X: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Calcula os coeficientes do MQO tradicional (Mínimos Quadrados Ordinários).
    
    Args:
        X (NDArray[np.float64]): Matriz de entrada com as variáveis independentes.
        y (NDArray[np.float64]): Matriz de saída com as variáveis dependentes.

    Returns:
        NDArray[np.float64]: Vetor de coeficientes estimados.
    """
    return np.linalg.pinv(X.T @ X) @ X.T @ y