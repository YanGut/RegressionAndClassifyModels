import numpy as np
from numpy.typing import NDArray
from typing import Dict, List
from sklearn.model_selection import train_test_split
from services.mqo_tradicional import fit_mqo_tradicional
from services.mqo_regularizado import fit_mqo_regularizado

def calcular_rss(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """
    Calcula a soma dos quadrados dos resíduos.
    
    Args:
        y_true (NDArray[np.float64]): Matriz de saída com os valores verdadeiros.
        y_pred (NDArray[np.float64]): Matriz de saída com os valores preditos.

    Returns:
        float: Soma dos quadrados dos resíduos.
    """
    return np.sum((y_true - y_pred) ** 2)

def validacao_monte_carlo(
    X: NDArray[np.float64], 
    y: NDArray[np.float64], 
    R: int = 500
) -> Dict[str, List[float]]:
    """
    Realiza a validação cruzada de Monte Carlo para diferentes valores de lambda.
    
    Args:
        X (NDArray[np.float64]): Matriz de entrada com as variáveis independentes.
        y (NDArray[np.float64]): Matriz de saída com as variáveis dependentes.
        R (int): Número de iterações para a validação cruzada.

    Returns:
        Dict[str, List[float]]: Dicionário com os resultados de RSS para cada modelo.
    """
    rss_results: Dict[str, List[float]] = {
        'MQO Tradicional': [],
        'Média Observáveis': [],
    }

    lambdas = [0, 0.25, 0.5, 0.75, 1]
    for lamb in lambdas:
        rss_results[f'MQO Regularizado (λ={lamb})'] = []

    for _ in range(R):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # MQO Tradicional
        b_hat_tradicional = fit_mqo_tradicional(X_train, y_train)
        y_pred_tradicional = X_test @ b_hat_tradicional
        rss_results['MQO Tradicional'].append(calcular_rss(y_test, y_pred_tradicional))

        # MQO Regularizado
        for lamb in lambdas:
            b_hat_regularizado = fit_mqo_regularizado(X_train, y_train, lamb)
            y_pred_regularizado = X_test @ b_hat_regularizado
            rss_results[f'MQO Regularizado (λ={lamb})'].append(calcular_rss(y_test, y_pred_regularizado))

        # Média dos observáveis
        media_observaveis = np.mean(y_train)
        y_pred_media = np.full_like(y_test, media_observaveis)
        rss_results['Média Observáveis'].append(calcular_rss(y_test, y_pred_media))

    return rss_results