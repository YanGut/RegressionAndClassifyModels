import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Dict

def load_data(filepath: str, sep: str = '\t') -> pd.DataFrame:
    """Carrega o dataset e define as colunas."""
    df = pd.read_csv(filepath, sep=sep)
    df.columns = ['velocidade do vento', 'potência gerada']
    return df

def prepare_data(df: pd.DataFrame) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Prepara os arrays de entrada e saída para a regressão."""
    velocidade_do_vento: NDArray[np.float64] = df['velocidade do vento'].values.reshape(-1, 1)
    potencia_gerada: NDArray[np.float64] = df['potência gerada'].values.reshape(-1, 1)
    X: NDArray[np.float64] = np.concatenate((np.ones_like(velocidade_do_vento), velocidade_do_vento), axis=1)
    y: NDArray[np.float64] = potencia_gerada
    return X, y

def fit_mqo_tradicional(X: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calcula os coeficientes do MQO tradicional."""
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def fit_mqo_regularizado(X: NDArray[np.float64], y: NDArray[np.float64], lamb: float) -> NDArray[np.float64]:
    """Calcula os coeficientes do MQO regularizado com Tikhonov."""
    identity: NDArray[np.float64] = np.eye(X.shape[1])
    return np.linalg.inv(X.T @ X + lamb * identity) @ X.T @ y

def fit_media_observaveis(y: NDArray[np.float64]) -> float:
    """Calcula a média dos valores observáveis como previsão constante."""
    return np.mean(y)

def plot_results(
    X: NDArray[np.float64], 
    y: NDArray[np.float64], 
    x_range: NDArray[np.float64], 
    b_hat_tradicional: NDArray[np.float64], 
    b_hats_tikhonov: Dict[float, NDArray[np.float64]], 
    media_observaveis: float
) -> None:
    """Plota os resultados dos modelos de regressão."""
    plt.scatter(X[:, 1], y, color='purple', label='Dados Observados')

    y_hat_tradicional: NDArray[np.float64] = np.concatenate((np.ones_like(x_range), x_range), axis=1) @ b_hat_tradicional
    plt.plot(x_range, y_hat_tradicional, color='blue', label='MQO Tradicional')

    for lamb, b_hat_reg in b_hats_tikhonov.items():
        y_hat_reg: NDArray[np.float64] = np.concatenate((np.ones_like(x_range), x_range), axis=1) @ b_hat_reg
        plt.plot(x_range, y_hat_reg, label=f'MQO Regularizado (λ={lamb})', linestyle='--')

    plt.axhline(y=media_observaveis, color='green', linestyle=':', label='Média dos Observáveis')

    plt.xlabel('Velocidade do Vento')
    plt.ylabel('Potência Gerada')
    plt.legend()
    plt.show()

def main() -> None:
    df: pd.DataFrame = load_data('aerogerador.dat')
    X, y = prepare_data(df)

    b_hat_tradicional: NDArray[np.float64] = fit_mqo_tradicional(X, y)

    lambdas: list[float] = [0, 0.25, 0.5, 0.75, 1]
    b_hats_tikhonov: Dict[float, NDArray[np.float64]] = {lamb: fit_mqo_regularizado(X, y, lamb) for lamb in lambdas}
    
    media_observaveis: float = fit_media_observaveis(y)

    x_range: NDArray[np.float64] = np.linspace(0, 15, 100).reshape(-1, 1)

    plot_results(X, y, x_range, b_hat_tradicional, b_hats_tikhonov, media_observaveis)

if __name__ == "__main__":
    main()
