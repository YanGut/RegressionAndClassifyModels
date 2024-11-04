import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

from utils.load_data import load_data
from utils.validacao_monte_carlo import validacao_monte_carlo
from services.mqo_tradicional import fit_mqo_tradicional
from services.mqo_regularizado import fit_mqo_regularizado

def prepare_data(df: pd.DataFrame) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Prepara os arrays de entrada e saída para a regressão.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados de entrada e saída.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: Tupla com os arrays de entrada e saída.
    """
    velocidade_do_vento: NDArray[np.float64] = df['velocidade do vento'].values.reshape(-1, 1)
    potencia_gerada: NDArray[np.float64] = df['potência gerada'].values.reshape(-1, 1)
    X: NDArray[np.float64] = np.concatenate((np.ones_like(velocidade_do_vento), velocidade_do_vento), axis=1)
    y: NDArray[np.float64] = potencia_gerada
    return X, y

def fit_media_observaveis(y: NDArray[np.float64]) -> float:
    """
    Calcula a média dos valores observáveis como uma previsão constante.
    
    Args:
        y (NDArray[np.float64]): Matriz de saída com as variáveis dependentes.

    Returns:
        float: Média dos valores observáveis.
    """
    return np.mean(y)

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

def calcular_estatisticas(rss_results: Dict[str, List[float]]) -> pd.DataFrame:
    """
    Calcula estatísticas de RSS (Residual Sum of Squares) para diferentes modelos.
    Args:
        rss_results (Dict[str, List[float]]): Um dicionário onde as chaves são nomes de modelos e os valores são listas de RSS para esses modelos.
    Returns:
        pd.DataFrame: Um DataFrame contendo as estatísticas calculadas para cada modelo, incluindo:
            - "Modelo": Nome do modelo.
            - "Média RSS": Média dos valores de RSS.
            - "Desvio Padrão RSS": Desvio padrão dos valores de RSS.
            - "Maior RSS": Maior valor de RSS.
            - "Menor RSS": Menor valor de RSS.
    """
    stats: Dict[str, List[float]] = {
    "Modelo": [],
    "Média RSS": [],
    "Desvio Padrão RSS": [],
    "Maior RSS": [],
    "Menor RSS": []
    }

    for model, rss_values in rss_results.items():
        stats["Modelo"].append(model)
        stats["Média RSS"].append(np.mean(rss_values))
        stats["Desvio Padrão RSS"].append(np.std(rss_values))
        stats["Maior RSS"].append(np.max(rss_values))
        stats["Menor RSS"].append(np.min(rss_values))

    return pd.DataFrame(stats)

def vizualizacao_inicial(X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
    """
    Visualiza os dados de entrada e saída.
    
    Args:
        X (NDArray[np.float64]): Matriz de entrada com as variáveis independentes.
        y (NDArray[np.float64]): Matriz de saída com as variáveis dependentes.

    Returns:
        None
    """
    plt.scatter(X[:, 1], y, color='purple')
    plt.xlabel('Velocidade do Vento')
    plt.ylabel('Potência Gerada')
    plt.show()

def plot_results(
    X: NDArray[np.float64], 
    y: NDArray[np.float64], 
    x_range: NDArray[np.float64], 
    b_hat_tradicional: NDArray[np.float64], 
    b_hats_tikhonov: Dict[float, NDArray[np.float64]], 
    media_observaveis: float
) -> None:
    """
    Plota os resultados dos modelos de regressão.
    
    Args:
        X (NDArray[np.float64]): Matriz de entrada com as variáveis independentes originais.
        y (NDArray[np.float64]): Matriz de saída com as variáveis dependentes originais.
        x_range (NDArray[np.float64]): Intervalo de valores para o eixo X.
        b_hat_tradicional (NDArray[np.float64]): Vetor de coeficientes estimados pelo MQO tradicional.
        b_hats_tikhonov (Dict[float, NDArray[np.float64]]): Dicionário com lambdas e vetores de coeficientes
            estimados pelo MQO regularizado.
        media_observaveis (float): Média dos valores observáveis.

    Returns:
        None
    """
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
    """
    Função principal para carregar dados, ajustar modelos e plotar os resultados.
    
    Carrega o dataset, prepara os dados, ajusta o modelo de MQO tradicional, 
    ajusta os modelos de MQO regularizado para diferentes valores de lambda, calcula a média dos valores observáveis 
    e chama a função de plotagem para visualizar os resultados.
    
    Args:
        None

    Returns:
        None
    """
    colunas = ['velocidade do vento', 'potência gerada']
    df: pd.DataFrame = load_data(filepath='aerogerador.dat', columns=colunas, transpose=False, sep='\t')
    X, y = prepare_data(df)
    
    vizualizacao_inicial(X, y)

    b_hat_tradicional: NDArray[np.float64] = fit_mqo_tradicional(X, y)

    lambdas: list[float] = [0, 0.25, 0.5, 0.75, 1]
    b_hats_tikhonov: Dict[float, NDArray[np.float64]] = {lamb: fit_mqo_regularizado(X, y, lamb) for lamb in lambdas}
    
    media_observaveis: float = fit_media_observaveis(y)

    x_range: NDArray[np.float64] = np.linspace(0, 15, 100).reshape(-1, 1)

    plot_results(X, y, x_range, b_hat_tradicional, b_hats_tikhonov, media_observaveis)
    
    resultados_rss = validacao_monte_carlo(X, y, R=500)
    
    estatisticas = calcular_estatisticas(resultados_rss)
    print("Estatísticas dos modelos:\n", estatisticas)

if __name__ == "__main__":
    main()
