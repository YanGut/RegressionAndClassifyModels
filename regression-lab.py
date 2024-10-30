import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Dict, List

def load_data(filepath: str, sep: str = '\t') -> pd.DataFrame:
    """
    Carrega o dataset e define as colunas.
    
    Args:
        filepath (str): Caminho para o arquivo.
        sep (str): Separador dos dados.

    Returns:
        pd.DataFrame: DataFrame com os dados de entrada e saída.
    """
    df: pd.DataFrame = pd.read_csv(filepath, sep=sep)
    df.columns = ['velocidade do vento', 'potência gerada']
    return df

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

        # Modelo MQO Tradicional
        b_hat_tradicional = fit_mqo_tradicional(X_train, y_train)
        y_pred_tradicional = X_test @ b_hat_tradicional
        rss_results['MQO Tradicional'].append(calculate_rss(y_test, y_pred_tradicional))

        # Modelo MQO Regularizado para cada lambda
        for lamb in lambdas:
            b_hat_regularizado = fit_mqo_regularizado(X_train, y_train, lamb)
            y_pred_regularizado = X_test @ b_hat_regularizado
            rss_results[f'MQO Regularizado (λ={lamb})'].append(calculate_rss(y_test, y_pred_regularizado))

        # Média dos observáveis
        media_observaveis = fit_media_observaveis(y_train)
        y_pred_media = np.full_like(y_test, media_observaveis)
        rss_results['Média Observáveis'].append(calculate_rss(y_test, y_pred_media))

    return rss_results

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
    df: pd.DataFrame = load_data('aerogerador.dat')
    X, y = prepare_data(df)
    
    vizualizacao_inicial(X, y)

    b_hat_tradicional: NDArray[np.float64] = fit_mqo_tradicional(X, y)

    lambdas: list[float] = [0, 0.25, 0.5, 0.75, 1]
    b_hats_tikhonov: Dict[float, NDArray[np.float64]] = {lamb: fit_mqo_regularizado(X, y, lamb) for lamb in lambdas}
    
    media_observaveis: float = fit_media_observaveis(y)

    x_range: NDArray[np.float64] = np.linspace(0, 15, 100).reshape(-1, 1)

    plot_results(X, y, x_range, b_hat_tradicional, b_hats_tikhonov, media_observaveis)

if __name__ == "__main__":
    main()
