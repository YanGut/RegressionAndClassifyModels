import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Tuple

def load_data(filepath: str) -> pd.DataFrame:
    """
    Carrega o dataset EMG e organiza as colunas.

    Args:
        filepath (str): Caminho para o arquivo.

    Returns:
        pd.DataFrame: DataFrame com dados dos sensores e classes.
    """
    df: pd.DataFrame = pd.read_csv(filepath, header=None).T  # Transpor para ter amostras nas linhas
    df.columns = ['sensor_corrugador_supercilio', 'sensor_zigomatico_maior', 'classe']
    return df

def prepare_data(df: pd.DataFrame) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Prepara os arrays X e Y para classificação.

    Args:
        df (pd.DataFrame): DataFrame com dados dos sensores e classes.

    Returns:
        Tuple[NDArray[np.float64], NDArray[np.int64]]: Matrizes X com características e Y com classes one-hot.
    """
    X: NDArray[np.float64] = df[['sensor_corrugador_supercilio', 'sensor_zigomatico_maior']].values
    Y: NDArray[np.int64] = np.zeros((len(df), 5), dtype=int)
    Y[np.arange(len(df)), df['classe'].astype(int) - 1] = 1
    return X, Y

def vizualizacao_inicial(X: NDArray[np.float64], Y: NDArray[np.int64]) -> None:
    """
    Visualiza as classes do dataset.

    Args:
        X (NDArray[np.float64]): Matriz de características.
        Y (NDArray[np.int64]): Matriz de classes one-hot.
    """
    plt.figure(figsize=(8, 6))
    for i in range(5):
        plt.scatter(X[Y[:, i] == 1, 0], X[Y[:, i] == 1, 1], label=f'Classe {i+1}')
    plt.xlabel('Sensor corrugador supercílio')
    plt.ylabel('Sensor zigomático maior')
    plt.legend()
    plt.show()

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

def fit_media_observaveis(y: NDArray[np.float64]) -> float:
    """
    Calcula a média dos valores observáveis como uma previsão constante.
    
    Args:
        y (NDArray[np.float64]): Matriz de saída com as variáveis dependentes.

    Returns:
        float: Média dos valores observáveis.
    """
    return np.mean(y)


def main() -> None:
    # Carregar o dataset e organizar as variáveis X e Y
    df = load_data('EMGsDataset.csv')
    X, Y = prepare_data(df)
    
    # Exibir as dimensões para validação
    print("Dimensões de X:", X.shape)
    print("Dimensões de Y:", Y.shape)
    print("Exemplo de X:", X[:5])
    print("Exemplo de Y:", Y[:5])
    
    vizualizacao_inicial(X, Y)
    
    # Treinar o modelo de regressão linear
    coeficientes = fit_mqo_tradicional(X, Y)
    print("Coeficientes do modelo:", coeficientes)
    

if __name__ == "__main__":
    main()
