import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split

from utils.load_data import load_data
from services.mqo_tradicional import fit_mqo_tradicional

def prepare_data_para_gaussiano_bayesiano(df: pd.DataFrame) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    X: NDArray[np.float64] = df[['sensor_corrugador_supercilio', 'sensor_zigomatico_maior']].values
    Y: NDArray[np.int64] = df['classe'].values - 1  # Ajuste para que as classes sejam 0, 1, 2, 3, 4
    return X, Y

def vizualizacao_inicial(X: NDArray[np.float64], Y: NDArray[np.int64]) -> None:
    plt.figure(figsize=(8, 6))
    for i in range(5):
        plt.scatter(X[Y == i, 0], X[Y == i, 1], label=f'Classe {i+1}')
    plt.xlabel('Sensor corrugador supercílio')
    plt.ylabel('Sensor zigomático maior')
    plt.legend()
    plt.title('Distribuição das Classes')
    plt.grid()
    plt.show()

def fit_gaussiano_regularizado(X: NDArray[np.float64], y: NDArray[np.int64], lambda_: float) -> Tuple[dict, dict]:
    classes = np.unique(y)
    medias = {}
    covariancias = {}

    for classe in classes:
        X_classe = X[y == classe]
        media_classe = np.mean(X_classe, axis=0)
        cov_classe = np.cov(X_classe, rowvar=False)
        cov_regularizada = (1 - lambda_) * cov_classe + lambda_ * np.eye(cov_classe.shape[0])

        medias[classe] = media_classe
        covariancias[classe] = cov_regularizada

    return medias, covariancias

def fit_gaussiano_covariancias_iguais(X: NDArray[np.float64], y: NDArray[np.int64]) -> Tuple[Dict[int, NDArray[np.float64]], NDArray[np.float64]]:
    """
    Ajusta o modelo Gaussiano com covariâncias iguais para todas as classes.
    """
    classes = np.unique(y)
    medias = {}
    covariancia_agregada = np.zeros((X.shape[1], X.shape[1]))

    for classe in classes:
        X_classe = X[y == classe]
        medias[classe] = np.mean(X_classe, axis=0)
        covariancia_agregada += np.cov(X_classe, rowvar=False)
    
    # Divide pela quantidade de classes para obter a covariância média
    covariancia_agregada /= len(classes)
    
    return medias, covariancia_agregada

def fit_gaussiano_cov_agregada(X: NDArray[np.float64], y: NDArray[np.int64]) -> Tuple[Dict[int, NDArray[np.float64]], NDArray[np.float64]]:
    """
    Ajusta o modelo Gaussiano com uma única covariância agregada para todas as classes.
    """
    classes = np.unique(y)
    medias = {}
    covariancia_agregada = np.cov(X, rowvar=False)  # Covariância agregada de todos os dados

    for classe in classes:
        X_classe = X[y == classe]
        medias[classe] = np.mean(X_classe, axis=0)

    return medias, covariancia_agregada

def fit_bayes_ingenuo(X: NDArray[np.float64], y: NDArray[np.int64]) -> Tuple[dict, dict]:
    classes = np.unique(y)
    medias = {}
    variancias = {}
    
    for classe in classes:
        X_classe = X[y == classe]
        medias[classe] = np.mean(X_classe, axis=0)
        variancias[classe] = np.var(X_classe, axis=0)
    
    return medias, variancias

def predizer_gaussiano(X_test: NDArray[np.float64], medias: Dict[int, NDArray[np.float64]], covariancias: Dict[int, NDArray[np.float64]]) -> NDArray[np.int64]:
    preds = []
    for x in X_test:
        scores = {}
        for classe, media in medias.items():
            diff = x - media
            cov_inv = np.linalg.pinv(covariancias[classe])
            scores[classe] = -0.5 * (diff.T @ cov_inv @ diff)
        preds.append(max(scores, key=scores.get))
    return np.array(preds)

def predizer_bayes_ingenuo(X_test: NDArray[np.float64], medias: Dict[int, NDArray[np.float64]], variancias: Dict[int, NDArray[np.float64]]) -> NDArray[np.int64]:
    preds = []
    for x in X_test:
        scores = {}
        
        for classe in medias.keys():
            # Calcule a probabilidade usando a distribuição normal
            probabilidade = np.prod(1 / np.sqrt(2 * np.pi * variancias[classe]) * np.exp(-(x - medias[classe]) ** 2 / (2 * variancias[classe])))
            scores[classe] = probabilidade
            
        pred_classe = max(scores, key=scores.get)
        preds.append(pred_classe)
    
    return np.array(preds)

def predizer_gaussiano_cov_agregada(X_test: NDArray[np.float64], medias: Dict[int, NDArray[np.float64]], cov_agregada: NDArray[np.float64]) -> NDArray[np.int64]:
    """
    Faz a predição para o modelo Gaussiano com covariância agregada.
    """
    preds = []
    cov_inv = np.linalg.pinv(cov_agregada)  # Inversa da covariância agregada

    for x in X_test:
        scores = {}
        for classe, media in medias.items():
            diff = x - media
            scores[classe] = -0.5 * (diff.T @ cov_inv @ diff)
        preds.append(max(scores, key=scores.get))
    
    return np.array(preds)

def testar_modelos(X: NDArray[np.float64], Y: NDArray[np.int64], num_simulacoes: int = 10):
    resultados = {
        "Modelo": [],
        "Média": [],
        "Desvio Padrão": [],
        "Maior Valor": [],
        "Menor Valor": []
    }

    for modelo in [
        "MQO", 
        "Gaussiano Tradicional (λ=0.0)", 
        "Classificador Gaussiano Cov. iguais (λ=1.0)", 
        "Classificador Gaussiano Cov. Agregada", 
        "Classificador de Bayes Ingênuo", 
        "Gaussiano Regularizado (λ=0.25)", 
        "Gaussiano Regularizado (λ=0.5)", 
        "Gaussiano Regularizado (λ=0.75)"
    ]:
        acuracias = []

        for _ in range(num_simulacoes):
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
            
            if modelo == "MQO":
                coeficientes = fit_mqo_tradicional(X_train, y_train)
                y_pred = np.dot(X_test, coeficientes)
                
                if y_pred.ndim == 1:  # Caso y_pred seja 1D, transforme para 2D
                    y_pred = y_pred.reshape(-1, 1) 
                    
                y_pred_classes = np.argmax(y_pred, axis=1)
            elif modelo == "Gaussiano Tradicional (λ=0.0)":
                lambda_ = 0.0
                medias, covariancias = fit_gaussiano_regularizado(X_train, y_train, lambda_)
                y_pred_classes = predizer_gaussiano(X_test, medias, covariancias)
            elif modelo == "Classificador Gaussiano Cov iguais (λ=1.0)":
                medias, covariancia_agregada = fit_gaussiano_covariancias_iguais(X_train, y_train)
                y_pred_classes = predizer_gaussiano(X_test, medias, {classe: covariancia_agregada for classe in medias.keys()})
            elif modelo == "Classificador Gaussiano Cov Agregada":
                medias, cov_agregada = fit_gaussiano_cov_agregada(X_train, y_train)
                y_pred_classes = predizer_gaussiano_cov_agregada(X_test, medias, cov_agregada)
            elif modelo == "Classificador de Bayes Ingênuo":
                medias, variancias = fit_bayes_ingenuo(X_train, y_train)
                y_pred_classes = predizer_bayes_ingenuo(X_test, medias, variancias)
            elif modelo == "Gaussiano Regularizado (λ=0.25)":
                lambda_ = 0.25
                medias, covariancias = fit_gaussiano_regularizado(X_train, y_train, lambda_)
                y_pred_classes = predizer_gaussiano(X_test, medias, covariancias)
            elif modelo == "Gaussiano Regularizado (λ=0.5)":
                lambda_ = 0.5
                medias, covariancias = fit_gaussiano_regularizado(X_train, y_train, lambda_)
                y_pred_classes = predizer_gaussiano(X_test, medias, covariancias)
            elif modelo == "Gaussiano Regularizado (λ=0.75)":
                lambda_ = 0.75
                medias, covariancias = fit_gaussiano_regularizado(X_train, y_train, lambda_)
                y_pred_classes = predizer_gaussiano(X_test, medias, covariancias)

            acuracia = np.mean(y_pred_classes == y_test)
            acuracias.append(acuracia)

        resultados["Modelo"].append(modelo)
        resultados["Média"].append(np.mean(acuracias))
        resultados["Desvio Padrão"].append(np.std(acuracias))
        resultados["Maior Valor"].append(np.max(acuracias))
        resultados["Menor Valor"].append(np.min(acuracias))

    return pd.DataFrame(resultados)

def main() -> None:
    colunas = ['sensor_corrugador_supercilio', 'sensor_zigomatico_maior', 'classe']
    df = load_data(filepath='EMGsDataset.csv', columns=colunas, transpose=True)
    X, Y = prepare_data_para_gaussiano_bayesiano(df)

    resultados = testar_modelos(X, Y)
    print(resultados)
    
    resultados.plot(x='Modelo', y=['Média', 'Desvio Padrão', 'Maior Valor', 'Menor Valor'], kind='bar', figsize=(12, 6))
    plt.title("Resultados dos Modelos")
    plt.ylabel("Acurácia")
    plt.show()

if __name__ == "__main__":
    main()
