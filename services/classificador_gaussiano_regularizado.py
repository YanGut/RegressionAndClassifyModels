import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict

def fit_gaussiano_regularizado(X: NDArray[np.float64], y: NDArray[np.int64], lambda_: float) -> Tuple[dict, dict, dict]:
    """
    Ajusta o modelo Gaussiano Regularizado.

    Args:
        X (NDArray[np.float64]): Dados de entrada.
        y (NDArray[np.int64]): Rótulos das classes.
        lambda_ (float): Parâmetro de regularização.

    Returns:
        Tuple[dict, dict, dict]: Médias, covariâncias e determinantes para cada classe.
    """
    classes = np.unique(y)
    medias = {}
    covariancias = {}
    determinantes = {}
    
    for classe in classes:
        X_classe = X[y == classe]
        media_classe = np.mean(X_classe, axis=0)
        cov_classe = np.cov(X_classe, rowvar=False)
        
        # Regularização de Friedman
        cov_regularizada = (1 - lambda_) * cov_classe + lambda_ * np.eye(cov_classe.shape[0])
        
        medias[classe] = media_classe
        covariancias[classe] = cov_regularizada
        determinantes[classe] = np.linalg.det(cov_regularizada)
    
    return medias, covariancias, determinantes

def fit_gaussiano_covariancias_iguais(X: NDArray[np.float64], y: NDArray[np.int64]) -> Tuple[dict, NDArray[np.float64]]:
    classes = np.unique(y)
    medias = {}
    covariancias = np.zeros((X.shape[1], X.shape[1]))
    
    for classe in classes:
        X_classe = X[y == classe]
        media_classe = np.mean(X_classe, axis=0)
        medias[classe] = media_classe
        covariancias += np.cov(X_classe, rowvar=False)
    
    # Calcule a covariância média
    covariancias /= len(classes)
    
    return medias, covariancias

def predizer_gaussiano(X_test: NDArray[np.float64], medias: Dict[int, NDArray[np.float64]], covariancias: Dict[int, NDArray[np.float64]], determinantes: Dict[int, float], lambda_: float) -> NDArray[np.int64]:
    """
    Prediz as classes de novas amostras usando o modelo Gaussiano Regularizado.
    
    Args:
        X_test (NDArray[np.float64]): Conjunto de dados de teste (N x p).
        medias (Dict[int, NDArray[np.float64]]): Médias de cada classe.
        covariancias (Dict[int, NDArray[np.float64]]): Covariâncias regularizadas de cada classe.
        determinantes (Dict[int, float]): Determinantes das covariâncias para cada classe.
        lambda_ (float): Parâmetro de regularização para escolher a função discriminante.
    
    Returns:
        NDArray[np.int64]: Vetor com as previsões de classe.
    """
    preds = []
    for x in X_test:
        scores = {}
        
        for classe, media in medias.items():
            diff = x - media
            cov_inv = np.linalg.inv(covariancias[classe])
            
            if lambda_ == 1:
                scores[classe] = diff.T @ cov_inv @ diff
            else:
                scores[classe] = -0.5 * np.log(determinantes[classe]) - 0.5 * (diff.T @ cov_inv @ diff)
        
        pred_classe = min(scores, key=scores.get)
        preds.append(pred_classe)
    
    return np.array(preds)

def fit_bayes_ingenuo(X: NDArray[np.float64], y: NDArray[np.int64]) -> Tuple[dict, dict]:
    classes = np.unique(y)
    medias = {}
    variancias = {}
    
    for classe in classes:
        X_classe = X[y == classe]
        medias[classe] = np.mean(X_classe, axis=0)
        variancias[classe] = np.var(X_classe, axis=0)
    
    return medias, variancias

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