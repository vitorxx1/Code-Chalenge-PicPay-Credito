import numpy as np
import pandas as pd
import eli5
import shap
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.stats import spearmanr, pearsonr, kendalltau
from eli5.sklearn import PermutationImportance
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

class ClipTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer personalizado para limitar os valores das variáveis 
    dentro dos limites inferior e superior definidos.
    """

    def __init__(self, clip_dict):
        self.clip_dict = clip_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_clipped = X.copy()
        
        for col, limits in self.clip_dict.items():
            if col in X_clipped.columns:
                X_clipped[col] = np.clip(X_clipped[col], limits["lower"], limits["upper"])
                
        return X_clipped
    
    
def select_best_features(
        df_features: pd.DataFrame,
        df_target: pd.Series,
        n_features: int
) -> pd.DataFrame:
    """
    Seleciona as melhores features para um modelo usando diferentes técnicas de seleção de variáveis
    utilizando uma variável random

    Parâmetros:
        df_features (pd.DataFrame): DataFrame com as variáveis explicativas.
        df_target (pd.Series): Variável resposta (target).
        n_features (int): Quantidade máxima de features com score maior que a variável random

    Retorna:
        pd.DataFrame: DataFrame com as variáveis selecionadas e seus scores.
    """
    # Criando uma feature aleatória
    np.random.seed(42)
    df_features = df_features.copy()
    df_features["random_feature"] = np.random.rand(df_features.shape[0])

    # Definição do tipo de problema (classificação ou regressão)
    is_classification = len(np.unique(df_target)) <= 10
    
    # Dicionário para armazenar os scores
    scores_dict = {"Feature": [], "Permutation Importance": [],"Mutual Information": [], 
                   "Spearman Correlation": [], "Pearson Correlation": [], "Kendall Correlation": []}

    # Treinando modelo
    x_train, x_val, y_train, y_val = train_test_split(df_features, df_target, test_size=0.2, random_state=42)
    
    if is_classification:
        model = RandomForestClassifier(random_state=42)
        model.fit(x_train,y_train)
    else:
        model = RandomForestRegressor(random_state=42)
        model.fit(x_train,y_train)
    
    # Permutation Importance
    perm = PermutationImportance(model, random_state=42).fit(x_val, y_val)
    pi_scores = perm.feature_importances_

    # Mutual Information
    if is_classification:
        mi_scores = mutual_info_classif(df_features, df_target, random_state=42)
    else:
        mi_scores = mutual_info_regression(df_features, df_target, random_state=42)

    # Correlação de Spearman e Correlação de Pearson
    spearman_scores = []
    pearson_scores = []
    kendall_scores = []

    for col in df_features.columns:
        spearman_corr, _ = spearmanr(df_features[col], df_target)
        pearson_corr, _ = pearsonr(df_features[col], df_target)
        kendall_corr, _ = kendalltau(df_features[col], df_target)
        spearman_scores.append(abs(spearman_corr))
        pearson_scores.append(abs(pearson_corr))
        kendall_scores.append(abs(kendall_corr))

    # Preenchendo o dicionário de scores
    for i, col in enumerate(df_features.columns):
        scores_dict["Feature"].append(col)
        scores_dict["Permutation Importance"].append(pi_scores[i])
        scores_dict["Mutual Information"].append(mi_scores[i])
        scores_dict["Spearman Correlation"].append(spearman_scores[i])
        scores_dict["Pearson Correlation"].append(pearson_scores[i])
        scores_dict["Kendall Correlation"].append(kendall_scores[i])

    # Criando o DataFrame de scores
    df_scores = pd.DataFrame(scores_dict)
    
    # Selecionando as features
    selected_vars_list = []
    for metric in df_scores.columns.drop("Feature"):
        selected_vars_list += df_scores[df_scores[metric] > df_scores[df_scores["Feature"] == "random_feature"][metric].values[0]]["Feature"].values.tolist()[:n_features]
    
    df_scores = df_scores[df_scores["Feature"].isin(selected_vars_list)]
    
    return df_scores

def plot_roc_curve(y_true, y_scores):
    """
    Plota a curva ROC e calcula a AUC.

    Parâmetros:
    - y_true: array-like, rótulos verdadeiros (0 ou 1).
    - y_scores: array-like, probabilidades da classe positiva.

    Retorna:
    - auc_score: valor da AUC (Área sob a curva ROC).
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Linha de referência
    plt.xlabel("Taxa de Falsos Positivos (FPR)")
    plt.ylabel("Taxa de Verdadeiros Positivos (TPR)")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    return auc_score


def plot_precision_recall_curve(y_true, y_scores):
    """
    Plota a Precision-Recall Curve e calcula a PRAUC.

    Parâmetros:
    - y_true: array-like, rótulos verdadeiros (0 ou 1).
    - y_scores: array-like, probabilidades da classe positiva.

    Retorna:
    - pr_auc_score: valor da PRAUC (Área sob a curva Precision-Recall).
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc_score = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Precision-Recall Curve (PRAUC = {pr_auc_score:.4f})', color='green')
    plt.xlabel("Recall")
    plt.ylabel("Precisão")
    plt.title("Curva Precision-Recall")
    plt.legend(loc="lower left")
    plt.grid()
    plt.show()

    return pr_auc_score


def plot_mean_by_category(df: pd.DataFrame, category_col: str, value_col: str):
    """
    Plota um gráfico de barras com a média de uma variável contínua para cada faixa categórica.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados.
    category_col (str): Nome da coluna categórica (faixas).
    value_col (str): Nome da variável contínua para calcular a média.

    Retorna:
    None: Exibe o gráfico.
    """
    try:
        # Verifica se as colunas existem no DataFrame
        if category_col not in df.columns or value_col not in df.columns:
            raise ValueError("As colunas especificadas não existem no DataFrame.")

        # Calcula a média da variável contínua para cada faixa categórica
        mean_values = df.groupby(category_col)[value_col].mean().reset_index()

        # Ordena as categorias se forem intervalos
        mean_values = mean_values.sort_values(by=category_col)

        # Configuração do gráfico
        plt.figure(figsize=(8, 4))
        sns.barplot(data=mean_values, x=category_col, y=value_col, palette="Blues")

        # Ajuste dos rótulos e título
        plt.xlabel("Faixa")
        plt.ylabel(f"Média de {value_col}")
        plt.title(f"Média de {value_col} por Faixa")
        plt.xticks(rotation=45)

        # Exibir o gráfico
        plt.show()

    except Exception as e:
        print(f"Erro ao gerar o gráfico: {e}")
        
def plot_stacked_bar_chart(df, faixa_col, cat_col):
    """
    Plota um gráfico de barras empilhadas com a quantidade de cada categoria por faixa.
    
    Parâmetros:
    df : pd.DataFrame
        DataFrame contendo os dados.
    faixa_col : str
        Nome da coluna correspondente às faixas.
    cat_col : str
        Nome da coluna categórica.
    """
    plt.figure(figsize=(8, 4))
    
    # Criando tabela de contingência
    contingency_table = df.pivot_table(index=faixa_col, columns=cat_col, aggfunc='size', fill_value=0)
    
    # Plotando gráfico de barras empilhadas
    contingency_table.plot(kind='bar', stacked=True, colormap='viridis', figsize=(12, 6))
    
    plt.xlabel(faixa_col)
    plt.ylabel("Quantidade")
    plt.title("Distribuição de Categorias por Faixa")
    plt.legend(title=cat_col)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    