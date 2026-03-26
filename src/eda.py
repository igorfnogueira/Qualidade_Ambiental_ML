"""Análise exploratória de dados (EDA) para o dataset ambiental."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Carrega o CSV e retorna um DataFrame."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {path.resolve()}")
    return pd.read_csv(path)


def carregar_dados(csv_path: str | Path) -> pd.DataFrame:
    """Alias em português para `load_dataset` (usado no notebook)."""
    return load_dataset(csv_path)


def validate_target_column(df: pd.DataFrame, target: str) -> None:
    """Garante que a coluna alvo existe; levanta ValueError caso contrário."""
    if target not in df.columns:
        raise ValueError(
            f'A coluna alvo "{target}" não foi encontrada no CSV. '
            f"Colunas disponíveis: {list(df.columns)}"
        )


def print_head(df: pd.DataFrame, n: int = 5) -> None:
    """Exibe as primeiras linhas."""
    print("\n=== PRIMEIRAS LINHAS ===")
    print(df.head(n))


def print_info(df: pd.DataFrame) -> None:
    """Exibe dtypes, contagem de não-nulos e uso de memória."""
    print("\n=== INFO ===")
    df.info()


def print_describe(df: pd.DataFrame) -> None:
    """Exibe estatísticas descritivas para todas as colunas."""
    print("\n=== DESCRIBE (todas as colunas) ===")
    print(df.describe(include="all"))


def print_null_counts(df: pd.DataFrame) -> None:
    """Exibe contagem de valores nulos por coluna."""
    print("\n=== VALORES NULOS POR COLUNA ===")
    print(df.isnull().sum())


def _numeric_feature_columns(df: pd.DataFrame, target: str | None) -> list[str]:
    """
    Colunas numéricas usadas como features nos gráficos.
    Exclui o alvo quando ele for numérico (ex.: classe codificada).
    """
    numeric = df.select_dtypes(include="number").columns.tolist()
    if target and target in df.columns and target in numeric:
        return [c for c in numeric if c != target]
    return numeric


def plot_histograms(
    df: pd.DataFrame,
    *,
    target: str | None = None,
    bins: int = 30,
    figsize_per_plot: tuple[float, float] = (4.0, 3.0),
) -> None:
    """
    Histogramas das variáveis numéricas (features) com Seaborn/Matplotlib.
    """
    cols = _numeric_feature_columns(df, target)
    if not cols:
        print("\n=== HISTOGRAMAS ===\nNenhuma coluna numérica para plotar.")
        return

    sns.set_theme(style="whitegrid")
    n = len(cols)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
        squeeze=False,
    )

    for i, col in enumerate(cols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        sns.histplot(data=df, x=col, bins=bins, kde=True, ax=ax, color="steelblue")
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle("Histogramas das variáveis numéricas", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(
    df: pd.DataFrame,
    *,
    method: str = "pearson",
    figsize: tuple[float, float] = (10, 8),
) -> None:
    """Matriz de correlação (apenas colunas numéricas) com heatmap."""
    corr = df.select_dtypes(include="number").corr(method=method)
    if corr.empty:
        print("\n=== MATRIZ DE CORRELAÇÃO ===\nSem colunas numéricas suficientes.")
        return

    sns.set_theme(style="white")
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(f"Matriz de correlação ({method})", fontsize=14, pad=12)
    plt.tight_layout()
    plt.show()


def plot_boxplots_outliers(
    df: pd.DataFrame,
    *,
    target: str | None = None,
    figsize_per_plot: tuple[float, float] = (5.0, 2.8),
) -> None:
    """
    Boxplots das variáveis numéricas para inspeção de outliers e dispersão.
    """
    cols = _numeric_feature_columns(df, target)
    if not cols:
        print("\n=== BOXPLOTS ===\nNenhuma coluna numérica para plotar.")
        return

    sns.set_theme(style="whitegrid")
    n = len(cols)
    ncols = min(2, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
        squeeze=False,
    )

    for i, col in enumerate(cols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        sns.boxplot(data=df, x=col, ax=ax, color="lightblue", width=0.35)
        ax.set_title(f"Boxplot — {col}", fontsize=10)
        ax.set_xlabel("")

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle("Boxplots (detecção visual de outliers)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def run_eda(df: pd.DataFrame, *, target: str | None = None) -> None:
    """Executa o relatório EDA completo: tabelas + histogramas, correlação e boxplots."""
    print_head(df)
    print_info(df)
    print_describe(df)
    print_null_counts(df)

    print("\n=== GERANDO GRÁFICOS (feche cada janela para continuar o script) ===")
    plot_histograms(df, target=target)
    plot_correlation_matrix(df)
    plot_boxplots_outliers(df, target=target)

