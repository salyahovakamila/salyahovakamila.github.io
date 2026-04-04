
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1. СОЗДАНИЕ И ОБРАБОТКА МАССИВОВ


def create_vector():
    """Создать массив от 0 до 9."""
    return np.arange(10)


def create_matrix():
    """Создать матрицу 5x5 со случайными числами [0,1]."""
    return np.random.rand(5, 5)


def reshape_vector(vec):
    """Преобразовать (10,) -> (2,5)"""
    return vec.reshape(2, 5)


def transpose_matrix(mat):
    """Транспонирование матрицы."""
    return mat.T


# 2. ВЕКТОРНЫЕ ОПЕРАЦИИ


def vector_add(a, b):
    """Сложение векторов одинаковой длины."""
    return a + b


def scalar_multiply(vec, scalar):
    """Умножение вектора на число."""
    return vec * scalar


def elementwise_multiply(a, b):
    """Поэлементное умножение."""
    return a * b


def dot_product(a, b):
    """Скалярное произведение."""
    return float(np.dot(a, b))


# 3. МАТРИЧНЫЕ ОПЕРАЦИИ


def matrix_multiply(a, b):
    """Умножение матриц."""
    return np.matmul(a, b)


def matrix_determinant(a):
    """Определитель матрицы."""
    return float(np.linalg.det(a))


def matrix_inverse(a):
    """Обратная матрица."""
    return np.linalg.inv(a)


def solve_linear_system(a, b):
    """Решить систему Ax = b"""
    return np.linalg.solve(a, b)



# 4. СТАТИСТИЧЕСКИЙ АНАЛИЗ


def load_dataset(path="data/students_scores.csv"):
    """Загрузить CSV и вернуть NumPy массив."""
    return pd.read_csv(path).to_numpy()


def statistical_analysis(data):
    """Статистический анализ одномерного массива."""
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "percentile_25": float(np.percentile(data, 25)),
        "percentile_75": float(np.percentile(data, 75)),
    }


def normalize_data(data):
    """Min-Max нормализация."""
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.zeros_like(data, dtype=float)
    return (data - min_val) / (max_val - min_val)



# 5. ВИЗУАЛИЗАЦИЯ


def plot_histogram(data):
    """Построить гистограмму распределения оценок."""
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=10, edgecolor="black", alpha=0.7)
    plt.title("Распределение оценок по математике")
    plt.xlabel("Оценка")
    plt.ylabel("Частота")
    plt.grid(axis="y", alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/histogram.png", dpi=300)
    plt.close()


def plot_heatmap(matrix):
    """Построить тепловую карту корреляции."""
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Тепловая карта корреляции")
    plt.tight_layout()
    plt.savefig("plots/heatmap.png", dpi=300)
    plt.close()


def plot_line(x, y):
    """Построить график зависимости: студент -> оценка."""
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", linewidth=2, markersize=6)
    plt.title("Зависимость оценки от номера студента")
    plt.xlabel("Номер студента")
    plt.ylabel("Оценка по математике")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/line.png", dpi=300)
    plt.close()