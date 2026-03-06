"""Скрипт для генерации графиков для отчёта."""
import numpy as np
import pandas as pd
from main import plot_histogram, plot_heatmap, plot_line, load_dataset

def main():
    # Загрузи данные
    data = load_dataset("data/students_scores.csv")
    
    # Гистограмма (оценки по математике - первый столбец)
    plot_histogram(data[:, 0])
    print("✓ Создан: plots/histogram.png")
    
    # Тепловая карта (корреляция между предметами)
    df = pd.DataFrame(data, columns=["math", "physics", "informatics"])
    corr_matrix = df.corr()
    plot_heatmap(corr_matrix)
    print("✓ Создан: plots/heatmap.png")
    
    # Линейный график (студент -> оценка по математике)
    students = np.arange(1, len(data) + 1)
    math_scores = data[:, 0]
    plot_line(students, math_scores)
    print("✓ Создан: plots/line.png")
    
    print("\n Все графики созданы в папке plots/")

if __name__ == "__main__":
    main()