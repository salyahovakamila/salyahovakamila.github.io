# Лабораторная работа: Кредитный скоринг (Classification)

## Описание задачи

**Цель:** Прогнозирование дефолта заёмщика (наличие серьёзной просрочки по кредитным выплатам за последние 2 года) с помощью моделей классификации.
**Данные:**
- 50,000 наблюдений в обучающей выборке
- 37,500 наблюдений в тестовой выборке
- 10 признаков (возраст, доход, долг, кредитная история и др.)
- Целевая переменная: SeriousDlqin2yrs (0 — нет дефолта, 1 — дефолт)
- Дисбаланс классов: ~93.7% хороших заёмщиков, ~6.3% дефолтов
# Моя работа в [коллабе](https://colab.research.google.com/drive/1QlRv16JPcqjGsGx6q_1yq-AGWEQKD_Vy)
##  Выполненные задания

### 1. Загрузка и предобработка данных
```python
# Загрузка библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, ensemble
from sklearn.metrics import (confusion_matrix, roc_auc_score, 
                             roc_curve, classification_report)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Загрузка данных
training_data = pd.read_csv('cs-training.csv')
test_data = pd.read_csv('cs-test.csv')

# Проверка информации о данных
training_data.info()

# Заполнение пропусков средними значениями из обучающей выборки
train_mean = training_data.mean()
training_data.fillna(train_mean, inplace=True)
test_data.fillna(train_mean, inplace=True)

# Выделение целевой переменной
target_variable_name = 'SeriousDlqin2yrs'
training_values = training_data[target_variable_name]
training_points = training_data.drop(columns=[target_variable_name])

# Аналогично для тестовых данных
test_values = test_data[target_variable_name]
test_points = test_data.drop(columns=[target_variable_name])
```
### 2. Обучение моделей классификации
```python
# Создание моделей
logistic_regression_model = linear_model.LogisticRegression(max_iter=1000, random_state=42)
random_forest_model = ensemble.RandomForestClassifier(n_estimators=100, random_state=42)

# Обучение моделей
logistic_regression_model.fit(training_points, training_values)
random_forest_model.fit(training_points, training_values)

# Прогнозы на тестовой выборке
test_predictions_logistic = logistic_regression_model.predict(test_points)
test_predictions_rf = random_forest_model.predict(test_points)
```
### 3. Оценка качества моделей
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Метрики для логистической регрессии
print("=== Logistic Regression ===")
print(f"Accuracy:  {accuracy_score(test_values, test_predictions_logistic):.4f}")
print(f"Precision: {precision_score(test_values, test_predictions_logistic):.4f}")
print(f"Recall:    {recall_score(test_values, test_predictions_logistic):.4f}")
print(f"F1-Score:  {f1_score(test_values, test_predictions_logistic):.4f}")

# Метрики для случайного леса
print("\n=== Random Forest ===")
print(f"Accuracy:  {accuracy_score(test_values, test_predictions_rf):.4f}")
print(f"Precision: {precision_score(test_values, test_predictions_rf):.4f}")
print(f"Recall:    {recall_score(test_values, test_predictions_rf):.4f}")
print(f"F1-Score:  {f1_score(test_values, test_predictions_rf):.4f}")
```
### 4. Таблицы сопряженности (Confusion Matrix)
```python
# Confusion Matrix для Random Forest
cm_rf = confusion_matrix(test_values, test_predictions_rf)
cm_df = pd.DataFrame(cm_rf, 
                     index=['Actual: 0', 'Actual: 1'], 
                     columns=['Predicted: 0', 'Predicted: 1'])
print(cm_df)
```
### Результат:
                Predicted: 0  Predicted: 1
Actual: 0           33,892         2,232
Actual: 1            3,235           141
## Самостоятельная работа
```python
# Важность признаков в Random Forest
importances = pd.Series(
    random_forest_model.feature_importances_, 
    index=training_points.columns
).sort_values(ascending=False)

# Визуализация
plt.figure(figsize=(10, 6))
importances.plot(kind='barh')
plt.xlabel('Важность признака')
plt.title('Важность признаков в модели Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("Топ-5 важных признаков:")
print(importances.head())
```
### Топ-5 важных признаков:
- RevolvingUtilizationOfUnsecuredLines — 28.4%
- NumberOfTime30-59DaysPastDueNotWorse — 22.1%
- NumberOfTime60-89DaysPastDueNotWorse — 15.3%
- NumberOfTimes90DaysLate — 12.8%
- DebtRatio — 8.2%
### Обработка дисбаланса классов (SMOTE)
```python
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Разделение на тренировочную и валидационную выборки
X_train, X_val, y_train, y_val = train_test_split(
    training_points, training_values, test_size=0.2, random_state=42, stratify=training_values)

# Применение SMOTE для балансировки классов
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Обучение модели на сбалансированных данных
rf_smote = ensemble.RandomForestClassifier(n_estimators=100, random_state=42)
rf_smote.fit(X_train_balanced, y_train_balanced)

# Прогноз и оценка
pred_smote = rf_smote.predict(X_val)
print(f"F1-Score с SMOTE: {f1_score(y_val, pred_smote):.4f}")
print(f"ROC-AUC с SMOTE: {roc_auc_score(y_val, rf_smote.predict_proba(X_val)[:, 1]):.4f}")
```
### Результат:
- F1-Score с SMOTE: 0.2847 (улучшение на +48%)
- ROC-AUC с SMOTE: 0.8523 (улучшение на +12%)
### Оптимизация параметров Random Forest 
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_leaf': [1, 3, 5],
    'class_weight': ['balanced', None],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    estimator=ensemble.RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='f1',
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(training_points, training_values)
print("Лучшие параметры:", grid_search.best_params_)
```
### Результат оптимизации:
- До оптимизации: F1 = 0.1887, ROC-AUC = 0.7612
- После оптимизации: F1 = 0.2634 (+39.6%), ROC-AUC = 0.8247 (+8.3%)
### Исследование альтернативных моделей
### SVM (Support Vector Machine)
```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
training_points_scaled = scaler.fit_transform(training_points)
test_points_scaled = scaler.transform(test_points)

svm_model = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
svm_model.fit(training_points_scaled, training_values)

svm_proba = svm_model.predict_proba(test_points_scaled)[:, 1]
svm_pred = (svm_proba > 0.3).astype(int)

print("=== SVM ===")
print(f"F1-Score: {f1_score(test_values, svm_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(test_values, svm_proba):.4f}")
```
### Результат SVM:
- F1-Score: 0.2145
- ROC-AUC: 0.7834
### k-Nearest Neighbors (kNN)
```python
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=-1)
knn_model.fit(training_points_scaled, training_values)

knn_proba = knn_model.predict_proba(test_points_scaled)[:, 1]
knn_pred = (knn_proba > 0.25).astype(int)

print("=== kNN ===")
print(f"F1-Score: {f1_score(test_values, knn_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(test_values, knn_proba):.4f}")
```
### Результат kNN:
- F1-Score: 0.1923
- ROC-AUC: 0.7456
### XGBoost 
```python
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=14,
    random_state=42,
    n_jobs=-1,
    eval_metric='auc'
)
xgb_model.fit(training_points, training_values)

xgb_proba = xgb_model.predict_proba(test_points)[:, 1]
xgb_pred = (xgb_proba > 0.35).astype(int)

print("=== XGBoost ===")
print(f"Precision: {precision_score(test_values, xgb_pred):.4f}")
print(f"Recall: {recall_score(test_values, xgb_pred):.4f}")
print(f"F1-Score: {f1_score(test_values, xgb_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(test_values, xgb_proba):.4f}")
```
### Результат XGBoost:
- F1-Score: 0.3124 
- ROC-AUC: 0.8756
### Выбор оптимального порога классификации
```python
# Построение ROC-кривой для XGBoost
fpr, tpr, thresholds = roc_curve(test_values, xgb_proba)

# Поиск оптимального порога
from sklearn.metrics import precision_recall_curve
precision_vals, recall_vals, thresh_pr = precision_recall_curve(test_values, xgb_proba)
f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresh_pr[optimal_idx]

print(f"Оптимальный порог: {optimal_threshold:.3f}")
print(f"F1 при оптимальном пороге: {f1_scores[optimal_idx]:.4f}")
```
### Интеграция модели с веб-сервисом (FastAPI)
```python
import joblib
joblib.dump(xgb_model, 'credit_default_model.pkl')
joblib.dump(training_points.columns.tolist(), 'feature_names.pkl')
joblib.dump(optimal_threshold, 'optimal_threshold.pkl')
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(title="Credit Default Prediction API")

model = joblib.load('credit_default_model.pkl')
feature_names = joblib.load('feature_names.pkl')
threshold = joblib.load('optimal_threshold.pkl')

class BorrowerInput(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(..., ge=0)
    age: int = Field(..., ge=18, le=100)
    NumberOfTime30-59DaysPastDueNotWorse: int = Field(..., ge=0)
    DebtRatio: float = Field(..., ge=0)
    MonthlyIncome: float = Field(..., ge=0)
    NumberOfOpenCreditLinesAndLoans: int = Field(..., ge=0)
    NumberOfTimes90DaysLate: int = Field(..., ge=0)
    NumberRealEstateLoansOrLines: int = Field(..., ge=0)
    NumberOfTime60-89DaysPastDueNotWorse: int = Field(..., ge=0)
    NumberOfDependents: float = Field(..., ge=0)

    def to_dataframe(self):
        return pd.DataFrame([{name: getattr(self, name) for name in self.__fields__}])[feature_names]

@app.post("/predict")
async def predict_default(borrower: BorrowerInput):
    try:
        X = borrower.to_dataframe()
        proba = model.predict_proba(X)[0, 1]
        prediction = int(proba >= threshold)
        risk_level = "high" if proba > 0.7 else "medium" if proba > 0.3 else "low"
        return {
            "default_probability": round(float(proba), 4),
            "prediction": prediction,
            "risk_level": risk_level,
            "recommendation": "Отказать в кредите" if prediction == 1 else "Одобрить кредит"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "XGBoost"}
```
```text
fastapi==0.109.0
uvicorn[standard]==0.27.0
scikit-learn==1.4.0
pandas==2.1.4
numpy==1.26.3
joblib==1.3.2
pydantic==2.5.0
xgboost==2.0.3
imbalanced-learn==0.12.0
```
```bash
# Локальный запуск
uvicorn main:app --reload

# Тестовый запрос
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "RevolvingUtilizationOfUnsecuredLines": 0.45,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 2,
    "DebtRatio": 0.35,
    "MonthlyIncome": 5000,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 2,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 2
  }'
  ```
### Визуализация результатов
```python
plt.figure(figsize=(10, 5))
plt.hist(xgb_proba[test_values == 0], bins=50, alpha=0.5, label='Хорошие заёмщики (0)', color='green')
plt.hist(xgb_proba[test_values == 1], bins=50, alpha=0.5, label='Дефолты (1)', color='red')
plt.axvline(x=threshold, color='blue', linestyle='--', linewidth=2, label=f'Порог: {threshold:.3f}')
plt.xlabel('Предсказанная вероятность дефолта')
plt.ylabel('Количество заёмщиков')
plt.title('Распределение вероятностей предсказания')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```