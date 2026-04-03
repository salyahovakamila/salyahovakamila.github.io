# Лабораторная работа: Предсказание цен на недвижимость (Regression)

## Описание задачи

**Цель:** Прогнозирование стоимости домов в округе Кинг (штат Вашингтон, США) с помощью регрессионных моделей.

**Данные:** 
- 21,613 наблюдений
- 21 переменная (признаки)
- Период: май 2014 - май 2015

# Моя работа в [коллабе](https://colab.research.google.com/drive/1LRIoC1BbeMT9DHjCdw5VAkrMHLMJzg0e?usp=sharing#scrollTo=xsG3XS6PG4Ag)
##  Выполненные задания

### 1. Загрузка и предобработка данных

```python
# Загрузка библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, ensemble
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Загрузка данных
training_data = pd.read_excel('predict_house_price_training_data.xlsx')
test_data = pd.read_excel('predict_house_price_test_data.xlsx')

# Проверка информации о данных
training_data.info()
# Результат: 15,129 строк, 16 столбцов, нет пропусков
target_variable_name = 'Целевая.Цена'

# Выделение целевой переменной
training_values = training_data[target_variable_name]

# Отделение входных признаков
training_points = training_data.drop(columns=[target_variable_name], axis=1)

# Аналогично для тестовых данных
test_values = test_data[target_variable_name]
test_points = test_data.drop(columns=[target_variable_name], axis=1)
# Создание моделей
linear_regression_model = linear_model.LinearRegression()
random_forest_model = ensemble.RandomForestRegressor(n_estimators=100, random_state=42)

# Обучение моделей
linear_regression_model.fit(training_points, training_values)
random_forest_model.fit(training_points, training_values)
# Прогнозы
test_predictions_linear = linear_regression_model.predict(test_points)
test_predictions_random_forest = random_forest_model.predict(test_points)

# Расчет метрик
mae_linear = mean_absolute_error(test_values, test_predictions_linear)
rmse_linear = np.sqrt(mean_squared_error(test_values, test_predictions_linear))

mae_rf = mean_absolute_error(test_values, test_predictions_random_forest)
rmse_rf = np.sqrt(mean_squared_error(test_values, test_predictions_random_forest))

print(f"Linear Regression - MAE: {mae_linear:.2f}, RMSE: {rmse_linear:.2f}")
print(f"Random Forest - MAE: {mae_rf:.2f}, RMSE: {rmse_rf:.2f}")
```
# Результаты:
Linear Regression: MAE = 126,852.51, RMSE = 201,883.24
Random Forest: MAE = 70,712.04, RMSE = 137,104.19
### Самостоятельная работа
# Наименее важные признаки (по feature_importances_):
 1)Количество этажей (0.0025)
 2)Год реновации (0.0024)
 3)Просмотрены ранее (0.0040)
 4)Спальни (0.0042)
 5)Вид на воду (0.0067)
```python
# Удаление незначимых признаков
features_to_drop = ['Количество этажей', 'Год реновации', 'Просмотрены ранее', 
                    'Спальни', 'Вид на воду']

training_points_reduced = training_points.drop(columns=features_to_drop)
test_points_reduced = test_points.drop(columns=features_to_drop)

# Переобучение модели
rf_reduced = ensemble.RandomForestRegressor(n_estimators=100, random_state=42)
rf_reduced.fit(training_points_reduced, training_values)
pred_reduced = rf_reduced.predict(test_points_reduced)

rmse_reduced = np.sqrt(mean_squared_error(test_values, pred_reduced))
print(f"RMSE после удаления признаков: {rmse_reduced:.2f}")
```
# Результат: RMSE изменился незначительно (~138,000), что подтверждает малую важность удаленных признаков.
### Оптимизация параметров Random Forest
```python
from sklearn.model_selection import GridSearchCV

# Сетка гиперпараметров
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Поиск лучших параметров
grid_search = GridSearchCV(
    estimator=ensemble.RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(training_points, training_values)

print("Лучшие параметры:", grid_search.best_params_)
print("Лучший RMSE:", np.sqrt(-grid_search.best_score_))
{
    'n_estimators': 200,
    'max_depth': 30,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt'
}
```
# Результат оптимизации:
1. До оптимизации: RMSE = 137,104.19
2. После оптимизации: RMSE = 128,450.33 (улучшение на 6.3%)

## XGBoost Regressor
```python
from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(training_points, training_values)
xgb_pred = xgb_model.predict(test_points)

mae_xgb = mean_absolute_error(test_values, xgb_pred)
rmse_xgb = np.sqrt(mean_squared_error(test_values, xgb_pred))

print(f"XGBoost - MAE: {mae_xgb:.2f}, RMSE: {rmse_xgb:.2f}")
```
# Результат XGBoost:
1. MAE = 62,340.15
2. RMSE = 118,230.47 

# LightGBM Regressor
```python
from lightgbm import LGBMRegressor

lgb_model = LGBMRegressor(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

lgb_model.fit(training_points, training_values)
lgb_pred = lgb_model.predict(test_points)

mae_lgb = mean_absolute_error(test_values, lgb_pred)
rmse_lgb = np.sqrt(mean_squared_error(test_values, lgb_pred))

print(f"LightGBM - MAE: {mae_lgb:.2f}, RMSE: {rmse_lgb:.2f}")
```
# Результат LightGBM:
1. MAE = 61,890.22
2. RMSE = 116,780.91
### Вывод: Градиентный бустинг (XGBoost/LightGBM) показал наилучшие результаты, превзойдя базовую модель Random Forest на 14-15%.
```python
import joblib

# Сохранение модели и метаданных
joblib.dump(lgb_model, 'house_price_model.pkl')
joblib.dump(training_points.columns.tolist(), 'feature_names.pkl')
# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="House Price Prediction API")

# Загрузка модели
model = joblib.load('house_price_model.pkl')
feature_names = joblib.load('feature_names.pkl')

class HouseFeatures(BaseModel):
    Жилая_площадь: float = Field(..., ge=0)
    Общая_площадь: float = Field(..., ge=0)
    Спальни: int = Field(..., ge=0)
    Ванные: float = Field(..., ge=0)
    Этажи: int = Field(..., ge=1)
    Состояние: int = Field(..., ge=1, le=5)
    Оценка_риелтора: int = Field(..., ge=1, le=13)
    Широта: float
    Долгота: float
    # ... остальные признаки

    def to_dataframe(self):
        data = {name.replace('_', ' '): getattr(self, name) 
                for name in self.__fields__}
        return pd.DataFrame([data])[feature_names]

@app.post("/predict")
async def predict_price(features: HouseFeatures):
    try:
        X = features.to_dataframe()
        prediction = model.predict(X)[0]
        return {
            "predicted_price": float(prediction),
            "currency": "USD",
            "confidence_interval": {
                "lower": float(prediction * 0.9),
                "upper": float(prediction * 1.1)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok", "model": "loaded"}
```
```text
fastapi==0.109.0
uvicorn[standard]==0.27.0
scikit-learn==1.4.0
pandas==2.1.4
numpy==1.26.3
joblib==1.3.2
pydantic==2.5.0
lightgbm==4.3.0
```
```bash
# Локальный запуск
uvicorn main:app --reload

# Тестовый запрос
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Жилая_площадь": 2000,
    "Общая_площадь": 2500,
    "Спальни": 3,
    "Ванные": 2.5,
    "Этажи": 2,
    "Состояние": 4,
    "Оценка_риелтора": 8,
    "Широта": 47.6,
    "Долгота": -122.3
  }'
```
### Визуализация результатов
# Scatter plot: Фактические vs Предсказанные цены
```python
plt.figure(figsize=(8, 8))
plt.scatter(test_values, lgb_pred, s=1, alpha=0.5)
plt.plot([0, 6*10**6], [0, 6*10**6], 'r--', linewidth=2)
plt.xlabel('Фактическая цена ($)', fontsize=12)
plt.ylabel('Предсказанная цена ($)', fontsize=12)
plt.title('LightGBM: Фактические vs Предсказанные цены', fontsize=14)
plt.show()

# Топ-10 важных признаков
importances = pd.Series(lgb_model.feature_importances_, 
                        index=training_points.columns)
top_features = importances.nlargest(10)

plt.figure(figsize=(10, 6))
top_features.plot(kind='barh')
plt.xlabel('Важность')
plt.title('Топ-10 важных признаков (LightGBM)')
plt.gca().invert_yaxis()
plt.show()
```
# Топ-5 важных признаков:
Жилая площадь (sqft_living) - 31.2%
Оценка риелтора (grade) - 28.6%
Широта (lat) - 17.4%
Долгота (long) - 7.3%
Общая площадь (sqft_lot) - 3.0%