# Название: Умный дом - интернет вещей
# Вопрос исследования: Какие переменные умного дома оказывают наибольшее влияние на использование электроэнергии?
# Полученное решение: Котёл отопления, винный погреб и холодильник - это комнаты и устройства, в которых используется больше энергии.
# 78,89% изменчивости данных объясняется этими переменными в наборе данных.

# Метод. регрессионный анализ случайных лесов.
# Результаты: Использовано CleanData собранная системой Интернета вещей

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import stats
import string as string
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Импорт файла необработанных данных во фрейм данных
#csv_path = "CleanData_per_month.csv"
csv_path = "CleanData_per_day.csv"
df = pd.read_csv(csv_path,   parse_dates=True)
nRow, nCol = df.shape
print(f'There are {nRow} rows and {nCol} columns')
df.head(12)

# Подготовка данных
df.columns
#df.dtypes
#CleanData.tail()

inputdf = df.filter(items=['month', 'day', 'year', 'use', 'Furnace',
                            'Home office', 'Living room', 'Wine cellar', 'Garage door', 'Kitchen',
                            'Barn', 'Well', 'Fridge', 'Microwave', 'Dishwasher', 'Temperature',
                            'Humidity', 'Visibility', 'ApparentTemperature', 'Pressure',
                            'WindSpeed', 'windBearing', 'Precip', 'DewPoint'])

# Запуск модели Случайного леса со всеми данными для исследования движущих переменных энергоэффективности с разрешением 1 мин
#my_data.dtypes
features=inputdf
type(features)
#features.iloc[:,5:].head(5)
#features.head(5)
#features.info()
features.columns

# Метки - это значения, которые мы хотим предсказать
labels = np.array(features['use'])
# Удалить ярлыки из функций
# ось 1 относится к столбцам
features= features.drop('use', axis = 1)
# Сохранение имен объектов для последующего использования
feature_list = list(features.columns)
# Преобразовать в массив NumPy
features = np.array(features)

# Использование Scikit-learn для разделения данных на обучающие и тестовые наборы
from sklearn.model_selection import train_test_split
# Разделить данные на обучающие и тестовые наборы
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 100)

print('Форма признаков обучения:', train_features.shape)
print('Форма меток обучения:', train_labels.shape)
print('Форма признаков тестирования:', test_features.shape)
print('Форма меток тестирования:', test_labels.shape)


# Импортируем используемую модель
from sklearn.ensemble import RandomForestRegressor
# Создание модели с 100 деревьями решений
#rf = RandomForestRegressor(n_estimators = 100, random_state = 100)
rf = RandomForestRegressor(n_estimators = 100, random_state = 100)
# Обучаем модель по выборки обучения
rf.fit(train_features, train_labels);

# Используйте метод прогнозирования леса на тестовых данных
predictions = rf.predict(test_features)
# Рассчитать абсолютные ошибки
errors = abs(predictions - test_labels)
# Вывести среднюю абсолютную ошибку (мАе)
print('Средняя абсолютная ошибка:', round(np.mean(errors), 2), 'кВт.')


# Рассчитать среднюю абсолютную процентную ошибку (MAPE)
mape = 100 * (errors / test_labels)
# Расчет и отображение точности
accuracy = 100 - np.mean(mape)
print('Точность:', round(accuracy, 2), '%.')


# Каковы основные переменные предиктора, которые регулируют мощность, используемую в умном доме?
# Импорт инструментов, необходимых для визуализации
from sklearn.tree import export_graphviz
#import pydot
# Вытащи одно дерево из леса
tree = rf.estimators_[5]

# Получить числовые значения функций
importances = list(rf.feature_importances_)
# Список кортежей с переменной и важностью
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# # Сортировать значения функций по наиболее важным
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Распечатать функцию и значение
[print('Переменные: {:20} Важность: {}'.format(*pair)) for pair in feature_importances];

#%matplotlib inline
# Установить стиль
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Сделать гистограмму
plt.bar(x_values, importances, orientation = 'vertical')
# Галочки меток для оси x
plt.xticks(x_values, feature_list, rotation='vertical')
# Оси метки и заголовок
plt.ylabel('Значимость'); plt.xlabel('Переменная'); plt.title('Значимость переменных');
plt.show()

# Обучаем новуюю модель Случайного леса с двумя самыми важными переменными
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=1000)
# Извлечь два самых важных признака (в нашем случае это печька и винный погреб)
important_indices = [feature_list.index('Furnace'), feature_list.index('Wine cellar')]
train_important = train_features[:, important_indices]
test_important = test_features[:, important_indices]
# Обучить Случайный лес
rf_most_important.fit(train_important, train_labels)
# Делать прогнозы и определять ошибку
predictions = rf_most_important.predict(test_important)
errors = abs(predictions - test_labels)
# Показать показатели производительности
print('Средняя абсолютная ошибка:', round(np.mean(errors), 2), 'кВт.')
mape = np.mean(100 * (errors / test_labels))
accuracy = 100 - mape
print('Точность:', round(accuracy, 2), '%.')

import datetime
# Даты тренировочных значений
months = features[:, feature_list.index('month')]
days = features[:, feature_list.index('day')]
years = features[:, feature_list.index('year')]

dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

# Датафрейм с истинными значениями и датами
true_data = pd.DataFrame(data = {'date': dates, 'actual': labels})

# Даты прогнозов
months = test_features[:, feature_list.index('month')]
days = test_features[:, feature_list.index('day')]
years = test_features[:, feature_list.index('year')]

# Колонка дат
test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]

# Конвертировать в объекты даты и времени
test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]

# Датафрейм с прогнозами и датами
predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predictions})

# График фактических значений
plt.plot(true_data['date'], true_data['actual'], 'b-', label = 'факт')
# График прогнозируемых значений
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'прогноз')
plt.xticks(rotation = '60');
plt.legend()
# Графические метки
plt.xlabel('Дата'); plt.ylabel('Используемая мощность [кВт]'); plt.title('Фактические и предсказанне значения');
plt.show()
