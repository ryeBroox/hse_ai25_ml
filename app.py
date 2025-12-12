import pickle
import streamlit as st
import pandas as pd
import plotly.express as px
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LinearRegression, Ridge
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import re

# Уменьшаем отступы
plt.rcParams['figure.autolayout'] = True


##### 0. конфиг страницы
st.set_page_config(
    page_title="Car Prices Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

##### 1. загрузка модели 
@st.cache_resource  # Кэшируем модель (загружается только один раз)
def load_pipeline():
    with open('models/full_pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)

    return pipeline


##### 2. загрузка CSV
@st.cache_data  # Кэшируем загруженные данные
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

def extract_torque_value(text):
    if pd.isna(text):  # обработка NaN
        return np.nan

    text = text.lower()

    # кейс 1 
    pattern1 = r'(\d+\.?\d*)\s*(nm|kgm|)\b'
    
    # кейс 2
    pattern2 = r'(\d+\.?\d*)\s*[@]'
    
    # тестим первый кейс
    match1 = re.search(pattern1, text)
    if match1:
        return float(match1.group(1))
    
    # если не первый кейс, то тестим второй
    match2 = re.search(pattern2, text)
    if match2:
        return float(match2.group(1))
    
    return np.nan


def extract_torque_unit(text):
    if pd.isna(text):  # обработка NaN
        return np.nan
    
    text = text.lower()
    
    if 'kgm' in text:  # определяем, что есть в строке - kgm или nm, так вытаскиваем ед измерения
        return 'kgm'
    if 'nm' in text:
        return 'nm'

    return np.nan


def prepare_features(df):
    """Приводим данные к формату обучения модели"""
    df_proc = df.copy()
    # Преобразуем признаки как в обучении:
    df_proc['mileage'] = df_proc.mileage.str.split(' ').str[0]
    df_proc['engine'] = df_proc.engine.str.split(' ').str[0]
    df_proc['max_power'] = df_proc.max_power.str.split(' ').str[0]

    for col in ['mileage', 'engine', 'max_power']:
        df_proc.loc[df_proc[col] == '', col] = np.nan

    # Применяем функции к колонке датафрейма
    df_proc['torque_value'] = df_proc['torque'].apply(extract_torque_value)
    df_proc['torque_unit'] = df_proc['torque'].apply(extract_torque_unit)
    
    df_proc.loc[(pd.isna(df_proc.torque_unit)) & (~pd.isna(df_proc.torque)), 'torque_unit'] = 'nm'

    df_proc.loc[df_proc.torque_unit == 'kgm', 'torque_value'] = df_proc.torque_value * 9.80665

    df_proc['torque'] = df_proc['torque_value']
    df_proc.drop(columns=['torque_value', 'torque_unit'], inplace=True)

    df_proc['mileage'] = df_proc['mileage'].astype(float)
    df_proc['engine'] = df_proc['engine'].astype(float)
    df_proc['max_power'] = df_proc['max_power'].astype(float)

    # считаем медианы по тренировочному датасету по условию в задании
    medians_d = {
        'mileage_median': df_proc['mileage'].median(),
        'engine_median': df_proc['engine'].median(),
        'max_power_median': df_proc['max_power'].median(),
        'torque_median': df_proc['torque'].median(),
        'seats_median': df_proc['seats'].median(),
    }   

    columns = ['mileage', 'engine', 'max_power', 'torque', 'seats']

    for col in columns:
        df_proc.loc[pd.isna(df_proc[col]), col] = medians_d[f'{col}_median']
    
    # удалим дубликаты по признакам и оставим первые строки (keep = 'first')
    df_proc = df_proc.drop_duplicates(subset=[col for col in df_proc.columns if col != 'selling_price'], keep='first').reset_index(drop=True)

    # сделаем нужные касты
    cols_to_int = ['engine', 'seats']

    for col in cols_to_int:
        df_proc[col] = df_proc[col].astype(int)

    df_proc.drop('name', axis=1, inplace=True)
    return df_proc

##### 5. Основная логика прилы

# В интерфейсе:
# Загружаем модель
pipeline = load_pipeline()

# Загрузка файла
uploaded_file = st.file_uploader("Загрузите CSV", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)

    # из модели вытаскиваем признаки:
    num_features = pipeline.named_steps['preprocessor']['num'].get_feature_names_out()
    cat_features = pipeline.named_steps['preprocessor']['cat'].get_feature_names_out()

    # Подготовка данных
    features = prepare_features(df)

    st.header("📊 EDA")

    # 1. Гистограмма
    st.subheader("Гистограмма распределения")
    selected_feature = st.selectbox(
        "Выберите признак для гистограммы:",
        features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    )

    fig1, ax1 = plt.subplots(figsize=(4, 2), dpi=100)
    sns.histplot(data=features, x=selected_feature, ax=ax1, kde=True)
    st.pyplot(fig1, use_container_width=False) 

    # 2. Корреляция
    st.subheader("Корреляция между признаками")

    col1, col2 = st.columns(2)
    with col1:
        feature_x = st.selectbox("Первый признак (X):", features.columns.tolist())
    with col2:
        feature_y = st.selectbox("Второй признак (Y):", features.columns.tolist())

    fig2, ax2 = plt.subplots(figsize=(4, 2), dpi=100) 
    sns.scatterplot(data=features, x=feature_x, y=feature_y, ax=ax2)
    st.pyplot(fig2, use_container_width=False) 

    # 3. Матрица корреляций
    if st.checkbox("Показать матрицу корреляций"):
        st.subheader("Матрица корреляций")
        corr_matrix = features.select_dtypes(include=['int64', 'float64']).corr()
        
        # Подгоняем размер под данные
        n_features = len(corr_matrix.columns)
        size = max(3, n_features * 0.3)  # динамический размер
        
        fig3, ax3 = plt.subplots(figsize=(size, size*0.8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax3,
                    annot_kws={"size": 4}, center=0, cbar_kws={"shrink": 0.8})  # уменьшаем шрифт и бар
        
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=6)  # размер цифр на шкале

        ax3.tick_params(axis='both', labelsize=6)  
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        st.pyplot(fig3, use_container_width=False)


    # 2. Блок прогнозирования
    st.header("🎯 Прогнозирование моделью")

    # Запрос индекса
    index_input = st.number_input(
        "Введите индекс объекта (0 - {max_idx}):".format(max_idx=len(features)-1),
        min_value=0,
        max_value=len(features)-1,
        value=0
    )

    if st.button("Сделать прогноз"):
        # Получаем объект по индексу
        object_data = features.iloc[[index_input]]  # DataFrame с одной строкой
        
        # Прогноз
        prediction = pipeline.predict(object_data)[0]
        
        # Вывод
        st.subheader("📋 Информация об объекте")
        st.dataframe(object_data.T.rename(columns={index_input: 'Значение'}))
        
        st.subheader("🎯 Результат прогноза")
        col_pred, col_real = st.columns(2)
        
        with col_pred:
            st.metric(
                label="Предсказанная цена",
                value=f"₽{prediction:,.2f}",
                delta=None
            )

    
    # 3. Блок информации о модели
    st.header("🤖 Информация о модели")

    # Информация о пайплайне
    st.subheader("Структура пайплайна")
    st.write("**Шаги:**")
    for i, (step_name, step_obj) in enumerate(pipeline.steps):
        st.write(f"{i+1}. **{step_name}**: {type(step_obj).__name__}")

    # Детали preprocessor
    if hasattr(pipeline.named_steps['preprocessor'], 'transformers'):
        st.write("**Препроцессор содержит:**")
        for name, transformer, cols in pipeline.named_steps['preprocessor'].transformers:
            if name != 'remainder':
                st.write(f"- {name}: {type(transformer).__name__} → {len(cols) if isinstance(cols, list) else '?'} признаков")

    # Информация о регрессоре
    regressor = pipeline.named_steps['regressor']
    st.write(f"**Регрессор:** {type(regressor).__name__}")
    st.write(f"- Параметры: {regressor.get_params()}")

    # График весов признаков
    st.subheader("📊 Веса признаков модели")

    try:
        # Получаем имена признаков после преобразования
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        coefficients = regressor.coef_
        
        # Создаем DataFrame для визуализации
        coef_df = pd.DataFrame({
            'Признак': feature_names,
            'Вес': coefficients,
            'abs_weight': abs(coefficients)
        }).sort_values('abs_weight', ascending=False)
        
        # Барплот
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(coef_df['Признак'], coef_df['Вес'])
        
        # Раскраска по знаку
        for bar, weight in zip(bars, coef_df['Вес']):
            bar.set_color('red' if weight < 0 else 'green')
        
        ax.set_xlabel('Вес признака')
        ax.set_title('Топ самых важных признаков')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Таблица с весами
        with st.expander("Посмотреть все веса"):
            st.dataframe(coef_df[['Признак', 'Вес']].round(4))
            
    except Exception as e:
        st.warning(f"Не удалось построить график весов: {e}")
    



