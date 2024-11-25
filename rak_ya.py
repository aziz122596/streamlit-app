import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch import nn

# Установка переменной окружения для OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Установка конфигурации страницы
st.set_page_config(
    page_title="Диагностика рака яичников",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Приложение для анализа данных и предсказания наличия рака яичников."
    }
)

# Установка кастомного CSS стиля
def set_css():
    st.markdown("""
        <style>
        .stApp {
            background-color: #f0f4f8;
        }
        h1 {
            color: #0073e6;
            text-align: center;
        }
        h3 {
            color: gray;
            text-align: center;
        }
        .css-1d391kg {
            background-color: rgba(255, 255, 255, 0.9);
        }
        </style>
    """, unsafe_allow_html=True)

set_css()

# Классификатор для анализа данных
class OvarianCancerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(OvarianCancerModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Загрузка модели
@st.cache_resource
def load_model():
    input_size = 10  # Количество входных признаков
    hidden_size = 16
    output_size = 2  # Два класса: 0 - отсутствует, 1 - присутствует
    model = OvarianCancerModel(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load('ovarian_cancer_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Предсказание с использованием модели
def predict(data, model):
    with torch.no_grad():
        input_tensor = torch.tensor(data, dtype=torch.float32)
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        return predicted.item()

# Главная функция
def main():
    st.markdown("<h1>Диагностика рака яичников</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Приложение для предсказания на основе машинного обучения</h3>", unsafe_allow_html=True)

    # Навигация по боковой панели
    with st.sidebar:
        st.title("Навигация")
        selected_option = st.radio(
            "Выберите действие:",
            ["Главная", "О приложении", "Контакты"]
        )

    # Главная страница
    if selected_option == "Главная":
        st.subheader("Загрузите данные для анализа")
        uploaded_file = st.file_uploader("Выберите файл CSV с данными", type=["csv"])

        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.write("Загруженные данные:")
            st.dataframe(data)

            # Проверяем, есть ли данные для анализа
            if st.button("Провести анализ"):
                # Подготовка данных
                try:
                    model = load_model()
                    input_data = data.values
                    predictions = [predict(row, model) for row in input_data]

                    # Добавляем предсказания в таблицу
                    data['Prediction'] = ['Рак не обнаружен' if pred == 0 else 'Рак обнаружен' for pred in predictions]
                    st.success("Анализ завершён!")
                    st.write("Результаты:")
                    st.dataframe(data)

                except Exception as e:
                    st.error(f"Ошибка анализа: {e}")

    # Страница "О приложении"
    elif selected_option == "О приложении":
        st.markdown("<h1>О приложении</h1>", unsafe_allow_html=True)
        st.write("""
        Это приложение разработано для анализа данных и предсказания наличия рака яичников.
        Используется обученная модель машинного обучения для обработки входных данных и классификации.
        """)

    # Страница "Контакты"
    elif selected_option == "Контакты":
        st.markdown("<h1>Контакты</h1>", unsafe_allow_html=True)
        st.write("""
        Если у вас есть вопросы, свяжитесь с нами:
        - Email: example@healthcare.com
        - Телефон: +7 (123) 456-78-90
        """)

if __name__ == "__main__":
    main()
