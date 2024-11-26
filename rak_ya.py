import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import joblib

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

# Классификатор для анализа данных
class OvarianCancerClassifier(nn.Module):
    def __init__(self, input_size):
        super(OvarianCancerClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

# Загрузка модели
@st.cache_resource
def load_model():
    input_size = joblib.load('input_size.joblib')  # Сохраняем размер входа
    model = OvarianCancerClassifier(input_size)
    model.load_state_dict(torch.load('vae_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Загрузка иммунизатора и масштабатора
@st.cache_resource
def load_preprocessing_objects():
    imputer = joblib.load('imputer.joblib')
    scaler = joblib.load('scaler.joblib')
    return imputer, scaler

# Предсказание с использованием модели
def predict(data, model):
    with torch.no_grad():
        input_tensor = torch.tensor(data, dtype=torch.float32)
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.numpy()

# Главная функция
def main():
    st.markdown("<h1 style='text-align: center; color: #4A90E2;'>Диагностика рака яичников</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Приложение для предсказания на основе машинного обучения</h3>", unsafe_allow_html=True)
    st.markdown("---")

    # Навигация по боковой панели
    with st.sidebar:
        st.image("https://vladimir.infodoctor.ru/resized/org/143/143/600z315_crop_Sadovaya-ambulatoriya_1.jpg", use_column_width=True)
        st.title("Навигация")
        selected_option = st.radio(
            "Перейти к:",
            ["Главная", "О приложении", "Контакты"]
        )

    # Главная страница
    if selected_option == "Главная":
        st.subheader("Загрузите данные для анализа")
        uploaded_file = st.file_uploader("Выберите CSV файл с данными", type=["csv"])

        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.write("Загруженные данные:")
            st.dataframe(data.head())

            # Удаление ненужных столбцов
            data = data.drop(columns=['Accession'], errors='ignore')

            # Предобработка данных
            X = data.copy()
            X = X.replace(r'[^0-9.-]', np.nan, regex=True).astype(float)

            # Загрузка иммунизатора и масштабатора
            imputer, scaler = load_preprocessing_objects()

            # Заполнение пропущенных значений
            if X.isnull().sum().sum() > 0:
                st.warning("В данных есть пропущенные значения. Они будут заполнены средним значением.")
            X_imputed = imputer.transform(X)

            # Масштабирование признаков
            X_scaled = scaler.transform(X_imputed)

            # Проверка на соответствие размерностей
            model = load_model()
            if X_scaled.shape[1] != model.fc1.in_features:
                st.error("Размерность входных данных не соответствует ожидаемой моделью.")
                return

            # Проверяем, есть ли данные для анализа
            if st.button("Провести анализ"):
                try:
                    # Прогнозирование
                    predictions = predict(X_scaled, model)

                    # Добавляем предсказания в таблицу
                    data['Результат'] = ['Рак не обнаружен' if pred == 0 else 'Рак обнаружен' for pred in predictions]
                    st.success("Анализ завершён!")
                    st.write("Результаты:")
                    st.dataframe(data)

                    # Кнопка для скачивания результатов
                    csv = data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Скачать результаты в формате CSV",
                        data=csv,
                        file_name='results.csv',
                        mime='text/csv',
                    )

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

