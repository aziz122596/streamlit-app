import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch import nn
import joblib

# Установка конфигурации страницы
st.set_page_config(
    page_title="Диагностика",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Приложение для анализа данных и классификации пациентов."
    }
)

# Класс модели VAE
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2)  # mu и logvar
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu, logvar = torch.chunk(encoded, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar

# Загрузка моделей и инструментов предобработки
@st.cache_resource
def load_model_and_tools():
    input_dim = joblib.load('input_dim2.joblib')
    imputer = joblib.load('imputer2.joblib')
    scaler = joblib.load('scaler2.joblib')
    expected_columns = joblib.load('expected_columns.joblib')  # Список ожидаемых столбцов

    vae = VAE(input_dim, latent_dim=16)
    vae.load_state_dict(torch.load('vae_model4.pth', map_location=torch.device('cpu'), weights_only=True))
    vae.eval()

    classifier = joblib.load('classifier1.joblib')
    return vae, classifier, imputer, scaler, expected_columns

# Классификация данных
def classify_data(latent_vectors, classifier):
    predictions = classifier.predict(latent_vectors)
    return ["Здоровый" if pred == 0 else "Больной" for pred in predictions]

# Главная функция приложения
def main():
    st.markdown("<h1 style='text-align: center; color: #4A90E2;'>Диагностика</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Классификация пациентов на основе VAE</h3>", unsafe_allow_html=True)
    st.markdown("---")

    # Боковая панель
    with st.sidebar:
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
            try:
                # Загрузка и отображение данных
                data = pd.read_csv(uploaded_file)
                st.write("Загруженные данные:", data.head())

                # Загрузка моделей и инструментов
                vae, classifier, imputer, scaler, expected_columns = load_model_and_tools()

                # Приведение данных к ожидаемой структуре
                data = data.reindex(columns=expected_columns, fill_value=0)

                # Обработка данных
                data = data.replace(r'[^0-9.-]', np.nan, regex=True).astype(float)
                data_imputed = imputer.transform(data)
                data_scaled = scaler.transform(data_imputed)

                # Извлечение латентных представлений
                data_tensor = torch.tensor(data_scaled, dtype=torch.float32)
                _, mu, _ = vae(data_tensor)
                latent_vectors = mu.numpy()

                # Классификация данных
                predictions = classify_data(latent_vectors, classifier)
                data['Результат'] = predictions

                # Отображение результатов
                st.success("Анализ завершён! Результаты отображены ниже:")
                st.dataframe(data)

                # Скачивание результатов
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Скачать результаты в формате CSV",
                    data=csv,
                    file_name='classification_results.csv',
                    mime='text/csv',
                )

            except Exception as e:
                st.error(f"Ошибка при обработке файла: {e}")

    # Страница "О приложении"
    elif selected_option == "О приложении":
        st.markdown("<h1>О приложении</h1>", unsafe_allow_html=True)
        st.write("""
            **Диагностика** — это приложение, разработанное для анализа медицинских данных 
            с использованием VAE. Оно позволяет классифицировать данные пациентов на "здоровых" и "больных".
        """)

    # Страница "Контакты"
    elif selected_option == "Контакты":
        st.markdown("<h1>Контакты</h1>", unsafe_allow_html=True)
        st.write("""
            Если у вас есть вопросы или предложения, свяжитесь с нами:
            - **Email:** support@diagnostic.com
            - **Телефон:** +7 (495) 123-45-67
        """)

# Запуск приложения
if __name__ == "__main__":
    main()
