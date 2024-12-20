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

# Класс модели VAE
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc3(z))
        h = torch.relu(self.fc4(h))
        return torch.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, x.shape[1]))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Загрузка модели
@st.cache_resource
def load_model():
    input_dim = joblib.load('input_dim.joblib')
    latent_dim = 16
    model = VAE(input_dim, latent_dim)
    model.load_state_dict(torch.load('vae_model2.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Загрузка иммунизатора и масштабатора
@st.cache_resource
def load_preprocessing_objects():
    imputer = joblib.load('imputer.joblib')
    scaler = joblib.load('scaler.joblib')
    return imputer, scaler

# Функция для получения латентных представлений данных
def get_latent_representation(data, model):
    with torch.no_grad():
        mu, _ = model.encode(torch.tensor(data, dtype=torch.float32))
        return mu.numpy()

# Главная функция приложения
def main():
    st.markdown("<h1 style='text-align: center; color: #4A90E2;'>Диагностика рака яичников</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Приложение для предсказания на основе машинного обучения</h3>", unsafe_allow_html=True)
    st.markdown("---")

    # Боковая навигация
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
                data = pd.read_csv(uploaded_file)
                st.write("Загруженные данные:")
                st.dataframe(data.head())

                # Проверка соответствия столбцов
                model = load_model()
                imputer, scaler = load_preprocessing_objects()
                
                # Загрузка имен признаков
                expected_columns = joblib.load('feature_names_in.joblib')
                
                # Проверка наличия всех необходимых столбцов
                missing_columns = set(expected_columns) - set(data.columns)
                additional_columns = set(data.columns) - set(expected_columns)
                
                if missing_columns:
                    st.error(f"В загруженных данных отсутствуют следующие необходимые столбцы: {', '.join(missing_columns)}")
                    st.stop()
                
                if additional_columns:
                    st.warning(f"В загруженных данных присутствуют дополнительные столбцы, которые будут проигнорированы: {', '.join(additional_columns)}")
                    # Удаление дополнительных столбцов
                    data = data[expected_columns]

                # Предобработка данных
                X = data.copy()
                X = X.replace(r'[^0-9.-]', np.nan, regex=True).astype(float)

                # Заполнение пропущенных значений
                if X.isnull().sum().sum() > 0:
                    st.warning("В данных есть пропущенные значения. Они будут заполнены средним значением.")
                X_imputed = imputer.transform(X)

                # Масштабирование признаков
                X_scaled = scaler.transform(X_imputed)

                # Проверка на соответствие размерностей
                if X_scaled.shape[1] != model.fc1.in_features:
                    st.error("Размерность входных данных не соответствует ожидаемой моделью.")
                    st.stop()

                # Проверяем, есть ли данные для анализа
                if st.button("Провести анализ"):
                    try:
                        # Получение латентных представлений
                        latent_vectors = get_latent_representation(X_scaled, model)

                        # Добавляем латентные представления в таблицу
                        for i in range(latent_vectors.shape[1]):
                            data[f'Latent_{i+1}'] = latent_vectors[:, i]

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

            except Exception as e:
                st.error(f"Ошибка при обработке файла: {e}")

    # Страница "О приложении"
    elif selected_option == "О приложении":
        st.markdown("<h1>О приложении</h1>", unsafe_allow_html=True)
        st.write("""
            **Диагностика рака яичников** — это приложение, разработанное для анализа биохимических данных пациентов с целью выявления признаков рака яичников. 
            Приложение использует вариационный автокодировщик (VAE) для извлечения латентных представлений из данных, что может помочь в обнаружении скрытых паттернов и аномалий.
            
            **Как это работает:**
            - Пользователь загружает CSV-файл с биохимическими данными.
            - Данные проходят предобработку: заполнение пропущенных значений и масштабирование.
            - Обученная модель VAE извлекает латентные признаки из данных.
            - Результаты отображаются в таблице и могут быть скачаны для дальнейшего анализа.
            
            **Примечание:**
            Данное приложение предназначено для исследовательских целей и не является медицинским инструментом диагностики. Для получения профессиональной медицинской консультации обратитесь к специалисту.
        """)

    # Страница "Контакты"
    elif selected_option == "Контакты":
        st.markdown("<h1>Контакты</h1>", unsafe_allow_html=True)
        st.write("""
            Если у вас есть вопросы или предложения, свяжитесь с нами:
            - **Email:** support@ovariancancerapp.com
            - **Телефон:** +7 (495) 123-45-67
            - **Адрес:** г. Москва, ул. Примерная, д. 1
        """)

# Запуск приложения
if __name__ == "__main__":
    main()
