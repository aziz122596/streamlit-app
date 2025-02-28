import os
import streamlit as st
import jwt
from datetime import datetime, timedelta

# Конфигурация страницы и установка кастомного CSS
st.set_page_config(
    page_title="Калькулятор свойств почвы с биоуглем",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "Приложение для расчёта свойств почвы с биоуглем"}
)

def set_css():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url(https://p1.zoon.ru/preview/UMTUl9g9WDauuEwF2o0CkQ/2400x1500x75/1/f/8/original_57dd9ac840c088373b94a409_5a0966316a8f1.jpg);
            background-size: cover;
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        }
        h1, h2, h3 {
            text-align: center;
            font-weight: 600;
        }
        h1 { color: #2c7c31; }
        h3 { color: #34495e; }
        .css-1d391kg { background-color: rgba(255, 255, 255, 0.85) !important; }
        .css-12ttj6m {
            background-color: rgba(255, 255, 255, 0.8) !important;
            backdrop-filter: blur(10px);
            border-radius: 10px;
            padding: 20px;
        }
        .stButton button {
            background-color: #2c7c31 !important;
            color: white !important;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            padding: 0.5em 1em;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #218838 !important;
        }
        .css-1c9n29x { font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True
    )

set_css()

# Настройки безопасности и базы пользователей
SECRET_KEY = "7c34571a98b4d2f6e8c1a9d5b3f7e2c4a8d5b9e2f6c3a7b4d8e5f1c9a6b3d5e"  # Задайте здесь или сгенерируйте новый ключ
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

USERS_DB = {
    "admin": {
        "username": "admin",
        "hashed_password": "hashed_admin_password"  # В реальном проекте используйте надёжное хеширование
    }
}

# Данные для расчёта
BIOCHAR_DATA = {
    'Дерново-подзолистая почва': {
        'Скорлупа кедрового ореха': {
            600: {
                'Теплопроводность, λ Вт/(м∙К)': {
                    'a': 6.4030501558E-10,
                    'b': -3.9017669371E-07,
                    'c': -0.0002202355,
                    'd': 0
                },
                'Удельное тепловое сопротивление, R (К∙м)/Вт': {
                    'a': 5.5336168852E-07,
                    'b': -3.7725112350E-04,
                    'c': 0.0222171741,
                    'd': 0
                },
                'Объемная теплоёмкость, Cv МДж/(м³∙К)': {
                    'a': -1.2456373835E-07,
                    'b': 8.1649215602E-05,
                    'c': -0.0032040359,
                    'd': 0
                },
                'Температуропроводность, а мм²/с': {
                    'a': -1.4161903760E-08,
                    'b': 8.8998094542E-06,
                    'c': -0.0002875819,
                    'd': 0
                }
            }
        },
        'Помет': {
            400: {
                'Теплопроводность, λ Вт/(м∙К)': {
                    'a': -1.2003705484E-07,
                    'b': 8.0134814372E-05,
                    'c': -0.0036293444,
                    'd': 0
                },
                'Удельное тепловое сопротивление, R (К∙м)/Вт': {
                    'a': -7.1261584317E-07,
                    'b': 3.9517129660E-04,
                    'c': 0.0261195476,
                    'd': 0
                },
                'Объемная теплоёмкость, Cv МДж/(м³∙К)': {
                    'a': 2.1126725917E-07,
                    'b': -1.4845068697E-04,
                    'c': 0.0095133799,
                    'd': 0
                },
                'Температуропроводность, а мм²/с': {
                    'a': -3.5776620248E-09,
                    'b': 2.3431064427E-06,
                    'c': -0.0000685733,
                    'd': 0
                }
            }
        }
    }
}

def create_access_token(data):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def calculate_property(coefs, dose):
    return coefs['a'] * dose**3 + coefs['b'] * dose**2 + coefs['c'] * dose + coefs['d']

# Форма авторизации
def login_view():
    st.subheader("Вход в систему")
    username = st.text_input("Имя пользователя", key="login_username")
    password = st.text_input("Пароль", type="password", key="login_password")
    if st.button("Войти"):
        user = USERS_DB.get(username)
        if user and password == "admin":
            token = create_access_token({"sub": user["username"]})
            st.session_state.token = token
            st.session_state.username = username
            st.success("Вход выполнен успешно")
        else:
            st.error("Неверное имя пользователя или пароль")

# Основное окно расчёта
def main_view():
    st.sidebar.title("Меню")
    if st.sidebar.button("Выйти"):
        st.session_state.pop("token")
        st.session_state.pop("username")
        st.experimental_rerun()

    st.header("Калькулятор свойств почвы с биоуглем")
    soil_types = list(BIOCHAR_DATA.keys())
    selected_soil = st.selectbox("Выберите тип почвы", soil_types)
    biochar_types = list(BIOCHAR_DATA[selected_soil].keys())
    selected_biochar = st.selectbox("Выберите тип биоугля", biochar_types)
    temperatures = list(BIOCHAR_DATA[selected_soil][selected_biochar].keys())
    selected_temp = st.selectbox("Выберите температуру", temperatures)
    dose = st.number_input("Введите дозу", value=1.0, step=0.1)

    if st.button("Рассчитать"):
        try:
            soil_data = BIOCHAR_DATA[selected_soil][selected_biochar][selected_temp]
            results = {}
            for property_name, coefficients in soil_data.items():
                results[property_name] = calculate_property(coefficients, dose)
            st.subheader("Результаты расчёта")
            st.json(results)
        except KeyError:
            st.error("Неверные параметры ввода")

def main():
    st.title("Приложение для расчёта свойств почвы с биоуглем")
    if "token" not in st.session_state:
        login_view()
    else:
        main_view()

if __name__ == "__main__":
    main()
