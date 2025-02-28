import os
import streamlit as st
import jwt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import time

# Set page configuration
st.set_page_config(
    page_title="Биоуголь калькулятор и консультант",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "Приложение для расчета свойств почвы с биоуглем и интерактивного общения"}
)

# Custom CSS for better styling
def set_css():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: linear-gradient(rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0.8)), 
                              url(https://p1.zoon.ru/preview/UMTUl9g9WDauuEwF2o0CkQ/2400x1500x75/1/f/8/original_57dd9ac840c088373b94a409_5a0966316a8f1.jpg);
            background-size: cover;
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        }
        
        h1, h2, h3 {
            text-align: center;
            font-weight: 600;
            margin-bottom: 1.5rem;
        }
        
        h1 { 
            color: #2c7c31; 
            font-size: 2.5rem;
            padding: 1rem 0;
            border-bottom: 2px solid #2c7c31;
        }
        
        h2 { 
            color: #0e6655; 
            font-size: 2rem;
        }
        
        h3 { 
            color: #34495e; 
            font-size: 1.5rem;
        }
        
        .stButton > button {
            background-color: #2c7c31 !important;
            color: white !important;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            padding: 0.6em 1.2em;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton > button:hover {
            background-color: #218838 !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        .css-1d391kg, .css-12ttj6m {
            background-color: rgba(255, 255, 255, 0.92) !important;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            margin-bottom: 1rem;
        }
        
        .st-bq {
            background-color: rgba(44, 124, 49, 0.1);
            border-left: 5px solid #2c7c31;
            padding: 1rem;
            border-radius: 5px;
        }
        
        .chat-message {
            padding: 1rem;
            border-radius: 12px;
            margin-bottom: 10px;
            display: flex;
            flex-direction: column;
        }
        
        .user-message {
            background-color: #dcf8c6;
            border-top-right-radius: 0;
            align-self: flex-end;
            margin-left: 20%;
        }
        
        .bot-message {
            background-color: #f2f2f2;
            border-top-left-radius: 0;
            align-self: flex-start;
            margin-right: 20%;
        }
        
        /* Стили для графиков и результатов */
        .chart-container {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .result-card {
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
            border-left: 5px solid #2c7c31;
            transition: all 0.3s ease;
        }
        
        .result-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        }
        
        .sidebar .sidebar-content {
            background: rgba(255, 255, 255, 0.95);
        }
        
        .css-1c9n29x { 
            font-weight: 600; 
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 4rem;
            white-space: pre-wrap;
            background-color: rgba(44, 124, 49, 0.05);
            border-radius: 5px 5px 0 0;
            padding: 1rem 2rem;
            font-weight: 600;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: rgba(44, 124, 49, 0.1);
            border-bottom: 4px solid #2c7c31;
        }
        
        /* Progress bar */
        .stProgress > div > div > div > div {
            background-color: #2c7c31;
        }
        
        /* Select boxes */
        .stSelectbox {
            border-radius: 8px;
        }
        
        /* Login form fields */
        div[data-testid="stForm"] {
            background-color: rgba(255, 255, 255, 0.9);
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            max-width: 500px;
            margin: 0 auto;
        }
        
        /* Metrics */
        div[data-testid="stMetric"] {
            background-color: rgba(255, 255, 255, 0.8);
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            border-left: 4px solid #2c7c31;
        }
        
        .metric-label {
            font-weight: 600;
            color: #2c7c31;
        }
        
        .metric-value {
            font-size: 1.8rem;
            color: #333;
        }
        
        </style>
        """,
        unsafe_allow_html=True
    )

set_css()

# Authentication configuration
SECRET_KEY = "7c34571a98b4d2f6e8c1a9d5b3f7e2c4a8d5b9e2f6c3a7b4d8e5f1c9a6b3d5e"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

USERS_DB = {
    "admin": {
        "username": "admin",
        "hashed_password": "hashed_admin_password"
    },
    "user": {
        "username": "user",
        "hashed_password": "hashed_user_password"
    }
}

# Biochar data with coefficients
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
    },
    'Чернозем': {
        'Скорлупа кедрового ореха': {
            600: {
                'Теплопроводность, λ Вт/(м∙К)': {
                    'a': 5.8030501558E-10,
                    'b': -3.5017669371E-07,
                    'c': -0.0001952355,
                    'd': 0
                },
                'Удельное тепловое сопротивление, R (К∙м)/Вт': {
                    'a': 4.9336168852E-07,
                    'b': -3.2725112350E-04,
                    'c': 0.0192171741,
                    'd': 0
                },
                'Объемная теплоёмкость, Cv МДж/(м³∙К)': {
                    'a': -1.0456373835E-07,
                    'b': 7.6649215602E-05,
                    'c': -0.0028040359,
                    'd': 0
                },
                'Температуропроводность, а мм²/с': {
                    'a': -1.2161903760E-08,
                    'b': 7.8998094542E-06,
                    'c': -0.0002475819,
                    'd': 0
                }
            }
        }
    }
}

# Extended knowledge base for chatbot
CHATBOT_KB = {
    "биоуголь": "Биоуголь является пористым углеродистым материалом, получаемым при пиролизе биомассы в условиях ограниченного доступа кислорода. Он обладает высокой пористостью и удельной поверхностью, что улучшает свойства почвы",
    
    "пиролиз": "Пиролиз представляет собой термический процесс разложения органических соединений без доступа кислорода при температурах от 300 до 900°C. Продуктами пиролиза являются твердый биоуголь, био-масло и синтез-газ",
    
    "почва": "Добавление биоугля способствует улучшению физических, химических и биологических свойств почвы. Он повышает водоудерживающую способность, аэрацию, улучшает структуру и способствует развитию полезных микроорганизмов",
    
    "теплопроводность": "Этот параметр характеризует способность материала проводить тепло. При внесении биоугля в почву теплопроводность обычно снижается, что помогает защитить корневую систему растений от перегрева",
    
    "удельное тепловое сопротивление": "Показатель, обратный теплопроводности. При увеличении этого параметра почва лучше сохраняет тепло. Биоуголь способствует повышению теплового сопротивления",
    
    "объемная теплоёмкость": "Показывает, сколько тепловой энергии способна накопить единица объема почвы при изменении температуры на один градус. Биоуголь может существенно влиять на этот параметр за счет своей пористой структуры",
    
    "температуропроводность": "Характеризует скорость изменения температуры в материале. Низкая температуропроводность означает, что почва медленно нагревается и охлаждается, создавая более стабильные условия для роста растений",
    
    "доза": "Оптимальная доза внесения биоугля зависит от типа почвы, культуры и целей применения. Обычно для полевых культур рекомендуют 0.5-5 т/га, для овощных и садовых культур до 20 т/га. Слишком высокая доза может приводить к иммобилизации питательных веществ",
    
    "применение": "Биоуголь применяют для улучшения почвы, секвестрации углерода, очистки воды и воздуха, а также в качестве кормовой добавки. В сельском хозяйстве его используют для повышения урожайности",
    
    "преимущества": "Биоуголь улучшает структуру почвы, повышает водоудерживающую способность, снижает вымывание питательных веществ, способствует секвестрации углерода и восстанавливает деградированные почвы",
    
    "исследования": "Исследования биоугля начались с изучения почв Terra Preta в Амазонии. Последующие работы выявили положительное влияние биоугля на физические и биологические свойства почвы, а современные исследования продолжают расширять знания о его применении",
    
    "экология": "Биоуголь считается экологически чистой технологией, так как позволяет секвестрировать углерод в течение длительного времени и помогает утилизировать органические отходы",
    
    "сырье": "Для производства биоугля используют древесину, сельскохозяйственные отходы, навоз и другие органические материалы. Его свойства зависят от типа сырья и условий пиролиза",
    
    "эффективность": "Эффективность применения биоугля зависит от типа почвы, климата, культуры, дозы внесения и характеристик самого биоугля. Наиболее заметный эффект наблюдается на бедных и деградированных почвах",
    
    "активация": "Процесс повышения реакционной способности биоугля, который может включать физическую обработку паром или CO2, а также химическую обработку кислотами или щелочами",
    
    "компост": "Компостирование биоугля с органическими отходами позволяет насытить его питательными веществами и микроорганизмами, что усиливает его положительное влияние на почву",
    
    "загрязнения": "Биоуголь способен адсорбировать тяжелые металлы, пестициды и другие загрязнители, что делает его полезным для очистки почвы и воды",
    
    "структура": "Биоуголь характеризуется высокой пористостью с порами различного размера, что обеспечивает большую удельную поверхность и улучшает адсорбционные свойства",
    
    "стандарты": "Существуют международные стандарты качества биоугля, такие как сертификаты European Biochar Certificate и International Biochar Initiative, которые определяют минимальные требования к продукту",
    
    "технологии": "Современные технологии производства биоугля включают реторты, реакторы с псевдоожиженным слоем, вращающиеся печи, микроволновые системы и другие установки. Выбор технологии зависит от масштаба производства и требуемых характеристик",
    
    "история": "Биоуголь применяли для улучшения почвы еще в древние времена, примером является феномен Terra Preta в Амазонии, созданный коренными жителями более 2500 лет назад. Современные исследования начались в 1980-х годах",
    
    "расчет": "Расчет параметров почвы с биоуглем выполняется по полиномиальным уравнениям вида a*x³ + b*x² + c*x + d, где коэффициенты зависят от типа почвы, биоугля и температуры пиролиза",
}

# Authentication functions
def create_access_token(data):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Calculation function
def calculate_property(coefs, dose):
    return coefs['a'] * dose**3 + coefs['b'] * dose**2 + coefs['c'] * dose + coefs['d']

# Function to generate data for plotting
def generate_plot_data(coefs, property_name, max_dose=10):
    doses = np.linspace(0, max_dose, 100)
    values = [calculate_property(coefs, dose) for dose in doses]
    return pd.DataFrame({
        'Доза биоугля (т/га)': doses,
        property_name: values
    })

# Improved chatbot function
def get_bot_response(user_input):
    text = user_input.lower()
    
    # Check for exact matches first
    for key, response in CHATBOT_KB.items():
        if key.lower() in text:
            return response
    
    # Fallback responses for more general questions
    if "привет" in text or "здравствуй" in text:
        return "Здравствуйте! Я ваш ассистент по вопросам биоугля и его применения. Чем могу помочь?"
    
    if "как" in text and ("рассчитать" in text or "вычислить" in text):
        return "Для расчета свойств почвы с биоуглем используйте калькулятор. Выберите тип почвы, биоугля, температуру пиролиза и введите дозу внесения"
    
    if "спасибо" in text:
        return "Всегда рад помочь! Если возникнут другие вопросы о биоугле, обращайтесь"
    
    if any(word in text for word in ["польза", "полезно", "эффект", "влияние"]):
        return "Биоуголь оказывает множество положительных эффектов, улучшая структуру почвы, повышая водоудерживающую способность, способствуя развитию полезных микроорганизмов и снижая вымывание питательных веществ"
    
    # Advanced semantic matching
    if any(phrase in text for phrase in ["произвести", "сделать", "получить"]):
        return "Биоуголь производится путем пиролиза биомассы при нагреве без доступа кислорода при температурах от 300 до 900°C. Применяются различные технологии производства"
    
    if any(phrase in text for phrase in ["внести", "применить", "добавить"]):
        return "Биоуголь вносят в почву до начала посевных работ, предварительно смешав с компостом или активировав иными способами. Рекомендуемые дозы зависят от типа почвы и культуры"
    
    # Default response
    return "Извините, информация по данному запросу отсутствует. Попробуйте сформулировать вопрос иначе или уточнить интересующую тему"

# Login view with improved UI
def login_view():
    st.markdown("<h2 style='text-align: center;'>Авторизация в системе</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            st.markdown("<p style='text-align: center; color: #2c7c31; font-weight: 600; font-size: 1.2rem;'>Введите данные для входа</p>", unsafe_allow_html=True)
            username = st.text_input("Имя пользователя", key="login_username")
            password = st.text_input("Пароль", type="password", key="login_password")
            submit = st.form_submit_button("Войти в систему")
            
            if submit:
                user = USERS_DB.get(username)
                if user and (password == "admin" or password == "user"):
                    with st.spinner("Вход в систему..."):
                        time.sleep(0.8)
                        token = create_access_token({"sub": user["username"]})
                        st.session_state.token = token
                        st.session_state.username = username
                        st.success("Вход выполнен успешно! Перенаправление")
                        time.sleep(1)
                        st.rerun()
                else:
                    st.error("Неверное имя пользователя или пароль")
        
        st.markdown("""
        <div style="text-align: center; margin-top: 20px; padding: 15px; background-color: rgba(44, 124, 49, 0.1); border-radius: 8px;">
            <p style="margin-bottom: 5px; color: #555;">Тестовые учетные данные</p>
            <p><strong>Пользователь</strong> admin или user</p>
            <p><strong>Пароль</strong> admin или user</p>
        </div>
        """, unsafe_allow_html=True)

# Enhanced calculator view with visualizations
def calc_view():
    st.markdown("<h2>Калькулятор свойств почвы с биоуглем</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.subheader("Параметры расчета")
        
        soil_types = list(BIOCHAR_DATA.keys())
        selected_soil = st.selectbox("Выберите тип почвы", soil_types)
        
        biochar_types = list(BIOCHAR_DATA[selected_soil].keys())
        selected_biochar = st.selectbox("Выберите тип биоугля", biochar_types)
        
        temperatures = list(BIOCHAR_DATA[selected_soil][selected_biochar].keys())
        selected_temp = st.selectbox("Выберите температуру пиролиза (°C)", temperatures)
        
        dose = st.slider("Введите дозу биоугля (т/га)", 0.0, 10.0, 1.0, 0.1)
        
        st.markdown("""
        <div style="background-color: rgba(44, 124, 49, 0.1); padding: 10px; border-radius: 5px; margin-top: 15px;">
            <p style="font-size: 0.9rem; margin-bottom: 0;">
                Примечание. Дозы внесения биоугля обычно составляют от 0.5 до 5 т/га для полевых культур и до 20 т/га для садовых и овощных культур
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        calculate_button = st.button("Рассчитать свойства")
        st.markdown("</div>", unsafe_allow_html=True)
    
    results_placeholder = st.empty()
    
    if calculate_button:
        with st.spinner("Выполняется расчет"):
            time.sleep(0.5)
            try:
                soil_data = BIOCHAR_DATA[selected_soil][selected_biochar][selected_temp]
                results = {}
                
                for prop, coefs in soil_data.items():
                    results[prop] = calculate_property(coefs, dose)
                
                plot_data = {}
                for prop, coefs in soil_data.items():
                    plot_data[prop] = generate_plot_data(coefs, prop)
                
                with results_placeholder.container():
                    st.markdown("<h3>Результаты расчета</h3>", unsafe_allow_html=True)
                    
                    properties = list(results.keys())
                    half = len(properties) // 2 + len(properties) % 2
                    
                    row1_props = properties[:half]
                    row2_props = properties[half:]
                    
                    cols = st.columns(len(row1_props))
                    for i, prop in enumerate(row1_props):
                        display_name = prop.split(',')[0]
                        cols[i].markdown(f"<div class='metric-label'>{display_name}</div>", unsafe_allow_html=True)
                        cols[i].markdown(f"<div class='metric-value'>{results[prop]:.5f}</div>", unsafe_allow_html=True)
                    
                    if row2_props:
                        cols = st.columns(len(row2_props))
                        for i, prop in enumerate(row2_props):
                            display_name = prop.split(',')[0]
                            cols[i].markdown(f"<div class='metric-label'>{display_name}</div>", unsafe_allow_html=True)
                            cols[i].markdown(f"<div class='metric-value'>{results[prop]:.5f}</div>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    st.markdown("<h3>Визуализация результатов</h3>", unsafe_allow_html=True)
                    
                    tabs = st.tabs(["Графики зависимости", "Сравнительная диаграмма", "Тепловая карта"])
                    
                    with tabs[0]:
                        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                        property_to_show = st.selectbox("Выберите свойство для отображения", list(results.keys()))
                        
                        selected_data = plot_data[property_to_show]
                        prop_display_name = property_to_show.split(',')[0]
                        
                        fig = px.line(
                            selected_data, 
                            x="Доза биоугля (т/га)", 
                            y=property_to_show,
                            title=f"Зависимость {prop_display_name} от дозы биоугля",
                            markers=True
                        )
                        
                        fig.update_layout(
                            xaxis_title="Доза биоугля (т/га)",
                            yaxis_title=property_to_show,
                            title_font=dict(size=18, color="#2c7c31"),
                            plot_bgcolor="rgba(255,255,255,0.9)",
                            paper_bgcolor="rgba(255,255,255,0)",
                            hovermode="x unified",
                            height=500
                        )
                        
                        fig.update_traces(
                            line=dict(color="#2c7c31", width=3),
                            marker=dict(size=8, color="#2c7c31")
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        current_val = results[property_to_show]
                        st.markdown(f"""
                        <div style='background-color: rgba(44, 124, 49, 0.1); padding: 15px; border-radius: 8px; margin-top: 10px;'>
                            <p style='font-weight: 600; margin-bottom: 5px;'>Текущее значение при дозе {dose} т/га</p>
                            <p style='font-size: 1.2rem; color: #2c7c31;'>{current_val:.5f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with tabs[1]:
                        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                        comparison_data = {
                            'Свойство': [],
                            'Значение': [],
                            'Единица измерения': []
                        }
                        
                        for prop, value in results.items():
                            prop_name = prop.split(',')[0]
                            unit = prop.split(',')[1].strip() if ',' in prop else ''
                            comparison_data['Свойство'].append(prop_name)
                            comparison_data['Значение'].append(value)
                            comparison_data['Единица измерения'].append(unit)
                        
                        df_comparison = pd.DataFrame(comparison_data)
                        
                        fig_bar = px.bar(
                            df_comparison,
                            x='Свойство',
                            y='Значение',
                            color='Свойство',
                            title=f"Сравнение свойств почвы при дозе биоугля {dose} т/га",
                            text='Значение',
                            color_discrete_sequence=px.colors.qualitative.G10
                        )
                        
                        fig_bar.update_layout(
                            showlegend=False,
                            plot_bgcolor="rgba(255,255,255,0.9)",
                            paper_bgcolor="rgba(255,255,255,0)",
                            xaxis_title="",
                            yaxis_title="Значение",
                            height=500
                        )
                        
                        fig_bar.update_traces(
                            texttemplate='%{y:.5f}',
                            textposition='outside'
                        )
                        
                        st.plotly_chart(fig_bar, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with tabs[2]:
                        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                        doses_range = np.linspace(0, 10, 11)
                        heatmap_data = []
                        
                        for dose_val in doses_range:
                            row_data = {'Доза': f"{dose_val:.1f} т/га"}
                            for prop, coefs in soil_data.items():
                                prop_name = prop.split(',')[0]
                                row_data[prop_name] = calculate_property(coefs, dose_val)
                            heatmap_data.append(row_data)
                        
                        df_heatmap = pd.DataFrame(heatmap_data)
                        df_heatmap = df_heatmap.set_index('Доза')
                        
                        df_norm = (df_heatmap - df_heatmap.min()) / (df_heatmap.max() - df_heatmap.min())
                        
                        fig_heatmap = px.imshow(
                            df_norm.transpose(),
                            labels=dict(x="Доза биоугля", y="Свойство", color="Относительное значение"),
                            x=df_heatmap.index,
                            y=df_norm.columns,
                            color_continuous_scale="Viridis",
                            title="Тепловая карта изменения свойств почвы в зависимости от дозы биоугля"
                        )
                        
                        fig_heatmap.update_layout(
                            height=500,
                            plot_bgcolor="rgba(255,255,255,0.9)",
                            paper_bgcolor="rgba(255,255,255,0)"
                        )
                        
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                        
                        st.markdown("""
                        <div style='background-color: rgba(44, 124, 49, 0.1); padding: 15px; border-radius: 8px; margin-top: 10px;'>
                            <p style='font-weight: 600; margin-bottom: 5px;'>О тепловой карте</p>
                            <p>Цветовая интенсивность отражает относительное изменение значений свойств почвы при различных дозах биоугля. Более темный цвет соответствует более высоким значениям</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.markdown("<h3>Рекомендации по применению</h3>", unsafe_allow_html=True)
                    
                    rec_text = f"""
                    <div class='result-card'>
                        <h4>Рекомендации для {selected_soil} при использовании биоугля из {selected_biochar.lower()}</h4>
                        <p>На основании расчетов при дозе <strong>{dose} т/га</strong> биоугля, полученного при температуре <strong>{selected_temp}°C</strong></p>
                        <ul>
                    """
                    
                    tc_value = results.get('Теплопроводность, λ Вт/(м∙К)', 0)
                    tr_value = results.get('Удельное тепловое сопротивление, R (К∙м)/Вт', 0)
                    hc_value = results.get('Объемная теплоёмкость, Cv МДж/(м³∙К)', 0)
                    
                    if tc_value < 0.3:
                        rec_text += "<li>Низкая теплопроводность помогает защитить корневую систему растений от перегрева в жаркий период</li>"
                    elif tc_value > 0.7:
                        rec_text += "<li>Высокая теплопроводность способствует быстрому прогреву почвы весной, что полезно для ранних посадок</li>"
                    
                    if tr_value > 3:
                        rec_text += "<li>Высокое тепловое сопротивление обеспечивает хорошую теплоизоляцию, полезную для защиты растений от резких перепадов температур</li>"
                    
                    if hc_value > 1.5:
                        rec_text += "<li>Повышенная объемная теплоёмкость позволяет почве аккумулировать больше тепла, создавая благоприятные условия в прохладный период</li>"
                    
                    rec_text += """
                        <li>Перед внесением биоугля рекомендуется провести его активацию (например, компостирование или замачивание в питательном растворе)</li>
                        <li>Более эффективным оказывается сочетание биоугля с органическими и минеральными удобрениями</li>
                        </ul>
                    </div>
                    """
                    
                    st.markdown(rec_text, unsafe_allow_html=True)
            
            except KeyError:
                st.error("Ошибка. Введены неверные параметры или данные для данной комбинации отсутствуют")
            except Exception as e:
                st.error(f"Произошла ошибка при расчете. {str(e)}")

# Enhanced chatbot view
def chat_view():
    st.markdown("<h2>Консультант по биоуглю</h2>", unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Здравствуйте! Я консультант по вопросам биоугля. Задайте интересующий вопрос о его свойствах или применении"
        })
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <div><strong>Вы</strong> {message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <div><strong>Консультант</strong> {message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_input = st.text_input(
                "Введите ваш вопрос",
                key="chat_input",
                placeholder="Например, что такое биоуголь или какие его преимущества",
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.button("Отправить", use_container_width=True)
    
    if send_button and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("Ассистент думает"):
            time.sleep(0.5)
            bot_response = get_bot_response(user_input)
        
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        st.rerun()
    
    st.markdown("<div style='margin-top: 20px;'><strong>Популярные вопросы</strong></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("❓ Что такое биоуголь?"):
            st.session_state.messages.append({"role": "user", "content": "Что такое биоуголь?"})
            bot_response = get_bot_response("биоуголь")
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            st.rerun()
    
    with col2:
        if st.button("🌱 Как применять биоуголь?"):
            st.session_state.messages.append({"role": "user", "content": "Как применять биоуголь?"})
            bot_response = get_bot_response("применение")
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            st.rerun()
    
    with col3:
        if st.button("📊 Оптимальная доза биоугля"):
            st.session_state.messages.append({"role": "user", "content": "Какая оптимальная доза биоугля?"})
            bot_response = get_bot_response("доза")
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            st.rerun()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Очистить историю чата", type="secondary"):
            st.session_state.messages = []
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "История чата очищена. Задайте новый вопрос о биоугле"
            })
            st.rerun()
    
    with col2:
        with st.expander("ℹ️ О возможностях консультанта"):
            st.markdown("""
            <h3>Темы для обсуждения</h3>
            <ul>
                <li>Общие сведения о биоугле и технологии его получения</li>
                <li>Свойства биоугля и влияние на почву</li>
                <li>Оптимальные дозы и способы применения</li>
                <li>Экологические аспекты использования</li>
                <li>Сырье и методы производства</li>
                <li>История применения и современные исследования</li>
            </ul>
            """, unsafe_allow_html=True)

# Info view for detailed information
def info_view():
    st.markdown("<h2>Информация о биоугле</h2>", unsafe_allow_html=True)
    
    tabs = st.tabs(["Что такое биоуголь", "Преимущества", "Производство", "Применение", "Исследования"])
    
    with tabs[0]:
        st.markdown("""
        <div class='result-card'>
            <h3>Что такое биоуголь</h3>
            <p>
                Биоуголь представляет собой твердый пористый углеродистый материал, получаемый путем пиролиза биомассы без доступа кислорода при температуре от 300°C до 900°C. Он создается для улучшения свойств почвы и длительного связывания углерода.
            </p>
            <p>
                Физико-химические свойства биоугля зависят от исходного сырья, температуры пиролиза, скорости нагрева и времени выдержки. Его высокая пористость и большая удельная поверхность делают его эффективным для улучшения структуры почвы.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.image("https://p1.zoon.ru/preview/UMTUl9g9WDauuEwF2o0CkQ/2400x1500x75/1/f/8/original_57dd9ac840c088373b94a409_5a0966316a8f1.jpg", 
                 caption="Пример биоугля из различного сырья")
    
    with tabs[1]:
        st.markdown("""
        <div class='result-card'>
            <h3>Преимущества применения биоугля</h3>
            <h4>Улучшение свойств почвы</h4>
            <ul>
                <li>Повышает водоудерживающую способность</li>
                <li>Улучшает структуру и аэрацию</li>
                <li>Снижает плотность и стабилизирует pH</li>
                <li>Уменьшает вымывание питательных веществ</li>
            </ul>
            <h4>Экологические преимущества</h4>
            <ul>
                <li>Секвестрирует углерод в течение длительного времени</li>
                <li>Снижает выбросы парниковых газов</li>
                <li>Помогает утилизировать органические отходы</li>
                <li>Защищает грунтовые воды от загрязнения</li>
            </ul>
            <h4>Агрономические и экономические преимущества</h4>
            <ul>
                <li>Повышает урожайность</li>
                <li>Снижает затраты на удобрения и полив</li>
                <li>Улучшает условия для полезных микроорганизмов</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        data = pd.DataFrame({
            'Категория': ['Урожайность', 'Водоудержание', 'pH почвы', 'Микробиологическая активность', 'Плотность почвы'],
            'Улучшение (%)': [25, 45, 15, 60, -20]
        })
        
        fig = px.bar(data, x='Категория', y='Улучшение (%)',
                     title='Средний эффект применения биоугля по литературным данным',
                     color='Улучшение (%)', 
                     color_continuous_scale=px.colors.sequential.Greens)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.markdown("""
        <div class='result-card'>
            <h3>Производство биоугля</h3>
            <h4>Основные технологии производства</h4>
            <ul>
                <li>Медленный пиролиз при температуре 400-500°C с выдержкой от часов до дней</li>
                <li>Быстрый пиролиз при температуре 400-600°C с выдержкой в секундах или минутах</li>
                <li>Газификация с ограниченным доступом кислорода при температуре 700-900°C</li>
                <li>Гидротермальная карбонизация при температуре 180-250°C под высоким давлением</li>
            </ul>
            <h4>Типы установок и сырье</h4>
            <ul>
                <li>Реторты, печи периодического и непрерывного действия</li>
                <li>Реакторы с псевдоожиженным слоем, вращающиеся печи, шнековые реакторы</li>
                <li>Древесина, сельскохозяйственные отходы, навоз, осадки сточных вод</li>
            </ul>
            <p>Качественный биоуголь должен соответствовать международным стандартам, таким как сертификаты European Biochar Certificate и стандарты International Biochar Initiative</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <h4 style="text-align: center;">Схема процесса производства биоугля</h4>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[3]:
        st.markdown("""
        <div class='result-card'>
            <h3>Применение биоугля</h3>
            <h4>В сельском хозяйстве</h4>
            <ul>
                <li>Внесение для улучшения свойств почвы</li>
                <li>Смешивание с компостом и органическими удобрениями</li>
                <li>Использование в системах беспочвенного выращивания</li>
                <li>Добавление в корма для животных</li>
            </ul>
            <h4>В экологии и ремедиации</h4>
            <ul>
                <li>Очистка загрязненных почв и вод</li>
                <li>Фильтрация сточных вод</li>
                <li>Ремедиация нарушенных территорий</li>
            </ul>
            <h4>Другие применения</h4>
            <ul>
                <li>Использование в строительных материалах</li>
                <li>Применение в косметике и медицине</li>
                <li>Использование в производстве суперконденсаторов и покрытий</li>
            </ul>
            <h4>Рекомендуемые дозы внесения</h4>
            <table style="width:100%; border-collapse: collapse; margin-top: 15px;">
                <tr style="background-color: rgba(44, 124, 49, 0.1);">
                    <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Тип почвы или применения</th>
                    <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Рекомендуемая доза (т/га)</th>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;">Полевые культуры</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">0.5 - 5</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;">Овощные культуры</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">5 - 20</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;">Садовые и ягодные культуры</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">2 - 10</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;">Городское озеленение</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">10 - 30</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;">Восстановление нарушенных земель</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">20 - 50</td>
                </tr>
            </table>
            <h4>Способы внесения</h4>
            <ul>
                <li>Равномерное распределение по поверхности с последующим заделыванием</li>
                <li>Внесение в посевные борозды или лунки</li>
                <li>Внесение с поливной водой в виде суспензии мелкодисперсного биоугля</li>
                <li>Предварительное смешивание с компостом или удобрениями</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[4]:
        st.markdown("""
        <div class='result-card'>
            <h3>Исследования биоугля</h3>
            <h4>История исследований</h4>
            <p>
                Научные исследования биоугля начались с изучения феномена Terra Preta в амазонских почвах. Результаты показали положительное влияние биоугля на физико-химические свойства почвы, увеличение урожайности и усиление микробиологической активности. Последующие работы расширили понимание его применения в сельском хозяйстве, а современные исследования направлены на оптимизацию условий производства и использования для различных типов почв.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Main navigation logic
if "token" not in st.session_state:
    login_view()
else:
    st.sidebar.markdown("<h2>Меню навигации</h2>", unsafe_allow_html=True)
    selected_page = st.sidebar.radio("Выберите раздел", ["Калькулятор", "Консультант", "Информация"])
    
    if st.sidebar.button("Выйти из системы"):
        st.session_state.clear()
        st.experimental_rerun()
    
    if selected_page == "Калькулятор":
        calc_view()
    elif selected_page == "Консультант":
        chat_view()
    elif selected_page == "Информация":
        info_view()
