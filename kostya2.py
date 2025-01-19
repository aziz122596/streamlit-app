import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Устанавливаем фон цвета хаки и выравниваем текст по центру
st.markdown(
    """
    <style>
    .stApp {
        background-color: #2F4F4F;
    }
    h1, h2, h3, h4, h5, h6, p, div, label {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def login():
    # Заголовок «Вход»
    st.markdown("<h2>Вход</h2>", unsafe_allow_html=True)

    # Разделяем на колонки, чтобы поле ввода было по центру
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("Имя пользователя")

    if 'login_button_clicked' not in st.session_state:
        st.session_state['login_button_clicked'] = False

    if st.button("Войти"):
        if username == 'admin':
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success(f"Вы успешно вошли как {username}")
            st.session_state['login_button_clicked'] = True
        else:
            st.error("Неправильное имя пользователя")

def main_app():
    st.title("🌱 Калькулятор свойств почвы с биоуглем")
    st.write(f"**Вы вошли как** {st.session_state['username']}")
    st.write("Это приложение рассчитывает изменения физических свойств почвы после внесения биоугля")

    st.sidebar.header("Параметры расчёта")
    dose_input = st.sidebar.number_input(
        "Введите дозу внесения биоугля (т/га)",
        min_value=0.0,
        value=1.0,
        step=0.5
    )

    # Данные коэффициентов
    biochar_data = {
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
                    },
                },
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
                    },
                },
            },
        },
    }

    soil_type = st.sidebar.selectbox("Выберите тип почвы", list(biochar_data.keys()))
    biochar_type = st.sidebar.selectbox("Выберите тип биоугля", list(biochar_data[soil_type].keys()))
    temperature = st.sidebar.selectbox(
        "Выберите температуру получения биоугля",
        list(biochar_data[soil_type][biochar_type].keys())
    )

    properties = biochar_data[soil_type][biochar_type][temperature]

    st.markdown("---")
    st.subheader(f"📊 Результаты расчёта при дозе внесения {dose_input} т/га")

    # Вкладки для свойств
    tabs = st.tabs(list(properties.keys()))
    for idx, (prop_name, coefs) in enumerate(properties.items()):
        with tabs[idx]:
            delta = calculate_soil_property_change(coefs, dose_input)
            st.metric(label=prop_name, value=f"{delta:.6f}")
            st.markdown("### График изменения")

            x_values = np.linspace(0, 100, 100)
            y_values = [calculate_soil_property_change(coefs, x) for x in x_values]
            fig, ax = plt.subplots()
            ax.plot(x_values, y_values, label=prop_name, color='green')
            ax.set_xlabel('Доза биоугля (т/га)')
            ax.set_ylabel('Изменение свойства')
            ax.set_title(prop_name)
            ax.legend()
            st.pyplot(fig)

    # Выход
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Выйти"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = ''

def calculate_soil_property_change(coefs, x):
    a = coefs['a']
    b = coefs['b']
    c = coefs['c']
    d = coefs['d']
    return a * x**3 + b * x**2 + c * x + d

def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['username'] = ''

    if st.session_state['logged_in']:
        main_app()
    else:
        login()

if __name__ == "__main__":
    main()
