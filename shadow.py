import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Настройка страницы
st.set_page_config(page_title="Анализ теневой экономики", page_icon=":bar_chart:", layout="wide")

@st.cache_data
def load_data():
    # Регионы, соответствующие данным в GeoJSON
    regions_geo = [
        "Republic of Karakalpakstan", "Andijan region", "Bukhara region", "Fergana region",
        "Jizzakh region", "Khorezm region", "Namangan region", "Navoiy region",
        "Samarkand region", "Surxondaryo region", "Tashkent region", "Qashqadaryo region"
    ]
    
    # Генерация данных
    data = {
        "Region": regions_geo,
        "Population": np.random.randint(500000, 5000000, size=len(regions_geo)),
        "GDP": np.random.uniform(500000, 2000000, size=len(regions_geo)),
        "Tax_Revenue": np.random.uniform(20000, 100000, size=len(regions_geo)),
        "Consumption": np.random.uniform(100000, 500000, size=len(regions_geo)),
        "Import": np.random.uniform(5000, 50000, size=len(regions_geo)),
        "Humanitarian_Aid": np.random.uniform(1000, 10000, size=len(regions_geo))
    }

    df = pd.DataFrame(data)
    df["Shadow_Economy_Percent"] = (
        (df["Consumption"] - df["Tax_Revenue"] - df["Import"] - df["Humanitarian_Aid"]) / df["GDP"]
    ) * 100
    df["Shadow_Economy_Percent"] = df["Shadow_Economy_Percent"].round(2)

    return df

@st.cache_data
def load_geojson():
    uzb_geojson = "https://raw.githubusercontent.com/azamat-jumaniyazov/uzbekistan-regions-geojson/main/uzbekistan_regions.json"
    return uzb_geojson

def plot_map(df, geojson_data):
    # Используем проекцию mercator
    fig = px.choropleth(
        df,
        geojson=geojson_data,
        locations="Region",
        featureidkey="properties.name",
        color="Shadow_Economy_Percent",
        hover_name="Region",
        hover_data=["Population", "GDP", "Tax_Revenue", "Consumption", "Import", "Humanitarian_Aid", "Shadow_Economy_Percent"],
        color_continuous_scale="Reds",
        projection="mercator",
        title=""
    )
    return fig

df = load_data()
uzb_geojson = load_geojson()

# Боковая панель
st.sidebar.title("Информация")
st.sidebar.markdown("#### О приложении")
st.sidebar.write("Приложение визуализирует уровень теневой экономики по регионам Узбекистана.")
st.sidebar.markdown("#### Контакты")
st.sidebar.write("Автор: ...")
st.sidebar.write("Email: example@example.com")
st.sidebar.markdown("#### Полезные ссылки")
st.sidebar.write("[О регионах Узбекистана](https://en.wikipedia.org/wiki/Regions_of_Uzbekistan)")

# Заголовок и описание
st.markdown("<h1 style='text-align: center;'>Оценка теневой экономики по регионам Узбекистана</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Выберите регион из списка ниже, чтобы увидеть детальную информацию.</p>", unsafe_allow_html=True)

# Карта
st.markdown("<h2 style='text-align: center;'>Интерактивная карта</h2>", unsafe_allow_html=True)
fig = plot_map(df, uzb_geojson)
st.plotly_chart(fig, use_container_width=True)

# Детальный просмотр по региону
st.markdown("<h2 style='text-align: center;'>Региональные показатели</h2>", unsafe_allow_html=True)
region_choice = st.selectbox("", df["Region"].unique())
region_data = df[df["Region"] == region_choice]

if not region_data.empty:
    st.markdown(f"**Регион:** {region_data['Region'].iloc[0]}")
    st.markdown(f"**Население:** {int(region_data['Population'].iloc[0])}")
    st.markdown(f"**ВВП:** {round(region_data['GDP'].iloc[0], 2)}")
    st.markdown(f"**Налоговые доходы:** {round(region_data['Tax_Revenue'].iloc[0], 2)}")
    st.markdown(f"**Потребление:** {round(region_data['Consumption'].iloc[0], 2)}")
    st.markdown(f"**Импорт:** {round(region_data['Import'].iloc[0], 2)}")
    st.markdown(f"**Гуманитарная помощь:** {round(region_data['Humanitarian_Aid'].iloc[0], 2)}")
    st.markdown(f"**Степень теневой экономики:** {region_data['Shadow_Economy_Percent'].iloc[0]}%")




