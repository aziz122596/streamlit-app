import streamlit as st
import pandas as pd
import numpy as np
import json
import folium
from streamlit_folium import folium_static
import requests
from shapely.geometry import shape

# Настройка страницы
st.set_page_config(page_title="Анализ теневой экономики", page_icon=":bar_chart:", layout="wide")

# Функция для загрузки GeoJSON из GitHub с кешированием
@st.cache_data
def load_geojson_from_github(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Проверка на ошибки HTTP
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка загрузки GeoJSON: {e}")
        st.stop()

# URL GeoJSON файла
geojson_url = "https://raw.githubusercontent.com/akbartus/GeoJSON-Uzbekistan/master/geojson/uzbekistan_regions.geojson"
uzb_geojson_data = load_geojson_from_github(geojson_url)

def load_data():
    regions_geo = [feature['properties']['name'] for feature in uzb_geojson_data['features']]
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
    df["Shadow_Economy_Percent"] = ((df["Consumption"] - df["Tax_Revenue"] - df["Import"] - df["Humanitarian_Aid"]) / df["GDP"]) * 100
    df["Shadow_Economy_Percent"] = df["Shadow_Economy_Percent"].round(2)
    return df

df = load_data()

# Функция для вычисления центроида без дополнительной проекции
def get_centroid(geojson_feature):
    geom = shape(geojson_feature['geometry'])
    centroid = geom.centroid
    # centroid.x = долгота, centroid.y = широта
    return [centroid.y, centroid.x]

city_coordinates = {}
for feature in uzb_geojson_data['features']:
    centroid = get_centroid(feature)
    if centroid is not None:
        city_coordinates[feature['properties']['name']] = centroid

# Создание карты folium
m = folium.Map(location=[41.3111, 69.2797], zoom_start=6)

folium.Choropleth(
    geo_data=uzb_geojson_data,
    data=df,
    columns=["Region", "Shadow_Economy_Percent"],
    key_on="feature.properties.name",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Теневая экономика (%)",
    highlight=True
).add_to(m)

folium.features.GeoJson(
    uzb_geojson_data,
    name='Hover',
    style_function=lambda x: {'color': 'black', 'fillOpacity': 0},
    highlight_function=lambda x: {'weight': 3, 'color': 'blue'},
    tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["Регион:"], localize=True)
).add_to(m)

for city, coords in city_coordinates.items():
    folium.CircleMarker(
        location=coords,
        radius=5,
        color="blue",
        fill=True,
        fill_color="blue",
        popup=city,
    ).add_to(m)

st.markdown("<h2 style='text-align: center;'>Интерактивная карта</h2>", unsafe_allow_html=True)
folium_static(m, width=700, height=500)

st.markdown("<h2 style='text-align: center;'>Региональные показатели</h2>", unsafe_allow_html=True)
region_choice = st.selectbox("Выберите регион:", df["Region"].unique(), label_visibility="collapsed")
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
