import streamlit as st
import pandas as pd
import numpy as np
import json
import folium
from streamlit_folium import folium_static

# Настройка страницы
st.set_page_config(page_title="Анализ теневой экономики", page_icon=":bar_chart:", layout="wide")

# Условный GeoJSON с тремя регионами (вымышленные координаты)
uzb_geojson_data = {
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "name": "Andijan region"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[72.0,40.0],[72.5,40.0],[72.5,40.5],[72.0,40.5],[72.0,40.0]]]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "name": "Bukhara region"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[64.0,39.5],[64.5,39.5],[64.5,40.0],[64.0,40.0],[64.0,39.5]]]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "name": "Samarkand region"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[66.5,39.8],[67.0,39.8],[67.0,40.3],[66.5,40.3],[66.5,39.8]]]
      }
    }
  ]
}

def load_data():
    regions_geo = ["Andijan region", "Bukhara region", "Samarkand region"]
    
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

df = load_data()

# Боковая панель
st.sidebar.title("Информация")
st.sidebar.markdown("#### О приложении")
st.sidebar.write("Приложение визуализирует уровень теневой экономики по регионам.")
st.sidebar.markdown("#### Контакты")
st.sidebar.write("Автор: Абдурахимов Азиз")
st.sidebar.write("Email: example@example.com")

# Заголовок и описание
st.markdown("<h1 style='text-align: center;'>Оценка теневой экономики</h1>", unsafe_allow_html=True)

# Создание карты folium
m = folium.Map(location=[40.5, 68.0], zoom_start=5)

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
    style_function=lambda x: {'color':'black','fillOpacity':0},
    highlight_function=lambda x: {'weight':3, 'color':'blue'},
    tooltip=folium.GeoJsonTooltip(
        fields=["name"],
        aliases=["Регион:"],
        localize=True
    )
).add_to(m)

region_centers = {
    "Andijan region": [40.25, 72.25],
    "Bukhara region": [39.75, 64.25],
    "Samarkand region": [40.05, 66.75]
}

for region, coords in region_centers.items():
    folium.CircleMarker(
        location=[coords[0], coords[1]],
        radius=5,
        color="red",
        fill=True,
        fill_color="red",
        popup=region
    ).add_to(m)

st.markdown("<h2 style='text-align: center;'>Интерактивная карта</h2>", unsafe_allow_html=True)
# Используем folium_static для статического отображения карты без мигания
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
