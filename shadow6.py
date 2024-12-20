import streamlit as st
import pandas as pd
import numpy as np
import json
import folium
from streamlit_folium import folium_static
from shapely.geometry import shape

# Настройка страницы
st.set_page_config(page_title="Анализ теневой экономики", page_icon=":bar_chart:", layout="wide")

# Примерный встроенный GeoJSON с тремя регионами
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
    regions_geo = [f['properties']['name'] for f in uzb_geojson_data['features']]
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
    df["Shadow_Economy_Percent"] = ((df["Consumption"] - df["Tax_Revenue"] - df["Import"] - df["Humanitarian_Aid"]) / df["GDP"])*100
    df["Shadow_Economy_Percent"] = df["Shadow_Economy_Percent"].round(2)
    return df

df = load_data()

# Функция для вычисления центроидов регионов
def get_centroid(geojson_feature):
    geom = shape(geojson_feature['geometry'])
    c = geom.centroid
    return [c.y, c.x]

region_centers = {}
for feature in uzb_geojson_data['features']:
    region_name = feature['properties']['name']
    region_centers[region_name] = get_centroid(feature)

# Боковая панель
st.sidebar.title("Информация")
st.sidebar.markdown("#### О приложении")
st.sidebar.write("Данное приложение визуализирует уровень теневой экономики по регионам.")
st.sidebar.markdown("#### Контакты")
st.sidebar.write("Автор: Абдурахимов Азиз")
st.sidebar.write("Email: example@example.com")

# Заголовок и описание (по центру)
st.markdown("<h1 style='text-align: center; margin-bottom:40px;'>Оценка теневой экономики по регионам</h1>", unsafe_allow_html=True)

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

# Добавляем подсказки при наведении на полигоны регионов
folium.features.GeoJson(
    uzb_geojson_data,
    name='Hover',
    style_function=lambda x: {'color': 'black','fillOpacity':0},
    highlight_function=lambda x: {'weight':3, 'color':'blue'},
    tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["Регион:"], localize=True)
).add_to(m)

# Добавляем маркеры для регионов с детальной информацией
for region, coords in region_centers.items():
    region_info = df[df["Region"] == region].iloc[0]
    # Формируем содержимое всплывающего окна (popup) с данными региона
    popup_html = f"""
    <div style="font-family:Arial; font-size:12px;">
    <h4 style='margin:5px; font-size:14px; font-weight:bold;'>{region}</h4>
    <table style="border:none; border-collapse: collapse;">
    <tr><td><b>Население:</b></td><td>{int(region_info['Population'])}</td></tr>
    <tr><td><b>ВВП:</b></td><td>{round(region_info['GDP'], 2)}</td></tr>
    <tr><td><b>Налоговые доходы:</b></td><td>{round(region_info['Tax_Revenue'], 2)}</td></tr>
    <tr><td><b>Потребление:</b></td><td>{round(region_info['Consumption'], 2)}</td></tr>
    <tr><td><b>Импорт:</b></td><td>{round(region_info['Import'], 2)}</td></tr>
    <tr><td><b>Гуманитарная помощь:</b></td><td>{round(region_info['Humanitarian_Aid'], 2)}</td></tr>
    <tr><td><b>Теневая экономика:</b></td><td>{region_info['Shadow_Economy_Percent']}%</td></tr>
    </table>
    </div>
    """

    folium.CircleMarker(
        location=coords,
        radius=6,
        color="blue",
        fill=True,
        fill_color="blue",
        tooltip="Кликните для подробностей",
        popup=folium.Popup(popup_html, max_width=300)
    ).add_to(m)

# Размещаем карту по центру
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.markdown("<h2 style='text-align: center;'>Интерактивная карта</h2>", unsafe_allow_html=True)
    folium_static(m, width=700, height=500)

st.markdown("<h2 style='text-align: center; margin-top:40px;'>Региональные показатели</h2>", unsafe_allow_html=True)
col4, col5, col6 = st.columns([1,2,1])
with col5:
    region_choice = st.selectbox("Выберите регион:", df["Region"].unique())
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
