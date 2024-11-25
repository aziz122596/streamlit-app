import os
# Устанавливаем переменную окружения для обхода ошибки OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
# Вызов st.set_page_config() должен быть первым вызовом Streamlit
st.set_page_config(
    page_title="Анализатор здоровья растений",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Приложение для классификации состояния растений и расчета спектральных индексов."
    }
)

import torch
from torch import nn
from torchvision import transforms
from PIL import Image, ExifTags
import numpy as np
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import io
import base64

# Дополнительные библиотеки для оформления
from streamlit_option_menu import option_menu

# Установка кастомного CSS стиля
def set_css():
    st.markdown("""
        <style>
        /* Устанавливаем фоновое изображение */
        .stApp {
            background-image: url(https://i.pinimg.com/736x/b1/73/aa/b173aafcd21c285a60cdff9ff39cb0d8.jpg);
            background-size: cover;
        }
        /* Стилизация заголовков */
        h1 {
            color: #28a745;
            text-align: center;
        }
        h3 {
            color: gray;
            text-align: center;
        }
        /* Стилизация боковой панели */
        .css-1d391kg {
            background-color: rgba(255, 255, 255, 0.8);
        }
        /* Стилизация основного контента */
        .css-12ttj6m {
            background-color: rgba(255, 255, 255, 0.8);
        }
        </style>
        """, unsafe_allow_html=True)

set_css()

# Определение модели
class PlantClassifier(nn.Module):
    def __init__(self):
        super(PlantClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        # Удалены лишние print()
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Функции для обработки и предсказания

# Индексы, требующие NIR-канал
def calculate_ndvi(image):
    nir = image[:, :, nir_index].astype(float)
    red = image[:, :, red_index].astype(float)
    numerator = nir - red
    denominator = nir + red + 1e-5
    ndvi = numerator / denominator
    return ndvi

def calculate_gndvi(image):
    nir = image[:, :, nir_index].astype(float)
    green = image[:, :, green_index].astype(float)
    numerator = nir - green
    denominator = nir + green + 1e-5
    gndvi = numerator / denominator
    return gndvi

def calculate_endvi(image):
    nir = image[:, :, nir_index].astype(float)
    green = image[:, :, green_index].astype(float)
    blue = image[:, :, blue_index].astype(float)
    numerator = (nir + green) - (2 * blue)
    denominator = (nir + green) + (2 * blue) + 1e-5
    endvi = numerator / denominator
    return endvi

def calculate_cvi(image):
    nir = image[:, :, nir_index].astype(float)
    red = image[:, :, red_index].astype(float)
    green = image[:, :, green_index].astype(float)
    cvi = (nir / green) * (red / green + 1e-5)
    return cvi

def calculate_osavi(image):
    nir = image[:, :, nir_index].astype(float)
    red = image[:, :, red_index].astype(float)
    numerator = 1.5 * (nir - red)
    denominator = nir + red + 0.16 + 1e-5
    osavi = numerator / denominator
    return osavi

# Индексы, основанные на RGB-каналах
def calculate_vari(image):
    red = image[:, :, red_index].astype(float)
    green = image[:, :, green_index].astype(float)
    blue = image[:, :, blue_index].astype(float)
    numerator = green - red
    denominator = green + red - blue + 1e-5
    vari = numerator / denominator
    return vari

def calculate_exg(image):
    red = image[:, :, red_index].astype(float)
    green = image[:, :, green_index].astype(float)
    blue = image[:, :, blue_index].astype(float)
    exg = 2 * green - red - blue
    return exg

def calculate_gli(image):
    red = image[:, :, red_index].astype(float)
    green = image[:, :, green_index].astype(float)
    blue = image[:, :, blue_index].astype(float)
    numerator = 2 * green - red - blue
    denominator = 2 * green + red + blue + 1e-5
    gli = numerator / denominator
    return gli

def calculate_rgbvi(image):
    red = image[:, :, red_index].astype(float)
    green = image[:, :, green_index].astype(float)
    blue = image[:, :, blue_index].astype(float)
    numerator = (green ** 2) - (red * blue)
    denominator = (green ** 2) + (red * blue) + 1e-5
    rgbvi = numerator / denominator
    return rgbvi

def calculate_vegetation_index(image_array, index_name):
    if index_name == 'NDVI':
        return calculate_ndvi(image_array)
    elif index_name == 'GNDVI':
        return calculate_gndvi(image_array)
    elif index_name == 'ENDVI':
        return calculate_endvi(image_array)
    elif index_name == 'CVI':
        return calculate_cvi(image_array)
    elif index_name == 'OSAVI':
        return calculate_osavi(image_array)
    elif index_name == 'VARI':
        return calculate_vari(image_array)
    elif index_name == 'ExG':
        return calculate_exg(image_array)
    elif index_name == 'GLI':
        return calculate_gli(image_array)
    elif index_name == 'RGBVI':
        return calculate_rgbvi(image_array)
    else:
        st.error("Неизвестный индекс.")
        return None

def predict_image(image, model, transform):
    transformed_image = transform(image).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(transformed_image)
        if output.dim() == 4:
            _, predicted = torch.max(output, 1)
            predicted_flat = predicted.view(-1)
            predicted_class = torch.mode(predicted_flat).values.item()
        elif output.dim() == 2:
            _, predicted = torch.max(output, 1)
            predicted_class = predicted.item()
        else:
            st.error("Неподдерживаемая форма выходного тензора модели.")
            return None
    class_names = ['Наблюдается Засуха', 'Оптимально']
    return class_names[predicted_class]

# Загрузка модели
@st.cache_resource
def load_model():
    model = PlantClassifier()
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Функция для извлечения метаданных изображения
def get_image_metadata(image):
    exif_data = image.getexif()
    metadata = {}
    if exif_data:
        for tag_id, value in exif_data.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            metadata[tag] = value
    return metadata

# Функции для расчёта площади
def calculate_gsd(flight_altitude, focal_length, sensor_width, sensor_height, image_width, image_height):
    gsd_width = (flight_altitude * sensor_width) / (focal_length * image_width)
    gsd_height = (flight_altitude * sensor_height) / (focal_length * image_height)
    return gsd_width, gsd_height

def calculate_image_area(gsd_width, gsd_height, image_width, image_height):
    ground_width = gsd_width * image_width  # Ширина покрытия на местности в метрах
    ground_height = gsd_height * image_height  # Высота покрытия на местности в метрах
    area = ground_width * ground_height  # Площадь в квадратных метрах
    return area, ground_width, ground_height

# Функция для отображения интерактивной карты
def interactive_map():
    st.subheader("Выбор региона на карте")
    st.write("Выберите место, откуда было сделано изображение.")

    tyumen_coords = [57.153033, 65.534328]
    map_object = folium.Map(location=tyumen_coords, zoom_start=5)
    folium.Marker(tyumen_coords, popup="Тюменская область").add_to(map_object)
    st_data = st_folium(map_object, width=700, height=500)
    # Получение координат при клике
    if st_data and 'last_clicked' in st_data and st_data['last_clicked']:
        coords = st_data['last_clicked']
        st.write(f"Вы выбрали координаты: {coords}")

# Главная функция Streamlit
def main():
    # Боковая панель навигации
    with st.sidebar:
        selected = option_menu(
            menu_title="Навигация",
            options=["Главная", "О приложении", "Контакты"],
            icons=["house", "info-circle", "envelope"],
            menu_icon="cast",
            default_index=0,
        )
        st.sidebar.title("Информация")
        st.sidebar.markdown("""
        **Классификация состояния растений и расчёт спектральных индексов**

        Этот инструмент позволяет анализировать изображения растений и определять их состояние, а также рассчитывать различные спектральные индексы.
        """)
        st.sidebar.markdown("---")
        st.sidebar.write("© 2024 Product by Aziz AIRI")

    if selected == "Главная":
        st.markdown("<h1>Анализатор здоровья растений</h1>", unsafe_allow_html=True)
        st.markdown("<h3>Анализ спектральных индексов</h3>", unsafe_allow_html=True)

        # Организация в колонки
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Загрузите изображение растения")
            uploaded_file = st.file_uploader("Выберите файл", type=["tif", "tiff", "jpg", "jpeg", "png"])

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                image_array = np.array(image)
                num_channels = image_array.shape[2] if len(image_array.shape) == 3 else 1

                st.write(f"Количество каналов в изображении: {num_channels}")

                # Инициализация индексов каналов
                global red_index, green_index, blue_index, nir_index
                red_index = 0
                green_index = 1
                blue_index = 2
                nir_index = None

                if num_channels >= 4:
                    # Предполагаем, что 4-й канал - NIR
                    nir_index = 3
                    st.success("Обнаружен NIR-канал. Доступны индексы, требующие NIR.")
                    available_indices = ['NDVI', 'GNDVI', 'ENDVI', 'CVI', 'OSAVI', 'VARI', 'ExG', 'GLI', 'RGBVI']
                elif num_channels == 3:
                    st.warning("NIR-канал не обнаружен. Доступны индексы на основе RGB.")
                    available_indices = ['VARI', 'ExG', 'GLI', 'RGBVI']
                else:
                    st.error("Изображение должно иметь как минимум 3 канала (RGB).")
                    return

                image = image.convert('RGB')
                # Отображение загруженного изображения
                st.image(image, caption="Загруженное изображение", use_column_width=True)

                # Выбор индекса для расчета
                st.subheader("Выберите спектральный индекс для расчета")
                selected_index = st.selectbox("Спектральный индекс:", available_indices)

                # Загрузка модели и предсказание
                model = load_model()
                transform = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                ])
                prediction = predict_image(image, model, transform)
                index_result = calculate_vegetation_index(image_array, selected_index)

                if prediction is not None:
                    st.success(f"**Предсказание:** {prediction}")
                    st.info(f"**Среднее значение {selected_index}:** {np.nanmean(index_result):.2f}")

                    # Отображение индекса в виде изображения
                    st.subheader(f"Карта {selected_index}")
                    fig, ax = plt.subplots()
                    index_image = (index_result - np.nanmin(index_result)) / (np.nanmax(index_result) - np.nanmin(index_result) + 1e-5)
                    ax.imshow(index_image, cmap='RdYlGn')
                    ax.axis('off')
                    st.pyplot(fig)

                    # Расчёт площади снимка
                    st.subheader("Расчёт площади покрытия снимка")

                    # Получение размеров изображения
                    image_width, image_height = image.size  # В пикселях

                    # Попытка получить фокусное расстояние из метаданных
                    metadata = get_image_metadata(image)
                    focal_length_exif = metadata.get('FocalLength', None)
                    if focal_length_exif is not None:
                        focal_length_value = focal_length_exif
                        if isinstance(focal_length_exif, tuple):
                            focal_length_value = focal_length_exif[0] / focal_length_exif[1]
                        st.write(f"Фокусное расстояние из метаданных: {focal_length_value:.2f} мм")
                    else:
                        st.write("Не удалось получить фокусное расстояние из метаданных.")
                        focal_length_value = 35.0  # Значение по умолчанию

                    # Ввод параметров
                    st.write("Введите параметры камеры и полёта:")
                    flight_altitude = st.number_input("Высота полёта (м):", min_value=1.0, value=100.0)
                    focal_length = st.number_input("Фокусное расстояние камеры (мм):", min_value=1.0, value=focal_length_value)
                    sensor_width = st.number_input("Ширина сенсора (мм):", min_value=1.0, value=36.0)
                    sensor_height = st.number_input("Высота сенсора (мм):", min_value=1.0, value=24.0)
                    # image_width и image_height уже получены из изображения

                    # Расчёт площади покрытия снимка
                    gsd_width, gsd_height = calculate_gsd(
                        flight_altitude, focal_length, sensor_width, sensor_height, image_width, image_height
                    )
                    area, ground_width, ground_height = calculate_image_area(
                        gsd_width, gsd_height, image_width, image_height
                    )
                    st.success(f"**Площадь покрытия снимка:** {area/1e6:.8f} км²")
                    st.info(f"**Размеры покрытия:** {ground_width:.2f} м (ширина) x {ground_height:.2f} м (высота)")

                else:
                    st.error("Ошибка при предсказании. Проверьте модель и данные.")

        with col2:
            st.subheader("Интерактивная карта")
            interactive_map()

    elif selected == "О приложении":
        st.markdown("<h1>О приложении</h1>", unsafe_allow_html=True)
        st.write("""
        **Анализатор здоровья растений** - это инструмент, который помогает фермерам и исследователям анализировать состояние растений с помощью современных технологий компьютерного зрения и спектрального анализа. С помощью этого приложения вы можете:
        - Загрузить изображение растения или поля.
        - Рассчитать различные спектральные индексы.
        - Получить предсказание состояния растений (например, засуха или оптимальное состояние).
        - Определить площадь покрытия снимка и другие полезные метрики.
        """)

    elif selected == "Контакты":
        st.markdown("<h1>Контакты</h1>", unsafe_allow_html=True)
        st.write("""
        Если у вас есть вопросы или предложения, пожалуйста, свяжитесь с нами:
        - Email: example@example.com
        - Телефон: +7 (123) 456-78-90
        """)

if __name__ == "__main__":
    main()
