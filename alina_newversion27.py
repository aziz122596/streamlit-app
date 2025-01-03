import os
# Устанавливаем переменную окружения для обхода ошибки OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
# Вызов st.set_page_config() должен быть первым вызовом Streamlit
st.set_page_config(
    page_title="Система оценки состояния угодий",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Приложение для анализа состояния угодий и расчета спектральных индексов"
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
from streamlit_option_menu import option_menu

# Установка кастомного CSS стиля
def set_css():
    st.markdown(
        """
        <style>
        .stApp {
            /* Если нужно поменять фоновую картинку, укажите свой URL */
            background-image: url(https://p1.zoon.ru/preview/UMTUl9g9WDauuEwF2o0CkQ/2400x1500x75/1/f/8/original_57dd9ac840c088373b94a409_5a0966316a8f1.jpg);
            background-size: cover;
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        }

        h1, h2, h3 {
            text-align: center;
            font-weight: 600;
        }

        /* Цвет главного заголовка */
        h1 {
            color: #2c7c31;
        }

        /* Цвет подзаголовков */
        h3 {
            color: #34495e;
        }

        /* Прозрачный фон у боковой панели */
        .css-1d391kg {
            background-color: rgba(255, 255, 255, 0.85) !important;
        }

        /* Основной блок */
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

        /* Стилизация selectbox */
        .css-1c9n29x {
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_css()

# Определение простой модели (пример)
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
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Функции для расчёта различных индексов
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
    # Добавлен маленький сдвиг, чтобы избежать деления на ноль
    cvi = (nir / green) * (red / green + 1e-5)
    return cvi

def calculate_osavi(image):
    nir = image[:, :, nir_index].astype(float)
    red = image[:, :, red_index].astype(float)
    numerator = 1.5 * (nir - red)
    denominator = nir + red + 0.16 + 1e-5
    osavi = numerator / denominator
    return osavi

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
        st.error("Неизвестный индекс")
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
            st.error("Неподдерживаемая форма выходного тензора модели")
            return None
    # Переформулировка классов с учётом пожеланий
    class_names = [
        "Удовлетворительно. Наблюдается засуха", 
        "Оптимально. Наблюдаются условия полива"
    ]
    return class_names[predicted_class]

@st.cache_resource
def load_model():
    model = PlantClassifier()
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

def get_image_metadata(image):
    exif_data = image.getexif()
    metadata = {}
    if exif_data:
        for tag_id, value in exif_data.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            metadata[tag] = value
    return metadata

def calculate_gsd(flight_altitude, focal_length, sensor_width, sensor_height, image_width, image_height):
    gsd_width = (flight_altitude * sensor_width) / (focal_length * image_width)
    gsd_height = (flight_altitude * sensor_height) / (focal_length * image_height)
    return gsd_width, gsd_height

def calculate_image_area(gsd_width, gsd_height, image_width, image_height):
    ground_width = gsd_width * image_width
    ground_height = gsd_height * image_height
    area = ground_width * ground_height
    return area, ground_width, ground_height

def interactive_map():
    st.subheader("Интерактивная карта")
    st.write("Выберите регион, соответствующий вашим угодьям")

    # Пример координат
    default_coords = [57.153033, 65.534328]
    map_object = folium.Map(location=default_coords, zoom_start=5)
    folium.Marker(default_coords, popup="Пример метки").add_to(map_object)
    st_data = st_folium(map_object, width=700, height=500)
    if st_data and 'last_clicked' in st_data and st_data['last_clicked']:
        coords = st_data['last_clicked']
        st.write(f"Вы выбрали координаты {coords}")

def main():
    with st.sidebar:
        hide_sidebar = st.checkbox("Скрыть боковую панель", value=False)
        if not hide_sidebar:
            selected = option_menu(
                menu_title="Навигация",
                options=["Главная", "О приложении", "Контакты"],
                icons=["house", "info-circle", "envelope"],
                menu_icon="cast",
                default_index=0
            )
            st.sidebar.title("Информация")
            st.sidebar.markdown(
                """
                Приложение для анализа состояния угодий и расчёта спектральных индексов
                """
            )
            st.sidebar.markdown("---")
            st.sidebar.write("© 2024 Product by Aziz AIRI")
        else:
            selected = "Главная"

    if selected == "Главная":
        st.markdown("<h1>Система оценки состояния угодий</h1>", unsafe_allow_html=True)
        st.markdown("<h3>Анализ изображений и спектральных снимков</h3>", unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Загрузите изображение своих угодий или полей")
            st.write("Limit 200MB per file (TIF, TIFF, JPG, JPEG, PNG)")
            uploaded_file = st.file_uploader("", type=["tif", "tiff", "jpg", "jpeg", "png"])

            if uploaded_file is not None:
                with st.spinner("Обработка изображения"):
                    image = Image.open(uploaded_file)
                    image_array = np.array(image)

                    if len(image_array.shape) < 3:
                        st.error("Изображение должно иметь не менее 3 каналов (RGB)")
                        return

                    num_channels = image_array.shape[2]
                    st.write(f"Число каналов в изображении {num_channels}")

                    global red_index, green_index, blue_index, nir_index
                    red_index = 0
                    green_index = 1
                    blue_index = 2
                    nir_index = None

                    # Проверка на наличие NIR-канала
                    if num_channels >= 4:
                        nir_index = 3
                        st.success("Обнаружен NIR-канал. Можно считать расширенные индексы")
                        # Расставляем ExG первым, потом VARI, чтобы соответствовать пожеланиям
                        available_indices = [
                            'ExG', 'VARI', 'NDVI', 'GNDVI', 'ENDVI', 'CVI', 'OSAVI', 'GLI', 'RGBVI'
                        ]
                    else:
                        st.warning("NIR-канал не обнаружен. Рассчитываются индексы для RGB")
                        available_indices = ['ExG', 'VARI', 'GLI', 'RGBVI']

                    image = image.convert('RGB')
                    st.image(image, caption="Загруженное изображение", use_column_width=True)

                    st.subheader("Выберите спектральный индекс для расчета")
                    selected_index = st.selectbox("", available_indices)

                    progress_bar = st.progress(0)
                    progress_bar.progress(30)

                    # Загрузка модели и предсказание состояния
                    model = load_model()
                    transform = transforms.Compose([
                        transforms.Resize((128, 128)),
                        transforms.ToTensor()
                    ])
                    prediction = predict_image(image, model, transform)
                    progress_bar.progress(60)

                    # Расчёт выбранного индекса
                    index_result = calculate_vegetation_index(image_array, selected_index)
                    progress_bar.progress(100)

                    if prediction is not None and index_result is not None:
                        st.success(f"Предсказанное состояние {prediction}")
                        mean_value = np.nanmean(index_result)
                        st.info(f"Среднее значение {selected_index} {mean_value:.2f}")

                        # Рекомендации в зависимости от класса
                        if "засуха" in prediction:
                            st.warning("Рекомендации")
                            st.write("- Необходим обильный полив")
                            st.write("- Для большей эффективности требуется внесение водоудерживающих добавок")
                        else:
                            st.success("Рекомендации")
                            st.write("- Для поддержания плодородия угодий подходит внесение минеральных и органических удобрений")
                            st.write("- Текущих проблем с поливом не выявлено")

                        tab1, tab2, tab3 = st.tabs(["Карта индекса", "Расчет площади угодий или полей", "Расчет площади засушливых участков"])

                        with tab1:
                            st.subheader(f"Карта {selected_index}")
                            # Нормируем для удобного отображения
                            normalized_index = (
                                (index_result - np.nanmin(index_result)) 
                                / (np.nanmax(index_result) - np.nanmin(index_result) + 1e-5)
                            )
                            fig, ax = plt.subplots()
                            im = ax.imshow(normalized_index, cmap='RdYlGn')
                            ax.axis('off')
                            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                            cbar.set_label(f"{selected_index} значение", rotation=270, labelpad=15)
                            st.pyplot(fig)

                        with tab2:
                            st.subheader("Расчет площади угодий или полей")
                            image_width, image_height = image.size
                            metadata = get_image_metadata(image)

                            # Извлекаем фокусное расстояние при наличии
                            focal_length_exif = metadata.get('FocalLength', None)
                            if focal_length_exif is not None:
                                focal_length_value = focal_length_exif
                                if isinstance(focal_length_exif, tuple):
                                    focal_length_value = focal_length_exif[0] / focal_length_exif[1]
                                st.write(f"Фокусное расстояние из метаданных {focal_length_value:.2f} мм")
                            else:
                                st.write("Фокусное расстояние из метаданных не найдено")
                                focal_length_value = 35.0

                            st.write("Введите параметры камеры и полёта")
                            flight_altitude = st.number_input("Высота полёта (м)", min_value=1.0, value=100.0)
                            focal_length = st.number_input("Фокусное расстояние камеры (мм)", min_value=1.0, value=focal_length_value)
                            sensor_width = st.number_input("Ширина сенсора (мм)", min_value=1.0, value=36.0)
                            sensor_height = st.number_input("Высота сенсора (мм)", min_value=1.0, value=24.0)

                            gsd_width, gsd_height = calculate_gsd(
                                flight_altitude, focal_length, sensor_width, sensor_height, image_width, image_height
                            )
                            area, ground_width, ground_height = calculate_image_area(
                                gsd_width, gsd_height, image_width, image_height
                            )
                            area_ha = area / 10000

                            st.write("Результаты")
                            met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                            with met_col1:
                                st.metric("Площадь", f"{area_ha:.2f} га")
                            with met_col2:
                                st.metric("Ширина", f"{ground_width:.2f} м")
                            with met_col3:
                                st.metric("Высота", f"{ground_height:.2f} м")
                            with met_col4:
                                st.metric("Покрытие", f"{area/1e6:.6f} км²")

                        with tab3:
                            st.subheader("Расчет площади засушливых участков")
                            st.write("Выберите порог индекса для обозначения засухи")
                            threshold = st.slider("Порог засухи", 0.0, 1.0, 0.5)

                            # Берём нормированную карту индекса
                            normalized_index = (
                                (index_result - np.nanmin(index_result)) 
                                / (np.nanmax(index_result) - np.nanmin(index_result) + 1e-5)
                            )
                            dryness_mask = normalized_index < threshold
                            dryness_pixels = dryness_mask.sum()

                            # Расчет площади одного пикселя
                            pixel_area_m2 = gsd_width * gsd_height
                            dryness_area_m2 = dryness_pixels * pixel_area_m2
                            dryness_area_ha = dryness_area_m2 / 10000
                            dryness_percent = 0
                            if area_ha > 0:
                                dryness_percent = (dryness_area_ha / area_ha) * 100

                            st.write("Засушливые участки")
                            st.write(f"Площадь {dryness_area_ha:.2f} га")
                            st.write(f"Это примерно {dryness_percent:.1f}% от общей площади")

                            # Рекомендуемая доза вводится как пример
                            recommended_dose_t_per_ha = 4.64
                            st.write("Пример расчета внесения водоудерживающих добавок")
                            st.write(f"Рекомендуется {recommended_dose_t_per_ha:.2f} т на 1 га засушливого участка")

        with col2:
            interactive_map()

    elif selected == "О приложении":
        st.markdown("<h1>О приложении</h1>", unsafe_allow_html=True)
        st.write(
            """
            Система оценки состояния угодий – это инструмент, который помогает фермерам, агрономам и исследователям 
            анализировать состояние своих угодий и растений с помощью технологий компьютерного зрения и спектрального анализа
            Приложение позволяет:
            - Рассчитывать спектральные индексы по изображению ваших угодий
            - Получать оценку состояния земельных участков (например, оптимальные условия полива или удовлетворительные при засухе)
            - Получать рекомендации по улучшению состояния своих угодий
            - Определять площадь угодий или полей и вносить необходимые добавки
            - Определять площадь засушливых участков и рассчитывать примерное количество водоудерживающих добавок
            """
        )

    elif selected == "Контакты":
        st.markdown("<h1>Контакты</h1>", unsafe_allow_html=True)
        st.write(
            """
            По вопросам и предложениям вы можете связаться с нами
            Email example@example.com
            Телефон +7 (123) 456-78-90
            """
        )

if __name__ == "__main__":
    main()
