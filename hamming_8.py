import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import numpy as np
import io
from fpdf import FPDF
import os

# Функция для установки CSS
def set_css():
    st.markdown("""
        <style>
        .stApp {
            background-image: url(https://i.pinimg.com/736x/b1/73/aa/b173aafcd21c285a60cdff9ff39cb0d8.jpg);
            background-size: cover;
        }
        </style>
        """, unsafe_allow_html=True)

set_css()

# Выбор языка
language = st.sidebar.selectbox("Выберите язык / Choose Language", ["Русский", "English"])

# Меню навигации
if language == "Русский":
    selected = st.sidebar.radio("Навигация", ["Главная", "О приложении", "Контакты"])
else:
    selected = st.sidebar.radio("Navigation", ["Home", "About", "Contacts"])

# Функция для отображения главной страницы
def main_page():
    # Заголовок и описание
    st.markdown(f"<h1 style='text-align: center; color: White;'>GenePurity Analyzer</h1>", unsafe_allow_html=True)
    if language == "Русский":
        st.markdown("""
        <p style='text-align: center;'>Приложение для анализа генетической чистоты сортов растений с использованием данных ДНК-маркеров. Приложение автоматически обрабатывает результаты молекулярных исследований, генерируя подробные текстовые отчеты с визуализацией данных в виде графиков и изображений. Оно упрощает процесс проверки чистоты сортов, обеспечивая точность, скорость и удобство анализа для научных и прикладных исследований.</p>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <p style='text-align: center;'>An application for analyzing the genetic purity of plant varieties using DNA marker data. The application automatically processes molecular research results, generating detailed text reports with data visualizations, including graphs and images. It simplifies the process of purity assessment, providing accuracy, speed, and convenience for scientific and applied research.</p>
        """, unsafe_allow_html=True)

    # Образец файла
    if language == "Русский":
        st.markdown("Вы можете скачать образец файла для примера:")
    else:
        st.markdown("You can download a sample file for example:")

    # Создаем образец данных
    sample_data = '''Sample-1;0;1;1;0
Sample-2;1;0;1;1
Sample-3;0;1;0;1
Sample-4;1;1;0;0
'''

    st.download_button(
        label="Скачать образец файла" if language == "Русский" else "Download sample file",
        data=sample_data,
        file_name='sample_data.csv',
        mime='text/csv'
    )

    # Загрузка файла данных
    uploaded_file = st.file_uploader(
        "Загрузите файл данных" if language == "Русский" else "Upload data file",
        type=["csv"]
    )

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, delimiter=';')
        st.write("Первые пять строк данных" if language == "Русский" else "First five rows of data")
        st.dataframe(data.head())

        data = data.apply(pd.to_numeric, errors='coerce')
        if data.empty:
            st.error("Загруженные данные пусты. Пожалуйста, загрузите корректный файл." if language == "Русский" else "Uploaded data is empty. Please upload a valid file.")
        elif data.isnull().any().any():
            st.error("В данных присутствуют некорректные значения." if language == "Русский" else "Data contains invalid values.")
        else:
            data_for_markers = data.T

            # Расчет расстояний Хэмминга и кластеризация
            hamming_distances = pdist(data_for_markers.values, metric='hamming')
            hamming_distance_matrix = squareform(hamming_distances)
            linked = linkage(hamming_distances, 'complete')

            # Получение порядка листьев для упорядочивания матрицы
            dendro = dendrogram(linked, labels=data_for_markers.index.tolist(), no_plot=True)
            ordered_sample_names = dendro['ivl']
            ordered_indices = [data_for_markers.index.tolist().index(name) for name in ordered_sample_names]
            ordered_distance_matrix = hamming_distance_matrix[:, ordered_indices][ordered_indices, :]

            st.markdown(f"<h1 style='text-align: center; color: white;'>{'Матрица расхождения Хэмминга' if language == 'Русский' else 'Hamming Distance Matrix'}</h1>", unsafe_allow_html=True)

            # Построение Heatmap
            plt.figure(figsize=(24, 12))
            sns.set(font_scale=1.2)
            sns.heatmap(
                ordered_distance_matrix,
                annot=False,
                cmap='magma',
                xticklabels=ordered_sample_names,
                yticklabels=ordered_sample_names,
                linewidths=.5,
                cbar_kws={'label': 'Расстояние Хэмминга' if language == 'Русский' else 'Hamming Distance'}
            )
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            plt.title('Матрица расхождения Хэмминга' if language == 'Русский' else 'Hamming Distance Matrix', fontsize=20)
            heatmap_buffer = io.BytesIO()
            plt.savefig(heatmap_buffer, format='png', bbox_inches='tight')
            heatmap_buffer.seek(0)
            st.pyplot(plt)
            plt.clf()

            # Построение дендрограммы
            plt.figure(figsize=(24, 12))
            dendro = dendrogram(
                linked,
                labels=data_for_markers.index.tolist(),
                leaf_rotation=90,
                leaf_font_size=12,
                color_threshold=0.5 * max(linked[:, 2])
            )
            plt.title('Дендрограмма кластеризации маркеров' if language == 'Русский' else 'Dendrogram of Marker Clustering', fontsize=20)
            plt.xlabel('Образцы' if language == 'Русский' else 'Samples', fontsize=16)
            plt.ylabel('Расстояние' if language == 'Русский' else 'Distance', fontsize=16)
            dendrogram_buffer = io.BytesIO()
            plt.savefig(dendrogram_buffer, format='png', bbox_inches='tight')
            dendrogram_buffer.seek(0)
            st.pyplot(plt)
            plt.clf()

            # Анализ и расчет показателей
            sample_names = data_for_markers.index.tolist()

            def extract_group_name(sample_name):
                return '-'.join(sample_name.split('-')[:-1])

            sample_groups = {}
            for idx, name in enumerate(sample_names):
                group_name = extract_group_name(name)
                if group_name not in sample_groups:
                    sample_groups[group_name] = []
                sample_groups[group_name].append(idx)

            # Вычисление общей однородности выборки
            upper_triangle_indices = np.triu_indices_from(hamming_distance_matrix, k=1)
            upper_triangle_values = hamming_distance_matrix[upper_triangle_indices]
            overall_average_distance = np.mean(upper_triangle_values)
            overall_homogeneity = 1 - overall_average_distance

            report_text = ''
            if language == "Русский":
                report_text += "Краткое резюме:\n"
                report_text += f"Общая однородность выборки составляет {overall_homogeneity*100:.2f}%.\n"
            else:
                report_text += "Summary:\n"
                report_text += f"Overall sample homogeneity is {overall_homogeneity*100:.2f}%.\n"

            group_homogeneities = {}
            for group_name, indices in sample_groups.items():
                submatrix = hamming_distance_matrix[np.ix_(indices, indices)]
                upper_triangle_indices = np.triu_indices_from(submatrix, k=1)
                upper_triangle_values = submatrix[upper_triangle_indices]
                average_distance = np.mean(upper_triangle_values)
                homogeneity = 1 - average_distance
                group_homogeneities[group_name] = homogeneity

            mean_group_homogeneity = np.mean(list(group_homogeneities.values()))
            if language == "Русский":
                report_text += f"Средняя однородность по всем видам составляет {mean_group_homogeneity*100:.2f}%.\n\n"
                report_text += "Определение чистоты сортов проведено с использованием метода Расхождения Хэмминга.\n\n"
            else:
                report_text += f"Average homogeneity across all groups is {mean_group_homogeneity*100:.2f}%.\n\n"
                report_text += "Purity assessment was conducted using the Hamming Distance method.\n\n"

            for group_name, indices in sample_groups.items():
                submatrix = hamming_distance_matrix[np.ix_(indices, indices)]
                upper_triangle_indices = np.triu_indices_from(submatrix, k=1)
                upper_triangle_values = submatrix[upper_triangle_indices]
                average_distance = np.mean(upper_triangle_values)
                homogeneity = 1 - average_distance
                if language == "Русский":
                    report_text += f"Выборка {group_name} – средняя неоднородность выборки = {average_distance*100:.2f}%, однородность = {homogeneity*100:.2f}%.\n"
                else:
                    report_text += f"Sample {group_name} – average heterogeneity = {average_distance*100:.2f}%, homogeneity = {homogeneity*100:.2f}%.\n"

                sub_dists = pdist(data_for_markers.values[indices], metric='hamming')
                sub_linkage = linkage(sub_dists, method='complete')
                max_d = 0.2
                labels = fcluster(sub_linkage, t=max_d, criterion='distance')
                n_clusters = np.max(labels)
                if n_clusters > 1:
                    if language == "Русский":
                        report_text += f"Выборка {group_name} разделяется на {n_clusters} кластера:\n"
                    else:
                        report_text += f"Sample {group_name} is divided into {n_clusters} clusters:\n"
                    for cluster_num in range(1, n_clusters+1):
                        cluster_indices = np.where(labels == cluster_num)[0]
                        cluster_sample_indices = [indices[i] for i in cluster_indices]
                        cluster_sample_names = [sample_names[idx] for idx in cluster_sample_indices]
                        cluster_submatrix = submatrix[np.ix_(cluster_indices, cluster_indices)]
                        upper_triangle_indices = np.triu_indices_from(cluster_submatrix, k=1)
                        upper_triangle_values = cluster_submatrix[upper_triangle_indices]
                        if len(upper_triangle_values) > 0:
                            average_distance = np.mean(upper_triangle_values)
                            homogeneity = 1 - average_distance
                            if language == "Русский":
                                report_text += f"Кластер {cluster_num}. Образцы: {', '.join(cluster_sample_names)}. Однородность = {homogeneity*100:.2f}%.\n"
                            else:
                                report_text += f"Cluster {cluster_num}. Samples: {', '.join(cluster_sample_names)}. Homogeneity = {homogeneity*100:.2f}%.\n"
                        else:
                            if language == "Русский":
                                report_text += f"Кластер {cluster_num} содержит один образец: {cluster_sample_names[0]}.\n"
                            else:
                                report_text += f"Cluster {cluster_num} contains one sample: {cluster_sample_names[0]}.\n"
                report_text += "\n"

            # Различия между группами
            group_names = list(sample_groups.keys())
            for i in range(len(group_names)):
                for j in range(i+1, len(group_names)):
                    group1 = group_names[i]
                    group2 = group_names[j]
                    indices1 = sample_groups[group1]
                    indices2 = sample_groups[group2]
                    inter_submatrix = hamming_distance_matrix[np.ix_(indices1, indices2)]
                    average_distance = np.mean(inter_submatrix)
                    if language == "Русский":
                        report_text += f"Различие между {group1} и {group2} составляет {average_distance*100:.2f}%.\n"
                    else:
                        report_text += f"Difference between {group1} and {group2} is {average_distance*100:.2f}%.\n"

            # Путь к файлу шрифта
            font_path = 'DejaVuSans.ttf'

            # Проверяем наличие файла шрифта
            if not os.path.isfile(font_path):
                st.error("Файл шрифта 'DejaVuSans.ttf' не найден. Поместите его в ту же папку, что и скрипт." if language == "Русский" else "Font file 'DejaVuSans.ttf' not found. Please place it in the same directory as the script.")
            else:
                # Генерация PDF с альбомной ориентацией
                pdf = FPDF(orientation='L')
                pdf.add_font('DejaVu', '', font_path, uni=True)
                pdf.add_page()
                pdf.set_font('DejaVu', '', size=12)
                for line in report_text.split('\n'):
                    pdf.multi_cell(0, 10, line)
                    pdf.ln()

                # Добавляем изображения
                pdf.add_page()
                pdf.set_font('DejaVu', '', 16)
                title = 'Матрица расхождения Хэмминга' if language == "Русский" else 'Hamming Distance Matrix'
                pdf.cell(0, 10, title, 0, 1, 'C')
                pdf.image(heatmap_buffer, x=10, y=20, w=pdf.epw, type='PNG')

                pdf.add_page()
                pdf.set_font('DejaVu', '', 16)
                title = 'Дендрограмма кластеризации маркеров' if language == "Русский" else 'Dendrogram of Marker Clustering'
                pdf.cell(0, 10, title, 0, 1, 'C')
                pdf.image(dendrogram_buffer, x=10, y=20, w=pdf.epw, type='PNG')

                # Сохраняем PDF в буфер
                pdf_buffer = io.BytesIO()
                pdf.output(pdf_buffer)
                pdf_buffer.seek(0)

                # Кнопка для скачивания PDF
                st.download_button(
                    label="Скачать PDF отчёт" if language == "Русский" else "Download PDF Report",
                    data=pdf_buffer,
                    file_name='report.pdf',
                    mime='application/pdf'
                )

# Отображение выбранной страницы
if selected == "Главная" or selected == "Home":
    main_page()
elif selected == "О приложении" or selected == "About":
    st.markdown(f"<h1>{'О приложении' if language == 'Русский' else 'About'}</h1>", unsafe_allow_html=True)
    if language == "Русский":
        st.write("""
        **GenePurity Analyzer** - это инструмент, который помогает исследователям анализировать генетическую чистоту сортов растений с использованием современных технологий молекулярной биологии и биоинформатики. С помощью этого приложения вы можете:
        - Загрузить данные ДНК-маркеров.
        - Получить подробный отчёт об однородности и неоднородности выборок.
        - Визуализировать данные с помощью матрицы расхождения Хэмминга и дендрограмм.
        - Сгенерировать PDF отчёт с результатами анализа.
        """)
    else:
        st.write("""
        **GenePurity Analyzer** is a tool that helps researchers analyze the genetic purity of plant varieties using modern molecular biology and bioinformatics technologies. With this application, you can:
        - Upload DNA marker data.
        - Obtain a detailed report on sample homogeneity and heterogeneity.
        - Visualize data using Hamming distance matrices and dendrograms.
        - Generate a PDF report with analysis results.
        """)
elif selected == "Контакты" or selected == "Contacts":
    st.markdown(f"<h1>{'Контакты' if language == 'Русский' else 'Contacts'}</h1>", unsafe_allow_html=True)
    if language == "Русский":
        st.write("""
        Если у вас есть вопросы или предложения, пожалуйста, свяжитесь с нами:
        - Email: example@example.com
        - Телефон: +7 (123) 456-78-90
        """)
    else:
        st.write("""
        If you have any questions or suggestions, please contact us:
        - Email: example@example.com
        - Phone: +1 (123) 456-7890
        """)
