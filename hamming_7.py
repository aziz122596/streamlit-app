import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, leaves_list
import numpy as np
import io
from fpdf import FPDF
import os

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

st.markdown("<h1 style='text-align: center; color: White;'>Анализ чистоты сорта</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Загрузите файл данных", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, delimiter=';')
    st.write("Первые пять строк данных")
    st.dataframe(data.head())

    data = data.apply(pd.to_numeric, errors='coerce')
    if data.empty:
        st.error("Загруженные данные пусты. Пожалуйста, загрузите корректный файл.")
    elif data.isnull().any().any():
        st.error("В данных присутствуют некорректные значения.")
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

        st.markdown("<h1 style='text-align: center; color: white;'>Матрица расхождения Хэмминга</h1>", unsafe_allow_html=True)

        # Улучшенный Heatmap в альбомной ориентации
        plt.figure(figsize=(24, 12))  # Увеличиваем ширину фигуры
        sns.set(font_scale=1.2)
        sns.heatmap(
            ordered_distance_matrix,
            annot=False,
            cmap='magma',
            xticklabels=ordered_sample_names,
            yticklabels=ordered_sample_names,
            linewidths=.5,
            cbar_kws={'label': 'Расстояние Хэмминга'}
        )
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.title('Матрица расхождения Хэмминга', fontsize=20)
        heatmap_buffer = io.BytesIO()
        plt.savefig(heatmap_buffer, format='png', bbox_inches='tight')
        heatmap_buffer.seek(0)
        st.pyplot(plt)
        plt.clf()

        # Улучшенная Дендрограмма в альбомной ориентации
        plt.figure(figsize=(24, 12))  # Увеличиваем ширину фигуры
        dendro = dendrogram(
            linked,
            labels=data_for_markers.index.tolist(),
            leaf_rotation=90,
            leaf_font_size=12,
            color_threshold=0.5 * max(linked[:, 2])  # Настройка порога цвета
        )
        plt.title('Дендрограмма кластеризации маркеров', fontsize=20)
        plt.xlabel('Образцы', fontsize=16)
        plt.ylabel('Расстояние', fontsize=16)
        dendrogram_buffer = io.BytesIO()
        plt.savefig(dendrogram_buffer, format='png', bbox_inches='tight')
        dendrogram_buffer.seek(0)
        st.pyplot(plt)
        plt.clf()

        # Анализ и расчет показателей (оставляем без изменений)
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
        report_text += "Краткое резюме:\n"
        report_text += f"Общая однородность выборки составляет {overall_homogeneity*100:.2f}%.\n"

        group_heterogeneities = {}
        group_homogeneities = {}
        for group_name, indices in sample_groups.items():
            submatrix = hamming_distance_matrix[np.ix_(indices, indices)]
            upper_triangle_indices = np.triu_indices_from(submatrix, k=1)
            upper_triangle_values = submatrix[upper_triangle_indices]
            average_distance = np.mean(upper_triangle_values)
            group_heterogeneities[group_name] = average_distance
            homogeneity = 1 - average_distance
            group_homogeneities[group_name] = homogeneity

        mean_group_homogeneity = np.mean(list(group_homogeneities.values()))
        report_text += f"Средняя однородность по всем видам составляет {mean_group_homogeneity*100:.2f}%.\n\n"

        report_text += "Определение чистоты сортов проведено с использованием метода Расхождения Хэмминга.\n\n"

        for group_name, indices in sample_groups.items():
            submatrix = hamming_distance_matrix[np.ix_(indices, indices)]
            upper_triangle_indices = np.triu_indices_from(submatrix, k=1)
            upper_triangle_values = submatrix[upper_triangle_indices]
            average_distance = np.mean(upper_triangle_values)
            homogeneity = 1 - average_distance
            report_text += f"Выборка {group_name} – средняя неоднородность выборки = {average_distance*100:.2f}%, однородность = {homogeneity*100:.2f}%.\n"

            sub_dists = pdist(data_for_markers.values[indices], metric='hamming')
            sub_linkage = linkage(sub_dists, method='complete')
            max_d = 0.2
            labels = fcluster(sub_linkage, t=max_d, criterion='distance')
            n_clusters = np.max(labels)
            if n_clusters > 1:
                report_text += f"Выборка {group_name} разделяется на {n_clusters} кластера:\n"
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
                        report_text += f"Кластер {cluster_num}. Образцы: {', '.join(cluster_sample_names)}. Однородность = {homogeneity*100:.2f}%.\n"
                    else:
                        report_text += f"Кластер {cluster_num} содержит один образец: {cluster_sample_names[0]}.\n"
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
                report_text += f"Различие между {group1} и {group2} составляет {average_distance*100:.2f}%.\n"

        # Путь к файлу шрифта
        font_path = 'DejaVuSans.ttf'

        # Проверяем наличие файла шрифта
        if not os.path.isfile(font_path):
            st.error("Файл шрифта 'DejaVuSans.ttf' не найден. Поместите его в ту же папку, что и скрипт.")
        else:
            # Генерация PDF с альбомной ориентацией
            pdf = FPDF(orientation='L')  # 'L' для альбомной ориентации
            pdf.add_font('DejaVu', '', font_path, uni=True)
            pdf.add_page()
            pdf.set_font('DejaVu', '', size=12)
            for line in report_text.split('\n'):
                pdf.multi_cell(0, 10, line)
                pdf.ln()

            # Добавляем изображения в альбомной ориентации
            pdf.add_page()
            pdf.set_font('DejaVu', '', 16)
            pdf.cell(0, 10, 'Матрица расхождения Хэмминга', 0, 1, 'C')
            pdf.image(heatmap_buffer, x=10, y=20, w=pdf.w - 20)

            pdf.add_page()
            pdf.set_font('DejaVu', '', 16)
            pdf.cell(0, 10, 'Дендрограмма кластеризации маркеров', 0, 1, 'C')
            pdf.image(dendrogram_buffer, x=10, y=20, w=pdf.w - 20)

            # Сохраняем PDF в буфер
            pdf_buffer = io.BytesIO()
            pdf.output(pdf_buffer)
            pdf_buffer.seek(0)

            # Кнопка для скачивания PDF
            st.download_button(
                label="Скачать PDF отчет",
                data=pdf_buffer,
                file_name='report.pdf',
                mime='application/pdf'
            )
