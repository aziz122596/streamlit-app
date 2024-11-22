import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform

# Заданные ранее данные

distance_matrix = np.array([
    [0.00, 0.15, 1.00, 1.00, 1.00, 1.00, 1.00],
    [0.15, 0.00, 0.48, 1.00, 1.00, 1.00, 1.00],
    [1.00, 0.48, 0.00, 0.43, 1.00, 1.00, 1.00],
    [1.00, 1.00, 0.43, 0.00, 0.50, 1.00, 1.00],
    [1.00, 1.00, 1.00, 0.50, 0.00, 0.00, 1.00],
    [1.00, 1.00, 1.00, 1.00, 0.00, 0.00, 1.00],
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.00]
])

# Список меток для первого набора данных
labelsA = [
    "Bothriocroton undatum", "Bothriocroton concolor", "Amblyomma sphenodonti", "Haemaphysalis flava",
    "Amblyomma elaphense", "Rhipicephalus sanguineus", "Amblyomma triguttatum"
]

# Используем бинарную матрицу на основе данных из tick_dict
binary_matrix = np.array(list({
    "ixodes": [9, 9, 1, 10, 5, 11, 13],
    "amblyomma": [1, 0, 1, 2, 1, 2, 2],
    "dermacentor": [5, 0, 2, 9, 7, 7, 1],
    "haemophysalis": [5, 5, 2, 10, 5, 8, 1],
    "hyalomma": [0, 2, 0, 2, 1, 2, 0],
    "rhipicephalus": [1, 2, 1, 5, 2, 4, 0],
    "ornithodoros": [0, 0, 0, 0, 0, 0, 0]
}.values()))

# Преобразование бинарной матрицы в матрицу расстояний (используем расстояния Хэмминга)
hamming_distances = np.array([np.sum(binary_matrix[i] != binary_matrix[j]) / len(binary_matrix[i])
                              for i in range(len(binary_matrix)) for j in range(len(binary_matrix))]).reshape(len(binary_matrix), len(binary_matrix))

# Список меток для второго набора данных
labelsB = ["ixodes", "amblyomma", "dermacentor", "haemophysalis", "hyalomma", "rhipicephalus", "ornithodoros"]

# Построение linkage для дендрограмм
linkage_1 = sch.linkage(pdist(distance_matrix), method='ward')
linkage_2 = sch.linkage(pdist(hamming_distances), method='ward')

# Построение дендрограмм и зеркальное отображение одной из них
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

dendro_1 = sch.dendrogram(linkage_1, labels=labelsA, ax=ax1, orientation='right')
dendro_2 = sch.dendrogram(linkage_2, labels=labelsB, ax=ax2, orientation='left')

ax1.set_title('Дендрограмма 1')
ax2.set_title('Дендрограмма 2')

# Соединение узлов линиями для сопоставления
for i, label in enumerate(dendro_1['ivl']):
    if label in dendro_2['ivl']:
        x1, y1 = 0, dendro_1['leaves'][i]
        x2, y2 = 1, dendro_2['leaves'][dendro_2['ivl'].index(label)]
        fig.lines.append(plt.Line2D([x1, x2], [y1, y2], color='grey', linestyle='--'))

plt.tight_layout()
plt.show()