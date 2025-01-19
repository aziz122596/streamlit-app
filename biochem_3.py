import streamlit as st
import pandas as pd
import numpy as np
from re import search
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from datetime import datetime

def load_data(file_path):
    """Загрузка данных из Excel файла."""
    df = pd.read_excel(file_path)
    # Проверяем наличие необходимых столбцов
    required_columns = ['исслед.материал']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"В файле отсутствуют следующие столбцы: {', '.join(missing_columns)}")
        return None
    return df

def split_data_by_material(df):
    """Разделение данных по типу материала."""
    materials = {
        'эритроциты': df[df['исслед.материал'] == 'эритроциты'],
        'тени': df[df['исслед.материал'] == 'тени'],
        'везикулы': df[df['исслед.материал'] == 'везикулы']
    }
    return materials

def calculate_values(df, cons_list, material_type, belok=1):
    """Универсальная функция расчета значений для всех типов материалов."""
    results = []
    current_date = datetime.now().strftime("%Y-%m-%d")  # Используем текущую дату
    
    for c in cons_list:
        H = [x for x in df.columns if search(c, x)]
        if not H:  # Пропускаем итерацию, если не найдены соответствующие столбцы
            continue
            
        vez = df.loc[:, H]
        vez.set_index(H[0], inplace=True)
        
        try:
            OAA = (vez.loc[1] + vez.loc[2]) / 2 - vez.loc[f'К{c}']
            MA = (vez.loc[f'Б{c}'] + vez.loc[f'Б{c},1']) / 2 - vez.loc[f'К{c}']
            NKA = OAA - MA
            MA_100 = MA / (OAA / 100)
            
            if material_type == 'эритроциты':
                factor = 0.23 * 3 * 2 * 10 * 1.1 * 10
            else:
                factor = 0.23 * 3 * 2 / belok
                
            summ = MA * factor
            summ2 = NKA * factor
            
            results.append({
                'date': current_date,
                'material_type': material_type,
                'concentration': float(c),  # Преобразуем в число для построения графиков
                'MA': summ.iloc[0],
                'NKA': summ2.iloc[0],
                'MA_100': MA_100.iloc[0]
            })
        except Exception as e:
            st.error(f"Ошибка в расчете для концентрации {c}мМ: {str(e)}")
            
    return pd.DataFrame(results)

def michaelis_menten(S, Vmax, Km):
    """Функция Михаэлиса-Ментен."""
    return Vmax * S / (Km + S)

def plot_michaelis_menten(df, enzyme_type):
    """Построение графика Михаэлиса-Ментен."""
    if df.empty:
        st.warning("Нет данных для построения графика")
        return
        
    S = df['concentration'].values
    v = df[enzyme_type].values
    
    try:
        # Фитирование кривой
        popt, _ = curve_fit(michaelis_menten, S, v, p0=[max(v), np.mean(S)])
        Vmax, Km = popt
        
        # Создание точек для построения теоретической кривой
        S_curve = np.linspace(0, max(S), 100)
        v_curve = michaelis_menten(S_curve, Vmax, Km)
        
        # Построение графика
        fig = go.Figure()
        
        # Экспериментальные точки
        fig.add_trace(go.Scatter(
            x=S, y=v,
            mode='markers',
            name='Экспериментальные данные',
            marker=dict(size=10)
        ))
        
        # Теоретическая кривая
        fig.add_trace(go.Scatter(
            x=S_curve, y=v_curve,
            mode='lines',
            name='Модель Михаэлиса-Ментен',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title=f'График Михаэлиса-Ментен для {enzyme_type}',
            xaxis_title='Концентрация субстрата (мМ)',
            yaxis_title='Скорость реакции',
            showlegend=True
        )
        
        st.plotly_chart(fig)
        
        # Вывод параметров
        st.write(f"Параметры уравнения Михаэлиса-Ментен:")
        st.write(f"Vmax = {Vmax:.2f}")
        st.write(f"Km = {Km:.2f} мМ")
        
    except Exception as e:
        st.error(f"Ошибка при построении графика Михаэлиса-Ментен: {str(e)}")

def plot_additional_graphs(df, material_type):
    """Построение дополнительных графиков анализа."""
    if df.empty:
        st.warning("Нет данных для построения графиков")
        return
        
    # График сравнения MA и NKA
    fig_comparison = px.line(
        df,
        x='concentration',
        y=['MA', 'NKA'],
        title=f'Сравнение MA и NKA для {material_type}',
        labels={'concentration': 'Концентрация (мМ)', 'value': 'Значение'},
        markers=True
    )
    st.plotly_chart(fig_comparison)
    
    # График MA_100
    fig_ma100 = px.line(
        df,
        x='concentration',
        y='MA_100',
        title=f'MA_100 для {material_type}',
        labels={'concentration': 'Концентрация (мМ)', 'MA_100': 'MA_100 (%)'},
        markers=True
    )
    st.plotly_chart(fig_ma100)

def main():
    st.title("Анализ биохимических данных")
    
    uploaded_file = st.file_uploader("Загрузите файл Excel", type=['xlsx'])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is None:
            return
            
        materials_data = split_data_by_material(df)
        cons_list = ["3", "6", "12"]
        
        # Контейнер для всех результатов
        all_results = pd.DataFrame()
        
        for material_type, data in materials_data.items():
            if not data.empty:
                belok = 0.278967742 if material_type in ['тени', 'везикулы'] else 1
                results = calculate_values(data, cons_list, material_type, belok=belok)
                all_results = pd.concat([all_results, results], ignore_index=True)
        
        if not all_results.empty:
            st.write("### Результаты расчетов")
            st.dataframe(all_results)
            
            # Выбор материала для анализа
            material_choice = st.selectbox(
                "Выберите материал для анализа:",
                all_results['material_type'].unique()
            )
            
            # Выбор типа фермента
            enzyme_choice = st.selectbox(
                "Выберите тип фермента:",
                ['MA', 'NKA']
            )
            
            # Фильтрация данных
            material_data = all_results[all_results['material_type'] == material_choice]
            
            # Построение графиков
            st.write(f"### График Михаэлиса-Ментен для {material_choice}")
            plot_michaelis_menten(material_data, enzyme_choice)
            
            st.write("### Дополнительные графики")
            plot_additional_graphs(material_data, material_choice)
            
            # Добавляем кнопку для скачивания результатов
            if st.button("Скачать результаты"):
                csv = all_results.to_csv(index=False)
                st.download_button(
                    label="Скачать CSV",
                    data=csv,
                    file_name="results.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()