import pandas as pd
from re import search
import os

os.chdir("c:\\Users\\User\\Desktop\\Biochem") 

os.listdir()

df = pd.read_excel("c:\\Users\\User\\Desktop\\Biochem\\БХ_данные_pandas.xlsx", sheet_name="12.11.2023")
# Создаем два отдельных DataFrame для 'эритроциты' 'тени' 'везикулы'
df_eritrociti = df[df['исслед.материал'] == 'эритроциты']
df_teni = df[df['исслед.материал'] == 'тени']
df_vezykuli = df[df['исслед.материал'] == 'везикулы']

# Списки концентраций
cons_list = ["3", "6", "12"]

# Функции для вычисления значений по формулам для каждого датафрейма отдельно
def calculate_values_erit(df, cons_list, sheet_date):
    results_erit = []
    for c in cons_list:
        H = [x for x in df.columns if search(c, x)]
        vez = df.loc[:, H]
        vez.set_index(H[0], inplace=True)

        OAA = (vez.loc[1] + vez.loc[2]) / 2 - vez.loc[f'К{c}']
        MA = (vez.loc[f'Б{c}'] + vez.loc[f'Б{c},1']) / 2 - vez.loc[f'К{c}']
        NKA = OAA - MA
        MA_100 = MA / (OAA / 100)
        summ = MA * 0.23 * 3 * 2 * 10 * 1.1 * 10
        summ2 = NKA * 0.23 * 3 * 2 * 10 * 1.1 * 10

        results_erit.append({
            'date': sheet_date,
            'material_type': 'эритроциты',
            'concentration': c + 'мМ',
            'MA': summ.iloc[0],
            'NKA': summ2.iloc[0]
        })
    return results_erit

def calculate_values_teni(df, cons_list, sheet_date):
    results_teni = []
    for c in cons_list:
        H = [x for x in df.columns if search(c, x)]
        vez = df.loc[:, H]
        vez.set_index(H[0], inplace=True)

        OAA = (vez.loc[1] + vez.loc[2]) / 2 - vez.loc[f'К{c}']
        MA = (vez.loc[f'Б{c}'] + vez.loc[f'Б{c},1']) / 2 - vez.loc[f'К{c}']
        NKA = OAA - MA
        MA_100 = MA / (OAA / 100)

        belok = 0.278967742

        summ = MA * 0.23 * 3 * 2 / belok
        summ2 = NKA * 0.23 * 3 * 2 / belok

        results_teni.append({
            'date': sheet_date,
            'material_type': 'тени',
            'concentration': c + 'мМ',
            'MA': summ.iloc[0],
            'NKA': summ2.iloc[0]
        })
    return results_teni

def calculate_values_vezykuli(df, cons_list, sheet_date):
    results_vezykuli = []
    for c in cons_list:
        H = [x for x in df.columns if search(c, x)]
        vez = df.loc[:, H]
        vez.set_index(H[0], inplace=True)

        OAA = (vez.loc[1] + vez.loc[2]) / 2 - vez.loc[f'К{c}']
        MA = (vez.loc[f'Б{c}'] + vez.loc[f'Б{c},1']) / 2 - vez.loc[f'К{c}']
        NKA = OAA - MA
        MA_100 = MA / (OAA / 100)

        belok_v = 0.278967742

        summ = MA * 0.23 * 3 * 2 / belok_v
        summ2 = NKA * 0.23 * 3 * 2 / belok_v

        results_vezykuli.append({
            'date': sheet_date,
            'material_type': 'тени',
            'concentration': c + 'мМ',
            'MA': summ.iloc[0],
            'NKA': summ2.iloc[0]
        })
    return results_vezykuli

# Вызываем функции для каждого датафрейма
res_erit_list = calculate_values(df_eritrociti, cons_list, 'эритроциты', '12.11.2023')
res_teni_list = calculate_values(df_teni, cons_list, 'тени', '12.11.2023')
res__list_vezykuli = calculate_values(df_vezykuli, cons_list, 'везикулы', '12.11.20233')

res_erit_list, res_teni_list, res__list_vezykuli