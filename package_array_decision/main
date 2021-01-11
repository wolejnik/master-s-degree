import numpy as np
import pandas as pd
from scipy.io import arff

def calculate_accuracy(df):
    attrs = df.iloc[:, 0:-1].columns
    unique_rows = df.drop_duplicates(keep='first')
    conflict_rows = unique_rows[unique_rows.duplicated(subset=attrs, keep=False)]
    unique_count = len(unique_rows.axes[0])
    conflict_count = len(conflict_rows.axes[0])
    accuracy = 1 - conflict_count / unique_count
    return accuracy

# Zadanie 1 Wczytaj dane z pliku ARFF.
data, meta = arff.loadarff('data/mushroom.arff')
df1 = pd.DataFrame(data)
df2 = pd.DataFrame(data).select_dtypes([np.object]).stack().str.decode('utf-8').unstack()

df1 = df1.drop(columns=df2.select_dtypes([np.object]).columns)
df = df2.join(df1)
# print(f'{data}\n{meta}\n')
print(f'{df}\n')

# Zadanie 2 Sprawdzić, czy w danych występują próbki z brakującymi wartościami atrybutów. Jeśli tak – usunie je.

counts_all_samples = str(len(df.axes[0]))
print(f'counts_all_samples : {counts_all_samples}')
counts_all_samples_deficient = np.max(df.isin(['?']).sum(axis=0))

df = df[df != "?"].dropna()
counts_all_samples_after_delete_deficient = str(len(df.axes[0]))
print(f'counts_all_samples_after_delete_deficient: {counts_all_samples_after_delete_deficient}')

# Zadanie 3 Dokonać dyskretyzacji atrybutów numerycznych.

for col in df:
    if df[col].dtype == float:
        df[col] = pd.cut(df[col].values,
                         [0, int(np.mean([min(df[col].values), max(df[col].values)])), max(df[col].values)]).codes
    else:
        classes = np.unique(df[col])
        df[col] = [np.where(classes == weather)[0][0] for weather in df[col]]
print(f'discretization  data: {df}')

# Zadanie 4 Wyznaczyć jakość zbioru danych i istotność atrybutów warunkowych.

n_conf_rows = df.drop_duplicates(subset=['population', 'cap-shape', 'cap-surface', 'cap-color'], keep=False)
# print(df[df != df_temp])
conf_rows = df[df.duplicated(subset=['population', 'cap-shape', 'cap-surface', 'cap-color'], keep=False)]
all_rows = len(df.axes[0])
# print(conf_rows)

quality_full = len(n_conf_rows) / all_rows

n_conf_rows1 = df.drop_duplicates(subset=['cap-shape', 'cap-surface', 'cap-color'], keep=False)
n_conf_rows2 = df.drop_duplicates(subset=['population', 'cap-surface', 'cap-color'], keep=False)
n_conf_rows3 = df.drop_duplicates(subset=['population', 'cap-shape', 'cap-color'], keep=False)
n_conf_rows4 = df.drop_duplicates(subset=['population', 'cap-shape', 'cap-surface'], keep=False)

quality_x1 = len(n_conf_rows1) / all_rows
quality_x2 = len(n_conf_rows2) / all_rows
quality_x3 = len(n_conf_rows3) / all_rows
quality_x4 = len(n_conf_rows4) / all_rows

r1 = (quality_full - quality_x1) / quality_full
r2 = (quality_full - quality_x2) / quality_full
r3 = (quality_full - quality_x3) / quality_full
r4 = (quality_full - quality_x4) / quality_full

print("Istotność atrybutu: " + df.columns[1] + " " + str(r1))
print("Istotność atrybutu: " + df.columns[2] + " " + str(r2))
print("Istotność atrybutu: " + df.columns[3] + " " + str(r3))
print("Istotność atrybutu: " + df.columns[4] + " " + str(r4))

# Zadanie 5 Wyznaczy najkrótszy redukt, gwarantujący założony poziom jakości zbioru.

df_tmp = df.copy()
accuracy = calculate_accuracy(df_tmp)
threshold=0.6
while accuracy > threshold:
    df_before_last_reduction = df_tmp.copy()

    relevances = []
    for column_name in df_tmp.iloc[:, 0:-1].columns:
        reduct = df_tmp.drop([column_name], axis=1)
        attrs = reduct.iloc[:, 0:-1].columns
        unique_rows = reduct.drop_duplicates(keep='first')
        conflict_rows = unique_rows.duplicated(subset=attrs, keep='first')
        unique_count = len(unique_rows.axes[0])
        conflict_count = len(conflict_rows.axes[0])
        relevance = conflict_count / unique_count
        relevances.append(relevance)
    column_index_to_drop =  np.argmin(relevances)

    column_name_to_drop = df_tmp.iloc[:, 0:-1].columns[column_index_to_drop]
    df_tmp = df_tmp.drop(column_name_to_drop, axis=1)
    accuracy = calculate_accuracy(df_tmp)

# df.drop_duplicates(subset=['population', 'cap-shape', 'cap-surface', 'cap-color'], keep=False)
print(f'Najkrótszy redukt\n{df_before_last_reduction}')
# Najkrótszy redukt => spore-print-color,  population,  habitat,  class

# Zadanie 6