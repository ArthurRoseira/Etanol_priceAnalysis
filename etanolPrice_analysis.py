import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
import math

# https://nbviewer.jupyter.org/github/leandrovrabelo/tsmodels/blob/master/notebooks/portugues/Princi%CC%81pios%20Ba%CC%81sicos%20para%20Prever%20Se%CC%81ries%20Temporais.ipynb

os.chdir('C:\\Users\\arthu\\OneDrive\\Ambiente de Trabalho\\Projects\\Python\\DataSeries\\Etanol- Esalq')
df = pd.read_excel("cepea-consulta-20210110095152.xls")
# Transform string to date format in Pandas
df['Data'] = pd.to_datetime(df['Data'], dayfirst=True)
df['À vista R$'] = df['À vista R$'].str.replace(",", ".")
df['À vista US$'] = df['À vista US$'].str.replace(",", ".")
df['À vista R$'] = df['À vista R$'].astype(float)
df['À vista US$'] = df['À vista US$'].astype(float)
print(df.head())
df.sort_values('Data')
df.set_index('Data', inplace=True)
df.columns = ['Preço R$', 'Preço US$']
df['Preço R$'] = df['Preço R$'] * 1000
print(df.head())
print(df.shape[0])
print(df.dtypes)
base_testSize = int(round(df.shape[0]*0.2, 0))
treino = df.iloc[:-base_testSize, 0:1].copy()
teste = df.iloc[-base_testSize:, 0:1].copy()
plt.figure(figsize=(18, 10))
plt.title('Preço do Etanol Hidratado - Base Esalq')
plt.plot(treino['Preço R$'], color='b')
plt.plot(teste['Preço R$'], color='orange')
plt.legend(['Treino', 'Teste'])
plt.xlabel('Data')
plt.ylabel('Preço')
# plt.show()
# Trend, sazonality and residue
# sazonality of 52 weeks or 1 year
season = seasonal_decompose(treino, period=52)
fig = season.plot()
fig.set_size_inches(16, 8)
# plt.show()
# Dickey Fuller Test for stationary series
adfinput = adfuller(treino['Preço R$'])
print(adfinput)
adftest = pd.Series(adfinput[0:4], index=['Teste Estatistico Dickey Fuller',
                                          'Valor-P', 'Lags Usados', 'Número de observações usadas'])
adftest = round(adftest, 4)
for key, value in adfinput[4].items():
    adftest["Valores Críticos (%s)" % key] = value.round(4)
print(adftest)
# KPSS Test
