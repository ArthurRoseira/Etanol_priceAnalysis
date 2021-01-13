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


def adfuller_test(serie, figsize=(18, 4), plot=True, title=''):
    if plot:
        serie.plot(figsize=figsize, title=title)
        plt.show()
    adf = adfuller(serie)
    output = pd.Series(adf[0:4], index=['Teste Estatistico Dickey Fuller', 'Valor-P',
                                        'Lags Usados', 'Número de observações usadas'])
    output = round(output, 4)
    for key, value in adf[4].items():
        output["Valores Críticos (%s)" % key] = value.round(4)
    return output


os.chdir('C:\\Users\\arthu\\OneDrive\\Ambiente de Trabalho\\Projects\\Python\\DataSeries\\Etanol- Esalq\\Etanol_priceAnalysis')
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
plt.show()
# Trend, sazonality and residue
# sazonality of 52 weeks or 1 year
season = seasonal_decompose(treino, period=52)
fig = season.plot()
fig.set_size_inches(16, 8)
plt.show()
# Dickey Fuller Test for stationary series
# https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html
adfinput = adfuller(treino['Preço R$'], autolag='AIC')
print(adfinput)
adftest = pd.Series(adfinput[0:4], index=['Teste Estatistico Dickey Fuller',
                                          'Valor-P', 'Lags Usados', 'Número de observações usadas'])
adftest = round(adftest, 4)
for key, value in adfinput[4].items():
    adftest["Valores Críticos (%s)" % key] = value.round(4)
print(adftest)
# KPSS Test
kpss_input = kpss(treino['Preço R$'], regression='c', nlags="auto")
kpss_test = pd.Series(kpss_input[0:3], index=[
                      'Teste Statistico KPSS', 'Valor-P', 'Lags Usados'])
kpss_test = round(kpss_test, 4)
for key, value in kpss_input[3].items():
    kpss_test["Valores Críticos (%s)" % key] = value
print(kpss_test)

# Stationary Series Transformation
plt.figure(figsize=(18, 10))
plt.title('Preços do Etanol Hidratado')
treino.loc['2011':'2012', 'Preço R$'].plot()
plt.show()
# Differencing Data Series function .diff()
plt.figure(figsize=(18, 10))
plt.title('Preços do Etanol Hidratado')
treino.loc['2011':'2012', 'Preço R$'].diff().dropna().plot()
plt.show()
print('Mostrando as 10 primeiras diferenciações')
print(treino['Preço R$'].diff().dropna().head(10))

# Adfuller Test For Diff series
adfuller1 = adfuller_test(treino['Preço R$'].diff().dropna(
), plot=True, title='Preços do Etanol com primeira diferenciação')
print(adfuller1)

# Inflation Price adjust !Important for prices
infl = pd.read_excel('IPCA.xls', sheet_name='IPCA')
infl['Data'] = pd.to_datetime(infl['Data'])
infl.set_index('Data', inplace=True)
print(infl.head(-10))
infl['Acumulado'] = pd.to_numeric(infl['Acumulado'])
inf_actual = infl.loc[infl.index.max()]
inf_actual = inf_actual['Acumulado']
# Creating New columns in training DB for the Month and Year
treino['Ano'] = treino.index.year
treino['Mês'] = treino.index.month
index = treino.index
print(treino.head(10))

# Merge Inflation DataFrame Into Training DB
treino = treino.merge(
    infl.loc[:, ['Ano', 'Mês', 'Acumulado']], how='left', on=['Ano', 'Mês'])
treino['Preço Ajustado'] = (
    treino['Preço R$']/treino['Acumulado']) * inf_actual
treino.set_index(index, inplace=True)
treino.dropna(inplace=True)
print(treino.loc[:, ['Preço R$',
                     'Acumulado', 'Preço Ajustado']].head())
adfuller2 = adfuller_test(treino.loc[:, 'Preço Ajustado'], plot=True,
                          title='Preço do Etanol ajustado pelo IPCA')
print(adfuller2)
treino.loc[:, ['Preço R$', 'Preço Ajustado']].plot(
    figsize=(18, 5), title='Preços Etanol - Original x Ajustado Inflação')
# # The ylim() function in pyplot module of matplotlib library is used to get or set the y-limits of the current axes.
plt.ylim([0, 2700])
plt.show()
