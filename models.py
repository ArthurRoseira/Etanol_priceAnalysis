from scipy.stats import boxcox
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
from sklearn.metrics import mean_squared_error, mean_absolute_error

# https://docs.oracle.com/cd/E16582_01/doc.91/e15111/und_forecast_levels_methods.htm#EOAFM00004


def error_check(orig, prev, nome_col='', nome_indice=''):
    vies = np.mean(orig - prev)
    mse = mean_squared_error(orig, prev)
    rmse = math.sqrt(mean_squared_error(orig, prev))
    mae = mean_absolute_error(orig, prev)
    mape = np.mean(np.abs((orig-prev)/orig))*100
    grupo_erro = [vies, mse, rmse, mae, mape]
    serie = pd.DataFrame(grupo_erro, index=[
        'VIÉS', 'MSE', 'RMSE', 'MAE', 'MAPE'], columns=[nome_col])
    serie.index.name = nome_indice
    return serie


def errorPLot(dados, figsize=(18, 8)):
    dados['Erro'] = dados.iloc[:, 0] - dados.iloc[:, 1]
    plt.figure(figsize=figsize)
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0))
    ax4 = plt.subplot2grid((2, 2), (1, 1))

    # Graph real x prev
    ax1.plot(dados.iloc[:, 0:2])
    ax1.legend(['Real', 'Prev'])
    ax1.set_title('Valores Reais vs Previstos')

    # Error x Prev
    ax2.scatter(dados.iloc[:, 1], dados.iloc[:, 2])
    ax2.set_xlabel('Valores Previstos')
    ax2.set_ylabel('Resíduo')
    ax2.set_title('Resíduo vs Valores Previstos')

    # QQ Plot
    sm.graphics.qqplot(dados.iloc[:, 2], line='r', ax=ax3)

    # Graph correlation
    plot_acf(dados.iloc[:, 2], lags=60, zero=False, ax=ax4)
    plt.tight_layout()
    plt.show()


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

# Merge Inflation DataFrame Into Training DB
treino = treino.merge(
    infl.loc[:, ['Ano', 'Mês', 'Acumulado']], how='left', on=['Ano', 'Mês'])
treino['Preço Ajustado'] = (
    treino['Preço R$']/treino['Acumulado']) * inf_actual
treino.set_index(index, inplace=True)
treino.dropna(inplace=True)

# Model using previous period value
# Just Used to generate error and validations values as base to other methods of prediction
# Cannot be used to do forecasting
# Training DB
simples_treino = treino[['Preço Ajustado']]
simples_treino.columns = ['Real']
simples_treino['Previsão'] = simples_treino['Real'].shift()
simples_treino.dropna(inplace=True)
simples_treino.plot()
plt.show()
erro_treino = error_check(
    simples_treino['Real'], simples_treino['Previsão'], nome_col='Simples', nome_indice='Base Treino')
errorPLot(simples_treino)
# Testing DB
simples_teste = teste[['Preço R$']]
simples_teste.columns = ['Real']
hist = [simples_treino.iloc[i, 0] for i in range(len(simples_treino))]
prev = []
for t in range(len(simples_teste)):
    yhat = hist[-1]
    obs = simples_teste.iloc[t, 0]
    prev.append(yhat)
    hist.append(obs)

simples_teste['Previsão'] = prev
erro_teste = error_check(
    simples_teste['Real'], simples_teste['Previsão'], nome_col='Simples', nome_indice='Base Teste')


# Simple Mean Model
# Training DB
ms_treino = treino[['Preço Ajustado']]
ms_treino.columns = ['Real']
ms_treino['Previsão'] = ms_treino['Real'].expanding().mean()
erro_treino['Média Simples'] = error_check(
    ms_treino['Real'], ms_treino['Previsão'])
errorPLot(ms_treino)
# Testing DB
ms_teste = teste[['Preço R$']]
ms_teste.columns = ['Real']
hist = [ms_treino.iloc[i, 0] for i in range(len(ms_treino))]
prev = []
for t in range(len(ms_teste)):
    # Mean of array Hist, all values
    yhat = np.mean(hist)
    obs = ms_teste.iloc[t, 0]
    prev.append(yhat)
    hist.append(obs)

ms_teste['Previsão'] = prev
erro_teste['Média Simples'] = error_check(
    ms_teste['Real'], ms_teste['Previsão'], nome_col='Mean Model')

# Simple Moving Mean
# Training DB
mm_treino = treino[['Preço Ajustado']]
mm_treino.columns = ['Real']
mm_treino['Previsão'] = mm_treino.rolling(5).mean()
mm_treino.dropna(inplace=True)
erro_treino['Média Movel'] = error_check(
    mm_treino['Real'], mm_treino['Previsão'])
# Testing DB
mm_teste = teste[['Preço R$']]
mm_teste.columns = ['Real']
hist = [mm_treino.iloc[i, 0] for i in range(len(mm_treino))]
prev = []
for t in range(len(mm_teste)):
    yhat = np.mean(hist[-5:])
    obs = mm_teste.iloc[t, 0]
    prev.append(yhat)
    hist.append(obs)

mm_teste['Previsão'] = prev
errorPLot(mm_teste)
erro_teste['Média Móvel'] = error_check(mm_teste['Real'], mm_teste['Previsão'])
print(erro_teste.head(10))
