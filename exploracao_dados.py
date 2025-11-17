# %%

import numpy as np
import matplotlib.pyplot as plt

# %%
# O dataset tem 7 anos de infromação (84 meses (7x12) + 3 meses de 2020)
# A lista vai de 1 a 87 + 1 para que o último dado seja englobado
# Valor de incremento é 1 

np.arange(1,88,1)

# %%
# Retorno dos dados em formato de array

dados = np.loadtxt('./data/apples_ts.csv', delimiter=',', usecols=np.arange(1, 88, 1))

# %%

dados 

# %% 
# Verifica a quantidade de dimensões do array

dados.ndim

# %%
# Quantidade de elementos de um array

dados.size

# %%
# Número de elementos em cada dimensão

dados.shape

# %%
# Troca linhas por colunas 
# Cada coluna representa os preços de determinada cidade
# Com exceção da primeira, que corresponde à data

dados.T

# %% 
# Salvar transposição

dados_transposto = dados.T

# %%
# Separar a primeira coluna (representa a data)

datas = dados_transposto[:,0]

# %%
# Separar as informações referente aos preços
# 5 localidades, que correspondem às 5 colunas,
# Intervalo vai de 1 a 6

precos = dados_transposto[:,1:6]

# %%
# No x, colocaremos as datas, no y, os preços da cidade de Moscow
# Para selecionar os valores dessa cidade utilizamos o índice 0
# Criar sequência de números com datas

datas = np.arange(1, 88)
plt.plot(datas,precos[:,0])

# %%

Moscow = precos[:,0]
Kaliningrad = precos[:,1]
Petersburg = precos[:,2]
Krasnodar = precos[:,3]
Ekaterinburg = precos[:,4]

# %%

Moscow.shape

# %%
# Separando por anos

Moscow_ano1 = Moscow[0:12] 
Moscow_ano2 = Moscow[12:24]
Moscow_ano3 = Moscow[24:36]
Moscow_ano4 = Moscow[36:48]

# %%

plt.plot(np.arange(1,13,1), Moscow_ano1)
plt.plot(np.arange(1,13,1), Moscow_ano2)
plt.plot(np.arange(1,13,1), Moscow_ano3)
plt.plot(np.arange(1,13,1), Moscow_ano4)
plt.legend(['ano1', 'ano2', 'ano3', 'ano4'])

# %%
# Verificar se 2 arrays são iguais

np.array_equal(Moscow_ano3, Moscow_ano4)

# %%
# Verificar se há uma grande diferença entre os valores de um ano para outro

np.allclose(Moscow_ano3, Moscow_ano4,0.01)
np.allclose(Moscow_ano3, Moscow_ano4,10)

# %%

plt.plot(datas, Kaliningrad)

# %%

Kaliningrad

# %%
# Verificar a quantidade de valores nulos

sum(np.isnan(Kaliningrad))

# %%
# Calcular a média entre o valor anterior e o posterior à NaN

np.mean([Kaliningrad[3],Kaliningrad[5]])

# %%
# Substituir NaN por este valor

Kaliningrad[4] = np.mean([Kaliningrad[3],Kaliningrad[5]])

# %% 

plt.plot(datas, Kaliningrad)

# %%
# O valor de y é equivalente ao preço das maçãs
# x corresponde ao mês (valor de 1 a 87)
# a é o coeficiente angular
# b, coeficiente linear, onde a reta corta o eixo y

x = datas
y = 2*x+80

# %% 

plt.plot(datas,Moscow)
plt.plot(x,y)

# %%

Moscow-y

# %%
# Para lidar com este problema de valores negativos e positivos
# podemos elevá-los ao quadrado utilizando a função power()

np.sum(np.power(Moscow-y,2))
# %%
# Calcular a raiz utilizando a função sqrt()
# Define a qualidade de nosso ajuste

np.sqrt(np.sum(np.power(Moscow-y,2)))

# %% 
# Tentar outro valor

y = 0.52*x+80

# %% 

plt.plot(datas,Moscow)
plt.plot(x,y)

# %%

np.sqrt(np.sum(np.power(Moscow-y,2)))

# Valor menor de ajuste
# %%
# Coeficientes angular e linear da reta, que nos permitem calcular o y e ajustar esta reta

Y = Moscow
X = datas
n = np.size(Moscow)

# %%
# Coeficientes angular (a)

a = (n*np.sum(X*Y) - np.sum(X)*np.sum(Y))/(n*np.sum(X**2) - np.sum(X)**2)
a
# %%
# Coeficiente linear (b)

b = np.mean(Y) - a*np.mean(X)
y = a*X+b
np.linalg.norm(Moscow-y)

# %%
