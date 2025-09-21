# supernovafit.py
# Por Lucas Pereira de Souza
# Programinha para plotar os gráficos da distância de supernovas vs redshift
# Seguindo o modelo lambda-cdm, para comparação de dados experimentais
# de surveis vs curva teórica vs best fit

# Vamos começar importando módulos importantes
import numpy as np
import scipy.integrate as integrate
import scipy.optimize as opt
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Vamos definir as funções que vamos utilizar
def E(z):
    Omo = 0.3
    return 1/np.sqrt(1 - Omo + Omo*(1 + z)**3 + (8e-5)*(1+z)**4)

# Definindo h:
def ho(z, h):
    return E(z)/h

# Um dl com o h, para a primeira parte do problema
def dl(z, h):
    return 3e3*(1+z)*integrate.quad(ho,0, z, args=(h))[0]
dl = np.vectorize(dl)

def dist(z, h):
    return 5.0*np.log10(dl(z,h)) + 25.0

# Define h para a parte 1
h = 0.7

dados = open('jla_mub_1.txt', 'r') # Abre o arquivo
lines = dados.readlines() # Lê as linhas
dados.close() # Fecha 
data = [item.split("  ") for item in lines] # Divide nos espaços em branco
data = np.asarray(data, dtype=np.float64, order='C') # Transforma em matriz de pontos flutuantes

z = [row[0] for row in data] # Pega a primeira coluna pro redshift
z2 = np.linspace(0.01, 1.3, 100) # Gera 100 pontos igualmente espaçados entre 0.01 e 1.3 p/ z teórico

distexp = [row[1] for row in data] # pega segunda coluna dos dados pra ser a distancia experimental
distteo = [dist(item,h) for item in z2] # calcula a distancia teorica

erroexp = [row[2] for row in data] # Pega a terceira coluna dos dados como erro experimental

# Para calcular o h como parametro livre através de best-fit
hp, sh = opt.curve_fit(dist, z, distexp, sigma=erroexp, absolute_sigma=True)
# com variância sh

y = [dist(item, hp) for item in z2] # para a distância após o best-fit
ey = 5.0/(hp*np.log(10))*np.sqrt(sh) # Para calculo do erro propagado para a nova distância

# Estas curvas abaixo servirão apenas por questão estética, para demonstrar no grafico o erro do y
y1 = []
y2 = []
e = ey[0].tolist()
for i in range(len(y)):
    yy = y[i].tolist()
    y1.append(yy[0] + e[0])
    y2.append(yy[0] - e[0])

print("O melhor ajuste para h será h = %.8f $\pm$ %.8f"%(hp,np.sqrt(sh))) # mostra o que calculamos com um textinho, pra
# ajudar a gente a se localizar melhor

area = [300.0*item for item in erroexp] # Isso vai servir de parametro pra a area dos pontos no grafico experimental
# Pra indicar visualmente quais medidas tiveram erro maior ou menor

# Plota a primeira figura e salva um arquivo desta
plt.figure(figsize=(12,8))
plt.scatter(z, distexp, s=area, c='r', label='Dados Experimentais')
plt.plot(z2,distteo,c='b',label='Curva Teórica')
plt.legend()
plt.xlabel("Redshift (z)")
plt.ylabel("Distância")
plt.title("Redshift vs Distância - Dados Experimentais vs Curva Teórica")
plt.savefig('Parte1.png')

# Plota a segunda figura e salva um arquivo desta
plt.figure(figsize=(12,8))
plt.scatter(z, distexp, s=area, c='r', label='Dados Experimentais')
plt.plot(z2,y,c='b',label=('Best-Fit h = %.8f $\pm$ %.8f'%(hp,np.sqrt(sh))))
plt.fill_between(z2,y1,y2) # Para o erro propagado
plt.legend()
plt.xlabel("Redshift (z)")
plt.ylabel("Distância")
plt.title("Redshift vs Distância - Dados Experimentais vs Best-Fit")
plt.savefig('Parte2.png')

plt.show()
# Fim
