## Przykład
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
sns.set()

print('-------------------------------------')
# x = np.random.randn(5)
# y = np.random.randn(5)
# print(f'randnx: {x}')
# print(f'randny: {y}')
#
# print(f'mean_x: {x.mean()}')
#
# print(f'std_x: {x.std()}')
#
# print(f'corel_xy:\n {np.corrcoef(x, y)}')

print('=======Rozklad normalny========')

## Przyklad

# mu = 0  #avg
# sigma = 1   #std
#
# x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
# plt.title('Rozklad normalny - funkcja gestosci')
# plt.plot(x, norm.pdf(x, mu, sigma))
# plt.show()

## Przyklad
# mu = 0  #avg
# sigma = 1   #std
#
# x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 500)
# params = [(0, 1), (0, 2), (0, 0.5)]
#
# plt.figure(figsize = (6, 6))
# for mu, sigma in params:
#     plt.plot(x, norm.pdf(x, mu, sigma), label = f'mu={mu}, sigma={sigma}')
# plt.legend()
# plt.show()

## Przyklad

# mu = 10  #avg
# sigma = 3   #std
#
# x = np.linspace(mu - 8 * sigma, mu + 8 * sigma, 500)
#
# plt.plot(x, norm.pdf(x, mu, sigma), label = f'mu={mu}, sigma={sigma}')
# plt.plot(x, norm.pdf(x, mu-mu, sigma), label = f'mu={mu-mu}, sigma={sigma}')
# plt.plot(x, norm.pdf(x, mu-mu, sigma/sigma), label = f'mu={mu-mu}, sigma={sigma/sigma}')
# plt.legend()
# plt.show()

print('========Rozklad normalny - dystrybuanta=========')
## Przyklad
# mu = 10  #avg
# sigma = 3   #std
#
# x = np.linspace(mu - 8 * sigma, mu + 8 * sigma, 500)
#
# plt.title('Rozklad normalny - dystrybuanta')
# plt.plot(x, norm.cdf(x, mu, sigma))
# plt.show()

print('========Rozklad normalny - funkcja przezycia=========')
## Przyklad

mu = 10  #avg
sigma = 3   #std

x = np.linspace(mu - 8 * sigma, mu + 8 * sigma, 500)

plt.title('Rozklad normalny - funkcja przezycia')
plt.plot(x, norm.sf(x, mu, sigma))
plt.show()