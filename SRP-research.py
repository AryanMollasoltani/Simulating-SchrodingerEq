import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
import scipy.constants as scp

En = np.linspace(1, 15, 15)

hbarx = scp.hbar
m_e = scp.electron_mass

N = 3000

L = 2743e-3
dx = L / N

m = 510998
hbar = 197.326980
potk = (m) / (hbar ** 2)


# (5*(np.cos(np.pi*x))**2)
# 1/2*(x-1/2)**2
# 100*np.abs(x-1/2)
# 1/2*2*10000*(x-1/2)**2
# 1/2*m*4*(x-1/2)**2
# 100*(-np.absolute(0.5-0.5*np.floor(3*x))+0.5)
# 1000*np.exp(-(x-0.7)**2/(2*0.05**2))
# np.sin(3*np.pi*x/L)
# np.sin(3*np.pi*x/L)+1
# 10*np.abs(1-np.floor(3*x/L)) endelig brønd
# np.abs(27/(np.tan(np.pi*x/L)))
# 1*(-np.abs(np.sin(5*np.pi*x/L))+1)
# np.abs(0/(np.tan(np.pi*x/L)))+(np.sin(10*np.pi*x/L))**2
# (-np.abs(np.sin(np.pi*x/L))+1)-np.sin(10*x)
# np.abs(0.1/(np.tan(np.pi*x/L)))+2.4*np.cos(22*np.pi*x/L)+3
# np.abs(27/(np.tan(np.pi*x/L)))
# 2.2*np.cos(22*np.pi*x/L)+3
# (5.2)*(-np.absolute(0.5-0.5*np.floor(3*x/L))+0.5)
# 55*np.absolute(L/2-x)
# 20*L/x
# 11*L*x
# 30*(np.absolute(L/2-x)-np.cos(43*np.pi*x/L))+30

def Theo(En):
    return ((En ** 2) * (hbarx ** 2) * np.pi ** 2) / (2 * m_e * ((L * 1e-9)) ** 2)


EnergyLevels = Theo(En) * (1 / (1.602e-19))


def get_potential(x, dx):
    return 55 * np.absolute(L / 2 - x)


def get_eigh_well():
    x = np.linspace(0, L, N)
    e = (-1 / (dx ** 2) * np.ones(len(x) - 3))
    d = (2 / (dx ** 2) + 2 * potk * (get_potential(x, dx)[1:-1]))
    return eigh_tridiagonal(d, e)


x = np.linspace(0, L, N)
w, v = get_eigh_well()

# plt.figure(dpi=1200)
# Normalize and plot Egenfunction

fig, ax1 = plt.subplots(figsize=(8, 8), dpi=80)
ax2 = ax1.twinx()

for i in range(0, 10, 2):
    v0 = (1 / np.sqrt(sum(v[:, i] ** 2) * dx)) * v[:, i]
    v_p = v0 ** 2
    v_L = v_p + (w[i] / (2 * potk))
    labl = '$\psi_{{{:2d}}}$'.format(i + 1)
    ax1.plot(x[1:-1], v_L, label=labl)

for i in range(10, 12):
    v0 = (1 / np.sqrt(sum(v[:, i] ** 2) * dx)) * v[:, i]
    v_p = v0 ** 2
    v_L = v_p + (w[i] / (2 * potk))
    labl = '$\psi_{{{:2d}}}$'.format(i + 1)
    ax1.plot(x[1:-1], v_L, label=labl)

Energilevels = (w[0:15] / (2 * potk))
print(Energilevels)
print(((6.626e-34 * 3e8) / ((Energilevels[11] - Energilevels[10]) * (1.602e-19))) * 1e9)

# Plot
ax2.plot(x, get_potential(x, dx), color='k', label='V(x)')
ax2.legend(bbox_to_anchor=(1.14, 0.09))
ax1.legend(bbox_to_anchor=(1, 1.05), loc='upper right', ncol=10)
ax1.set_xlabel('x[nm]', fontsize=20)
ax2.set_ylabel('$V[eV]$', fontsize=20)
ax1.set_ylabel('$E_n[eV]+|\psi_n|^2$', fontsize=20)
ax2.set_ylim(0, 400)
ax1.set_ylim(-1)
ax1.set_xlim(0, L)
ax2.set_xlim(0, L)

plt.figure(figsize=(2, 4), dpi=100)
plt.eventplot(positions=[EnergyLevels, Energilevels], orientation='vertical', colors='r', linewidths=1,
              linelengths=0.95)
plt.ylabel('$E_n[eV]$')
# plt.xlabel('$Energiniveauer$')
plt.xlim((-1, 2))
plt.xticks(ticks=[-1, 0, 1, 2], labels=['', '$V=0$', 'simul', ''])
ax = plt.gca()
plt.show()



