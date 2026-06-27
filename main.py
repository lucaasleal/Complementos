import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

t, s, tau = sp.symbols('t s tau')

R, L, C, V_0 = sp.symbols('R L C V_0', positive = True)

i = sp.Function('i')
v = sp.Function('v')

u = sp.Heaviside(t)

I_s = sp.symbols('I_s')

term_R = sp.laplace_transform(R * i(t), t, s, noconds=True)
term_L = sp.laplace_transform(L * i(t).diff(t), t, s, noconds=True)
term_int = I_s / (C * s)


term_R = term_R.subs(sp.LaplaceTransform(i(t), t, s), I_s).subs(i(0), 0)
term_L = term_L.subs(sp.LaplaceTransform(i(t), t, s), I_s).subs(i(0), 0)
left_eq  = V_0 / s
right_eq = term_R + term_L + term_int

I_s = sp.factor(sp.solve(sp.Eq(left_eq, right_eq), I_s)[0])
print(I_s)

#===========================================================
#Receber valores usuário
r = sp.Rational(input("Insira resistência:"))
l = sp.Rational(input("Insira indutância:"))
c = sp.Rational(input("Insira capacitância:"))
v_0 = sp.Rational(input("Insira fonte:"))

I_s_sol = sp.factor(I_s.subs({R: r, L: l, C: c, V_0 : v_0}))
print("I(s) = ", I_s_sol)


#===========================================================
#Polos complexos
polos = sp.roots(sp.denom(I_s_sol), s)
i_t = 0

print(polos)

for polo, mult in polos.items():
    phi = (s-polo)**mult * I_s_sol * sp.exp(s*t)
    
    res = sp.diff(phi, s, mult - 1).subs(s, polo) / sp.factorial(mult-1)
    res = sp.simplify(res)
    i_t += res
    print(f"Resíduo do Polo (s = {polo}): {res}")


#Plotagem da função
i = sp.lambdify(t, i_t) 
t = np.linspace(0, 5, 1000)


plt.figure(figsize=(8, 5))
plt.plot(t, i(t))
plt.grid(True)
plt.xlabel('tempo (s)')
plt.ylabel('i(t)')
plt.title('Função de Saída')
plt.savefig("funcao")
plt.show()

#Plotagem da resposta em função da frequencia
'''
w = sp.symbols('w', real=True)

I_jw = sp.simplify(I_s_sol.subs(s, sp.I*w))

mod = sp.Abs(I_jw)

f = sp.lambdify(w, mod, "numpy")

omega = np.linspace(-10, 20, 1000)

plt.figure(figsize=(8,5))
plt.plot(omega, f(omega))
plt.grid(True)
plt.xlabel(r'$\omega$ (rad/s)')
plt.ylabel(r'$|I(j\omega)|$')
plt.title("Resposta na frequência")
plt.savefig("Complexo")
plt.show()
'''

#POLOS

plt.figure(figsize=(7,7))

for polo, mult in polos.items():
    x = float(sp.re(sp.N(polo)))
    y = float(sp.im(sp.N(polo)))
    plt.scatter(x, y, color='red', marker='x', s=120)
    plt.text(x + 0.1, y + 0.1, f"{sp.N(polo,3)}")

    max_coord = 0

for polo in polos:
    max_coord = max(max_coord,
        abs(float(sp.re(sp.N(polo)))),
        abs(float(sp.im(sp.N(polo))))
        )
margem = 1

plt.xlim(-(max_coord + margem), max_coord + margem)
plt.ylim(-(max_coord + margem), max_coord + margem)

plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)

plt.xlabel("Parte Real (σ)")
plt.ylabel("Parte Imaginária (jω)")
plt.title("Polos no Plano Complexo")

plt.grid(True)

maior_real = max(float(sp.re(sp.N(p))) for p in polos)

# Pequena margem para ficar à direita dos polos
gamma = maior_real + 0.5

# Intervalo da integração (mesmo intervalo do gráfico)
ymin, ymax = plt.ylim()

plt.plot(
    [gamma, gamma],
    [ymin, ymax],
    'b--',
    linewidth=2,
    label=fr'$\Re(s)=\gamma={gamma:.2f}$'
)

plt.legend()

plt.savefig("polos.png", dpi=300)
plt.show()





#PLOT 3D
from mpl_toolkits.mplot3d import Axes3D
# Definição automática da região de plot

reais = np.array([float(sp.re(sp.N(p))) for p in polos])
imag = np.array([float(sp.im(sp.N(p))) for p in polos])

# Caso exista apenas um polo
if len(reais) == 1:
    span_real = 1
else:
    span_real = max(reais) - min(reais)

if len(imag) == 1:
    span_imag = 1
else:
    span_imag = max(imag) - min(imag)

# Margem proporcional ao tamanho da região
margem_real = max(1.0, span_real)
margem_imag = max(1.0, span_imag)

sigma = np.linspace(
    min(reais) - margem_real,
    max(reais) + margem_real,
    300
)

omega = np.linspace(
    min(imag) - margem_imag,
    max(imag) + margem_imag,
    300
)

SIGMA, OMEGA = np.meshgrid(sigma, omega)

# Avaliação da função

F = sp.lambdify(s, I_s_sol, "numpy")

S = SIGMA + 1j*OMEGA

Z = np.abs(F(S))

# Remove infinitos dos polos
Z[np.isinf(Z)] = np.nan
Z[np.isnan(Z)] = np.nan

# Escala automática para não deixar um polo dominar o gráfico
limite = np.nanpercentile(Z, 99)

Z = np.clip(Z, 0, limite)

# Plot

fig = plt.figure(figsize=(11,8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    SIGMA,
    OMEGA,
    Z,
    cmap="viridis",
    edgecolor="none"
)

# Marca os polos
for p in polos:
    xr = float(sp.re(sp.N(p)))
    yi = float(sp.im(sp.N(p)))

    ax.scatter(
        xr,
        yi,
        limite,
        color="red",
        marker="x",
        s=100
    )

ax.set_xlabel("Parte Real (σ)")
ax.set_ylabel("Parte Imaginária (jω)")
ax.set_zlabel(r"$|I(s)|$")
ax.set_title("Superfície de |I(s)|")

fig.colorbar(surf, shrink=0.6, label=r"$|I(s)|$")

plt.tight_layout()
plt.show()
