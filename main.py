import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

t, s, tau = sp.symbols('t s tau')
t = sp.symbols('t', real=True) 

R, L, C, V_0 = sp.symbols('R L C V_0', positive=True)

i = sp.Function('i')
v = sp.Function('v')

u = sp.Heaviside(t)

I_s = sp.symbols('I_s')

term_R = sp.laplace_transform(R * i(t), t, s, noconds=True)
term_L = sp.laplace_transform(L * i(t).diff(t), t, s, noconds=True)
term_int = I_s / (C * s)

term_R = term_R.subs(sp.LaplaceTransform(i(t), t, s), I_s).subs(i(0), 0)
term_L = term_L.subs(sp.LaplaceTransform(i(t), t, s), I_s).subs(i(0), 0)
left_eq = V_0 / s
right_eq = term_R + term_L + term_int

I_s = sp.factor(sp.solve(sp.Eq(left_eq, right_eq), I_s)[0])
print(I_s)


def calcula(r, l, c, v0):
    r, l, c, v0 = [sp.nsimplify(x, rational=True) for x in (r, l, c, v0)]

    I_s_sol = sp.factor(I_s.subs({R: r, L: l, C: c, V_0: v0}))
    print("I(s) = ", I_s_sol)

    den = sp.denom(I_s_sol)
    num = sp.numer(I_s_sol)

    polos_exatos = sp.roots(den, s)
    polos = {sp.N(polo): mult for polo, mult in polos_exatos.items()}
    i_t = 0

    print(polos)

    for polo, mult in polos.items():
        # Divisao polinomial exata: Q(s) = D(s) / (s-polo)^mult,
        # cancelando o fator do proprio polo ANTES de montar phi.
        Q = sp.quo(sp.Poly(den, s), sp.Poly((s - polo) ** mult, s)).as_expr()
        phi = num * sp.exp(s * t) / Q

        res = sp.diff(phi, s, mult - 1).subs(s, polo) / sp.factorial(mult - 1)
        res = sp.simplify(res)
        i_t += res
        print(f"Residuo do Polo (s = {polo}): {res}")

    i_t = sp.simplify(sp.expand_complex(i_t))

    return I_s_sol, polos, i_t

# ============================================================
# Interface interativa - 3 graficos separados
# ============================================================

fig = plt.figure(figsize=(14, 8))

ax_time = fig.add_axes([0.05, 0.60, 0.38, 0.32])
ax_poles = fig.add_axes([0.05, 0.18, 0.38, 0.32])
ax3d = fig.add_axes([0.48, 0.13, 0.48, 0.78], projection="3d")

plt.subplots_adjust(bottom=0.25)

axR = plt.axes([0.06, 0.09, 0.88, 0.02])
axL = plt.axes([0.06, 0.06, 0.88, 0.02])
axC = plt.axes([0.06, 0.03, 0.88, 0.02])
axV = plt.axes([0.06, 0.00, 0.88, 0.02])

sR = Slider(axR, "R", 0.1, 10, valinit=1)
sL = Slider(axL, "L", 0.1, 10, valinit=1)
sC = Slider(axC, "C", 0.1, 10, valinit=1)
sV = Slider(axV, "V0", 1, 20, valinit=5)


def update(val=None):
    r, l, c, v = sR.val, sL.val, sC.val, sV.val

    I, polos, it = calcula(r, l, c, v)

    ax_time.clear()
    ax_poles.clear()
    ax3d.clear()

    reais = [float(sp.re(sp.N(p))) for p in polos]
    imag = [float(sp.im(sp.N(p))) for p in polos]

    # --- Resposta temporal ---
    reais_abs = [abs(x) for x in reais if abs(x) > 1e-8]
    tf = 5 / min(reais_abs) if reais_abs else 10

    tempo = np.linspace(0, tf, 800)
    f = sp.lambdify(t, it, "numpy")
    ax_time.plot(tempo, f(tempo))
    ax_time.grid(True)
    ax_time.set_xlabel("tempo (s)")
    ax_time.set_ylabel("i(t)")
    ax_time.set_title("Resposta temporal")

    # --- Polos no plano complexo ---
    for p in polos:
        x = float(sp.re(sp.N(p)))
        y = float(sp.im(sp.N(p)))
        ax_poles.scatter(x, y, c='r', marker='x', s=90)
        ax_poles.text(x + 0.1, y + 0.1, f"{sp.N(p, 3)}", fontsize=8)

    m = max([1] + [abs(val) for val in reais + imag])
    ax_poles.set_xlim(-m - 1, m + 1)
    ax_poles.set_ylim(-m - 1, m + 1)
    ax_poles.axhline(0, color='k', linewidth=1)
    ax_poles.axvline(0, color='k', linewidth=1)
    ax_poles.grid(True)
    ax_poles.set_xlabel("Parte Real (\u03c3)")
    ax_poles.set_ylabel("Parte Imaginaria (j\u03c9)")
    ax_poles.set_title("Polos no plano complexo")

    # --- Superficie |I(s)| ---
    sig = np.linspace(-m - 1, m + 1, 120)
    omg = np.linspace(-m - 1, m + 1, 120)
    S, O = np.meshgrid(sig, omg)
    F = sp.lambdify(s, sp.Abs(I), "numpy")

    with np.errstate(divide='ignore', invalid='ignore'):
        Z = np.abs(F(S + 1j * O))

    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0)
    limite = np.nanpercentile(Z, 99) if np.any(Z) else 1.0
    Z = np.clip(Z, 0, limite)

    ax3d.plot_surface(S, O, Z, cmap="viridis", edgecolor="none")
    ax3d.set_xlabel("Parte Real (\u03c3)")
    ax3d.set_ylabel("Parte Imaginaria (j\u03c9)")
    ax3d.set_zlabel("|I(s)|")
    ax3d.set_title("Superficie de |I(s)|")

    fig.canvas.draw_idle()


for sl in (sR, sL, sC, sV):
    sl.on_changed(update)

update()
plt.show()
