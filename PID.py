# -*- coding: utf-8 -*-
"""rlc_pid_animado_botoes.py

# 🔌 Simulador RLC + PID — Análise Completa
**Animação interativa com sliders em tempo real**

- Diagrama Polo-Zero: RLC (malha aberta) vs Malha Fechada com PID
- Superfície 3D de |V(s)|   ← (antes era |I(s)|)
- Resposta v(t) em malha aberta (SymPy)  ← (antes era i(t))
- Simulação PID em tempo real
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")  # Backend para execução local
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from collections import deque
import sympy as sp
import warnings
warnings.filterwarnings("ignore")

print(f"matplotlib {matplotlib.__version__} | backend: {matplotlib.get_backend()}")

BG   = "#0f0f13"
CARD = "#16161e"
GRID = "#2a2a3a"
TXT  = "#cdd6f4"
SUB  = "#6c7086"
BLUE = "#89b4fa"
RED  = "#f38ba8"
GRN  = "#a6e3a1"
AMB  = "#fab387"
TEAL = "#94e2d5"
PURP = "#cba6f7"

def style_ax(ax, title):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=SUB, labelsize=7.5)
    for sp_ in ax.spines.values():
        sp_.set_color(GRID); sp_.set_linewidth(0.5)
    ax.xaxis.label.set_color(SUB)
    ax.yaxis.label.set_color(SUB)
    ax.title.set_color(TXT)
    ax.grid(True, color=GRID, linewidth=0.4, alpha=0.8)
    ax.set_title(title, fontsize=9, pad=5, fontweight="bold")

print("✅ Tema carregado.")

# ── Simbólico ─────────────────────────────────────────────────────────
t_sym, s_sym = sp.symbols('t s', real=True)
R_sym, L_sym, C_sym, V0_sym = sp.symbols('R L C V_0', positive=True)
I_sym = sp.symbols('I_s')

# I(s) = V0*C / (LCs² + RCs + 1)
_left  = V0_sym / s_sym
_right = I_sym * (R_sym + L_sym * s_sym + 1 / (C_sym * s_sym))
I_s_geral = sp.factor(sp.solve(sp.Eq(_left, _right), I_sym)[0])

# V(s) = Vc(s) = I(s) / (C·s)  →  tensão no capacitor
V_s_geral = sp.factor(I_s_geral / (C_sym * s_sym))


def calcular_rlc(r, l, c, v0):
    """Retorna I_s_num, V_s_num, polos e i(t) (inversa de Laplace de I(s))."""
    r, l, c, v0 = [sp.nsimplify(x, rational=True) for x in (r, l, c, v0)]

    # ── I(s) — para o gráfico i(t) e polo-zero ────────────────────
    I_s_num = sp.factor(I_s_geral.subs({R_sym: r, L_sym: l, C_sym: c, V0_sym: v0}))
    den_i = sp.denom(I_s_num)
    num_i = sp.numer(I_s_num)
    polos = {sp.N(p): m for p, m in sp.roots(den_i, s_sym).items()}

    # Inversa de Laplace de I(s) via resíduos
    i_t = sp.Integer(0)
    for polo, mult in polos.items():
        Q   = sp.quo(sp.Poly(den_i, s_sym), sp.Poly((s_sym - polo)**mult, s_sym)).as_expr()
        phi = num_i * sp.exp(s_sym * t_sym) / Q
        res = sp.diff(phi, s_sym, mult - 1).subs(s_sym, polo) / sp.factorial(mult - 1)
        i_t += sp.simplify(res)
    i_t = sp.simplify(sp.expand_complex(i_t))

    # ── V(s) — apenas para a superfície 3D ───────────────────────
    V_s_num = sp.factor(V_s_geral.subs({R_sym: r, L_sym: l, C_sym: c, V0_sym: v0}))

    return I_s_num, V_s_num, polos, i_t


def _pz_numericos(r, l, c, kp, ki, kd):
    """Polos/zeros de malha aberta (RLC) e malha fechada (PID)."""
    a2, a1, a0 = l*c, r*c, 1.0
    sq = np.sqrt(complex(a1**2 - 4*a2*a0))
    polos_rlc = [(-a1+sq)/(2*a2), (-a1-sq)/(2*a2)]
    b3, b2, b1, b0 = c*l, c*r+kd, kp+1.0, ki
    polos_mf  = list(np.roots([b3, b2, b1, b0]))
    zeros_mf  = list(np.roots([kd, kp, ki])) if abs(kd) > 1e-12 else \
                ([complex(-ki/kp)] if abs(kp) > 1e-12 else [])
    # V(s) = I(s)/Cs → zero extra em s=0 para malha aberta
    # (polo em s=0 cancela com o zero do numerador de I(s))
    # Os polos de V(s) são os mesmos de I(s); não há zeros na MA
    zeros_rlc = []
    return polos_rlc, zeros_rlc, polos_mf, zeros_mf


def calc_zeta(R, L, C):
    return R / (2.0*np.sqrt(L/C)), 1.0/np.sqrt(L*C)

print("✅ Funções simbólicas prontas.")

# ═══════════════════════════════════════════════════════════════════════
# SIMULADOR ANIMADO
# ═══════════════════════════════════════════════════════════════════════

plt.close("all")

# ── Parâmetros iniciais ────────────────────────────────────────────────
R0, L0, C0, V0_0       = 2.0, 1.0, 0.5, 5.0
KP0, KI0, KD0, VREF0   = 5.0, 2.0, 1.0, 1.0
DT, WINDOW, INTERVAL_MS = 0.02, 400, 40

# ── Estado PID ────────────────────────────────────────────────────────
state = dict(vc=0.0, iL=0.0, integral=0.0, prev_err=0.0,
             t=0.0, running=False, vref=VREF0,
             last_vref=VREF0, u_raw=0.0, perturbacoes=[])
buf_t  = deque(maxlen=WINDOW); buf_vc = deque(maxlen=WINDOW)
buf_ref= deque(maxlen=WINDOW); buf_u  = deque(maxlen=WINDOW)

cache = {}

# ─────────────────────────────────────────────────────────────────────
# FIGURA
# ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12), facecolor=BG)
fig.canvas.header_visible = False

outer = gridspec.GridSpec(3, 1, figure=fig,
                          left=0.05, right=0.97, top=0.95, bottom=0.20,
                          height_ratios=[1.0, 1.45, 1.80], hspace=0.50)
top_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.30)
mid_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.28)
bot_gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[2])

ax_pz    = fig.add_subplot(top_gs[0, 0])
ax_it    = fig.add_subplot(top_gs[0, 1])
ax_3d_ma = fig.add_subplot(mid_gs[0, 0], projection="3d")
ax_3d_mf = fig.add_subplot(mid_gs[0, 1], projection="3d")
ax_pid   = fig.add_subplot(bot_gs[0, 0])

style_ax(ax_pz,  "Polo-Zero: RLC vs MF+PID")
style_ax(ax_it,  "Resposta i(t) — Malha Aberta")
style_ax(ax_pid, "Tensão Vc(t) — Malha Fechada (PID)")

for ax_3d in (ax_3d_ma, ax_3d_mf):
    ax_3d.set_facecolor(CARD)
    ax_3d.tick_params(colors=SUB, labelsize=6)
ax_3d_ma.set_title("|V(s)| — Malha Aberta (RLC, sem PID)", fontsize=9, color=TXT, fontweight="bold")  # ← atualizado
ax_3d_mf.set_title("|Vc(s)| — Malha Fechada (com PID)",    fontsize=9, color=TXT, fontweight="bold")

fig.text(0.5, 0.975, "Simulador RLC + PID — Análise Completa",
         ha="center", va="top", color=TXT, fontsize=13, fontweight="bold")

ln_vc,  = ax_pid.plot([], [], color=BLUE, lw=2,   label="Vc (saída)")
ln_ref, = ax_pid.plot([], [], color=SUB,  lw=1.2, ls="--", label="Vref")
ln_u,   = ax_pid.plot([], [], color=GRN,  lw=1.4, label="u(t) PID")
ax_pid.legend(loc="upper right", fontsize=7.5, facecolor=CARD,
              edgecolor=GRID, labelcolor=TXT, framealpha=0.9)
vlines_pid = []

# ─────────────────────────────────────────────────────────────────────
# SLIDERS
# ─────────────────────────────────────────────────────────────────────
SL_BG = "#1a1a24"
def mk_sl(rect, label, lo, hi, init):
    ax_ = fig.add_axes(rect, facecolor=SL_BG)
    sl  = Slider(ax_, label, lo, hi, valinit=init,
                 valstep=(hi-lo)/250, color=BLUE, track_color=GRID)
    sl.label.set_color(TXT);   sl.label.set_fontsize(7.5)
    sl.valtext.set_color(AMB); sl.valtext.set_fontsize(7.5)
    return sl

sl_R    = mk_sl([0.05, 0.155, 0.17, 0.017], "R (Ω)",  0.1, 10.0, R0)
sl_L    = mk_sl([0.05, 0.130, 0.17, 0.017], "L (H)",  0.1,  5.0, L0)
sl_C    = mk_sl([0.05, 0.105, 0.17, 0.017], "C (F)",  0.05, 2.0, C0)
sl_V0   = mk_sl([0.05, 0.080, 0.17, 0.017], "V₀ (V)", 1.0, 20.0, V0_0)
sl_Kp   = mk_sl([0.30, 0.155, 0.17, 0.017], "Kp",     0.0, 30.0, KP0)
sl_Ki   = mk_sl([0.30, 0.130, 0.17, 0.017], "Ki",     0.0, 20.0, KI0)
sl_Kd   = mk_sl([0.30, 0.105, 0.17, 0.017], "Kd",     0.0, 10.0, KD0)
sl_vref = mk_sl([0.55, 0.080, 0.40, 0.026], "Vref(V)",-3.0,  5.0, VREF0)
sl_vref.valtext.set_color(BLUE); sl_vref.valtext.set_fontsize(9)

def mk_btn(rect, texto, cor):
    ax_ = fig.add_axes(rect, facecolor=SL_BG)
    btn = Button(ax_, texto, color=cor, hovercolor=GRID)
    btn.label.set_color(TXT)
    btn.label.set_fontsize(8)
    return btn

btn_play = mk_btn([0.05, 0.01, 0.15, 0.035], "Play/Pause", "#1a2a1a")
btn_rst  = mk_btn([0.22, 0.01, 0.12, 0.035], "Reset", "#2a1a1a")
btn_calc = mk_btn([0.36, 0.01, 0.25, 0.035], "Recalcular (SymPy)", "#1a1a2a")

# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
def _mf_transfer_function(polos_mf, zeros_mf, s_sym):
    num = sp.Integer(1)
    for z in zeros_mf:
        num *= (s_sym - sp.nsimplify(z.real) - sp.nsimplify(z.imag) * sp.I)
    den = sp.Integer(1)
    for p in polos_mf:
        den *= (s_sym - sp.nsimplify(p.real) - sp.nsimplify(p.imag) * sp.I)
    if den == 1:
        den = s_sym - s_sym + 1
    return num / den


def _plot_pz_surface_3d(ax_3d, F_expr, polos, zeros, m, color_p, color_z, titulo, zlabel="|F(s)|"):
    ax_3d.clear()
    ax_3d.set_facecolor(CARD)
    ax_3d.tick_params(colors=SUB, labelsize=6)
    ax_3d.set_title(titulo, fontsize=9, color=TXT, fontweight="bold")

    sig = np.linspace(-m-0.5, m+0.5, 80)
    omg = np.linspace(-m-0.5, m+0.5, 80)
    SG, OM = np.meshgrid(sig, omg)
    try:
        F = sp.lambdify(s_sym, sp.Abs(F_expr), "numpy")
        with np.errstate(divide="ignore", invalid="ignore"):
            Z = np.abs(F(SG + 1j*OM)).astype(float)
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0)
        lim = np.nanpercentile(Z, 98.5) if np.any(Z) else 1.0
        Z = np.clip(Z, 0, lim if lim > 0 else 1.0)
        ax_3d.plot_surface(SG, OM, Z, cmap="inferno", edgecolor="none", alpha=0.85)
        zmax = float(Z.max()) if np.any(Z) else 1.0
    except Exception:
        zmax = 1.0

    for p in polos:
        ax_3d.scatter(p.real, p.imag, 0, marker="x", s=70, lw=2.2,
                      color=color_p, depthshade=False, zorder=10)
    for z in zeros:
        ax_3d.scatter(z.real, z.imag, 0, marker="o", s=45, lw=1.6,
                      facecolors="none", edgecolors=color_z, depthshade=False, zorder=10)

    ax_3d.set_xlabel("σ", color=SUB, fontsize=7, labelpad=2)
    ax_3d.set_ylabel("jω", color=SUB, fontsize=7, labelpad=2)
    ax_3d.set_zlabel(zlabel, color=SUB, fontsize=7, labelpad=2)   # ← label dinâmico
    ax_3d.set_zlim(0, max(zmax, 1e-6))


# ─────────────────────────────────────────────────────────────────────
# RENDER ANALÍTICO
# ─────────────────────────────────────────────────────────────────────
def render_analise():
    r, l, c, v0 = sl_R.val, sl_L.val, sl_C.val, sl_V0.val
    kp, ki, kd  = sl_Kp.val, sl_Ki.val, sl_Kd.val

    # Retorna I(s) + i(t) para o gráfico superior, e V(s) para o 3D
    I_s_num, V_s_num, polos_sym, i_t = calcular_rlc(r, l, c, v0)
    cache.update({"I_s": I_s_num, "V_s": V_s_num, "i_t": i_t})

    polos_rlc, zeros_rlc, polos_mf, zeros_mf = _pz_numericos(r, l, c, kp, ki, kd)
    cache.update({"p_rlc": polos_rlc, "z_rlc": zeros_rlc,
                  "p_mf": polos_mf,   "z_mf": zeros_mf})

    all_pts = polos_rlc + zeros_rlc + polos_mf + zeros_mf
    m = max(1.5, *(abs(x) for x in [p.real for p in all_pts] + [p.imag for p in all_pts]))
    cache["m"] = m

    # ── Polo-Zero ─────────────────────────────────────────────────────
    ax_pz.clear()
    style_ax(ax_pz, "Polo-Zero: RLC vs MF+PID")
    ax_pz.axvspan(-m-0.8, 0, alpha=0.07, color=GRN, zorder=0)
    ax_pz.axhline(0, color=GRID, lw=0.8)
    ax_pz.axvline(0, color=GRID, lw=0.8)

    def lbl(pt, ox, oy, color):
        re_, im_ = pt.real, pt.imag
        t_ = f"{re_:.2f}{im_:+.2f}j" if abs(im_) > 1e-6 else f"{re_:.3f}"
        ax_pz.text(re_+ox, im_+oy, t_, fontsize=6.5, color=color, alpha=0.88)

    for p in polos_rlc:
        ax_pz.scatter(p.real, p.imag, marker="x", s=170, lw=2.5, color=BLUE, zorder=6)
        lbl(p,  0.04*m,  0.04*m, BLUE)
    for z in zeros_rlc:
        ax_pz.scatter(z.real, z.imag, marker="o", s=90, lw=2,
                      facecolors="none", edgecolors=BLUE, zorder=6)
        lbl(z,  0.04*m,  0.04*m, BLUE)
    for p in polos_mf:
        ax_pz.scatter(p.real, p.imag, marker="x", s=170, lw=2.5, color=AMB, zorder=6)
        lbl(p,  0.04*m, -0.09*m, AMB)
    for z in zeros_mf:
        ax_pz.scatter(z.real, z.imag, marker="o", s=90, lw=2,
                      facecolors="none", edgecolors=AMB, zorder=6)
        lbl(z,  0.04*m,  0.04*m, AMB)

    legend_elems = [
        Line2D([0],[0], marker="x", color=BLUE, lw=0, ms=9, mew=2.5, label="Polo RLC (MA)"),
        Line2D([0],[0], marker="o", color=BLUE, lw=0, ms=7, mew=2, mfc="none", label="Zero RLC (MA)"),
        Line2D([0],[0], marker="x", color=AMB,  lw=0, ms=9, mew=2.5, label="Polo MF+PID"),
        Line2D([0],[0], marker="o", color=AMB,  lw=0, ms=7, mew=2, mfc="none", label="Zero MF+PID"),
    ]
    ax_pz.legend(handles=legend_elems, loc="upper right",
                 fontsize=7, facecolor=CARD, edgecolor=GRID, labelcolor=TXT, framealpha=0.92)
    ax_pz.set_xlim(-m-0.6, m+0.6); ax_pz.set_ylim(-m-0.6, m+0.6)
    ax_pz.set_xlabel("σ (parte real)"); ax_pz.set_ylabel("jω (parte imaginária)")

    # ── i(t) — Corrente em malha aberta (original) ─────────────────────
    ax_it.clear()
    style_ax(ax_it, "Resposta i(t) — Malha Aberta")
    reais_abs = [abs(p.real) for p in polos_rlc if abs(p.real) > 1e-8]
    tf = min(5/min(reais_abs), 60.0) if reais_abs else 10.0
    tempo = np.linspace(0, tf, 900)
    try:
        f = sp.lambdify(t_sym, i_t, "numpy")
        iv = np.real(f(tempo)).astype(float)
        iv = np.nan_to_num(iv)
        ax_it.plot(tempo, iv, color=TEAL, lw=1.8)
        ax_it.fill_between(tempo, 0, iv, alpha=0.12, color=TEAL)
    except Exception as e:
        ax_it.text(0.5, 0.5, str(e), transform=ax_it.transAxes,
                   color=RED, ha="center", va="center", fontsize=7)
    ax_it.axhline(0, color=GRID, lw=0.8)
    ax_it.set_xlabel("tempo (s)"); ax_it.set_ylabel("i(t)  [A]")

    # ── 3D — Malha Aberta: |V(s)| (tensão no capacitor) ───────────────
    _plot_pz_surface_3d(
        ax_3d_ma, V_s_num, polos_rlc, zeros_rlc, m,
        color_p=BLUE, color_z=BLUE,
        titulo="|V(s)| — Malha Aberta (RLC, sem PID)",   # ← atualizado
        zlabel="|V(s)|",                                  # ← atualizado
    )

    # ── 3D — Malha Fechada: |Vc(s)| ───────────────────────────────────
    Vc_s_num = _mf_transfer_function(polos_mf, zeros_mf, s_sym)
    cache["Vc_s"] = Vc_s_num
    _plot_pz_surface_3d(
        ax_3d_mf, Vc_s_num, polos_mf, zeros_mf, m,
        color_p=AMB, color_z=AMB,
        titulo="|Vc(s)| — Malha Fechada (com PID)",
        zlabel="|Vc(s)|",
    )

    fig.canvas.draw_idle()

# ─────────────────────────────────────────────────────────────────────
# PASSO DE SIMULAÇÃO PID
# ─────────────────────────────────────────────────────────────────────
def sim_step():
    s = state
    R, L, C   = sl_R.val, sl_L.val, sl_C.val
    Kp, Ki, Kd = sl_Kp.val, sl_Ki.val, sl_Kd.val
    vref = sl_vref.val
    if abs(vref - s["last_vref"]) > 0.05:
        s["perturbacoes"].append(s["t"]); s["last_vref"] = vref
    s["vref"] = vref
    err = vref - s["vc"]
    s["integral"] = np.clip(s["integral"] + err*DT, -20, 20)
    deriv = (err - s["prev_err"]) / DT; s["prev_err"] = err
    u_raw = Kp*err + Ki*s["integral"] + Kd*deriv
    Vin = np.clip(u_raw, -15, 15); s["u_raw"] = u_raw
    diL = (Vin - R*s["iL"] - s["vc"]) / L
    s["iL"] += diL*DT; s["vc"] += (s["iL"]/C)*DT; s["t"] += DT
    buf_t.append(s["t"]); buf_vc.append(s["vc"])
    buf_ref.append(vref); buf_u.append(Vin)

# ─────────────────────────────────────────────────────────────────────
# ANIMAÇÃO
# ─────────────────────────────────────────────────────────────────────
def update_anim(_frame):
    if not state["running"]:
        return ln_vc, ln_ref, ln_u

    for _ in range(3):
        sim_step()

    t_arr  = np.array(buf_t);  vc_arr = np.array(buf_vc)
    r_arr  = np.array(buf_ref); u_arr  = np.array(buf_u)

    ln_vc.set_data(t_arr, vc_arr)
    ln_ref.set_data(t_arr, r_arr)
    ln_u.set_data(t_arr, u_arr)

    if len(t_arr) > 1:
        span = max(t_arr[-1] - t_arr[0], 5.0)
        ax_pid.set_xlim(t_arr[0], t_arr[0]+span)
    if len(vc_arr) > 5:
        lo = min(vc_arr.min(), r_arr.min(), u_arr.min()) - 0.5
        hi = max(vc_arr.max(), r_arr.max(), u_arr.max()) + 0.5
        ax_pid.set_ylim(lo, hi)

    for vl in vlines_pid: vl.remove()
    vlines_pid.clear()
    recent = [p for p in state["perturbacoes"] if p >= state["t"] - WINDOW*DT]
    state["perturbacoes"] = recent
    for tp in recent:
        vlines_pid.append(ax_pid.axvline(tp, color=AMB, lw=0.7, ls=":", alpha=0.7))

    return ln_vc, ln_ref, ln_u


# ─────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────
def on_playpause(event):
    state["running"] = not state["running"]
    fig.canvas.draw_idle()

def on_reset(event):
    state.update(dict(vc=0.0, iL=0.0, integral=0.0, prev_err=0.0,
                      t=0.0, running=False, vref=sl_vref.val,
                      last_vref=sl_vref.val, u_raw=0.0, perturbacoes=[]))
    buf_t.clear(); buf_vc.clear(); buf_ref.clear(); buf_u.clear()
    for ln in [ln_vc, ln_ref, ln_u]: ln.set_data([], [])
    for vl in vlines_pid: vl.remove()
    vlines_pid.clear()
    fig.canvas.draw_idle()

def on_calc(event):
    print("⏳ Recalculando análise simbólica…")
    render_analise()
    print("✅ Pronto.")

btn_play.on_clicked(on_playpause)
btn_rst.on_clicked(on_reset)
btn_calc.on_clicked(on_calc)

# ─────────────────────────────────────────────────────────────────────
# MÉTODOS DE SINTONIA
# ─────────────────────────────────────────────────────────────────────
def sintonia_zn_malha_aberta(R, L, C):
    wn = 1.0 / np.sqrt(L * C)
    zeta = R / (2.0 * np.sqrt(L / C))
    K   = 1.0 / R if R > 0 else 1.0
    tau = 2.0 * zeta / wn
    Td  = 1.0 / wn * 0.5
    Kp  = 1.2 * tau / (K * Td)
    Ti  = 2.0 * Td
    Td2 = 0.5 * Td
    Ki  = Kp / Ti if Ti > 0 else 0
    Kd  = Kp * Td2
    return Kp, Ki, Kd

def sintonia_zn_malha_fechada(R, L, C):
    wn  = 1.0 / np.sqrt(L * C)
    Ku  = 2.0 * np.sqrt(L / C) / R if R > 0 else 10.0
    Pu  = 2.0 * np.pi / wn
    Kp  = 0.6  * Ku
    Ki  = Kp / (0.5 * Pu)
    Kd  = Kp * (0.125 * Pu)
    return Kp, Ki, Kd

def sintonia_cohen_coon(R, L, C):
    wn  = 1.0 / np.sqrt(L * C)
    zeta = R / (2.0 * np.sqrt(L / C))
    K   = 1.0 / R if R > 0 else 1.0
    tau = max(2.0 * zeta / wn, 0.01)
    Td  = max(1.0 / (wn * 2.0), 0.001)
    r   = Td / tau
    Kp  = (1.0/K) * (1.33 + r/4.0)
    Ti  = tau * (2.25*r + 0.833*r**2) / (1.25*r + 0.5*r**2 + 1.0)
    Td2 = Td * 0.37 / (1.0 + 0.2/r) if r > 0 else Td
    Ki  = Kp / Ti if Ti > 0 else 0
    Kd  = Kp * Td2
    return Kp, Ki, Kd

def sintonia_imc(R, L, C):
    wn   = 1.0 / np.sqrt(L * C)
    zeta = R / (2.0 * np.sqrt(L / C))
    K    = 1.0 / R if R > 0 else 1.0
    tau  = max(2.0 * zeta / wn, 0.01)
    lam  = tau / 2.0
    Kp   = (2.0*tau) / (2.0*K*lam)
    Ki   = Kp / tau
    Kd   = 0.0
    return Kp, Ki, Kd

def sintonia_chien_hrones_reswick(R, L, C):
    wn   = 1.0 / np.sqrt(L * C)
    zeta = R / (2.0 * np.sqrt(L / C))
    K    = 1.0 / R if R > 0 else 1.0
    tau  = max(2.0 * zeta / wn, 0.01)
    Td   = max(1.0 / (wn * 2.0), 0.001)
    Kp   = 0.6 * tau / (K * Td)
    Ki   = Kp / tau
    Kd   = Kp * Td * 0.5
    return Kp, Ki, Kd

METODOS = {
    "ZN Malha Aberta":      sintonia_zn_malha_aberta,
    "ZN Malha Fechada":     sintonia_zn_malha_fechada,
    "Cohen-Coon":           sintonia_cohen_coon,
    "IMC":                  sintonia_imc,
    "Chien-Hrones-Reswick": sintonia_chien_hrones_reswick,
}
METODOS_LISTA = list(METODOS.keys())
state["metodo_idx"] = 0

ax_m = fig.add_axes([0.64, 0.01, 0.32, 0.035], facecolor=SL_BG)
btn_metodo = Button(ax_m, f"Sintonizar: {METODOS_LISTA[0]}", color="#1a1a2a", hovercolor=GRID)
btn_metodo.label.set_color(TEAL)
btn_metodo.label.set_fontsize(8)

def on_metodo(event):
    idx = (state["metodo_idx"] + 1) % len(METODOS_LISTA)
    state["metodo_idx"] = idx
    nome   = METODOS_LISTA[idx]
    func   = METODOS[nome]
    R, L, C = sl_R.val, sl_L.val, sl_C.val
    kp, ki, kd = func(R, L, C)
    kp = float(np.clip(kp, sl_Kp.valmin, sl_Kp.valmax))
    ki = float(np.clip(ki, sl_Ki.valmin, sl_Ki.valmax))
    kd = float(np.clip(kd, sl_Kd.valmin, sl_Kd.valmax))
    sl_Kp.set_val(kp); sl_Ki.set_val(ki); sl_Kd.set_val(kd)
    btn_metodo.label.set_text(f"Sintonizar: {nome}")
    print(f"[{nome}]  Kp={kp:.3f}  Ki={ki:.3f}  Kd={kd:.3f}")
    fig.canvas.draw_idle()

btn_metodo.on_clicked(on_metodo)

# ─────────────────────────────────────────────────────────────────────
# INÍCIO
# ─────────────────────────────────────────────────────────────────────
print("⏳ Calculando análise inicial (SymPy)…")
render_analise()

anim = FuncAnimation(fig, update_anim, interval=INTERVAL_MS,
                     blit=False, cache_frame_data=False)

print("✅ Pronto!")
print("   → Clique em ▶ Play/Pause para iniciar/pausar a simulação PID")
print("   → ∫ Recalcular atualiza o polo-zero, v(t) e as superfícies 3D (MA / MF)")
print("   → ↺ Reset zera o estado do circuito")
plt.show()