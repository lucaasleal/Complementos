import sympy as sp

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
r = float(input("Insira resistência:"))
l = float(input("Insira indutância:"))
c = float(input("Insira capacitância:"))
v_0 = float(input("Insira fonte:"))

I_s_sol = sp.factor(I_s.subs({R: r, L: l, C: c, V_0 : v_0}))
print("I(s) = ", I_s_sol)


#===========================================================
#Polos complexos
polos = sp.solve(I_s_sol, s, domain=sp.S.Complexes)

print(polos)

