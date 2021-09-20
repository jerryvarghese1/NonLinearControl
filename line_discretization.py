# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 11:08:56 2021

@author: vargh
"""

import control
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sympy import init_printing

init_printing()


t_min = 0
t_max = 2*np.pi

segs = 50

t_disc = np.linspace(t_min, t_max, segs)
t_step = (t_max - t_min)/segs

func_0 = 0
func_dot_0 = 5

func_list = list(np.zeros([len(t_disc)]))
func_laplace = list(np.zeros([len(t_disc)]))
data_list = np.zeros([len(t_disc)])

t, s, x_0, x_dot_0 = sp.symbols('t, s, x_0, x_dot_0')
x = sp.Function('x')(t)
X = sp.Function('X')(s)

x_doub_dot = sp.Derivative(x, t, 2)
x_dot = sp.Derivative(x, t, 1)

LHS = x_doub_dot + sp.exp(x_dot)
RHS = 0
diff_eq = sp.Eq(LHS, RHS)

f_non_lin = sp.exp(x)
f_lin = sp.diff(f_non_lin, x)

for b in range(len(t_disc)):
    print('.', end= '')
    
    t_cur = t_disc[b]
    
    y_point = f_non_lin.subs(x, t_cur)
    
    local_mult = f_lin.subs(x, t_cur)
    
    local_LHS = x_doub_dot + local_mult*x_dot
    lap_eq_RHS = 0
    
    lap_eq_LHS = local_LHS.subs(x_doub_dot, s**2*X-s*func_0 - func_dot_0).subs(x_dot, s*X-func_0).subs(x, X)   
    
    local_laplace = sp.solve(sp.Eq(lap_eq_LHS, lap_eq_RHS), X)[0]
    
    func_laplace[b] = local_laplace
    func_list[b] = sp.inverse_laplace_transform(local_laplace, s, t)
    
    data_list[b] = sp.re(func_list[b].subs(t, t_disc[b]).subs(sp.Heaviside(0.0), 0).evalf())


plt.plot(t_disc, data_list)

log_data = np.log(data_list)
log_time = np.log(t_disc)

diff1 = np.diff(log_data)/t_step
diff2 = np.diff(diff1)/t_step

print('\n')


