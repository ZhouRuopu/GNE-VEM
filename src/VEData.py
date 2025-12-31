#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from FEData import *
"""
Global variables defining the VEM model

"""
"""
Original data in FEData

Title = None
nsd = 0
ndof = 0
nnp = 0
nel = 0
nen = 0
neq = 0
ngp = 2
nd = 0
nbe = 0

f = None
d = None
K = None

# boundary conditions
flags = None
e_bc = None
n_bc = None

# force conditions
P = None
b = None

# material
D = None
G = 0.0

# The type of plane problem is set to plane stress by default
plane_strain = 0

# define the mesh
x = None
y = None
IEN = None

ID = None
LM = None

# parameter for postprocess
counter = None
nodestress = None
compute_stress = None
plot_mesh = None
plot_disp = None
print_disp = None
plot_nod = None
plot_stress_xx = None
plot_mises = None
plot_tex = None
fact = 1
"""

ven = None # 各单元节点总数
stab_var = 1.0 # 稳定性系数
SI = [] # shape index
area = []
d_real = []