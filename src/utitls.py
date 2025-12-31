#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides utilities used by FE analysis.
  1. gauss: Gauss quadrature rules.
  2. assembly: Global stiffness matrix and nodal force vector assembly.
  3. solvedr: Solving the stiffness equations by the reduction approach.

Created on Sun Apr 24 18:56:57 2020

@author: xzhang@tsinghua.edu.cn
"""

import numpy as np
import FEData as model
import VEData as VEmodel


def gauss(ngp):
    """
    Get Gauss points in the parent element domain [-1, 1] and
    the corresponding weights.

    Args:
        ngp : (int) number of Gauss points.

    Returns: w,gp
        w  : weights.
        gp : Gauss points in the parent element domain.
    """
    gp = None
    w = None
    if ngp == 1:
        gp = [0]
        w = [2]
    elif ngp == 2:
        gp = [-0.57735027, 0.57735027]
        w = [1, 1]
    elif ngp == 3:
        gp = [-0.7745966692, 0.7745966692, 0.0]
        w = [0.5555555556, 0.5555555556, 0.8888888889]
    else:
        raise ValueError("The given number (ngp = {}) of Gauss points is too large and not implemented".format(ngp))
    return w, gp


def assembly(e, ke, fe):
    """
    Assemble element stiffness matrix and nodal force vector.

    Args:
        e   : (int) Element number
        ke  : (numpy(nen,nen)) element stiffness matrix
        fe  : (numpy(nen,1)) element nodal force vector
    """
    for loop1 in range(model.nen*model.ndof):
        i = model.LM[loop1, e]-1
        model.f[i] += fe[loop1]   # assemble nodal force vector

        for loop2 in range(model.nen*model.ndof):
            j = model.LM[loop2, e]-1
            model.K[i, j] += ke[loop1, loop2]   # assemble stiffness matrix


def solvedr():
    """
    Partition and solve the system of equations

    Returns:
        f_E : (numpy.array(nd,1)) Reaction force vector
    """
    nd = model.nd
    neq = model.neq
    K_E = model.K[0:nd, 0:nd]
    K_F = model.K[nd:neq, nd:neq]
    K_EF = model. K[0:nd, nd:neq]
    f_F = model.f[nd:neq]
    d_E = model.d[0:nd]

    if (neq > nd):
        model.Cond_K = np.linalg.cond(K_F)
        # print('\nCondition number of stiffness matrix: ', model.Cond_K)

    # solve for d_F
    d_F = np.linalg.solve(K_F, f_F - K_EF.T @ d_E)

    # reconstruct the global displacement d
    model.d = np.append(d_E,d_F)

    # compute the reaction r
    f_E = K_E@d_E + K_EF@d_F - model.f[:model.nd]

    '''
    # write to the workspace
    print('\nFEM solution d')
    print(model.d)
    print('\nreaction f = \n', f_E)
    '''

    return f_E

'''
def assembly4VEM(e, ke, fe):
    """
    Assemble element stiffness matrix and nodal force vector.

    Args:
        e   : (int) Element number
        ke  : (numpy(nen,nen)) element stiffness matrix
        fe  : (numpy(nen,1)) element nodal force vector
    """
    for loop1 in range(VEmodel.ven[e]*VEmodel.ndof):
        i = VEmodel.LM[e][loop1]-1
        VEmodel.f[i] += fe[loop1]   # assemble nodal force vector

        for loop2 in range(VEmodel.ven[e]*VEmodel.ndof):
            j = VEmodel.LM[e][loop2]-1
            VEmodel.K[i, j] += ke[loop1, loop2]   # assemble stiffness matrix
'''

def assembly4VEM(e, ke, fe):
    """
    Assemble element stiffness matrix and nodal force vector.

    Args:
        e   : (int) Element number
        ke  : (numpy(nen,nen)) element stiffness matrix
        fe  : (numpy(nen,1)) element nodal force vector
    """
    for loop1 in range(VEmodel.ven[e]*VEmodel.ndof):
        i = VEmodel.LM[e][loop1]-1
        VEmodel.f[i] += fe[loop1]   # assemble nodal force vector

        for loop2 in range(VEmodel.ven[e]*VEmodel.ndof):
            j = VEmodel.LM[e][loop2]-1
            VEmodel.K[i, j] += ke[loop1, loop2]   # assemble stiffness matrix

def solvedr4VEM():
    """
    Partition and solve the system of equations

    Returns:
        f_E : (numpy.array(nd,1)) Reaction force vector
    """
    nd = VEmodel.nd
    neq = VEmodel.neq
    K_E = VEmodel.K[0:nd, 0:nd]
    K_F = VEmodel.K[nd:neq, nd:neq]
    K_EF = VEmodel. K[0:nd, nd:neq]
    f_F = VEmodel.f[nd:neq]
    d_E = VEmodel.d[0:nd]

    if (neq > nd):
        VEmodel.Cond_K = np.linalg.cond(K_F)
        print('\nCondition number of stiffness matrix: ', VEmodel.Cond_K)

    try:
        # 先正常求解，如奇异则使用伪逆
        try:
            # solve for d_F
            d_F = np.linalg.solve(K_F, f_F - K_EF.T @ d_E)

            # reconstruct the global displacement d
            d_solve = np.append(d_E, d_F)

            # compute the reaction r
            f_E = K_E@d_E + K_EF@d_F - VEmodel.f[:VEmodel.nd]


        except:
            # 矩阵奇异，正则化求解
            try:
                reg_param = 1e-10
                K_F_reg = K_F + reg_param * np.eye(*K_F.shape, device=K_F.device, dtype=K_F.dtype)

                # solve for d_F using Regularization matrix
                d_F_reg = np.linalg.solve(K_F_reg, f_F - K_EF.T @ d_E)
            except:
                reg_param = 1e-8
                K_F_reg = K_F + reg_param * np.eye(*K_F.shape, device=K_F.device, dtype=K_F.dtype)

                # solve for d_F using Regularization matrix
                d_F_reg = np.linalg.solve(K_F_reg, f_F - K_EF.T @ d_E)

            d_solve = np.append(d_E, d_F_reg)

            # compute the reaction r
            f_E = K_E@d_E + K_EF@d_F_reg - VEmodel.f[:VEmodel.nd]
    except:
        # 正则化失败，尝试使用伪逆
        # 伪逆
        K_F_pinv = np.linalg.pinv(K_F)
        d_F_pinv = K_F_pinv @ (f_F - K_EF.T @ d_E)
        d_solve = np.append(d_E, d_F_pinv)

        # compute the reaction r
        f_E = K_E@d_E + K_EF@d_F_pinv - VEmodel.f[:VEmodel.nd]

    # reconstruct the global displacement d
    VEmodel.d = d_solve

    # write to the workspace
    # print('\nVEM solution d')
    # print(VEmodel.d)
    # print('\nVEM reaction f = \n', f_E)

    return f_E
