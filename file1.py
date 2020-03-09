# -*- coding: utf-8 -*-
"""
@author: Yann d'Argenlieu et Yassin Mesrati

fortement inspiré des TD de différences finies UE3.1 à l'ENSTA Bretagne
"""

import numpy as np
# import numpy.linalg
# import scipy.sparse as sp
# import scipy.sparse.linalg
import time
# import matplotlib.pyplot as plt


def laplacien(p):
    lap = np.zeros((p ** 2, p ** 2))
    lap = -4 * np.diag(np.ones(p ** 2))

    # [1111111] (p '1')
    vec_sup_inf = np.ones(p)
    # [1111110] (p-1 '1', and '0')
    vec_sup_inf[-1] = 0
    # [1111110]*[1111111].T
    # array of shape (p,p), NOT  matrix
    vec_sup_inf = vec_sup_inf * np.ones((p, 1))
    # flatten the matrix, ie [[1 0][1 0]]->[1 0 1 0]
    # remove the trailing 0
    vec_sup_inf = vec_sup_inf.flatten()[:-1]
    lap += np.diag(vec_sup_inf, 1)
    lap += np.diag(vec_sup_inf, -1)
    lap += np.diag(np.ones(p * (p - 1)), p)
    lap += np.diag(np.ones(p * (p - 1)), -p)
    return lap

def A_static(N):
    p = int(np.sqrt(N))
    h = 1./(p+1)
    A = 1. /(h*h)*laplacien(p)
    return A

def fonction_G(N,gamma):
    p = int(np.sqrt(N))
    h = 1./(p+1)
    A = h*h

    # Question 2 i.e. vorticite nulle
    GG = np.zeros((N,1))

    # Question 3
    GG = np.zeros((N,1))
    for i in [p/2-1,p/2]:
        for j in [p/2-1,p/2]:
            k = int(i + j * p)
            GG[k] = - gamma/(4*A)


    # k = 0
    # for j in range(p):
    #     for i in range(p):
    #         x = (i+1)*h
    #         y = (j+1)*h
    #         GG[k]= (-2*epsilon*(y*(y-1) + x*(x-1)) + alpha*y*(y-1)*(2*x-1) + beta*x*(x-1)*(2*y-1) + c*x*y*(x-1)*(y-1))
    #         k = k+1

    return GG

def etude():
    for N in [10*10]:
        print("------------")
        print("N = ",N)
        start = time.time()
        A = A_static(N)
        G = fonction_G(N, 1)
        end = time.time()
        print(" Temps de création des matrices : ", end-start )
        print (" Conditionnement de la matrice : ",np.linalg.cond(A))
        start = time.time()
        U_approx = np.linalg.solve(A,G)
        end = time.time()
        print(U_approx)
        print(" Temps d'execution de solve: ", end-start )

if __name__ == '__main__':
    print('ok')
    etude()