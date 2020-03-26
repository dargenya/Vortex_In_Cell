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
import matplotlib.pyplot as plt


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

def derive_x(p):
    # [1111111] (p '1')
    vec_sup_inf=np.ones(p)
    # [1111110] (p-1 '1', and '0')
    vec_sup_inf[-1] = 0
    # [1111110]*[1111111].T
    # array of shape (p,p), NOT  matrix
    vec_sup_inf=vec_sup_inf*np.ones((p,1))
    # flatten the matrix, ie [[1 0][1 0]]->[1 0 1 0]
    # remove the trailing 0
    vec_sup_inf=vec_sup_inf.flatten()[:-1]
    DD  = np.diag(vec_sup_inf,1)
    DD -= np.diag(vec_sup_inf,-1)
    return DD

def derive_y(p):
    DD  = np.diag(np.ones(p*(p-1)),p)
    DD -= np.diag(np.ones(p*(p-1)),-p)
    return DD

def A_static(N):
    p = int(np.sqrt(N))
    h = 1./(p+1)
    A = 1. /(h*h)*laplacien(p)
    return A

def fonction_G(N,gamma): # second membre -1*omega
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

# la solution exacte est un logarithme complexe. On ne comparera seulement les vitesses exactes et approchees
def Sp_exact(N, gamma, pos): #vitesse exacte à pos pour un trourbillon au centre
    p = int(np.sqrt(N))
    pos = np.array(pos)
    tourb = np.ones(2) * np.mean([p/2-1,p/2])
    C = gamma/(2*np.pi)
    
    x, y = pos - tourb
    
    r = np.sqrt(x**2 + y**2)
    if r==0:
        print(r=0)
    theta = np.arctan2(y,x)
    
    u = -C*np.sin(theta)/r
    v = C*np.cos(theta)/r
    return np.array([u,v])
    


N = 10*10
gamma = 1
print("------------")
print("N = ",N)
start = time.time()
A = A_static(N)
G = fonction_G(N, gamma)
end = time.time()
print(" Temps de création des matrices : ", end-start )
print (" Conditionnement de la matrice : ",np.linalg.cond(A))
start = time.time()
U_approx = np.linalg.solve(A,G)
end = time.time()
#print(U_approx)
print(" Temps d'execution de solve: ", end-start )

#Q5/ On calcul le champs de vitesse
p = int(np.sqrt(N))
h = 1. / (p + 1)
Spx = (+1 / (2 * h) * derive_y(p)) @ U_approx
Spy = (-1 / (2 * h) * derive_x(p)) @ U_approx

# Spx = np.array(Spx)
# Spy = np.array(Spy)

X = []
Y = []
for k in range(N):
    i, j = k % p, k // p
    X.append(i)
    Y.append(j)

X = np.array(X)
Y = np.array(Y)

#convergence
print('--- Convergence ---')
Spx_ex = np.empty(0)
Spy_ex = np.empty(0)
for x in X:
    for y in Y:
        pos = [x,y]
        a, b = Sp_exact(N, gamma, pos)
        Spx_ex = np.append(Spx_ex,a)
        Spy_ex = np.append(Spy_ex,b)

error_Spx = Spx_ex - Spx
error_Spy = Spy_ex - Spy        
print('std of speeds are: \n Spx : {} \n Spy : {}'.format(np.std(error_Spx),np.std(error_Spy)))
        

fig, ax = plt.subplots()
q = ax.quiver(X, Y, Spx, Spy, units='xy', scale=1)

plt.grid()

ax.set_aspect('equal')

ax.xaxis.set_ticks([k for k in range(0,p)])
ax.yaxis.set_ticks([k for k in range(0,p)])

# plt.xlim(-5, 5)
# plt.ylim(-5, 5)

plt.title('Q5/ Vecteur vitesse gamma=1', fontsize=10)

#plt.savefig('how_to_plot_a_vector_in_matplotlib_fig3.png', bbox_inches='tight')
plt.show()

# Q6/

#u_x = 0
#u_y = 0
#for i in [p / 2 - 1, p / 2 -1]:
#    for j in [p / 2 - 1, p / 2 -1]:
#        k = int(i + j * p)
#        u_x += (Spx[k])/4
#        u_y += (Spx[k]) / 4
    
u_x = []
u_y = []
#fills in in order: bottom-left, then  top-left, bottom-right, bottom-left
for i in [p / 2 - 1, p / 2 ]: #left/right
    for j in [p / 2 - 1, p / 2 ]: #top/bottom
        k = int(i + j * p)
        u_x.append( (Spx[k]) )
        u_y.append( (Spx[k]) )
u_x = np.array(u_x)
u_y = np.array(u_y)

A = (h/2)*(h/2) * np.ones(4)

# vitesses au niveau du troubillon:
ux = (np.sum(A*u_x)) / np.sum(A)
uy = (np.sum(A*u_y)) / np.sum(A)

# Q7/




# def reforme(U):
#     #Permet de passer d une colonne mono indice à la matrice correspondante
#     N = len(U)
#     p = int(np.sqrt(N))
#     M = np.zeros((p, p))
#     for k in range(len(U)):
#         i,j = k%p , k//p
#         M[i,j] = U[k]
#     return(M)


