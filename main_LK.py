#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed 21 June 2023

@author: vm
"""

import numpy as np
from  quantum_box_LK import muB, get_material_parameters, make_box_LK, compute_eigenvalues, get_ga
import time

# set material

mat = 'Si'
mat_params = get_material_parameters(mat)
print('Material parameters: {}'.format(mat_params))

# set lattice parameters (lengths in nm)

L, W, H = 30, 30, 10
ax, ay, az = 1/2, 1/2, 1/2

lat_params = {"L": L, "W": W, "H": H, "ax": ax, "ay": ay, "az": az}
print('Lattice parameters (lengths in nm): {}'.format(lat_params))

nx, ny, nz = int(L/ax - 1), int(W/ay - 1), int(H/az - 1)
print('Number of points in the lattice: {0}*{1}*{2}={3}'.format(nx, ny, nz, nx*ny*nz))

F = [0, 0, 0]

def main_g(a='z', F=F):
    print('Computing g{0} for Ey = {1:.2f} mV/nm, Ez = {2:.2f} mV/nm'.format(a, F[1], F[2]))
    t1 = time.time()
    g = get_ga(mat_params, lat_params, a, F)
    t2 = time.time()
    print('g{0} = {1:.6f} (done in {2:.2f}s).'.format(a, g, t2 - t1))

def main_Ey_g(mat_params=mat_params, lat_params=lat_params):
    Fys = np.arange(0, 20.1, 0.1)
    Fz = 0.0
    l = ['x', 'y', 'z']
    for i, a in enumerate(l):
        if a == 'x':
            B = [0.01, 0.0, 0.0]
        elif a == 'y':
            B = [0.0, 0.01, 0.0]
        elif a == 'z':
            B = [0.0, 0.0, 0.01]
        syst = make_box_LK(mat_params, lat_params, B)
        gs = []
        for Fy in Fys:
            F = [0, Fy, Fz]
            print('Computing g{0} for Ey = {1:.2f} mV/nm, Ez = {2:.2f} mV/nm'.format(a, F[1], F[2]))
            t1 = time.time()
            evals = compute_eigenvalues(syst, F)
            g = (evals[1] - evals[0])/(muB*np.linalg.norm(B))
            t2 = time.time()
            print('g{0} = {1:.6f} (done in {2:.2f}s).'.format(a, g, t2 - t1))
            gs.append(g)
        gs = np.array(gs)
        np.save('Ey_g'+a+'.npy', [Fys, gs])

if __name__ == "__main__":
    main_Ey_g()
