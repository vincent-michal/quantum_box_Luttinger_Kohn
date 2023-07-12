#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed 21 June 2023

@author: vm
"""

from  quantum_box_LK import get_material_parameters, get_ga
import time


# set material

mat = 'Si'
mat_params = get_material_parameters(mat)
print('Material parameters: {}'.format(mat_params))

# set lattice parameters (lengths in nm)

L = 30
W = 30
H = 10
ax, ay, az = 1, 1, 1/2

lat_params = {"L": L, "W": W, "H": H, "ax": ax, "ay": ay, "az": az}
print('Lattice parameters (lengths in nm): {}'.format(lat_params))

nx, ny, nz = int(L/ax - 1), int(W/ay - 1), int(H/az - 1)
print('Number of points in the lattice: {0}*{1}*{2}={3}'.format(nx, ny, nz, nx*ny*nz))

F = [0, 0, 0]

def main(a='z', F=F):
    print('Computing g{0} for Ey = {1:.2f} mV/nm, Ez = {2:.2f} mV/nm'.format(a, F[1], F[2]))
    t1 = time.time()
    g = get_ga(mat_params, lat_params, a, F)
    t2 = time.time()
    print('g{0} = {1:.6f} (done in {2:.2f}s).'.format(a, g, t2 - t1))
    
def main_Ey(mat_params, lat_params):
    Fys = np.arange(0, 20.1, 0.1)
    Fz = 0.0
    l = ['x', 'y', 'z']
    for i, a in enumerate(l):
        gs = []
        for Fy in Fys:
            F = [0, Fy, Fz]
            g = get_ga(mat_params, lat_params, a, F)
            gs.append(g)
        gs = np.array(gs)
        np.save('Ey_g'+a+'.npy', [Fys, gs])

def plot():
    Ey1, gx = np.load('Ey_gx.npy')
    Ey2, gy = np.load('Ey_gy.npy')
    Ey3, gz = np.load('Ey_gz.npy')
    plt.plot(Ey1, gx, Ey2, gy, Ey3, gz)
    #plt.xlim(0, 12)
    #plt.ylim(0, 5)
    plt.show()

if __name__ == "__main__":
    main()
