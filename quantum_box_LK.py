#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 17:59:17 2023

@author: vm
"""

import numpy as np
import scipy as sp
from numpy import pi, matmul
import kwant
#import kwant.continuum
#import tinyarray
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt
from scipy.constants import physical_constants
from material_parameters import parameters
import time

# Bohr radius in nm

a0 = 1e9*physical_constants["Bohr radius"][0]

# Hartree energy in meV

EH = 1e3*physical_constants["Hartree energy in eV"][0]

# Bohr magneton in meV/T

muB = 1e3*physical_constants["Bohr magneton in eV/T"][0]

# magnetic flux h/e in Tesla.nm^2

phi0 = 2e18*physical_constants["mag. flux quantum"][0]

#

mu = 0.5 * EH * a0**2

#

sqrt2 = np.sqrt(2)
sqrt3 = np.sqrt(3)

# Spin-3/2 matrices:

id4 = np.identity(4)

Jx = np.array(
    [[0, sqrt3/2, 0, 0], 
     [sqrt3/2, 0, 1, 0], 
     [0, 1, 0, sqrt3/2], 
     [0, 0, sqrt3/2, 0]])

Jy = np.array(
    [[0, -1j*sqrt3/2, 0, 0], 
     [1j*sqrt3/2, 0, -1j, 0], 
     [0, 1j, 0, -1j*sqrt3/2], 
     [0, 0, 1j*sqrt3/2, 0]])

Jz = np.array(
    [[3/2, 0, 0, 0], 
     [0, 1/2, 0, 0], 
     [0, 0, -1/2, 0], 
     [0, 0, 0, -3/2]])

Jyz = 0.5*(matmul(Jy, Jz) + matmul(Jz, Jy))
Jzx = 0.5*(matmul(Jz, Jx) + matmul(Jx, Jz))
Jxy = 0.5*(matmul(Jx, Jy) + matmul(Jy, Jx))
Jx2 = matmul(Jx, Jx)
Jy2 = matmul(Jy, Jy)
Jz2 = matmul(Jz, Jz)
Jx3 = matmul(Jx2, Jx)
Jy3 = matmul(Jy2, Jy)
Jz3 = matmul(Jz2, Jz)

def get_material_parameters(mat):
    for row in parameters:
        if row["Material"] == mat:
            return row
        
def vec_pot(B, y, z):
    return [z*B[1] - y*B[2], -z*B[0], 0]

def make_box_LK(mat_params, lat_params, B):
    ka, qu, ga1, ga2, ga3 = mat_params["ka"], mat_params["qu"], mat_params["ga1"], mat_params["ga2"], mat_params["ga3"]
    L, W, H, ax, ay, az = lat_params["L"], lat_params["W"], lat_params["H"], lat_params["ax"], lat_params["ay"], lat_params["az"]
    nx, ny, nz = int(L/ax - 1), int(W/ay - 1), int(H/az - 1)
    lat = kwant.lattice.general([(ax, 0, 0), (0, ay, 0), (0, 0, az)], norbs = 4)
    syst = kwant.Builder()
    def potential(site, F):
        x, y, z = site.pos[0], site.pos[1], site.pos[2]
        return - F[0] * x - F[1] * y - F[2] * z
    def onsite(site, F):
        return potential(site, F) * id4 + 2 * muB * (ka * (Jx*B[0] + Jy*B[1] + Jz*B[2]) + qu * (Jx3*B[0] + Jy3*B[1] + Jz3*B[2])) \
            + mu * ((2*ga1 + 5*ga2) * (1/ax**2 + 1/ay**2 + 1/az**2)*id4 - 4 * ga2 * (Jx2/ax**2 + Jy2/ay**2 + Jz2/az**2))
    print('Building the Hamiltonian...')
    t1 = time.time()
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                syst[lat(i, j, k)] = onsite
                y, z = ay*j, az*k
                if i > 0:
                    syst[lat(i, j, k), lat(i - 1, j, k)] = (mu/ax**2) * (-(ga1 + 5*ga2/2) * id4  + 2 * ga2 * Jx2) \
                    + (mu/ax) * ((ga1 + 5*ga2/2) * id4  - 2 * ga2 * Jx2) * (2j*pi/phi0) * vec_pot(B, y, z)[0] \
                    - (mu/ax) * 2 * ga3 * (2j*pi/phi0) * vec_pot(B, y, z)[1] * Jxy
                if j > 0:
                    syst[lat(i, j, k), lat(i, j - 1, k)] = (mu/ay**2) * (-(ga1 + 5*ga2/2) * id4  + 2 * ga2 * Jy2) \
                    + (mu/ay) * ((ga1 + 5*ga2/2) * id4  - 2 * ga2 * Jy2) * (1j*pi/phi0) * (vec_pot(B, y - ay, z)[1] + vec_pot(B, y, z)[1]) \
                    - (mu/ay) * 2 * ga3 * (1j*pi/phi0) * (vec_pot(B, y - ay, z)[0] + vec_pot(B, y, z)[0]) * Jxy
                if k > 0:
                    syst[lat(i, j, k), lat(i, j, k - 1)] = (mu/az**2) * (-(ga1 + 5*ga2/2) * id4  + 2 * ga2 * Jz2) \
                    - (mu/az) * 2 * ga3 * (1j*pi/phi0) * ((vec_pot(B, y, z - az)[0] + vec_pot(B, y, z)[0])*Jzx + (vec_pot(B, y, z - az)[1] + vec_pot(B, y, z)[1])*Jyz)
                if i > 0 and j > 0:
                    syst[lat(i, j, k), lat(i - 1, j - 1, k)] = (mu/(ax*ay)) * ga3 * Jxy
                    syst[lat(i, j - 1, k), lat(i - 1, j, k)] = -(mu/(ax*ay)) * ga3 * Jxy
                if i > 0 and k > 0:
                    syst[lat(i, j, k), lat(i - 1, j, k - 1)] = (mu/(az*ax)) * ga3 * Jzx
                    syst[lat(i, j, k - 1), lat(i - 1, j, k)] = -(mu/(az*ax)) * ga3 * Jzx
                if j > 0 and k > 0:
                    syst[lat(i, j, k), lat(i, j - 1, k - 1)] = (mu/(ay*az)) * ga3 * Jyz
                    syst[lat(i, j, k - 1), lat(i, j - 1, k)] = -(mu/(ay*az)) * ga3 * Jyz
    t2 = time.time()
    print('Done in {0:.2f}s'.format(t2-t1))
    syst = syst.finalized()
    return syst

def compute_eigenvalues(syst, F, N=8):
    ham = syst.hamiltonian_submatrix(params=dict(F=F), sparse=True)
    evals = sla.eigsh(ham, k=N, which='SA')[0]
    evals = np.array(sorted(evals))
    return evals

def get_ga(mat_params, lat_params, a, F):
    if a == 'x':
        B = [0.01, 0.0, 0.0]
    elif a == 'y':
        B = [0.0, 0.01, 0.0]
    elif a == 'z':
        B = [0.0, 0.0, 0.01]
    syst = make_box_LK(mat_params, lat_params, B)
    evals = compute_eigenvalues(syst, F)
    g = (evals[1] - evals[0])/(muB*np.linalg.norm(B))
    return g