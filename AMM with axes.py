# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:07:17 2018

@author: cpierer

This script develops code written originally by Steffen Lohrey to adapt the 
Alonso-Muth-Mills monocentric city model for star-shaped city with linear 
transport axes.
"""


import bisect
import math as ma
import numpy as np
import scipy as sp 

###############################################################################
#                       The AMM model adapted for axes.                       #
###############################################################################

## The model takes as inputs the following values:
# r  = a 1D array, corresponding to a vector from the city center
# m  = transport costs
# y  = income of the residents
# u0 = initial utility
# alpha, beta = values for the Utility function satisfying alpha + beta = 1
# N  = population
# k  = number of axes
# Ra = Agricultural rent
# width = the width of the transport axes


def AMM_axes(r, m, y, u0, alpha, beta, N,k,Ra,width): 
    Nstar = 0
    ustar = u0 
    while np.abs(N - Nstar)/N > 0.03:
        rc, rci = city_boundary(r, m, y, Ra,k)
        i = bisect.bisect_left(r,rc)
        Tr = m*r[0:i]
        psi = bid_rent(y, Tr, ustar, alpha, beta)
        s, rho = dwelling_surface(y, Tr, ustar, alpha, beta)
        circumcircradius = 1/2*ma.tan(ma.pi/k)
        j = bisect.bisect_right(r,circumcircradius)
        Nstar = k*population_1D(rho[j:i], r[j:i],width)
        z = y - m*r[0:i] - psi[0:i] * s[0:i]
        ustar = ustar * (1 + 0.1 * (Nstar/N - 1))
    ucd =  alpha * np.log(z[0:rci]) + beta * np.log(s[0:rci])
    if np.abs(ustar-ucd[0])/ustar > 0.005:
        print ("Error in AMM function. Utilities not equal!")
    return rho,rc,rci,ustar, Nstar

# The population is supposed to reside entirely outside the city center, which
# is a central business district (cbd). To find the part of the city that lies outside
# the cbd we assume that the polygon coincides with the circumscribed circle. 
    
class AMM:
        def __init__(self,r,cost,densityprofile,cityboundary,cityboundary_index,axesnumber,utility,population):
            self.r = r
            self.cost = cost
            self.densityprofile = densityprofile
            self.cityboundary = cityboundary
            self.cityboundary_index = cityboundary_index
            self.axesnumber = axesnumber
            self.utility = utility

def bid_rent(y, Tr, u, alpha, beta):
    psi = alpha**(alpha / beta) * beta * (y - Tr)**(1/beta) * np.exp(-u/beta)
    return psi

# psi (= bid rent) is assumed to be equal to rent R. See Fujita, 1989: "Urban Econonmic theory"

def dwelling_surface(y, Tr, u, alpha, beta):        
    s = alpha**(-alpha / beta) * (y - Tr)**(-alpha/beta) * np.exp(u/beta)
    rho = 1./s
    return s, rho

# See Creutzig, 2014: "How fuel prices determine public transport infrastructure, modalshares and urban form"
# in Urban Climate, 10:63–76.

def utility_1D(y, Tr, r,rci, N, alpha, beta):
    u = -1 * beta*np.log(N) / (sp.integrate.simps(1/ ( alpha**(-alpha/beta) * (y - Tr)**(-alpha/beta)), x = r[0:rci]))
    return u

def consumption(r, m, psi, s, y):
    z = y - m*r - psi*s
    return z

def population_1D(rho, r,width):     # Expects rho, r bounded and in equal spacings
    N = sp.integrate.simps(rho, x = r)*width   # x is arg* of sp.integrate.simps
    return N

def population_2D(rho, r):     # Expects rho, r bounded and in equal spacings
    N = sp.integrate.simps(2*np.pi*rho*r, x = r)   # x is arg* of sp.integrate.simps
    return N

def city_boundary(r, m, y, Ra,k):        #Determine city boundary by evaluating budget equation, divided by number of axes. 
    for ii in np.arange(0,np.size(r), dtype = np.int32):
        budget = y - m *r[ii] - 1 -Ra*1    #(minimum consumption and living space are both defined as 1)
        if budget <= 0 or ii == np.size(r)-1:   # i.e.the money spent on transport equals the income: No more money can be spent on either housing or consumption
            rci = ii - 1 
            rc = r[rci]/k
            break
    return rc, rci
    
# The interior angle of the park triangles depends on the number of axes:
def interiorangle(k):
    gamma = 2*ma.pi/k 
    return gamma
