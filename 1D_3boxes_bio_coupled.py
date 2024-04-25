#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 09:29:07 2023

@author: eb711
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from math import sin, cos, tan, pi, acos, sqrt, factorial, exp, log, floor
from time import perf_counter
import pandas as pd
import sys


# -------------------------------- #
# Read in csv file with paramaters #
# -------------------------------- #

df = pd.read_csv("NPZD_params2.csv")

# Assign parameter names and make parameters universal in the code (=)
Vp0         = df.iloc[0][1]                 # max rate photosynthesis at zero degrees C, g C (g chl)-1 h-1
alpha_bio   = df.iloc[1][1]                 # initial slope of P-I curve, g C (g chl)-1 h-1 (W m-2)-1
kN          = df.iloc[2][1]                 # half sat. constant: N, mmol m-3
mP          = df.iloc[3][1]                 # phyto. mortality rate, linear, d-1
mP2         = df.iloc[4][1]                 # phyto. mortality rate, quadratic, (mmol N m-3)-1 d-1
Imax        = df.iloc[5][1]                 # zooplankton max. ingestion, d-1
kz          = df.iloc[6][1]                 # zoo. half sat. for intake, mmol N m-3
phiP        = df.iloc[7][1]                 # zoo. preference: P
phiD        = df.iloc[8][1]                 # zoo. preference: D
betaz       = df.iloc[9][1]                 # zoo. absorption efficiency: N
kNz         = df.iloc[10][1]                # zoo. net production efficiency: N
mz          = df.iloc[11][1]                # zoo. mortality rate, linear, d-1
mz2         = df.iloc[12][1]                # zoo. mortality, quadratic, (mmol N m-3)-1 d-1
VD          = df.iloc[13][1]                # detritus sinking rate, m d-1
mD          = df.iloc[14][1]                # detritus remineralisation rate, d-1
t_vent      = 5*365            # ventilation timescale, d
Wmix        = 0.13
CtoChl      = df.iloc[16][1]                # carbon to chlorophyll ratio, g g-1
g0          = 0.89
T0          = 30 + 273.15

   
# -------------------------------------------------- #
# Read in initial conditions and run characteristics #
# -------------------------------------------------- #
    
#init <- read.csv("./NPZD_extra.txt", quote="'")
df2 = pd.read_csv("NPZD_extra2.csv")
Pinit        = df2.iloc[0][1]     # initial P, mmol N m-3
Ninit        = df2.iloc[1][1]     # initial N, mmol N m-3
Zinit        = df2.iloc[2][1]     # inital  Z, mmol N m-3
Dsinit       = df2.iloc[3][1]     # initial Ds, mmol N m-3
kw           = df2.iloc[4][1]     # light attenuation coeff.: water,  m-1
kc           = df2.iloc[5][1]     # light attenuation coeff.: phytoplankton, m2 (mmol N)-1
delta_t      = df2.iloc[6][1]     # time step, d (choose at time step that divides exactly into 1 as an integer)
nyears       = 20     # run duration, years
flag_stn     = int(df2.iloc[8][1])     # choice of station; 1=India (60N 20W), 2=Biotrans (47N 20W), 3=Kerfix (50 40?S 68 25?E), 4=Papa (50N, 145W)
flag_LI      = int(df2.iloc[9][1])    # 1=numeric; 2=Evans & Parslow (1985); 3=Anderson (1993)
flag_atten   = int(df2.iloc[10][1])    # 1: kw+kcP (one layer for ML); 2: Anderson (1993) (spectral, layers 0-5,5-23,>23 m
flag_irrad   = int(df2.iloc[11][1])    # 1=triangular over day; 2=sinusoidal over day
flag_PIcurve = int(df2.iloc[12][1])   # 1=Smith fn; 2=exponential fn
flag_grazing = int(df2.iloc[13][1])   # (1) phihat_i = phi_i, (2) phihat_i = phi_i*P_i
flag_outtype = int(df2.iloc[14][1])   # output choice: 0=none, 1=last year only, 2=whole simulation
flag_outfreq = int(df2.iloc[15][1])   # output frequency: 1=every day, 0=every time step
flag_integ   = int(df2.iloc[16][1])     # choice of integration method: 0=Euler, 1=Runge Kutta 4

# ----------------------------------- #
# Read in Stations data from CSV file #
# ----------------------------------- #

df3 = pd.read_csv("stations_forcing2.csv")
MLDi = np.zeros((13,), dtype=float)
SSTi = np.zeros((13,), dtype=float)

if flag_stn == 1 :                 # Station India (60N 20W)
    for i in range(0,13):
        MLDi[i] = df3.iloc[i][0]   # i,0 is first column in the text file
        SSTi[i] = df3.iloc[i][1]   # i,1 is second column in the text file
        latitude = 60.0            # latitude, degrees
        clouds = 6.0               # cloud fraction, oktas
        e0 = 12.0                  # atmospheric vapour pressure
        aN = 0.0074                # coeff. for N0 as fn depth
        bN = 10.85                 # coeff. for N0 as fn depth
elif flag_stn == 2 :               # Station Biotrans (47 N 20 W)
    for i in range(0,13):
        MLDi[i] = df3.iloc[i][2]
        SSTi[i] = df3.iloc[i][3]
        latitude = 47.0
        clouds = 6.0
        e0 = 12.0
        aN = 0.0174
        bN = 4.0
elif flag_stn == 3 :               # Station Kerfix (50 40?S 68 25?E)
    for i in range(0,13):
        MLDi[i] = df3.iloc[i][6]
        SSTi[i] = df3.iloc[i][7]
        latitude = -50.67
        clouds = 6.0
        e0 = 12.0
        aN = 0
        bN = 26.1
elif flag_stn == 4 :               # Station Papa (50N 145W)
    for i in range(0,13):
        MLDi[i] = df3.iloc[i][4]
        SSTi[i] = df3.iloc[i][5]
        latitude = 50.0
        clouds = 6.0
        e0 = 12.0
        aN = 0.0
        bN = 14.6


# Convert latitude to radians
latradians = latitude*pi/180.0


# ------------------------------------------------------------------------------- #
# Mixed layer depth: set up 366 unit array for one year                           #
# Fitting sinusoid around starting temperatire and MLD                            #
# ------------------------------------------------------------------------------- #

MLD = np.zeros((366,), dtype=float)   # mixed layer depth (m)
SST = np.zeros((366,), dtype=float)

for j in range(0, 366):
    MLD[j] = 100 + 50*math.cos(2*math.pi*j/365 + 75)
    
for j in range(0, 366):
    SST[j] = 6 - 3*math.sin(2*math.pi*j/365)
    
    
MLD[0] = MLD[365]           # note: position 1 in the array corresponds to t=0
SST[0] = SST[365]
"""

# ------------------------------------------------------------------------------- #
# Mixed layer depth: set up 366 unit array for one year                           #
# Interpolation used here from monthly data (assumed to be middle of month)       #
# ------------------------------------------------------------------------------- #

MLD = np.zeros((366,), dtype=float)   # mixed layer depth (m)
SST = np.zeros((366,), dtype=float) 

for istep in range(0, 366): #seq(1,365)) {   # interpolate data from monthly to daily values
    rdiv = 365.0/12.0            # monthly data
    quot, remdr = divmod(istep, rdiv)
    remdr = remdr/rdiv
    n1=int(quot)
    if n1 == 12 :
        n1 = 1
    n2 = n1+1
    iday = istep+15          # set MLD data for middle (day 15) of month
    if iday > 365 :
        iday = iday-365

    MLD[iday] = MLDi[n1] + (MLDi[n2]-MLDi[n1])*remdr    # linear interpolation of data
    SST[iday] = SSTi[n1] + (SSTi[n2]-SSTi[n1])*remdr


#for istep in range(365,2,-1) : #{      # shift array because first position corresponds to t=0
#    MLD[istep] = MLD[istep-1]
#    SST[istep] = SST[istep-1]
#}
MLD[0] = MLD[365]           # note: position 1 in the array corresponds to t=0
SST[0] = SST[365]
"""
######################################################
# -------------------------------------------------- #
# Variables specific to model: adjust accordingly    #
# -------------------------------------------------- #
######################################################

nSvar = 4      # no. state variables
nfluxmax = 10  # max no. of flux terms associated with any one state variable
nDvar = 20     #GLOBAL # max no. auxilliary variables or output/plotting (these are defined in the Y array: FNget_flux function)

X = (Pinit,Ninit,Zinit,Dsinit)    # array X holds values of state variables: assign initial conditions

Svarname = ("P","N","Z","D")      # array of text strings (keep them short) for state variables
Svarnames = ' '.join(Svarname)   # concatenated string of state variable names

#Y = np.zeros((20,), dtype=float)          # array Y holds values of auxiliary variables
Y = np.zeros((nDvar,), dtype=float)          # array Y holds values of auxiliary variables

kPAR = np.zeros((4,), dtype=float)        # light extinction coeff., m-1
zbase = np.zeros((4,), dtype=float)       # depth of base of layer; for element 1 in array (surface) this is zero
Ibase = np.zeros((4,), dtype=float)       # irradiance leaving the base of layer; for element 1, this is surface irradiance, W m-2
zdep = np.zeros((4,), dtype=float)        # depth of the layer, m
aphybase = np.zeros((4,), dtype=float)    # a (absorption by chlorophyll) at base of layer; only required when using Anderson (1993) calculation of photosynthesis


"""
Setting Parameters for physical model

"""

lamda = 1  # climate feedback
m = 5.1352*10**18  # mass of dry air
Kg = 5*10**(-5)
P = 1  # atmospheric pressure in atm
Ma = 28.97*10**(-3)  # mean molecular mass
Na = m/Ma  # number of moles in the atmosphere
alpha = 5.35  # radiative forcing coefficient

rho_o = 1025  # referenced ocean density
rho_a = 1  # air density
cp_o = 4*10**3  # ocean specific heat capacity
cp_a = 1*10**3  # air specific heat capacity
Sref = 34.5  # referenced ocean salinity
Tref = 6 + 273.15  # referenced ocean temperature - we will have it fluctuate between 6 degrees and 12 degrees. Going to use Papa data
                   # temperature peaks at around august/ september
                   
Ar_a = 3.4*10**14  # area of atmospheric layer
Ar_o = Ar_a  # area of ocean layer

h_a = 10*10**3  #  thickness of atmospheric layer
h_ocean = 4000  # thickness of ocean
h_m = 100  # thickness of mixed layer - we will have this fluctuate between 50m and 150m. 50m in June/ July and 150m in jan/december
h_d = 150  # thickness of ocean interior
h_ab = h_ocean - h_m - h_d

CC = 50  # air-sea heat transfer parameter 
tc1 = t_vent*24*3600  # time scale for ventilation of mixed layer and thermocline in seconds
tc2 = 200*365*24*3600  # time scale for ventilation of thermocline and ocean interior
Dt = 3600*24  # time step: a day

Pho = 2.157*10**(-6)  # mol/kg Inorganic Phosphate for the ocean; choose value; remains unchanged
Sil = 2.157*10**(-6)  # mol/kg Silicate value for the ocean; choose value; remains unchanged

CN_P = 6.625
CN_Z = 5.625
CN_D = 6.625
DRC = 0.8
ARC = 0.1


print("----------------- End of declarations and initialisation -----------------")

### End of variables specific to model 

# ---------------------------------------------------------------------------------- #
# Function get_flux: the specification of the ecosystem model is primarily handled   #
#  here. The terms in the differential equations are calculated and transferred to   #
#  matrix, flux[i,j], which is passed back to the core code for integration.         #
#  The user can define auxiliary variables, Y[i], that are written to output         #
#  files (state variables are automatically stored for output).                      #
# ---------------------------------------------------------------------------------- #


def get_flux(X, daynow, SST_day, tc_input):      # X is the array that carries state variables
#def get_flux(X, daynow):      # X is the array that carries state variables
# (note that all functions begin with "FN...", although this is not obligatory in R

    P, N, Z, Ds = X         # unpack state variables: P, N, Z, D 

    # Environmental forcing for current day of year

    noonparnow = noonparcalc(daynow,latradians,clouds,e0)  # noon irradiance (PAR), W m-2
    daylnow = daylcalc(daynow,latradians)                  # day length, h  

    #MLD1 = MLD[daynow]                                       # mixed layer depth at start of day
    MLD1 = MLD[daynow-1]      # NB: Correction in Python index starts from 0
    #MLD2 = MLD[daynow+1]                                     # mixed layer depth at end of day
    MLD2 = MLD[daynow]        # NB: Correction in Python index starts from 0

    MLDnow = MLD1+(istepnow-1)/nstepsday*(MLD2-MLD1)         # MLD at start of current time step
    N0 = aN*MLDnow + bN                                      # DIN immediately below mixed layer

    Y[0:] = 0.  # re-initialise Y-array to zero

    VpT = Vp0*1.066**SST_day[daynow-1]             # maximum photosynthetic (growth) rate (Eppley function of sea surface temperature)
    
    #VpT = g0*1.47**(((SST_day[daynow -1] + 273.15) - T0)/10)     # photosynthetic (growth) rate (Q10 model of sea surface temperature)
    
    ### Light attenuation in water column 
    # Major choice here regarding calculation of light attenuation in the water column:
    # (1) Mixed layer as single layer, with light attenuation calculated according to k_tot = k_w + k_c*P 
    # (2) Mixed layer split into three layers according to Anderson (1993): 0-5, 5-23, >23m, with a
    #       separate extinction coefficient for each layer as a function of chlorophyll

    chl = P*6.625*12.0/CtoChl     # chlorophyll, mg m-3 (Redfield ratio of 6.625 mol C mol N-1 assumed for C:N of phytoplankton)
    ss = sqrt(chl)                # square root of chlorophyll

    # Four unit array for the following variables handles three layers; first unit (0) in array is for surface properties, 
    #  elements 1,2,3 (2,3,4) handle layers 1,2,3 respectively (if only one layer needed (option 1 or light attenuation), then
    #  elements 2,3 (3,4) are redundant

    kPAR[0:] = 0.                   # light extinction coeff., m-1
    zbase[0:] = 0.                  # depth of base of layer; for element 1 in array (surface) this is zero                 
    Ibase[0:] = 0.                  # irradiance leaving the base of layer; for element 1, this is surface irradiance, W m-2
    zdep[0:] = 0.                   # # depth of the layer, m
    aphybase[0:] = 0.               # a (absorption by chlorophyll) at base of layer; only required when using Anderson (1993) calculation of photosynthesis

    # flag_atten is choice of light attenuation scheme: (1) single layer k_w+k_c*P, (2) three layer, extinction coefficients as fn chl (Anderson, 1993)
    jnlay = 0

    if (flag_atten == 1):    # option (1)
        jnlay = 1            # mixed layer as a single layer
        zbase[0] = 0.
        zbase[1] = MLDnow
        zdep[1] = MLDnow
        kPAR[1] = kw + kc*P
        Ibase[0] = noonparnow                      # irradiance entering water column
        Ibase[1] = Ibase[0]*exp(-kPAR[1]*MLDnow)   # irradiance at base of mixed layer
  
    elif (flag_atten==2):     # option (2): MLD separated into depth ranges 0-5m, 5-23m and >23m  Anderson (1993)

        Ibase[0] = noonparnow   # irradiance entering water column
        kPAR[1] = 0.13096 + 0.030969*ss + 0.042644*ss**2 - 0.013738*ss**3 + 0.0024617*ss**4 - 0.00018059*ss**5     # extinction coefficients (Anderson, 1993)
        kPAR[2] = 0.041025 + 0.036211*ss + 0.062297*ss**2 - 0.030098*ss**3 + 0.0062597*ss**4 - 0.00051944*ss**5
        kPAR[3] = 0.021517 + 0.050150*ss + 0.058900*ss**2 - 0.040539*ss**3 + 0.0087586*ss**4 - 0.00049476*ss**5
        zbase[0] = 0.
        Ibase[0] = noonparnow      # irradiance entering water column

        # Three layers only if MLD > 23.0m, otherwise one or two layers:
  
        if (MLDnow <= 5.0):
            jnlay = 1
            zbase[1] = MLDnow
            zdep[1] = MLDnow
            Ibase[1] = Ibase[0]*exp(-kPAR[1]*zdep[1])    # irradiance leaving layer 1
        elif (MLDnow > 5 and MLDnow <= 23.0):
            jnlay = 2
            zbase[1] = 5.0
            zdep[1] = 5.0 
            Ibase[1] = Ibase[0]*exp(-kPAR[1]*5.0)        # irradiance leaving layer 1
            zbase[2] = MLDnow
            zdep[2] = MLDnow-5.0
            Ibase[2] = Ibase[1]*exp(-kPAR[2]*zdep[2])    # irradiance leaving layer 2
        elif (MLDnow > 23.0):
            jnlay = 3
            zbase[1] = 5.0
            zdep[1] = 5.0 
            Ibase[1] = Ibase[0]*exp(-kPAR[1]*5.0)        # irradiance leaving layer 1
            zbase[2] = 23.0
            zdep[2] = 23.0-5.0
            Ibase[2] = Ibase[1]*exp(-kPAR[2]*zdep[2])    # irradiance leaving layer 2
            zbase[3] = MLDnow
            zdep[3] = MLDnow-23.0
            Ibase[3] = Ibase[2]*exp(-kPAR[3]*zdep[3])    # irradiance leaving layer 3

    ### Calculate L_I (light limitation of growth, 0 <= L_I <= 1)

    L_Isum = 0.     # L_I is calculated as a weighted sum over the total mixed layer

    for ilay in range(1,jnlay+1):         # loop over layers; element 1 (2) in array corresponds to first layer

        # Call function for calculating photosynthesis
        if (flag_LI == 1): 
            # numeric integration for light over time (through day) with analytic depth integrals
            L_I = LIcalcNum(zdep[ilay],Ibase[ilay-1],Ibase[ilay],kPAR[ilay],alpha_bio,VpT,daylnow,flag_irrad,flag_PIcurve)
       # elif (flag_LI == 2):               # Evans and Parslow (1985): triangular light, Smith fn for P-I curve
       #     L_I = FNLIcalcEP85(zdep[ilay],Ibase[ilay-1],Ibase[ilay],kPAR[ilay],alpha,VpT,daylnow)
       # elif (flag_LI == 3):              # Anderson (1993): sinusoidal light, exponential fn for P-I curve, alpha spectrally dependent
       #     aphybase[1] = 0.36796 + 0.17537*ss - 0.065276*ss**2 + 0.013528*ss**3 - 0.0011108*ss**4           # a (chl absorption) at ocean surface as function chl
       #     ahash = FNaphy(ss,zbase[ilay-1],zbase[ilay],aphybase[ilay-1])                                 # change in a with depth
       #     aphybase[ilay] = aphybase[ilay-1]+ahash                                                       # a at base of layer
       #     aphyav = aphybase[ilay-1]+ahash*0.5                                                           # average a in layer (from which alpha is calculated: alpha = a*alphamax
       #     L_I = FNLIcalcA93(zdep[ilay],Ibase[ilay-1],Ibase[ilay],kPAR[ilay],alpha,VpT,daylnow,aphyav)
  
        L_Isum = L_Isum + L_I*zdep[ilay]      # multiply by layer depth in order to set up weighted average for total mixed layer

    L_I = L_Isum/MLDnow                   # weighted average for mixed layer

    ### Calculate L_N (nutrient limitation of growth, 0 <= L_I <= 1)
    L_N = N/(kN+N)                  # nutrient limitation of growth rate (0 <= L_N <= 1)

    ### Grazing 
    if (flag_grazing == 1):           # phihat_i = phi_i (grazing preferences)
        phiPhat = phiP                
        phiDhat = phiD
    else: 
        phiPhat = phiP*P              # phihat_i = phi_i*P_i
        phiDhat = phiD*Ds

    intakespP = Imax*phiPhat*P/(kz**2+phiPhat*P+phiDhat*Ds)    # specific intake: phytoplankton, d-1
    intakespD = Imax*phiDhat*Ds/(kz**2+phiPhat*P+phiDhat*Ds)   # specific intake: detritus, d-1
 
    # terms in the differential equations  (unit: mmol N m-3 d-1)          

    Pgrow = VpT*L_I*L_N*24/CtoChl*P                                                  # P growth
    Pgraz = intakespP*Z                                                              # P grazed
    Dgraz = intakespD*Z                                                              # D grazed
    Pmort = mP*P                                                                     # P mortality: linear
    Pmort2 = mP2*P*P                                                                 # P mortality: quadratic
    Pmix  = ((0-P)/delta_t)*(math.exp(-delta_t/tc_input) - 1)                     # P loss: mixing
    #Pmix  = Wmix*P/MLDnow
    Zmix  = ((0-Z)/delta_t)*(math.exp(-delta_t/tc_input) - 1)                     # Z loss: mixing     
    #Zmix  = Wmix*Z/MLDnow
    Nmix  = ((N0-N)/delta_t)*(math.exp(-delta_t/tc_input) - 1)                    # N net input: mixing
    #Nmix  = -Wmix*(N0 - N)/MLDnow
    Zgrow = betaz*kNz*(Pgraz+Dgraz)                                                  # Z growth
    Zexc  = betaz*(1.0-kNz)*(Pgraz+Dgraz)                                            # Z excretion
    Zpel  = (1.0-betaz)*(Pgraz+Dgraz)                                                # Z faecal pettet production
    Zmort = mz*Z                                                                     # Z mortality: linear
    Zmort2 = mz2*Z**2                                                                # Z mortality: quadratic
    Dmix  = ((0-Ds)/delta_t)*(math.exp(-delta_t/tc_input) - 1)                    # D loss: mixing
    #Dmix  = Wmix*Ds/MLDnow
    Dsink = VD*Ds/MLDnow                                                             # D loss: sinking
    Dremin = mD*Ds                                                                   # D loss: remineralisation
  
    #### Entrainment/dilution (when depth of mixed layer increases)

    if (MLD2-MLD1 > 0.0):          
        Nadd = (MLD2-MLD1)*N0/MLDnow       # entrainment: N
        Ndilute = N*(MLD2-MLD1)/MLDnow     # dilution: N already present in ML
        Pdilute = P*(MLD2-MLD1)/MLDnow     # dilution: P
        Zdilute = Z*(MLD2-MLD1)/MLDnow     # dilution: Z
        Ddilute = Ds*(MLD2-MLD1)/MLDnow    # dilution: Ds                             
    else:                     
        Nadd = 0.0                         # detrainment, but concentration in ML unchanged
        Ndilute = 0.0
        Pdilute = 0.0
        Zdilute = 0.0
        Ddilute = 0.0
    #}

    ### Transfer terms as calculated above are trasferred to matrix flux[i,j] for processing by the core code
    # (number of terms for any one state variable should not exceed nfluxmax as specified in core code)
                                                                                                 
    # Phytoplankton
    flux[0][0] = Pgrow                 # P growth
    flux[1][0] = -Pgraz                # P grazed
    flux[2][0] = -Pmort                # P non-grazing mortality: linear
    flux[3][0] = -Pmort2               # P non-grazing mortality: quadratic
    flux[4][0] = -Pmix                 # P loss: mixing
    flux[5][0] = -Pdilute              # P loss: dilution  

              
    # Nitrate
    flux[0][1] = -Nmix                  # N net input: mixing
    flux[1][1] = Zexc                  # Z excretion
    flux[2][1] = Dremin                # D remineralisation
    flux[3][1] = -Pgrow                # uptake by P
    flux[4][1] = Nadd                  # N input: entrainment
    flux[5][1] = -Ndilute              # N loss: dilution
  
    # Zooplankton
    flux[0][2] = Zgrow                 # Z growth
    flux[1][2] = -Zmort                # Z mortality: linear
    flux[2][2] = -Zmort2               # Z mortality: quadratic
    flux[3][2] = -Zdilute              # Z loss: dilution
    flux[4][2] = -Zmix                 # Z loss: mixing

    # Detritus
    flux[0][3] = Pmort                 # D from P mortality (linear)
    flux[1][3] = Pmort2                # D from P mortality (quadratic)
    flux[2][3] = Zmort                 # D from Z mortality (linear)
    flux[3][3] = Zpel                  # D from Z faecal pellets
    flux[4][3] = -Dgraz                # D grazed
    flux[5][3] = -Dmix                 # D loss: mixing
    flux[6][3] = -Dsink                # D loss: sinking
    flux[7][3] = -Dremin               # D loss: remineralisation
    flux[8][3] = -Ddilute              # D loss: dilution

    # User-definied auxiliary variables for writing to output files: number stored = nDvar (core code)
    # If you want to increase or decrease the number of units in this array, this is controlled by nDvar in core code

    Y[0]  = L_I                  
    Y[1]  = L_N
    Y[2]  = Pgrow/P
    Y[3]  = MLDnow
    Y[4]  = noonparnow
    Y[5]  = chl
    Y[6]  = SST_day[daynow]
    #Y[6]  = SST[daynow]
    Y[7]  = Pgrow*MLDnow
    Y[8]  = Pgrow
    Y[9] = Pgraz
    Y[10] = Pmort+Pmort2
    Y[11] = 0
    Y[12] = 0
    Y[13] = 0
    Y[14] = 0
    Y[15] = 0
    Y[16] = 0
    Y[17] = 0
    Y[18] = 0
    Y[19] = 0
    
    return flux      # matrix flux is passed back to the core code


# ------------------------------------------------------------------------------------------------------ #
# L_I: Calculation of daily photosynthesis: numeric intergration (time) and analytic integration (depth) #
# Diel cycle of irradiance: choice of (1) triangular vs (2) sinusoidal (flag_irrad)                      #
# P-I curve: choice of (1) Smith fn vs (2) exponential fn  (flag_PIcurve)                                #
# numeric integration (time) uses nstepst+1 steps and trapezoidal rule for first and last steps          #
#   (the user may manually change nstepst in the code if so desired)                                     #
# -------------------------------------------------------------------------------------------------------#

def daylcalc(jday, latradians):
    """
    Calculation of day length as a function of day of year and latitude

    Parameters
    ----------
    jday : day of the year in julian days
    latradians : latitude in radians

    Returns
    -------
    day_length : TYPE
        DESCRIPTION.

    """
    
    declin = 23.45 * math.sin(2 * math.pi * (284 + jday) / 365) * math.pi/180  # calculating solar declination angle
    
    day_length = 2 * math.acos( -1 * math.tan(latradians) * math.tan(declin)) * 12/(math.pi)
    
    return (day_length)


def noonparcalc(jday, latradians, clouds, e0):
    """
    Calculation of noon irradiance
    
    """
    albedo = 0.04  # earth's ocean albedo
    solarconst = 1368  # solar constant, W/m^2 - incoming solar radiation that would be incident on a perpendicular, immediately outside the atmosphere
    parrac = 0.43  # photosynthesis active radiation (PAR) fraction
    declin = 23.45 * math.sin(2 * math.pi * (284 + jday) * 0.00274) * math.pi/180  # calculating solar declination angle
    coszen = math.sin(latradians) * math.sin(declin) + math.cos(latradians) * math.cos(declin)  # cosine of zenith angle
    zen = math.acos(coszen) * 180 / math.pi  # zenith angle, degrees
    Rvector = 1/math.sqrt(1 + 0.033 * math.cos(2 * math.pi * jday * 0.00274))  # Earth's radius vector
    Iclear = solarconst * coszen**2 / (Rvector**2) / (1.2 * coszen + e0 * (1 + coszen) * 0.001 + 0.0455)  # irradiance at ocean surface, clear sky
    cfac = (1 - 0.62 * clouds * 0.125 + 0.0019 * (90 - zen))  # cloud factor (atmospheric transmission)
    Inoon = Iclear * cfac * (1-albedo)  # noon irradiance: total solar
    noonparnow = parrac * Inoon  # noon irradiance: photosynthesis active radiation 
    
    return(noonparnow)
 

def LIcalcNum(zdepth, Iin, Iout, kPARlay, alpha_bio, Vp, daylnow, choice_irrad, choice_PI):
    """
    Calculation of daily photosynthesis: numeric integration (time) and analytic integration (depth)
    
    NEED TO ADD FURTHER DESCRIPTIONS/ EXPLANATIONS ETC.

    Parameters
    ----------
    zdepth : TYPE
        DESCRIPTION.
    Iin : TYPE
        DESCRIPTION.
    Iout : TYPE
        DESCRIPTION.
    kPARlay : TYPE
        DESCRIPTION.
    alpha_bio : TYPE
        DESCRIPTION.
    Vp : TYPE
        DESCRIPTION.
    daylnow : TYPE
        DESCRIPTION.
    choiceirrad : TYPE
        DESCRIPTION.
    choicePI : TYPE
        DESCRIPTION.

    Returns
    -------
    Lim_I : 

    """
    sumps = 0.0                     # for calculating sum of photosynthesis
    range_rad = pi/2.0
    nstepst = 10                    # the day from sunrise to midday will be divided into nstepst steps (can be altered if so desired)

    for itme in range(1,nstepst + 1):   # time loop     (no need to calculate itme = 0 because irradiance zero at sunrise)
        #print("itme=", itme)
        if itme == nstepst:            # last step (note use of trapezoidal rule for integration)
            rtrapt = 0.5
        else:       
            rtrapt = 1.0               # other steps

        if choice_irrad == 1:        # coice of either (1) triangular or (2) sinusoidal irradiance over day
            Icorr = itme/nstepst                    # irradiance multiplier for time of day: triangular
        elif choice_irrad == 2:
            Icorr = sin(itme/nstepst*range_rad)     # irradiance multiplier for time of day: sinusoidal

        if choice_PI == 1:           # Smith function (the most straightforward choice in terms of calculating analytic depth integral)
            sumpsz = 0.
            x0 = alpha_bio*Icorr*Iin
            xH = alpha_bio*Icorr*Iin*exp(-kPARlay*zdepth)
            sumpsz = Vp/kPARlay/zdepth*(log(x0+(Vp**2+x0**2)**0.5)-log(xH+(Vp**2+xH**2)**0.5))

        else:                       # exponential function (Platt et al., 1990)               
            # Default setting here is analytic depth integral. Based on a factorial sequence, we found that this does not always perform well and so, if desired, 
            # the user can switch to an analytic depth integral by setting (manually in the code) choiceanalytic <- 2
            choiceanalytic = 1         # 1=analytic solution; 2=numeric solution; set manually in the code
            if choiceanalytic == 1:    # analytic solution
                sumpsz = 0.
                istar1 = alpha * Icorr * Iin/Vp
                istar2 = alpha * Icorr * Iout/Vp
                # Analytic solution is an infinite series
                for ifact in range(1, 16+1):  # 16 units in factorial array (by all means increase this number by altering the code; run time will increase accordingly)
                    sumpsz = sumpsz + (-1)**(ifact+1)/(ifact*factorial(ifact))*(istar1**ifact-istar2**ifact)
                
                sumpsz = sumpsz*Vp/kPARlay/zdepth

            else:   # numeric solution

                nstepsz = 100            #no. of steps: depth
                sumpsz = 0.0

                for izz in range(0, nstepsz+1):     # depth loop
                    if izz == 0 or izz == nstepsz:
                        rtrapz = 0.5
                    else:        # trapezoidal rule
                        rtrapz = 1.0
       
                    z = izz/nstepsz*zdepth
                    Iz = Iin*Icorr*exp(-kPARlay*z)
                    psnow = Vp*(1.0-exp(-alpha*Iz/Vp))
                    sumpsz = sumpsz + psnow*rtrapz

            sumpsz = sumpsz/nstepsz

        sumps = sumps + sumpsz*rtrapt       # sum photosynthesis over daylight hours

    Lim_I = sumps/nstepst*daylnow/24.0  # take account of zero PS during night
    Lim_I = Lim_I/Vp                    # dimensionless
    
    return(Lim_I)

def SST_sine(SST_avg):
    """
    

    Parameters
    ----------
    SST_avg : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    SST_degrees = SST_avg - 273.15
    
    for j in range(0, 366):
        SST[j] = SST_degrees - 0.5*SST_degrees*math.sin(2*math.pi*j/365)
        
    return(SST)
    

def equilibrium(T, S, pCO2eq, pt, sit, ta, pHlocal):
    """
    Calculates dic_eq and pH_eq for a given atmospheric pCO2
    Based on efficient solver following Follows et al. (2006)

    Parameters
    ----------
    T : temperature (degrees K)
    S : salinity (PSU)
    pCO2eq : atmospheric reference pCO2 level for which to find equilibrium DIC
    pt : inorganic phosphate (mol/ kg)
    sit : inorganic silicate (mol/kg)
    ta : total alkalinity
    pHlocal : local pH for first guess of [H+]

    Returns
    -------
    dic_eq : equilibrium total inorganic carbon (mol/ kg)
    Hnew : estimate of new hydrogen ion content
    pH : final pH
    k1 : equiloibrium coefficient (CHECK) 
    k2 : equilibrium coefficient (CHECK)
    ff : fugacity 
    """
    # setting first guess for [H]
    hg = 10**(-pHlocal)
    
    #print("hg = " , hg)
    
    # estimate concentration of borate based on salinity
    scl = S/1.80655
    bt = 0.000232 * scl/10.811
    
    # some definitions to make calculations neater
    S2 = S*S
    sqrtS = math.sqrt(S)
    invT = 1/T
    T1 = T/100
    
    # Coefficient algorithms as used in OCMIP2 protocols
    # K1, K2 Millero (1995) using Mehrbach data
    
    k1 = 10**(-1 * (3670.7 * invT - 62.008 + 9.7944*math.log(T) - 0.0118*S + 0.000116*S2))
    
    k2 = 10**(-1*(1394.7*invT + 4.777 - 0.0184*S + 0.000118*S2))
    
    # K1p, K2p, K3p DOE (1994)
    
    k1p = math.exp(-4576.752*invT + 115.525 - 18.453*math.log(T) + (-106.736*invT + 0.69171)*sqrtS + (-0.65643*invT - 0.01844)*S)
    
    k2p = math.exp(-8814.715*invT + 172.0883 - 27.927*math.log(T) + (-160.34*invT + 1.3566)*sqrtS + (0.37335*invT - 0.05778)*S)
    
    k3p = math.exp(-3070.75*invT - 18.141 + (17.27039*invT + 2.81197)*sqrtS + (-44.99486*invT - 0.09984)*S)
    
    # Kb, Millero (1995) using data from Dickson
    
    kb = math.exp((-8966.90 - 2890.53*sqrtS - 77.942*S + 1.728*S**1.5 - 0.0996*S2)*invT + (148.0248 + 137.1942*sqrtS + 1.62142*S) + (-24.4344 - 25.085*sqrtS - 0.2474*S)*math.log(T) + 0.053105*T*sqrtS)
    
    # Kw, Millweo (1995)
    kw = math.exp(-13847.26*invT + 148.9652 - 23.6521*math.log(T) + (118.67*invT - 5.977 + 1.0495*math.log(T))*sqrtS - 0.01615*S)
    
    # Ksi, Millero (1995)
    
    I = (19.924*S)/(1000 - 1.005*S)
    
    ksi = math.exp(-8904.2*invT + 117.385 - 19.334*math.log(T) +	(-458.79*invT + 3.5913)*(I**0.5) + (188.74*invT - 1.5998)*I + (-12.1652*invT + 0.07871)*(I*I) + math.log(1.0 - 0.001005*S))
    
    # fugacity, Weiss and Price, Marine Chem, 8, 347 (1990)
    
    ff = math.exp(-162.8301 + 218.2968/(T1) + 90.9241*math.log(T1) - 1.47696*(T1*T1) + S*(.025695 - .025225*(T1) + 0.0049867*(T1*T1)))
    
    # First guess of [H+]: from last timestep OR fixed for cold start
    # We iterate for accurate solution

    for i in range(0, 21):
        # Estimate contributions to total alk from borate, silicate, phosphate
        bohg = (bt*kb)/(hg + kb)
        siooh3g = (sit*ksi)/(ksi + hg)
        denom = (hg**3) + (k1p*hg**2) + (k1p*k2p*hg) + (k1p*k2p*k3p)
        h3po4g = (pt*hg**3)/denom
        h2po4g = (pt*k1p*hg**2)/denom
        hpo4g = (pt*k1p*k2p*hg)/denom
        po4g = (pt*k1p*k2p*k3p)/denom
        
        # Estimate carbonate alkalinity
        cag = ta - bohg - (kw/hg) + hg - hpo4g - 2*po4g + h3po4g - siooh3g
        
        # Estimate hydrogen ion conc
        # Here, we solve a system of 3 simultaneous equations, where the 3 unknowns are: [HCO3], [CO3], [H+]
        # We end up with a quadratic in terms of [H+] which we can solve
        Hnew = 0.5*(k1*ff*pCO2eq/cag + math.sqrt((k1*ff*pCO2eq/cag)*(k1*ff*pCO2eq/cag) + 8*k1*k2*ff*pCO2eq/cag))
        hg = Hnew
    
    # Evaluate DIC equilibrium
    dic_eq = pCO2eq*ff*(1.0 + (k1/Hnew) + (k1*k2/(Hnew*Hnew)))
    
    # Calculate final pH
    pH = -math.log(Hnew, 10)
    
    return(dic_eq, Hnew, pH, k1, k2, ff)


def CO2_follows(T, S, dic, pt, sit, ta, pHlocal):
    """
    Calculate CO2 and new pH for given DIC
    Efficient solver following Follows et al. (2006), On the solution of the Carbonate Chemistry system in ocean biogeochemistry models

    Parameters
    ----------
    T : Temperature (degrees K)
    S : Salinity (PSU)
    dic : total inorganic carbon (mol/kg)
    pt : inorganic phosphate (mol/kg)
    sit : inorganic silicate (mol/kg)
    ta : total alkalinity (mol/kg)
    pHlocal : local pH value used for first gyess of [H+]

    Returns
    -------
    CO2 : aqueous carbon dioxide
    Hnew : estimate of new hydrogen ion content
    pH : final pH
    k1 : equiloibrium coefficient (CHECK) 
    k2 : equilibrium coefficient (CHECK)
    ff : fugacity

    """
    """
    
    # Setting first guess for {h}
    hg = 10**(-pHlocal)
    print(hg)
    # Estimated concentration of borate based on salinity
    scl = S/1.80655
    bt = 0.000232 * scl/10.811
    
    # Some definitions
    S2 = S*S
    sqrtS = S**0.5
    invT = 1.0/T
    T1 = T/100
    
    # Coefficient algorithms as used in OCMIP2 protocols
    # K1, K2 Millero 91995) using Mehrbach data
    k1 = 10**(-1*(3670.7*invT - 62.008 + 9.7944*np.log(T) - 0.0118*S + 0.000116*S2))
    
    k2 = 10**(-1*(1394.7*invT + 4.777 - 0.0184*S + 0.000118*S2))
    
    # K1p, K2p, K3p DOE (1994)
    k1p = np.exp(-4576.752*invT + 115.525 - 18.43*np.log(T) + (-106.763*invT + 0.69171)*sqrtS + (-0.65643*invT - 0.01844)*S)
    
    k2p = np.exp(-8814.715*invT + 172.0883 - 27.927*np.log(T) +	(-160.34*invT + 1.3566)*sqrtS +	(0.37335*invT - 0.05778)*S)

    k3p = np.exp(-3070.75*invT - 18.141 + (17.27039*invT + 2.81197) * sqrtS + (-44.99486*invT - 0.09984)*S)
    
    # Kb, Millero (1995) using data from Dickson
    kb = np.exp((-8966.90 - 2890.53*sqrtS - 77.942*S + 1.728*S**1.5 - 0.0996*S2)*invT + (148.0248 + 137.1942*sqrtS + 1.62142*S) + (-24.4344 - 25.085*sqrtS - 0.2474*S) * np.log(T) + 0.053105*T*sqrtS)
    
    # Kw, Millero (1995)
    kw = np.exp(-13847.26*invT + 148.9652 - 23.6521*np.log(T) + (118.67*invT - 5.977 + 1.0495*np.log(T)) * sqrtS - 0.01615*S)
    
    # Ksi, Millero (1995)
    I = (19.924*S)/(1000 - 1.005*S)
    ksi = np.exp(-8904.2*invT + 117.385 - 19.334*np.log(T) + (-458.79*invT + 3.5913)*I**0.5 + (188.74*invT - 1.5998)*I + (-12.1652*invT + 0.07871)*I**2 + np.log(1.0 - 0.001005*S))
    
    # Fugacity, Weiss and Price, Marine Chem, 8, 347 (1900)
    ff = np.exp(-162.8301 + 218.2968/(T1) + 90.9241*np.log(T1) - 1.47696*(T1*T1) + S*(.025695 - .025225*(T1) + 0.0049867*(T1*T1)))
    
    #print("ff = ", ff)
    """
    # setting first guess for [H]
    hg = 10**(-pHlocal)
    
    #print("hg = " , hg)
    
    # estimate concentration of borate based on salinity
    scl = S/1.80655
    #print(scl)
    bt = 0.000232 * scl/10.811
    #print(bt)
    
    # some definitions to make calculations neater
    S2 = S*S
    sqrtS = math.sqrt(S)
    invT = 1/T
    T1 = T/100
    
    #print(S2, sqrtS, invT, T1)
    
    # Coefficient algorithms as used in OCMIP2 protocols
    # K1, K2 Millero (1995) using Mehrbach data
    
    k1 = 10**(-1 * (3670.7 * invT - 62.008 + 9.7944*math.log(T) - 0.0118*S + 0.000116*S2))
    
    k2 = 10**(-1*(1394.7*invT + 4.777 - 0.0184*S + 0.000118*S2))
    
    #print(k1, k2)
    # K1p, K2p, K3p DOE (1994)
    
    k1p = math.exp(-4576.752*invT + 115.525 - 18.453*math.log(T) + (-106.736*invT + 0.69171)*sqrtS + (-0.65643*invT - 0.01844)*S)
    
    k2p = math.exp(-8814.715*invT + 172.0883 - 27.927*math.log(T) + (-160.34*invT + 1.3566)*sqrtS + (0.37335*invT - 0.05778)*S)
    
    k3p = math.exp(-3070.75*invT - 18.141 + (17.27039*invT + 2.81197)*sqrtS + (-44.99486*invT - 0.09984)*S)
    
    # Kb, Millero (1995) using data from Dickson
    
    kb = math.exp((-8966.90 - 2890.53*sqrtS - 77.942*S + 1.728*S**1.5 - 0.0996*S2)*invT + (148.0248 + 137.1942*sqrtS + 1.62142*S) + (-24.4344 - 25.085*sqrtS - 0.2474*S)*math.log(T) + 0.053105*T*sqrtS)
    
    # Kw, Millweo (1995)
    kw = math.exp(-13847.26*invT + 148.9652 - 23.6521*math.log(T) + (118.67*invT - 5.977 + 1.0495*math.log(T))*sqrtS - 0.01615*S)
    
    # Ksi, Millero (1995)
    
    I = (19.924*S)/(1000 - 1.005*S)
    
    ksi = math.exp(-8904.2*invT + 117.385 - 19.334*math.log(T) +	(-458.79*invT + 3.5913)*math.sqrt(I) + (188.74*invT - 1.5998)*I + (-12.1652*invT + 0.07871)*(I*I) + math.log(1.0 - 0.001005*S))
    
    # fugacity, Weiss and Price, Marine Chem, 8, 347 (1990)
    
    ff = math.exp(-162.8301 + 218.2968/(T1) + 90.9241*math.log(T1) - 1.47696*(T1*T1) + S*(.025695 - .025225*(T1) + 0.0049867*(T1*T1)))
    
    
    # First guess of [H+]: from last timestep OR fixed for cold start
    # --- here iterate for accurate solution
    
    for i in range(0, 21):
        # Estimate contributions to toal alk from borate, silicate, phosphate
        bohg = (bt*kb)/(hg+kb)
        siooh3g = (sit*ksi)/(ksi + hg)
        denom = hg*hg*hg + k1p*hg*hg + k1p*k2p*hg + k1p*k2p*k3p
        h3po4g = pt*hg*hg*hg/denom
        h2po4g = pt*k1p*hg*hg/denom
        hpo4g = pt*k1p*k2p*hg/denom
        po4g = pt*k1p*k2p*k3p/denom
        
        # Estimate carbonate alkalinity
        cag = ta - bohg - (kw/hg) + hg - hpo4g - 2*po4g + h3po4g - siooh3g
        
        # Estimate hydrogen ion conc
        # Again solving system of simultaneous equations but with different unknowns
        # Equations below are exactly those presented in Follows et al. (2006)
        gamm = dic/cag
        dummy = ((1-gamm)*(1-gamm))*(k1*k1) - 4*k1*k2*(1 - 2*gamm)
        Hnew = 0.5*((gamm - 1)*k1 + math.sqrt(dummy))
        hg = Hnew
        
    # Evaluate CO2
    CO2 = dic/(1.0 + (k1/Hnew) + (k1*k2/(Hnew*Hnew)))
    
    # Caluclate final pH
    pH = -math.log(Hnew, 10)
    
    return(CO2, Hnew, pH, k1, k2, ff)


# ----------------------------------- #
# Basic setup for background biology  #
# ----------------------------------- #

ndays = int(nyears*365)                          # no. of days in simulation
nsteps = int(floor(ndays/delta_t+delta_t/1000))                   # no. of time steps in simulation
nstepsday = int(floor(1/delta_t+delta_t/1000))                    # no. of time steps per day
nwrite = int(flag_outfreq*(ndays+1)+(1-flag_outfreq)*(nsteps+1))

#tm <- array(dim=c(nwrite), rep(0.0, times=nwrite))    # array to store time for output
tm = np.zeros((nwrite,), dtype=int)

#flux <- matrix(0,nrow=nfluxmax,ncol=nSvar)            # matrix to hold update terms for state variables
flux = np.zeros((nfluxmax,nSvar))

#fluxyear <- matrix(0,nrow=nfluxmax,ncol=nSvar)        # matrix to sum fluxes over year
fluxyear = np.zeros((nfluxmax,nSvar))
fluxstep = np.zeros((nfluxmax,nSvar))
fluxyear3D = np.ones((nfluxmax,nSvar,20))

#Svar <- matrix(nrow=nwrite, ncol=nSvar)               # matrix to store values of state variables
Svar = np.zeros((nwrite,nSvar))

#Dvar <- matrix(nrow=nwrite, ncol=nDvar)               # matrix to store values of auxiliary variables
Dvar = np.zeros((nwrite,nDvar))

#Svar[1,] <- X                                    # store initial values of state variables (position 1 in array corresponds to t=0)
Svar[0][:] = X

tme = 0       # time=0: start of simultion
tm[1] = tme
 
#Xout <- signif(X,digits=4)     # truncate X to 4 significant figures, ready for writing to output file
Xout = X
if flag_outtype != 0 :
    #write(c(tme,Xout),file="out_statevars.txt",ncolumns=nSvar+1,sep=",",append=FALSE)  # write inital values of state vars to file
    #print(tme, Xout)
    pass

#icount = 1      # records position in array for writing output
icount = 0   # For numpy arrays starts from index 0
istep = 0

# ------------------------------------- #
# Basic setup for background biology    #
# ------------------------------------- #

nstepsday = 2
nstepsday = int(floor(1/delta_t+delta_t/1000))
fluxyear = np.zeros((nfluxmax,nSvar))
fluxstep = np.zeros((nfluxmax,nSvar))
fluxyear3D = np.zeros((20, nfluxmax, nSvar))
Ystep = np.zeros((nDvar,), dtype=float)


"""
INITIAL BIOLOGY RUN FOR BACKGROUND SIGNAL

"""

for iyear in range(0, 20):  # trying to restructure loop so that we first loop of the years - this is because the biological model and physical model communicate with each other after each year


    fluxyear[:][:] = 0.
    
    SST_day = SST_sine(Tref)
    
    
    for iday in range(1,366):
        daynow = iday                            # global variable to pass to functions

        for istepnow in range(1,nstepsday+1):       # loop over time steps within one day (1,nstepsday)
            
            istep = istep + 1
            istepplus = istep + 1               # (Dvar (auxiliary variables) stored at beginning of each time step)
                        
            #fluxstep <- rep(0.0,times=nfluxmax*nSvar)  # matrix to sum up fluxes over time step
            fluxstep[:][:] = 0.0
                        
            #Ystep <- rep(0.0,times=nDvar)              # matrix to store average value of Y for each time step
            Ystep[:] = 0.

            # -------------------------- #
            # Numerical integration      #
            # -------------------------- # 

            flux1 = get_flux(X, daynow, SST_day, t_vent)*delta_t        # 1st iteration
            #flux1 = get_flux(X, daynow)*delta_t        # 1st iteration

            #deltaX1 = colSums(flux1)             # sum of flux terms, for each state variable
            deltaX1 = np.sum(flux1, axis=0)

            if flag_integ == 0:                  # Euler
                fluxstep = fluxstep + flux1
                Ystep = Ystep + Y
            else:                                # Runge Kutta 4
                fluxstep = fluxstep + flux1/6
                Ystep = Ystep + Y/6

            if flag_integ == 1:                   # Runge Kutta only

                flux2 = get_flux(X + deltaX1/2, daynow, SST_day, t_vent)*delta_t       # 2nd iteration
                #flux2 = get_flux(X + deltaX1/2, daynow)*delta_t       # 2nd iteration
                deltaX2 = np.sum(flux2, axis=0)
                fluxstep = fluxstep + flux2/3
                Ystep = Ystep + Y/3

                flux3 = get_flux(X + deltaX2/2, daynow, SST_day, t_vent)*delta_t       # 3rd iteration
                #flux3 = get_flux(X + deltaX2/2, daynow)*delta_t       # 3rd iteration
                #deltaX3 = colSums(flux3)
                deltaX3 = np.sum(flux3, axis=0)
                fluxstep = fluxstep + flux3/3
                Ystep = Ystep + Y/3
  
                flux4 = get_flux(X + deltaX3, daynow, SST_day, t_vent)*delta_t         # 4th iteration
                #flux4 = get_flux(X + deltaX3, daynow)*delta_t         # 4th iteration
                #deltaX4 = colSums(flux4)
                deltaX4 = np.sum(flux4, axis=0)
                fluxstep = fluxstep + flux4/6
                Ystep = Ystep + Y/6

            # sum iterations
            if flag_integ == 0:               # Euler
                deltaX = deltaX1
            else:       #Euler method  
                deltaX = (deltaX1 + 2*deltaX2 + 2*deltaX3 + deltaX4)/6    # Runge Kutta
            #}
    
            X = X + deltaX    # update state varialbes

            fluxyear = fluxyear + fluxstep

            tme = tme + delta_t    # update time

            # -------------------------- #
            # Write to output files      #
            # -------------------------- # 

            # 1st time step awkward: state vars already output for t=0; output fluxes and auxiliary variables for t=0 as calculated for first time step

            if istep == 1:
                Dvar[icount,:] = Ystep[:]
                
                '''
                if flag_outtype == 2 or (flag_outtype==1 and nyears==1):                  # only needed if whole simulation from t=0 is to be written to files
                    #Dvarwrite <- signif(Dvar[icount,],digits=4)        # truncated to 4 significant figures: auxiliary variables
                    Dvarwrite = Dvar[icount,]
                    #fluxwrite <- signif(fluxstep,digits=4)             # truncated to 4 significant figures: fluxes
                    fluxwrite = fluxstep
                    tmenow = 0.0     # actually, it is time 1 time step, but for the purposes of continuity in output files, set to zero
                    write(c(tmenow,Dvarwrite),file="out_aux.txt",ncolumns=nDvar+1,sep=",",append=FALSE)
                    for (iw in seq(1,nSvar)) {
                        if (iw==1) {
                            fluxwritecat <- fluxwrite[,1]
                        } else {
                            fluxwritecat <- c(fluxwritecat,fluxwrite[,iw])
                        }
                    }
                    write(c(tmenow,fluxwritecat),file="out_fluxes.txt",ncolumns=nSvar*nfluxmax+1,sep=",",append=FALSE)
                #}
            #}
'''

            # all other time steps
            if istepnow == nstepsday or flag_outfreq == 0 :
                icount = icount + 1                            # records position in output arrays
                tm[icount] = tme                            # time
                Svar[icount,:] = X[:]                            # state variables
                Dvar[icount,:] = Ystep[:]                        # auxiliary variables
        
    fluxyear3D[iyear][:][:]=fluxyear

Pgrow_initial = fluxyear3D[:,0,0]
Dremin_initial = fluxyear3D[:,7,3]
DOC_to_deep_initial = fluxyear3D[:,4,0] + fluxyear3D[:,5,0] + fluxyear3D[:,2,2] + fluxyear3D[:,3,2] + fluxyear3D[:,4,2] + fluxyear3D[:,5,3] + fluxyear3D[:,6,3] + fluxyear3D[:,8,3]

fig, axes = plt.subplots(1,1)
axes.plot(Pgrow_initial, label = "Pgrow")
axes.plot(Dremin_initial, label = "Dremin")
axes.plot(DOC_to_deep_initial, label = "DOC to deep")
axes.legend()
axes.set_xlabel("Time (years)")
axes.set_ylabel("Bio FLuxes (mol/ m^2/ year)")


"""
TIMING FOR SIMULATIONS

"""

# get start time

st = perf_counter()

"""
Set initial (pre-industrial) values for carbon start from equilibrium

"""

chi_pre = 280*10**(-6)  # pre-industrial mixing ratio
chi = [chi_pre]
chi_year = [chi_pre]
Ac = 2372*10**(-6)  # pre-industrial alkalinity
Acm = [0] # initialise alkalinity array for mixed layer (but shouldn't Acm[0] be the pre-iindsutrial alkalinity?)
Acd = [0] # initialise alkalinity array for mixed layer (but shouldn't Acd[0] be the pre-iindsutrial alkalinity?)
Acab = [0]

# start from equilibrium

[dic_eq, Heq, pHeq, K1m0, K2m0, Kom0] = equilibrium(Tref, Sref, chi[0], Pho, Sil, Ac, 8)

pHm = [pHeq]  # mixed layer pH at pre-indsutrial
Hm = [Heq]
Hd =[0]
Hab = [0]
pHd = [pHeq]  # deep layer pH at pre-industrial
pHab = [pHeq]
pH_all = pHeq
dic_m_pre = dic_eq  # mixed layer DIC at pre-industrial
dic_m = [dic_m_pre]
dic_d_pre = dic_eq  # interior DIC at pre-industrial
dic_d = [dic_d_pre]
dic_ab_pre = dic_eq
dic_ab = [dic_ab_pre]
DIC_to_DOC_m = [0]  # cumulative DOC anomaly in mixed layer
DIC_to_DOC_d = [0]  # cumulative DOC anomaly in deep ocean
DOC_flux_m = [0]  # DOC yearly flux in mixed layer
DOC_flux_d = [0]  # DOC yearly flux in deep ocean

K1m = [K1m0]
K2m = [K2m0]
Kom = [Kom0]

# carbon partitioning

CO2_m0 = dic_m[0]/(1 + (K1m[0]/Hm[0]) + (K1m[0]*K2m[0]/(Hm[0]**2)))
CO2_m = [CO2_m0]
HCO3_m0 = CO2_m[0]*K1m[0]/Hm[0]
HCO3_m = [HCO3_m0]
CO3_m0 = CO2_m[0]*(K1m[0]*K2m[0])/(Hm[0]**2)
CO3_m = [CO3_m0]
CO3_d = [0]
HCO3_d = [0]
CO3_ab = [0]
HCO3_ab = [0]


# initial carbon fluxes

Fa_pre = -rho_o*Kg*(CO2_m[0] - Kom[0]*P*chi[0])  # should be zero if starting from equilibrium
print("Fa_pre = ", Fa_pre)

Fd_pre = (dic_d[0]*rho_o*h_d - dic_m[0]*rho_o*h_d)*(math.exp(-Dt/tc1) - 1)  # should be zero if starting from equilbrium
print("Fd_pre = ", Fd_pre)

Fab_pre = (dic_ab[0]*rho_o*h_ab - dic_d[0]*rho_o*h_ab)*(math.exp(-Dt/tc2) - 1)  # should be zero if starting from equilbrium
print("Fab_pre = ", Fab_pre)


"""
Setting the cumulative emissions

"""

# emssions with constant rate of 20 Pg C/ year for 100 years
# emissions vector size controls how long the run us - here 1000 years

nyears = 800

ndays = nyears*365

Iem = np.ones((ndays+1))  # Initialise emissions array with ones
#Iem = np.zeros((ndays+1))
Iem_year = np.ones((nyears+1))  # Initialise yearly emissions array with ones
#Iem_year = np.zeros((ndays+1))
Rate_em = 20  # Emission rate in PgC/ year


print("Iem = " , Iem)   #Some print statements to check model is working
print("size of Iem = " , len(Iem))

# Choose how long emissions last for - here 100 years

emtime = 100

Iem[0:emtime*365] = Rate_em/365*(np.arange(1,emtime*365+1))



N = nyears - emtime
x = np.arange(0, (emtime + 1)*Rate_em, Rate_em, dtype=int)
Iem_year = np.pad(x, (0, N), mode="constant", constant_values=(0, Rate_em*emtime))


print(np.arange(1,emtime*365+1))
print(Iem[0:10])

print("Iem on last year with emissions = " , Iem[emtime*365-1])

# AFter the 100 years emissions cease and the cumulative emissions remain constant

Iem[emtime*365:len(Iem)] = Iem[emtime*365-1]*Iem[emtime*365:len(Iem)]
#Iem_year[emtime:len(Iem_year)] = Iem_year[emtime-1]*Iem_year[emtime:len(Iem)]

print("Iem after emissions cease = " , Iem[emtime*365:len(Iem)])

print("Iem in the transition when emissions cease = " , Iem[(emtime*365-6):(emtime*365+4)])

print("Iem_year = ", Iem_year )


# Initial values for the carbon and heat fluxes associated with emissions (should be zero in pre-industrial since there are no emissions)
Fa = [0]
Fd = [0]
Fab = [0]
Heat_d = [0]
Heat_ab = [0]
NTOA = [0]
N = [0]
DTa = [0]
DTm = [0]
DTm_year = [0]
DTd = [0]
DTd_year = [0]
DTab = [0]
DTab_year = [0]
delta_temp = [0]  # change in temperature each yaer
DR = [0]
Natm = [0]
Tm = [Tref]  # average mixed layer temperature per year
Td = [Tref]  # average deep layer temperature per year
Tab = [Tref]  # average abyssal layer temperature per year
t_vent1_phys = [tc1]
t_vent2_phys = [tc2]
t_vent_bio = [t_vent]

dic_m_year = [dic_m_pre]
dic_d_year = [dic_d_pre]
dic_ab_year = [dic_ab_pre]
delta_dic_bio_m = [0]
delta_dic_bio_d = [0]
delta_dic_bio_ab = [0]

# PHYTOPLANKTON
Pgrow = [fluxyear3D[19,0,0]]
Pgraz = [fluxyear3D[19,1,0]]
Pmort = [fluxyear3D[19,2,0]]
Pmort2 = [fluxyear3D[19,3,0]]
Pmix = [fluxyear3D[19,4,0]]
Pdilute = [fluxyear3D[19,5,0]]

# NITRATE
Nmix = [fluxyear3D[19,0,1]]
Zexc = [fluxyear3D[19,1,1]]
Dremin_src = [fluxyear3D[19,2,1]]
Pgrow_sink = [fluxyear3D[19,3,1]]
Nadd = [fluxyear3D[19,4,1]]
Ndilute = [fluxyear3D[19,5,1]]

# ZOOPLANKTON
Zgrow = [fluxyear3D[19,0,2]]
Zmort = [fluxyear3D[19,1,2]]
Zmort2 = [fluxyear3D[19,2,2]]
Zdilute = [fluxyear3D[19,3,2]]
Zmix = [fluxyear3D[19,4,2]]

# DETRITUS
Pmort_src = [fluxyear3D[19,0,3]]
Pmort2_src = [fluxyear3D[19,1,3]]
Zmort_src = [fluxyear3D[19,2,3]]
Zpel = [fluxyear3D[19,3,3]]
Dgraz = [fluxyear3D[19,4,3]]
Dmix = [fluxyear3D[19,5,3]]
Dsink = [fluxyear3D[19,6,3]]
Dremin = [fluxyear3D[19,7,3]]
Ddilute = [fluxyear3D[19,8,3]]

# CHECK SIGNS FOR DOC INVENTORIES IN MIXED LAYER AND DEEP OCEAN

#DOC_invent_m = [(fluxyear3D[19,0,0] + fluxyear3D[19,7,3] + fluxyear3D[19,4,0] + fluxyear3D[19,5,0] + fluxyear3D[19,2,2] + fluxyear3D[19,3,2] + fluxyear3D[19,4,2] + fluxyear3D[19,5,3] + fluxyear3D[19,6,3] + fluxyear3D[19,8,3])*CN_org/(rho_o*1000)]
doc_m_pre = (fluxyear3D[19,0,0]*CN_P + fluxyear3D[19,7,3]*CN_D)/(rho_o*1000)
doc_m = [doc_m_pre]
doc_sink_pre = -(fluxyear3D[19,4,0]*CN_P + fluxyear3D[19,5,0]*CN_P + fluxyear3D[19,2,2]*CN_Z + fluxyear3D[19,3,2]*CN_Z + fluxyear3D[19,4,2]*CN_Z + fluxyear3D[19,5,3]*CN_D + fluxyear3D[19,6,3]*CN_D + fluxyear3D[19,8,3]*CN_D)/(1000)*h_m*Ar_o   # Amount of DOC that goes below mixed layer (mol C) 
dic_bio_d_pre = doc_sink_pre*DRC/(rho_o*h_d*Ar_o)
dic_bio_d = [dic_bio_d_pre]
dic_bio_ab_pre = doc_sink_pre*ARC/(rho_o*h_ab*Ar_o)
dic_bio_ab = [dic_bio_ab_pre]
doc_below_pre = (doc_sink_pre - dic_bio_d_pre*rho_o*h_d*Ar_o - dic_bio_ab_pre*rho_o*h_ab*Ar_o)/((h_ab + h_d)*Ar_o*rho_o)
doc_below = [doc_below_pre]
delta_DOC_total = [0]

xtra_DOC_m = [0]
xtra_DOC_below = [0]
xtra_DIC_d = [0]
xtra_DIC_ab = [0]
doc_deep = [0]

# CHECKING DOC BUDGET CLOSES IN PRE-INDUSTRIAL

I_doc_below = doc_below_pre*(h_d + h_ab)*Ar_o*rho_o   # mol C
I_dic_bio_d = dic_bio_d_pre*(h_d*Ar_o*rho_o)
I_dic_bio_ab = dic_bio_ab_pre*(h_ab*Ar_o*rho_o)

dic_below = I_doc_below + I_dic_bio_d + I_dic_bio_ab

print("doc sink pre = ", doc_sink_pre)
print("dic belwo mixed layer = ", dic_below)

"""
FLUX ANOMALIES

"""

# PHYTOPLANKTON
Delta_Pgrow = [0]
Delta_Pgraz = [0]
Delta_Pmort = [0]
Delta_Pmort2 = [0]
Delta_Pmix = [0]
Delta_Pdilute = [0]

# NITRATE
Delta_Nmix = [0]
Delta_Zexc = [0]
Delta_Dremin_src = [0]
Delta_Pgrow_sink = [0]
Delta_Nadd = [0]
Delta_Ndilute = [0]

# ZOOPLANKTON
Delta_Zgrow = [0]
Delta_Zmort = [0]
Delta_Zmort2 = [0]
Delta_Zdilute = [0]
Delta_Zmix = [0]

# DETRITUS
Delta_Pmort_src = [0]
Delta_Pmort2_src = [0]
Delta_Zmort_src = [0]
Delta_Zpel = [0]
Delta_Dgraz = [0]
Delta_Dmix = [0]
Delta_Dsink = [0]
Delta_Dremin = [0]
Delta_Ddilute = [0]

"""
YEARLY FLUX CHANGES

"""
dPgrow = [0]
dDremin = [0]
dZgrow =[0]
dNmix =[0]
    
dZmort2 = [0]
dPmix = [0]
dZmix = [0]
dDmix = [0]
dDsink = [0]
dPdilute = [0]
dZdilute = [0]
dDdilute = [0]


"""
CUMULATIVE FLUXES

"""

# PHYTOPLANKTON
Pgrow_sum = [0]
Pgraz_sum = [0]
Pmort_sum = [0]
Pmort2_sum = [0]
Pmix_sum = [0]
Pdilute_sum = [0]

# NITRATE
Nmix_sum = [0]
Zexc_sum = [0]
Dremin_src_sum = [0]
Pgrow_sink_sum = [0]
Nadd_sum = [0]
Ndilute_sum = [0]

# ZOOPLANKTON
Zgrow_sum = [0]
Zmort_sum = [0]
Zmort2_sum = [0]
Zdilute_sum = [0]
Zmix_sum = [0]

# DETRITUS
Pmort_src_sum = [0]
Pmort2_src_sum = [0]
Zmort_src_sum = [0]
Zpel_sum = [0]
Dgraz_sum = [0]
Dmix_sum = [0]
Dsink_sum = [0]
Dremin_sum = [0]
Ddilute_sum = [0]

Fa_year = [0]
Fd_year = [0]

"""
BOX MODEL SOLVER

"""

# -------------------------- #
# Basic setup for main run   #
# -------------------------- #

ndays = int(nyears*365)                          # no. of days in simulation
nsteps = int(floor(ndays/delta_t+delta_t/1000))                   # no. of time steps in simulation
nstepsday = int(floor(1/delta_t+delta_t/1000))                    # no. of time steps per day
nwrite = int(flag_outfreq*(ndays+1)+(1-flag_outfreq)*(nsteps+1))

#tm <- array(dim=c(nwrite), rep(0.0, times=nwrite))    # array to store time for output
tm = np.zeros((nwrite,), dtype=int)

#flux <- matrix(0,nrow=nfluxmax,ncol=nSvar)            # matrix to hold update terms for state variables
flux = np.zeros((nfluxmax,nSvar))

#fluxyear <- matrix(0,nrow=nfluxmax,ncol=nSvar)        # matrix to sum fluxes over year
fluxyear = np.zeros((nfluxmax,nSvar))
fluxstep = np.zeros((nfluxmax,nSvar))

#Svar <- matrix(nrow=nwrite, ncol=nSvar)               # matrix to store values of state variables
Svar = np.zeros((nwrite,nSvar))

#Dvar <- matrix(nrow=nwrite, ncol=nDvar)               # matrix to store values of auxiliary variables
Dvar = np.zeros((nwrite,nDvar))

#Svar[1,] <- X                                    # store initial values of state variables (position 1 in array corresponds to t=0)
Svar[0][:] = X

tme = 0       # time=0: start of simultion
tm[1] = tme
 
#Xout <- signif(X,digits=4)     # truncate X to 4 significant figures, ready for writing to output file
Xout = X
if flag_outtype != 0 :
    #write(c(tme,Xout),file="out_statevars.txt",ncolumns=nSvar+1,sep=",",append=FALSE)  # write inital values of state vars to file
    #print(tme, Xout)
    pass

#icount = 1      # records position in array for writing output
icount = 0   # For numpy arrays starts from index 0
istep = 0



#sys.exit()


##############################    
# -------------------------- #
# Time loop starts here      #
# -------------------------- #
##############################

print("========START TIME LOOP===========")


for iyear in range(0, nyears):  # trying to restructure loop so that we first loop of the years - this is because the biological model and physical model communicate with each other after each year


    fluxyear[:][:] = 0.
    
    SST_day = SST_sine(Tm[iyear])
    #SST_day = SST_sine(Tm[0])   # No temperature coupling with biology

    #update ventilation timescale for biology
    t_vent_bio.append(t_vent + 100*t_vent*((DTm_year[iyear] - DTd_year[iyear])/Tref))
    #t_vent_bio.append(t_vent_bio[0]/(1 - DTm_year[iyear]/Tref*30))
    #t_vent_bio.append(t_vent_bio[0])
    
    DTm_sum = 0   # initialise DTm_sum value for summing up over the year
    DTd_sum = 0
    DTab_sum = 0

    
    if iyear > 0:
        dic_m[iyear*365] = dic_m_year[iyear]
        dic_d[iyear*365] = dic_d_year[iyear]
        dic_ab[iyear*365] = dic_ab_year[iyear]
    
    else:
        dic_m[iyear*365] = dic_m[0]
        dic_d[iyear*365] = dic_d[0]
        dic_ab[iyear*365] = dic_ab[0]
    
    #print("DTm_sum at start of year = ", DTm_sum)
    
    print("year = ", iyear)
    

    for kday in range(1, 366):  # trying to loop over all the days of the year and get the same resutls for the physical model
                                # somethign is going wrong her with the indexing
        
            chi.append(chi[iyear*365 + kday -1] - Dt*(Ar_o)*Fa[iyear*365 + kday -1]/Na + (Iem[iyear*365 + kday] - Iem[iyear*365 + kday -1])*10**15/(12.01*Na))
    
            #update ventialtion timescale
            t_vent1_phys.append(tc1 + 100*tc1*((DTm[iyear*365 + kday -1] - DTd[iyear*365 + kday -1])/Tref))
            t_vent2_phys.append(tc2 + 50*tc2*(((h_m*(DTm[iyear*365 + kday -1]) + h_d*(DTd[iyear*365 + kday -1]))/(h_m + h_d)) - (DTab[iyear*365 + kday -1]))/Tref) 
            #t_vent1_phys.append(t_vent1_phys[0]/(1 - DTm[iyear*365 + kday -1]/Tref*30))
            #t_vent2_phys.append(t_vent2_phys[0]/(1 - DTm[iyear*365 + kday -1]/Tref*30))
            #t_vent1_phys.append(t_vent1_phys[0])
            #t_vent2_phys.append(t_vent2_phys[0])
            
            # heat budget
            DR.append(alpha*(math.log(chi[iyear*365 + kday]/chi[0])))
            NTOA.append((DR[iyear*365 + kday] - lamda*DTa[iyear*365 + kday -1]))
            Natm.append(CC*(DTa[iyear*365 + kday -1] - DTm[iyear*365 + kday -1]))
            N.append(NTOA[iyear*365 + kday] + Natm[iyear*365 + kday])
            DTm.append(DTm[iyear*365 + kday -1] + (Dt/(Ar_o*h_m*rho_o*cp_o))*(-Heat_d[iyear*365 + kday -1]) + (Ar_o*Dt/(Ar_o*h_m*rho_o*cp_o))*N[iyear*365 + kday])
            DTd.append(DTd[iyear*365 + kday -1] + (Dt/(Ar_o*(h_d)*rho_o*cp_o))*(Heat_d[iyear*365 + kday -1]) + (Dt/(Ar_o*h_d*rho_o*cp_o))*(-Heat_ab[iyear*365 + kday -1]))
            DTab.append(DTab[iyear*365 + kday -1] + (Dt/(Ar_o*h_ab*rho_o*cp_o))*(Heat_ab[iyear*365 + kday -1]))
            DTa.append((CC*DTm[iyear*365 + kday] + rho_a*cp_a*h_a*DTa[iyear*365 + kday -1]/Dt)/(rho_a*cp_a*h_a/Dt + CC))
            Heat_d.append(((Ar_o*h_d*rho_o*cp_o)/Dt)*(DTd[iyear*365 + kday] - DTm[iyear*365 + kday])*(math.exp(-Dt/t_vent1_phys[iyear*365 + kday]) - 1))
            Heat_ab.append(((Ar_o*h_ab*rho_o*cp_o)/Dt)*(DTab[iyear*365 + kday] - DTd[iyear*365 + kday])*(math.exp(-Dt/t_vent2_phys[iyear*365 + kday]) - 1))
            
            
            # carbon budget
            dic_m.append(dic_m[iyear*365 + kday -1] + Dt*(1/(rho_o*h_m))*Fa[iyear*365 + kday -1] - Dt*(1/(rho_o*h_m))*Fd[iyear*365 + kday -1])
            dic_d.append(dic_d[iyear*365 + kday -1] + Dt*(1/(rho_o*h_d))*Fd[iyear*365 + kday -1] - Dt*(1/(rho_o*h_d))*Fab[iyear*365 + kday -1])
            dic_ab.append(dic_ab[iyear*365 + kday -1] + Dt*(1/(rho_o*h_ab))*Fab[iyear*365 + kday -1])
            
            
            # assumes alkalinity does not change
            # choose or not to ignore the effect of warming to Ko, K1 and K2, if you want to
            # to consider this effect, instead of Tref use Tref+DTm[k] in the following line
            
            [co2new, Hnew, pHnew, K1, K2, Ko] = CO2_follows(Tref, Sref, dic_m[iyear*365 + kday], Pho, Sil, Ac, pHm[iyear*365 + kday -1])
            CO2_m.append(co2new)
            Hm.append(Hnew)
            pHm.append(pHnew)
            HCO3_m.append(CO2_m[iyear*365 + kday]*K1/Hm[iyear*365 + kday])
            CO3_m.append(CO2_m[iyear*365 + kday]*K1*K2/(Hm[iyear*365 + kday]**2))
            Kom.append(Ko)
            K1m.append(K1)
            K2m.append(K2)
            Acm.append(HCO3_m[iyear*365 + kday] + 2*CO3_m[iyear*365 + kday])
            
            
            [co2new, Hnew, pHnew, K1, K2, Ko] = CO2_follows(Tref, Sref, dic_d[iyear*365 + kday], Pho, Sil, Ac, pHm[iyear*365 + kday -1])
            pHd.append(pHnew)
            Hd.append(Hnew)
            HCO3_d.append(co2new*K1/Hd[iyear*365 + kday])
            CO3_d.append(co2new*(K1*K2)/(Hd[iyear*365 + kday]**2))
            Acd.append(HCO3_d[iyear*365 + kday] + 2*CO3_d[iyear*365 + kday])
            
            [co2new, Hnew, pHnew, K1, K2, Ko] = CO2_follows(Tref, Sref, dic_ab[iyear*365 + kday], Pho, Sil, Ac, pHm[iyear*365 + kday -1])
            pHab.append(pHnew)
            Hab.append(Hnew)
            HCO3_ab.append(co2new*K1/Hab[iyear*365 + kday])
            CO3_ab.append(co2new*(K1*K2)/(Hab[iyear*365 + kday]**2))
            Acab.append(HCO3_ab[iyear*365 + kday] + 2*CO3_ab[iyear*365 + kday])
            
            Fa.append((-rho_o*Kg*(CO2_m[iyear*365 + kday] - Kom[iyear*365 + kday]*P*chi[iyear*365 + kday])))
            Fd.append(((dic_d[iyear*365 + kday]*rho_o*h_d - dic_m[iyear*365 + kday]*rho_o*h_d)*(math.exp(-Dt/t_vent1_phys[iyear*365 + kday]) - 1)/Dt))
            Fab.append(((dic_ab[iyear*365 + kday]*rho_o*h_ab - dic_d[iyear*365 + kday]*rho_o*h_ab)*(math.exp(-Dt/t_vent2_phys[iyear*365 + kday]) - 1)/Dt))
            DTm_sum = DTm_sum + DTm[iyear*365 + kday]
            DTd_sum = DTd_sum + DTd[iyear*365 + kday]
            DTab_sum = DTab_sum + DTab[iyear*365 + kday]
            

            
            
    for iday in range(1,366):
        daynow = iday                            # global variable to pass to functions

        for istepnow in range(1,nstepsday+1):       # loop over time steps within one day (1,nstepsday)
            
            istep = istep + 1
            istepplus = istep + 1               # (Dvar (auxiliary variables) stored at beginning of each time step)
                        
            #fluxstep <- rep(0.0,times=nfluxmax*nSvar)  # matrix to sum up fluxes over time step
            fluxstep[:][:] = 0.0
                        
            #Ystep <- rep(0.0,times=nDvar)              # matrix to store average value of Y for each time step
            Ystep[:] = 0.

            # -------------------------- #
            # Numerical integration      #
            # -------------------------- # 

            flux1 = get_flux(X, daynow, SST_day, t_vent_bio[iyear+1])*delta_t        # 1st iteration
            #flux1 = get_flux(X, daynow, SST, t_vent_bio[iyear+1])*delta_t        # 1st iteration

            #deltaX1 = colSums(flux1)             # sum of flux terms, for each state variable
            deltaX1 = np.sum(flux1, axis=0)

            if flag_integ == 0:                  # Euler
                fluxstep = fluxstep + flux1
                Ystep = Ystep + Y
            else:                                # Runge Kutta 4
                fluxstep = fluxstep + flux1/6
                Ystep = Ystep + Y/6

            if flag_integ == 1:                   # Runge Kutta only

                flux2 = get_flux(X + deltaX1/2, daynow, SST_day, t_vent_bio[iyear+1])*delta_t       # 2nd iteration
                #flux2 = get_flux(X + deltaX1/2, daynow, SST, t_vent_bio[iyear+1])*delta_t       # 2nd iteration
                deltaX2 = np.sum(flux2, axis=0)
                fluxstep = fluxstep + flux2/3
                Ystep = Ystep + Y/3

                flux3 = get_flux(X + deltaX2/2, daynow, SST_day, t_vent_bio[iyear+1])*delta_t       # 3rd iteration
                #flux3 = get_flux(X + deltaX2/2, daynow, SST, t_vent_bio[iyear+1])*delta_t       # 3rd iteration
                #deltaX3 = colSums(flux3)
                deltaX3 = np.sum(flux3, axis=0)
                fluxstep = fluxstep + flux3/3
                Ystep = Ystep + Y/3
  
                flux4 = get_flux(X + deltaX3, daynow, SST_day, t_vent_bio[iyear+1])*delta_t         # 4th iteration
                #flux4 = get_flux(X + deltaX3, daynow, SST, t_vent_bio[iyear+1])*delta_t         # 4th iteration
                #deltaX4 = colSums(flux4)
                deltaX4 = np.sum(flux4, axis=0)
                fluxstep = fluxstep + flux4/6
                Ystep = Ystep + Y/6

            # sum iterations
            if flag_integ == 0:               # Euler
                deltaX = deltaX1
            else:       #Euler method  
                deltaX = (deltaX1 + 2*deltaX2 + 2*deltaX3 + deltaX4)/6    # Runge Kutta
            #}
    
            X = X + deltaX    # update state varialbes

            fluxyear = fluxyear + fluxstep

            tme = tme + delta_t    # update time

            # -------------------------- #
            # Write to output files      #
            # -------------------------- # 

            # 1st time step awkward: state vars already output for t=0; output fluxes and auxiliary variables for t=0 as calculated for first time step

            if istep == 1:
                Dvar[icount,:] = Ystep[:]
                
                '''
                if flag_outtype == 2 or (flag_outtype==1 and nyears==1):                  # only needed if whole simulation from t=0 is to be written to files
                    #Dvarwrite <- signif(Dvar[icount,],digits=4)        # truncated to 4 significant figures: auxiliary variables
                    Dvarwrite = Dvar[icount,]
                    #fluxwrite <- signif(fluxstep,digits=4)             # truncated to 4 significant figures: fluxes
                    fluxwrite = fluxstep
                    tmenow = 0.0     # actually, it is time 1 time step, but for the purposes of continuity in output files, set to zero
                    write(c(tmenow,Dvarwrite),file="out_aux.txt",ncolumns=nDvar+1,sep=",",append=FALSE)
                    for (iw in seq(1,nSvar)) {
                        if (iw==1) {
                            fluxwritecat <- fluxwrite[,1]
                        } else {
                            fluxwritecat <- c(fluxwritecat,fluxwrite[,iw])
                        }
                    }
                    write(c(tmenow,fluxwritecat),file="out_fluxes.txt",ncolumns=nSvar*nfluxmax+1,sep=",",append=FALSE)
                #}
            #}
'''

            # all other time steps
            if istepnow == nstepsday or flag_outfreq == 0 :
                icount = icount+1                            # records position in output arrays
                tm[icount] = tme                             # time
                Svar[icount,:] = X[:]                            # state variables
                Dvar[icount,:] = Ystep[:]                        # auxiliary variables
    
    #fluxyear3D[iyear][:][:]=fluxyear
    
    #print("DTm_sum = ", DTm_sum)
    
    # calculating average change in mixed layer temperature over the year
    
    DTm_average = DTm_sum/365
    DTm_year.append(DTm_average)
    DTd_average = DTd_sum/365
    DTd_year.append(DTd_average)
    DTab_average = DTab_sum/365
    DTab_year.append(DTab_average)
    Tm.append(Tref + DTm_average)
    Td.append(Tref + DTd_average)
    Td.append(Tref + DTab_average)
    delta_temp.append(Tm[iyear + 1] - Tm[iyear])
    
    # PHYTOPLANKTON
    
    Pgrow.append((fluxyear[0][0]))
    Pgraz.append((fluxyear[1][0]))
    Pmort.append((fluxyear[2][0]))
    Pmort2.append((fluxyear[3][0]))
    Pmix.append((fluxyear[4][0]))
    Pdilute.append((fluxyear[5][0]))
    
    # NITRATE
    
    Nmix.append((fluxyear[0][1]))
    Zexc.append((fluxyear[1][1]))
    Dremin_src.append((fluxyear[2][1]))
    Pgrow_sink.append((fluxyear[3][1]))
    Nadd.append((fluxyear[4][1]))
    Ndilute.append((fluxyear[5][1]))
    
    # ZOOPLANKTON
    
    Zgrow.append((fluxyear[0][2]))
    Zmort.append((fluxyear[1][2]))
    Zmort2.append((fluxyear[2][2]))
    Zdilute.append((fluxyear[3][2]))
    Zmix.append((fluxyear[4][2]))
    
    # DETRITUS
    
    Pmort_src.append((fluxyear[0][3]))
    Pmort2_src.append((fluxyear[1][3]))
    Zmort_src.append((fluxyear[2][3]))
    Zpel.append((fluxyear[3][3]))
    Dgraz.append((fluxyear[4][3]))
    Dmix.append((fluxyear[5][3]))
    Dsink.append((fluxyear[6][3]))
    Dremin.append((fluxyear[7][3]))
    Ddilute.append((fluxyear[8][3]))
    
    
    Delta_Pgrow.append((fluxyear[0][0] - fluxyear3D[19,0,0])*CN_P/(1000*rho_o))
    Delta_Dremin.append((fluxyear[7][3] - fluxyear3D[19,7,3])*CN_D/(1000*rho_o))
    Delta_Zgrow.append((fluxyear[0][2] - fluxyear3D[19,0,2])*CN_Z/(1000*rho_o))
    Delta_Nmix.append((fluxyear[0][1] - fluxyear3D[19,0,1])/(1000*rho_o))     # mol N kg^-1
    
    Delta_Zmort2.append((fluxyear[2][2] - fluxyear3D[19,2,2])*CN_Z/(1000*rho_o))
    Delta_Pmix.append((fluxyear[4][0] - fluxyear3D[19,4,0])*CN_P/(1000*rho_o))
    Delta_Zmix.append((fluxyear[4][2] - fluxyear3D[19,4,2])*CN_Z/(1000*rho_o))
    Delta_Dmix.append((fluxyear[5][3] - fluxyear3D[19,5,3])*CN_D/(1000*rho_o))
    Delta_Dsink.append((fluxyear[6][3] - fluxyear3D[19,6,3])*CN_D/(1000*rho_o))
    Delta_Pdilute.append((fluxyear[5][0] - fluxyear3D[19,5,0])*CN_P/(1000*rho_o))
    Delta_Zdilute.append((fluxyear[3][2] - fluxyear3D[19,3,2])*CN_Z/(1000*rho_o))
    Delta_Ddilute.append((fluxyear[8][3] - fluxyear3D[19,8,3])*CN_D/(1000*rho_o))
    
    dPgrow.append(Delta_Pgrow[iyear] - Delta_Pgrow[iyear-1])
    dDremin.append(Delta_Dremin[iyear] - Delta_Dremin[iyear-1])
    dZgrow.append(Delta_Zgrow[iyear] - Delta_Zgrow[iyear-1])
    dNmix.append(Delta_Nmix[iyear] - Delta_Nmix[iyear-1])
    
    dZmort2.append(Delta_Zmort2[iyear] - Delta_Zmort2[iyear-1])
    dPmix.append(Delta_Pmix[iyear] - Delta_Pmix[iyear-1])
    dZmix.append(Delta_Zmix[iyear] - Delta_Zmix[iyear-1])
    dDmix.append(Delta_Dmix[iyear] - Delta_Dmix[iyear-1])
    dDsink.append(Delta_Dsink[iyear] - Delta_Dsink[iyear-1])
    dPdilute.append(Delta_Pdilute[iyear] - Delta_Pdilute[iyear-1])
    dZdilute.append(Delta_Zdilute[iyear] - Delta_Zdilute[iyear-1])
    dDdilute.append(Delta_Ddilute[iyear] - Delta_Ddilute[iyear-1])
    
    """
    DOC_to_deep = (Delta_Zmort2[iyear] + Delta_Pmix[iyear] + Delta_Zmix[iyear] + Delta_Dmix[iyear] + Delta_Dsink[iyear] + Delta_Pdilute[iyear] + Delta_Zdilute[iyear] + Delta_Ddilute[iyear])*(h_m)   # DOC THAT SINKS TO DEEP OCEAN FROM MIXED LAYER AS AN INVENTORY 
    
    Pgrow_sum.append(Pgrow_sum[iyear-1] + Delta_Pgrow[iyear])
    Zgrow_sum.append(Zgrow_sum[iyear-1] + Delta_Zgrow[iyear])
    Dremin_sum.append(Dremin_sum[iyear-1] + Delta_Dremin[iyear])
    Nmix_sum.append(Nmix_sum[iyear-1] + Delta_Nmix[iyear])
    
    Pmix_sum.append(Pmix_sum[iyear-1] + Delta_Pmix[iyear])
    Zmix_sum.append(Zmix_sum[iyear-1] + Delta_Zmix[iyear])
    Dmix_sum.append(Dmix_sum[iyear-1] + Delta_Dmix[iyear])
    Dsink_sum.append(Dsink_sum[iyear-1] + Delta_Dsink[iyear])

    total_DOC = Delta_Pgrow[iyear] + Delta_Dremin[iyear] + (DOC_to_deep)*DRC/(h_d)
    extra_DOC_m = Delta_Pgrow[iyear] + Delta_Dremin[iyear] + DOC_to_deep/(h_m)
    extra_DOC_d = -DOC_to_deep*(1-DRC)/(h_d)
    extra_DIC_d = -DOC_to_deep*DRC/(h_d)
    """
    
    
    DOC_to_deep = (dZmort2[iyear] + dPmix[iyear] + dZmix[iyear] + dDmix[iyear] + dDsink[iyear] + dPdilute[iyear] + dZdilute[iyear] + dDdilute[iyear])*(h_m*Ar_o*rho_o)   # DOC THAT SINKS TO DEEP OCEAN FROM MIXED LAYER AS AN INVENTORY (mol C)  
    
    Pgrow_sum.append(Pgrow_sum[iyear-1] + Delta_Pgrow[iyear])
    Zgrow_sum.append(Zgrow_sum[iyear-1] + Delta_Zgrow[iyear])
    Dremin_sum.append(Dremin_sum[iyear-1] + Delta_Dremin[iyear])
    Nmix_sum.append(Nmix_sum[iyear-1] + Delta_Nmix[iyear])
    
    Pmix_sum.append(Pmix_sum[iyear-1] + Delta_Pmix[iyear])
    Zmix_sum.append(Zmix_sum[iyear-1] + Delta_Zmix[iyear])
    Dmix_sum.append(Dmix_sum[iyear-1] + Delta_Dmix[iyear])
    Dsink_sum.append(Dsink_sum[iyear-1] + Delta_Dsink[iyear])

    #total_DOC = Delta_Pgrow[iyear] + Delta_Dremin[iyear] + (DOC_to_deep)*DRC/(h_d*Ar_o*rho_o)
    extra_DOC_m = dPgrow[iyear] + dDremin[iyear] + DOC_to_deep/(h_m*Ar_o*rho_o)
    #extra_DIC_d = -DOC_to_deep*DRC/(h_d*Ar_o*rho_o)
    #extra_DIC_ab = -DOC_to_deep*ARC/(h_ab*Ar_o*rho_o)
    extra_DIC_d = (-DOC_to_deep + (doc_below[iyear] - doc_below[0])*(h_d + h_ab)*Ar_o*rho_o)*DRC/(h_d*Ar_o*rho_o)
    extra_DIC_ab = (-DOC_to_deep + (doc_below[iyear] - doc_below[0])*(h_d + h_ab)*Ar_o*rho_o)*ARC/(h_ab*Ar_o*rho_o)
    #extra_DOC_below = (-DOC_to_deep - (-DOC_to_deep*DRC) - (-DOC_to_deep*ARC))/((h_d + h_ab)*Ar_o*rho_o)
    #extra_DOC_d = -DOC_to_deep*(1-DRC)/(h_d*Ar_o*rho_o)
    #extra_DOC_d = (-DOC_to_deep - extra_DIC_d*(h_d*Ar_o*rho_o))/(h_d*Ar_o*rho_o)
    
    #total_DOC_below = extra_DIC_d*(h_d*Ar_o*rho_o) + extra_DIC_ab*(h_ab*Ar_o*rho_o) + extra_DOC_below*((h_ab + h_d)*Ar_o*rho_o)
    
    #extra_DOC_m = 0    # no carbon coupling between the two models
    #extra_DOC_d = 0
    #extra_DIC_d = 0
    
    #print("DOC to deep = ", DOC_to_deep)
    #print("total DOC belwo = ", total_DOC_below)
    #print("Delta Pgrow = ", Delta_Pgrow)
    #print("DOC to deep = ", DOC_to_deep)
    #print("Delta_Dremin = ", Delta_Dremin)
    #print("extra DOC m = ", extra_DOC_m)
    #print("extra DOC below = ", extra_DOC_below)
    #print("extra DIC d = ", extra_DIC_d)
    #print("extra DIC ab = ", extra_DIC_ab)
    #print("dic_m = ", dic_m[(iyear + 1)*365])
    #print("dic_d = ", dic_d[(iyear + 1)*365])
    
    chi_year.append(chi[(iyear + 1)*365])
    
    dic_m_year.append(dic_m[(iyear + 1)*365] - (dPgrow[iyear] + dDremin[iyear]))
    #dic_m_year.append(dic_m[(iyear + 1)*365] + extra_DOC_m)
    dic_d_year.append(dic_d[(iyear + 1)*365] + extra_DIC_d)
    dic_ab_year.append(dic_ab[(iyear + 1)*365] + extra_DIC_ab)
    
    delta_dic_bio_m.append(delta_dic_bio_m[iyear] - (dPgrow[iyear] + dDremin[iyear]))  # ANOMALY
    delta_dic_bio_d.append(delta_dic_bio_d[iyear] + extra_DIC_d)  # ANOMALY
    delta_dic_bio_ab.append(delta_dic_bio_ab[iyear] + extra_DIC_ab)
    
    
    doc_m.append(doc_m[iyear] + extra_DOC_m)   # Total inventory (NOT ANOMALY) of DOC in the mixed layer
    doc_below.append((-DOC_to_deep + doc_below[iyear]*(h_d + h_ab)*Ar_o*rho_o - ((-DOC_to_deep + (doc_below[iyear] - doc_below[0])*(h_d + h_ab)*Ar_o*rho_o)*DRC) - ((-DOC_to_deep + (doc_below[iyear] - doc_below[0])*(h_d + h_ab)*Ar_o*rho_o)*ARC))/((h_d + h_ab)*Ar_o*rho_o))   # Total inventory (NOT ANOMALY) of DOC in the deep ocean
    dic_bio_d.append(dic_bio_d[iyear] + extra_DIC_d)  # Total inventory (NOT ANOMALY) of DIC from biology in deep ocean
    dic_bio_ab.append(dic_bio_ab[iyear] + extra_DIC_ab)
    
    
    
    #print("dic_m_year = ", dic_m_year)
    #print("dic_d_year = ", dic_d_year)
    
    #print("Tm = ", Tm[iyear+1])
    
    #print("flux year = ", fluxyear)
    
    

print("END OF LOOP")


"""
Diagnostics

"""

# estimate carbon inventories
"""
dic_m = np.array(dic_m)
dic_d = np.array(dic_d)

chi = np.array(chi)

Ia = Na*chi*12.01/10**15
Im = dic_m*h_m*Ar_o*rho_o*12.01*10**(-15)
Id = dic_d*h_d*Ar_o*rho_o*12.01*10**(-15)
Iocean = Im + Id


Iocean_0 = np.ones((ndays+1))
Ia_0 = np.ones((ndays+1))

Iocean_0 = Iocean_0*Iocean[0]
print(Iocean_0)

Ia_0 = Ia_0*Ia[0]
print(Ia_0)

Delta_Iocean = Iocean - Iocean_0
Delta_Ia = Ia - Ia_0



x1 = np.linspace(0., ndays/365, num=ndays+1)
x2 = np.linspace(0., ndays/365 - 1, num=ndays-1)

fig, axes = plt.subplots(1,1)

axes.plot(x1, Delta_Iocean + Delta_Ia, label = "\u0394Iocean + \u0394Ia")
axes.plot(x1, Iem, label = "Iem")
axes.plot(x1, Delta_Iocean, label = "\u0394IIocean")
axes.plot(x1, Delta_Ia, label = "\u0394IIa")
axes.legend()
axes.xaxis.set_ticks(np.linspace(0, ndays/365, 11))
axes.set_xlabel("Time (years)")
axes.set_ylabel("Carbon Inventory (PgC)")
fig.savefig("carbon_budget.png")

fig, axes = plt.subplots(1,1)

axes.plot(x1, pHm, label = "pH_ml")
axes.plot(x1, pHd, label = "pH_deep")
axes.legend()

DTa = np.array(DTa)
DR = np.array(DR)

fig, axes = plt.subplots(1,1)

axes.plot(x1, DTa/DR)
axes.xaxis.set_ticks(np.linspace(0, ndays/365, 11))
axes.set_xlabel("Time (years)")
axes.set_ylabel("thermal response")


fig, axes = plt.subplots(1,1)

axes.plot(x1, 1000*DR/Iem)
axes.xaxis.set_ticks(np.linspace(0, ndays/365, 11))
axes.set_xlabel("Time (years)")
axes.set_ylabel("carbon response")

fig, axes = plt.subplots(1,1)

axes.plot(x1, 1000*DTa/Iem)
axes.xaxis.set_ticks(np.linspace(0, ndays/365, 11))
axes.set_xlabel("Time (years)")
axes.set_ylabel("TCRE")

fig, axes = plt.subplots(1,1)

axes.plot(x1, DTm)

"""

### TESTS ####
"""
print("dic_m_year = ", dic_m_year)
print("dic_d_year = ", dic_d_year)
print("dic ab year = ", dic_ab_year)
print("doc m = ", doc_m)
print("doc belwo = ", doc_below)
print("chi = ", chi_year)
"""

# estimate carbon inventories on a yearly basis

dic_m_year = np.array(dic_m_year)
dic_d_year = np.array(dic_d_year)
dic_ab_year = np.array(dic_ab_year)
#DIC_to_DOC_m = np.array(DIC_to_DOC_m)
#DIC_to_DOC_d = np.array(DIC_to_DOC_d)
Delta_Pgrow = np.array(Delta_Pgrow)
Delta_Dremin = np.array(Delta_Dremin)
delta_dic_bio_m = np.array(delta_dic_bio_m)    #ANOMALY
delta_dic_bio_d = np.array(delta_dic_bio_d)    #ANOMALY
delta_dic_bio_ab = np.array(delta_dic_bio_ab)
doc_m = np.array(doc_m)
doc_below = np.array(doc_below)
dic_bio_d = np.array(dic_bio_d)
dic_bio_ab = np.array(dic_bio_ab)
t_vent1_phys = np.array(t_vent1_phys)
t_vent2_phys = np.array(t_vent2_phys)

t_vent1_phys = t_vent1_phys/(60*60*24*365)
t_vent2_phys = t_vent2_phys/(60*60*24*365)

chi_year = np.array(chi_year)

Ia_year = Na*chi_year*12.01/10**15
Im_DIC = dic_m_year*h_m*Ar_o*rho_o*12.01*10**(-15)      # total DIC in mixed layer after all transfers
Id_DIC = dic_d_year*h_d*Ar_o*rho_o*12.01*10**(-15)      # total DIC in thermocline after all transfers
Iab_DIC = dic_ab_year*h_ab*Ar_o*rho_o*12.01*10**(-15)   # total DIC in abyss after all transfers
Delta_Im_DIC_bio = delta_dic_bio_m*h_m*Ar_o*rho_o*12.01*10**(-15)      # DIC from bio in mixed layer ANOMALY
Delta_Id_DIC_bio = delta_dic_bio_d*h_d*Ar_o*rho_o*12.01*10**(-15)      # DIC from bio in thermocline ANOMALY
Delta_Iab_DIC_bio = delta_dic_bio_ab*h_ab*Ar_o*rho_o*12.01*10**(-15)   # DIC from bio in abyss ANOMALY

I_DIC = Im_DIC + Id_DIC + Iab_DIC

# calculating the TOTAL DOC carbon inventory (NOT anomaly)

Im_DOC = (doc_m)*h_m*Ar_o*rho_o*12.01*10**(-15)
Ibelow_DOC = (doc_below)*(h_d + h_ab)*Ar_o*rho_o*12.01*10**(-15)

I_DOC = Im_DOC + Ibelow_DOC

# creating array for the yearly change in temperature

delta_temp = np.array(delta_temp)


Ia_0_year = np.ones((nyears+1))
Im_DIC_phys_0 = np.ones((nyears+1))
Id_DIC_phys_0 = np.ones((nyears+1))
Iocean_0_year = np.ones((nyears+1))
Im_DIC_0 = np.ones((nyears+1))
Id_DIC_0 = np.ones((nyears+1))
Iab_DIC_0 = np.ones((nyears+1))
I_DIC_0 = np.ones((nyears+1))

Im_DOC_0 = np.ones((nyears+1))
Ibelow_DOC_0 = np.ones((nyears+1))
I_DOC_0 = np.ones((nyears+1))

#print(Iocean_0_year)

Ia_0_year = Ia_0_year*Ia_year[0]
#print(Ia_0_year)


Im_DIC_0 = Im_DIC_0*Im_DIC[0]
Id_DIC_0 = Id_DIC_0*Id_DIC[0]
Iab_DIC_0 = Iab_DIC_0*Iab_DIC[0]
I_DIC_0 = I_DIC_0*I_DIC[0]

Im_DOC_0 = Im_DOC_0*Im_DOC[0]
Ibelow_DOC_0 = Ibelow_DOC_0*Ibelow_DOC[0]
I_DOC_0 = I_DOC_0*I_DOC[0]


Delta_Ia_year = Ia_year - Ia_0_year

Delta_Im_DIC = Im_DIC - Im_DIC_0
Delta_Id_DIC = Id_DIC - Id_DIC_0
Delta_Iab_DIC = Iab_DIC - Iab_DIC_0
Delta_I_DIC = I_DIC - I_DIC_0

Delta_Im_DOC = Im_DOC - Im_DOC_0
Delta_Ibelow_DOC = Ibelow_DOC - Ibelow_DOC_0
Delta_I_DOC = I_DOC - I_DOC_0


#x1 = np.linspace(0., ndays/365, num=ndays+1)
#x2 = np.linspace(0., ndays/365 - 1, num=ndays-1)

#Id_DIC_phys = Delta_Id_DIC - Delta_Id_DIC_bio
#Im_DIC_phys = Delta_Im_DIC - Delta_Im_DIC_bio

fig, axes = plt.subplots(1,1)

axes.plot(Delta_I_DIC + Delta_I_DOC + Delta_Ia_year, label = "\u0394Iocean + \u0394Ia")
axes.plot(Iem_year, label = "Iem")
axes.plot(Delta_I_DIC, label = "\u0394Iocean (DIC)")
axes.plot(Delta_I_DOC, label = "\u0394Iocean (DOC)")
axes.plot(Delta_Ia_year, label = "\u0394Ia")
#axes.plot(Delta_Im_sol + Delta_Id_sol, label = "\u0394Iocean without biology")
axes.legend()
#axes.xaxis.set_ticks(np.linspace(0, ndays/365, 11))
axes.set_xlabel("Time (years)")
axes.set_ylabel("Carbon Inventory (PgC)")
fig.savefig("carbon_budget.png", dpi=500)

fig, axes = plt.subplots(1,1)

axes.plot(pHm, label = "pH_ml")
axes.plot(pHd, label = "pH_deep")
axes.legend()

DTa = np.array(DTa)
DR = np.array(DR)

fig, axes = plt.subplots(1,1)

axes.plot(DTa/DR)
#axes.xaxis.set_ticks(np.linspace(0, ndays/365, 11))
axes.set_xlabel("Time (years)")
axes.set_ylabel("thermal response")


fig, axes = plt.subplots(1,1)

axes.plot(1000*DR/Iem)
#axes.xaxis.set_ticks(np.linspace(0, ndays/365, 11))
axes.set_xlabel("Time (years)")
axes.set_ylabel("carbon response")

fig, axes = plt.subplots(1,1)

axes.plot(1000*DTa/Iem)
#axes.xaxis.set_ticks(np.linspace(0, ndays/365, 11))
axes.set_xlabel("Time (years)")
axes.set_ylabel("TCRE")

#fig, axes = plt.subplots(1,1)
#axes.plot(DTm)

fig, axes = plt.subplots(1,1)
axes.plot(Delta_Pgrow, label = "Pgrow")
axes.set_xlabel("Time (years)")
axes.set_ylabel("Pgrow flux anomaly in mixed layer (mol/ m^2/ year)")

fig, axes = plt.subplots(1,1)
axes.plot(Delta_Zgrow, label = "Zgrow")
axes.set_xlabel("Time (years)")
axes.set_ylabel("Zgrow flux anomaly in mixed layer (mol/ m^2/ year)")

fig, axes = plt.subplots(1,1)
axes.plot(Delta_Dremin, label="Dremin")
axes.set_ylabel("Dremin flux anomaly in mixed layer(mol/ m^2/ year)")
axes.set_xlabel("Time (years)")

fig, axes = plt.subplots(1,1)
axes.plot(Delta_Pmix, label="Pmix")
axes.set_ylabel("Pmix flux anomaly in deep layer(mol/ m^2/ year)")
axes.set_xlabel("Time (years)")

fig, axes = plt.subplots(1,1)
axes.plot(Delta_Zmix, label="Zmix")
axes.set_ylabel("Zmix flux anomaly in deep layer(mol/ m^2/ year)")
axes.set_xlabel("Time (years)")

fig, axes = plt.subplots(1,1)
axes.plot(Delta_Dmix, label="Dmix")
axes.set_ylabel("Dmix flux anomaly in deep layer(mol/ m^2/ year)")
axes.set_xlabel("Time (years)")

fig, axes = plt.subplots(1,1)
axes.plot(Delta_Dsink, label="Dsink")
axes.set_ylabel("Dsink flux anomaly in deep layer(mol/ m^2/ year)")
axes.set_xlabel("Time (years)")

fig, axes = plt.subplots(1,1)
axes.plot(Pgrow, label = "Pgrow")
axes.set_xlabel("Time (years)")
axes.set_ylabel("Pgrow flux")

fig, axes = plt.subplots(1,1)
axes.plot(Zgrow, label = "Zgrow")
axes.set_xlabel("Time (years)")
axes.set_ylabel("Zgrow flux")

fig, axes = plt.subplots(1,1)
axes.plot(Dremin, label="Dremin")
axes.set_ylabel("Dremin flux")
axes.set_xlabel("Time (years)")

fig, axes = plt.subplots(1,1)
axes.plot(Pmix, label="Pmix")
axes.set_ylabel("Pmix flux")
axes.set_xlabel("Time (years)")

fig, axes = plt.subplots(1,1)
axes.plot(Zmix, label="Zmix")
axes.set_ylabel("Zmix flux")
axes.set_xlabel("Time (years)")

fig, axes = plt.subplots(1,1)
axes.plot(Dmix, label="Dmix")
axes.set_ylabel("Dmix flux")
axes.set_xlabel("Time (years)")

fig, axes = plt.subplots(1,1)
axes.plot(Dsink, label="Dsink")
axes.set_ylabel("Dsink flux")
axes.set_xlabel("Time (years)")

fig, axes = plt.subplots(1,1)
axes.plot(DIC_to_DOC_m)
axes.set_ylabel("DIC to DOC in mixed layer (mol/ m^2/ year)")
axes.set_xlabel("Time (years)")

fig, axes = plt.subplots(1,1)
axes.plot(DIC_to_DOC_d)
axes.set_ylabel("DIC to DOC in deep layer (mol/ m^2/ year)")
axes.set_xlabel("Time (years)")

#fig, axes = plt.subplots(1,1)
#axes.plot(Fa)
#axes.set_ylabel("Atmosphere - mixed layer flux (mol/ m^2/ s)")
#axes.set_xlabel("Time (years)")

#fig, axes = plt.subplots(1,1)
#axes.plot(Fd)
#axes.set_ylabel("Mixed layer - deep layer flux (mol/ m^2/ s)")
#axes.set_xlabel("Time (years)")

fig, axes = plt.subplots(1,1)
axes.plot(DTa, label="Change in atmosphere temperature")
axes.plot(DTm, label="Change in mixed layer temperature")
axes.plot(DTd, label="Change in deep layer temperature")
axes.legend()

fig, axes = plt.subplots(1,1)
axes.plot(Delta_Id_DIC)
axes.plot(Ibelow_DOC)

fig, axes = plt.subplots(1,1)
axes.plot(Delta_Im_DIC)
axes.plot(Im_DOC)

"""
WRITING DATA INTO CSV FILES

"""
# Defining arrays for the time axes
x1 = np.arange(0, nyears+1)  # for results with yearly time increments
x2 = np.linspace(0., ndays/365, num=ndays+1)  # for results with daily time increments

# create a dataframe with x and y1 columns

print("CREATING DATAFRAMES")

df_Iem = pd.DataFrame(data=[x1,Iem_year]).T
df_Iem.columns = ['x1','Iem']

df_Delta_Id_DIC = pd.DataFrame(data=[x1, Delta_Id_DIC]).T
df_Delta_Id_DIC.columns = ['x1','Delta_Id_DIC_coupled']

df_Delta_Im_DIC = pd.DataFrame(data=[x1, Delta_Im_DIC]).T
df_Delta_Im_DIC.columns = ['x1','Delta_Im_DIC_coupled']

df_Delta_Iab_DIC = pd.DataFrame(data=[x1, Delta_Iab_DIC]).T
df_Delta_Iab_DIC.columns = ['x1','Delta_Iab_DIC_coupled']

df_Delta_Im_DIC_bio = pd.DataFrame(data=[x1, Delta_Im_DIC_bio]).T
df_Delta_Im_DIC_bio.columns = ['x1', 'Delta_Im_DIC_bio_coupled']

df_Delta_Id_DIC_bio = pd.DataFrame(data=[x1, Delta_Id_DIC_bio]).T
df_Delta_Id_DIC_bio.columns = ['x1', 'Delta_Id_DIC_bio_coupled']

df_Delta_Iab_DIC_bio = pd.DataFrame(data=[x1, Delta_Iab_DIC_bio]).T
df_Delta_Iab_DIC_bio.columns = ['x1', 'Delta_Iab_DIC_bio_coupled']

df_Ibelow_DOC = pd.DataFrame(data=[x1, Ibelow_DOC]).T
df_Ibelow_DOC.columns = ['x1','Ibelow_DOC_coupled']

df_Delta_Ibelow_DOC = pd.DataFrame(data=[x1, Delta_Ibelow_DOC]).T
df_Delta_Ibelow_DOC.columns = ['x1','Delta_Ibelow_DOC_coupled']

df_Im_DOC = pd.DataFrame(data=[x1, Im_DOC]).T
df_Im_DOC.columns = ['x1','Im_DOC_coupled']

df_Delta_Im_DOC = pd.DataFrame(data=[x1, Delta_Im_DOC]).T
df_Delta_Im_DOC.columns = ['x1','Delta_Im_DOC_coupled']

df_Delta_I_DOC = pd.DataFrame(data=[x1,Delta_I_DOC]).T
df_Delta_I_DOC.columns = ['x1','Delta_I_DOC_coupled']
#df_I_DOC.columns = ['x1','I_DOC_tventconst']
#print(df_I_DOC)

df_Delta_I_DIC = pd.DataFrame(data=[x1,Delta_I_DIC]).T
df_Delta_I_DIC.columns = ['x1','Delta_I_DIC_coupled']
#df_I_DIC.columns = ['x1','I_DIC_tventconst']
#rint(df_I_DIC)

df_Delta_Ia = pd.DataFrame(data=[x1,Delta_Ia_year]).T
df_Delta_Ia.columns = ['x1','Delta_Ia_coupled']
#df_Ia.columns = ['x1','Ia_tventconst']
#print(df_Ia)

df_delta_temp = pd.DataFrame(data=[x1, delta_temp]).T
df_delta_temp.columns = ['x1', 'delta_temp_coupled']

df_dic_m = pd.DataFrame(data=[x1,dic_m_year]).T
df_dic_m.columns = ['x1','dic_m_coupled']
#df_dic_m.columns = ['x1','dic_m_tventconst']
#print(df_dic_m)

df_dic_d = pd.DataFrame(data=[x1,dic_d_year]).T
df_dic_d.columns = ['x1','dic_d_coupled']
#df_dic_d.columns = ['x1','dic_d_tventconst']
#print(df_dic_d)

df_dic_ab = pd.DataFrame(data=[x1,dic_ab_year]).T
df_dic_ab.columns = ['x1','dic_ab_coupled']

df_DTa = pd.DataFrame(data=[x2,DTa]).T
df_DTa.columns = ['x2','DTa_coupled']
#df_DTa.columns = ['x2','DTa_tventconst']
#print(df_DTa)

df_DTm = pd.DataFrame(data=[x2,DTm]).T
df_DTm.columns = ['x2','DTm_coupled']
#df_DTm.columns = ['x2','DTm_tventconst']
#print(df_DTm)

df_DTd = pd.DataFrame(data=[x2,DTd]).T
df_DTd.columns = ['x2','DTd_coupled']
#df_DTd.columns = ['x1','DTd_tventconst']
#print(df_DTd)

df_DTab = pd.DataFrame(data=[x2,DTab]).T
df_DTab.columns = ['x2','DTab_coupled']

df_Delta_Pgrow = pd.DataFrame(data=[x1, Delta_Pgrow]).T
df_Delta_Pgrow.columns = ['x1','Delta_Pgrow_coupled']
#df_Delta_Pgrow.columns = ['x1','Delta_Pgrow_tventconst']

df_Delta_Zgrow = pd.DataFrame(data=[x1, Delta_Zgrow]).T
df_Delta_Zgrow.columns = ['x1','Delta_Zgrow_coupled']
#df_Delta_Zgrow.columns = ['x1','Delta_Zgrow_tventconst']

df_Delta_Dremin = pd.DataFrame(data=[x1, Delta_Dremin]).T
df_Delta_Dremin.columns = ['x1','Delta_Dremin_coupled']
#df_Delta_Dremin.columns = ['x1','Delta_Dremin_tventconst']

df_Delta_Nmix = pd.DataFrame(data=[x1, Delta_Nmix]).T
df_Delta_Nmix.columns = ['x1','Delta_Nmix_coupled']
#df_Delta_Nmix.columns = ['x1','Delta_Nmix_tventconst']

df_Delta_Pmix = pd.DataFrame(data=[x1, Delta_Pmix]).T
df_Delta_Pmix.columns = ['x1','Delta_Pmix_coupled']
#df_Delta_Pmix.columns = ['x1','Delta_Pmix_tventconst']

df_Delta_Zmix = pd.DataFrame(data=[x1, Delta_Zmix]).T
df_Delta_Zmix.columns = ['x1','Delta_Zmix_coupled']
#df_Delta_Zmix.columns = ['x1','Delta_Zmix_tventconst']

df_Delta_Dmix = pd.DataFrame(data=[x1, Delta_Dmix]).T
df_Delta_Dmix.columns = ['x1','Delta_Dmix_coupled']
#df_Delta_Dmix.columns = ['x1','Delta_Dmix_tventconst']

df_Delta_Dsink = pd.DataFrame(data=[x1, Delta_Dsink]).T
df_Delta_Dsink.columns = ['x1','Delta_Dsink_coupled']
#df_Delta_Dsink.columns = ['x1','Delta_Dsink_tventconst']

df_Pgrow = pd.DataFrame(data=[x1, Pgrow]).T
df_Pgrow.columns = ['x1','Pgrow_coupled']

df_Pgraz = pd.DataFrame(data=[x1, Pgraz]).T
df_Pgraz.columns = ['x1','Pgraz_coupled']

df_Pmort = pd.DataFrame(data=[x1, Pmort]).T
df_Pmort.columns = ['x1','Pmort_coupled']

df_Pmort2 = pd.DataFrame(data=[x1, Pmort2]).T
df_Pmort2.columns = ['x1','Pmort2_coupled']

df_Pmix = pd.DataFrame(data=[x1, Pmix]).T
df_Pmix.columns = ['x1','Pmix_coupled']

df_Pdilute = pd.DataFrame(data=[x1, Pdilute]).T
df_Pdilute.columns = ['x1','Pdilute_coupled']

df_Nmix = pd.DataFrame(data=[x1, Nmix]).T
df_Nmix.columns = ['x1','Nmix_coupled']

df_Zexc = pd.DataFrame(data=[x1, Zexc]).T
df_Zexc.columns = ['x1','Zexc_coupled']

df_Dremin_src = pd.DataFrame(data=[x1, Dremin_src]).T
df_Dremin_src.columns = ['x1','Dremin_src_coupled']

df_Pgrow_sink = pd.DataFrame(data=[x1, Pgrow_sink]).T
df_Pgrow_sink.columns = ['x1','Pgrow_sink_coupled']

df_Nadd = pd.DataFrame(data=[x1, Nadd]).T
df_Nadd.columns = ['x1','Nadd_coupled']

df_Ndilute = pd.DataFrame(data=[x1, Ndilute]).T
df_Ndilute.columns = ['x1','Ndilute_coupled']

df_Zgrow = pd.DataFrame(data=[x1, Zgrow]).T
df_Zgrow.columns = ['x1','Zgrow_coupled']

df_Zmort = pd.DataFrame(data=[x1, Zmort]).T
df_Zmort.columns = ['x1','Zmort_coupled']

df_Zmort2 = pd.DataFrame(data=[x1, Zmort2]).T
df_Zmort2.columns = ['x1','Zmort2_coupled']

df_Zdilute = pd.DataFrame(data=[x1, Zdilute]).T
df_Zdilute.columns = ['x1','Zdilute_coupled']

df_Zmix = pd.DataFrame(data=[x1, Zmix]).T
df_Zmix.columns = ['x1','Zmix_coupled']

df_Pmort_src = pd.DataFrame(data=[x1, Pmort_src]).T
df_Pmort_src.columns = ['x1','Pmort_src_coupled']

df_Pmort2_src = pd.DataFrame(data=[x1, Pmort2_src]).T
df_Pmort2_src.columns = ['x1','Pmort2_src_coupled']

df_Zmort_src = pd.DataFrame(data=[x1, Zmort_src]).T
df_Zmort_src.columns = ['x1','Zmort_src_coupled']

df_Zpel = pd.DataFrame(data=[x1, Zpel]).T
df_Zpel.columns = ['x1','Zpel_coupled']

df_Dgraz = pd.DataFrame(data=[x1, Dgraz]).T
df_Dgraz.columns = ['x1','Dgraz_coupled']

df_Dmix = pd.DataFrame(data=[x1, Dmix]).T
df_Dmix.columns = ['x1','Dmix_coupled']

df_Dsink = pd.DataFrame(data=[x1, Dsink]).T
df_Dsink.columns = ['x1','Dsink_coupled']

df_Dremin = pd.DataFrame(data=[x1, Dremin]).T
df_Dremin.columns = ['x1','Dremin_coupled']

df_Ddilute = pd.DataFrame(data=[x1, Ddilute]).T
df_Ddilute.columns = ['x1','Ddilute_coupled']

df_Pgrow_sum = pd.DataFrame(data=[x1, Pgrow_sum]).T
df_Pgrow_sum.columns = ['x1','Pgrow_sum_coupled']
#df_Pgrow_sum.columns = ['x1','Pgrow_sum_tventconst']

df_Zgrow_sum = pd.DataFrame(data=[x1, Zgrow_sum]).T
df_Zgrow_sum.columns = ['x1','Zgrow_sum_coupled']
#df_Zgrow_sum.columns = ['x1','Zgrow_sum_tventconst']

df_Dremin_sum = pd.DataFrame(data=[x1, Dremin_sum]).T
df_Dremin_sum.columns = ['x1','Dremin_sum_coupled']
#df_Dremin_sum.columns = ['x1','Dremin_sum_tventconst']

df_Nmix_sum = pd.DataFrame(data=[x1, Nmix_sum]).T
df_Nmix_sum.columns = ['x1','Nmix_sum_coupled']
#df_Nmix_sum.columns = ['x1','Nmix_sum_tventconst']

df_Pmix_sum = pd.DataFrame(data=[x1, Pmix_sum]).T
df_Pmix_sum.columns = ['x1','Pmix_sum_coupled']
#df_Pmix_sum.columns = ['x1','Pmix_sum_tventconst']

df_Zmix_sum = pd.DataFrame(data=[x1, Zmix_sum]).T
df_Zmix_sum.columns = ['x1','Zmix_sum_coupled']
#df_Zmix_sum.columns = ['x1','Zmix_sum_tventconst']

df_Dmix_sum = pd.DataFrame(data=[x1, Dmix_sum]).T
df_Dmix_sum.columns = ['x1','Dmix_sum_coupled']
#df_Dmix_sum.columns = ['x1','Dmix_sum_tventconst']

df_Dsink_sum = pd.DataFrame(data=[x1, Dsink_sum]).T
df_Dsink_sum.columns = ['x1','Dsink_sum_coupled']
#df_Dsink_sum.columns = ['x1','Dsink_sum_tventconst']

df_t_vent1_phys = pd.DataFrame(data=[x2, t_vent1_phys]).T
df_t_vent1_phys.columns = ['x2','t_vent1_phys_coupled']

df_t_vent2_phys = pd.DataFrame(data=[x2, t_vent2_phys]).T
df_t_vent2_phys.columns = ['x2','t_vent2_phys_coupled']

print("WRITING INTO CSVs")

# write to a csv
df_Iem.to_csv("Results/Iem.csv", index=None)

df_Delta_Id_DIC.to_csv("Results/Delta_Id_DIC_coupled.csv", index=None)

df_Delta_Im_DIC.to_csv("Results/Delta_Im_DIC_coupled.csv", index=None)

df_Delta_Iab_DIC.to_csv("Results/Delta_Iab_DIC_coupled.csv", index=None)

df_Delta_Im_DIC_bio.to_csv("Results/Delta_Im_DIC_bio_coupled.csv", index=None)

df_Delta_Id_DIC_bio.to_csv("Results/Delta_Id_DIC_bio_coupled.csv", index=None)

df_Delta_Iab_DIC_bio.to_csv("Results/Delta_Iab_DIC_bio_coupled.csv", index=None)

df_Ibelow_DOC.to_csv("Results/Ibelow_DOC_coupled.csv", index=None)

df_Delta_Ibelow_DOC.to_csv("Results/Delta_Ibelow_DOC_coupled.csv", index=None)

df_Im_DOC.to_csv("Results/Im_DOC_coupled.csv", index=None)

df_Delta_Im_DOC.to_csv("Results/Delta_Im_DOC_coupled.csv", index=None)

df_Delta_I_DOC.to_csv("Results/Delta_I_DOC_coupled.csv", index=None)
#df_I_DOC.to_csv("Results/I_DOC_tvent2.csv", index=None)
#df_I_DOC.to_csv("Results/I_DOC_tventconst.csv", index=None)

df_Delta_I_DIC.to_csv("Results/Delta_I_DIC_coupled.csv", index=None)
#df_I_DIC.to_csv("Results/I_DIC_tvent2.csv", index=None)
#df_I_DIC.to_csv("Results/I_DIC_tventconst.csv", index=None)

df_Delta_Ia.to_csv("Results/Delta_Ia_coupled.csv", index=None)
#df_Ia.to_csv("Results/Ia_tvent2.csv", index=None)
#df_Ia.to_csv("Results/Ia_tventconst.csv", index=None)

df_delta_temp.to_csv("Results/delta_temp_coupled.csv", index=None)

df_dic_m.to_csv("Results/dic_m_coupled.csv", index=None)
#df_dic_m.to_csv("Results/dic_m_tvent2.csv", index=None)
#df_dic_m.to_csv("Results/dic_m_tventconst.csv", index=None)

df_dic_d.to_csv("Results/dic_d_coupled.csv", index=None)
#df_dic_d.to_csv("Results/dic_d_tvent2.csv", index=None)
#df_dic_d.to_csv("Results/dic_d_tventconst.csv", index=None)

df_dic_ab.to_csv("Results/dic_ab_coupled.csv", index=None)

df_DTa.to_csv("Results/DTa_coupled.csv", index=None)
#df_DTa.to_csv("Results/DTa_tvent2.csv", index=None)
#df_DTa.to_csv("Results/DTa_tventconst.csv", index=None)

df_DTm.to_csv("Results/DTm_coupled.csv", index=None)
#df_DTm.to_csv("Results/DTm_tvent2.csv", index=None)
#df_DTm.to_csv("Results/DTm_tventconst.csv", index=None)

df_DTd.to_csv("Results/DTd_coupled.csv", index=None)
#df_DTd.to_csv("Results/DTd_tvent2.csv", index=None)
#df_DTd.to_csv("Results/DTd_tventconst.csv", index=None)

df_DTab.to_csv("Results/DTab_coupled.csv", index=None)

df_Delta_Pgrow.to_csv("Results/Delta_Pgrow_coupled.csv", index=None)
#df_Delta_Pgrow.to_csv("Results/Delta_Pgrow_tventconst.csv", index=None)

df_Delta_Zgrow.to_csv("Results/Delta_Zgrow_coupled.csv", index=None)
#df_Delta_Zgrow.to_csv("Results/Delta_Zgrow_tventconst.csv", index=None)

df_Delta_Dremin.to_csv("Results/Delta_Dremin_coupled.csv", index=None)
#df_Delta_Dremin.to_csv("Results/Delta_Dremin_tventconst.csv", index=None)

df_Delta_Pmix.to_csv("Results/Delta_Pmix_coupled.csv", index=None)
#df_Delta_Pmix.to_csv("Results/Delta_Pmix_tventconst.csv", index=None)

df_Delta_Zmix.to_csv("Results/Delta_Zmix_coupled.csv", index=None)
#df_Delta_Zmix.to_csv("Results/Delta_Zmix_tventconst.csv", index=None)

df_Delta_Nmix.to_csv("Results/Delta_Nmix_coupled.csv", index=None)
#df_Delta_Nmix.to_csv("Results/Delta_Nmix_tventconst.csv", index=None)

df_Delta_Dmix.to_csv("Results/Delta_Dmix_coupled.csv", index=None)
#df_Delta_Dmix.to_csv("Results/Delta_Dmix_tventconst.csv", index=None)

df_Delta_Dsink.to_csv("Results/Delta_Dsink_coupled.csv", index=None)
#df_Delta_Dsink.to_csv("Results/Delta_Dsink_tventconst.csv", index=None)

df_Pgrow.to_csv("Results/Pgrow_coupled.csv", index=None)

df_Pgraz.to_csv("Results/Pgraz_coupled.csv", index=None)

df_Pmort.to_csv("Results/Pmort_coupled.csv", index=None)

df_Pmort2.to_csv("Results/Pmort2_coupled.csv", index=None)

df_Pmix.to_csv("Results/Pmix_coupled.csv", index=None)

df_Pdilute.to_csv("Results/Pdilute_coupled.csv", index=None)

df_Nmix.to_csv("Results/Nmix_coupled.csv", index=None)

df_Zexc.to_csv("Results/Zexc_coupled.csv", index=None)

df_Dremin_src.to_csv("Results/Dremin_src_coupled.csv", index=None)

df_Pgrow_sink.to_csv("Results/Pgrow_sink_coupled.csv", index=None)

df_Nadd.to_csv("Results/Nadd_coupled.csv", index=None)

df_Ndilute.to_csv("Results/Ndilute_coupled.csv", index=None)

df_Zgrow.to_csv("Results/Zgrow_coupled.csv", index=None)

df_Zmort.to_csv("Results/Zmort_coupled.csv", index=None)

df_Zmort2.to_csv("Results/Zmort2_coupled.csv", index=None)

df_Zdilute.to_csv("Results/Zdilute_coupled.csv", index=None)

df_Zmix.to_csv("Results/Zmix_coupled.csv", index=None)

df_Pmort_src.to_csv("Results/Pmort_src_coupled.csv", index=None)

df_Pmort2_src.to_csv("Results/Pmort2_src_coupled.csv", index=None)

df_Zmort_src.to_csv("Results/Zmort_src_coupled.csv", index=None)

df_Zpel.to_csv("Results/Zpel_coupled.csv", index=None)

df_Dgraz.to_csv("Results/Dgraz_coupled.csv", index=None)

df_Dmix.to_csv("Results/Dmix_coupled.csv", index=None)

df_Dsink.to_csv("Results/Dsink_coupled.csv", index=None)

df_Dremin.to_csv("Results/Dremin_coupled.csv", index=None)

df_Ddilute.to_csv("Results/Ddilute_coupled.csv", index=None)

df_Pgrow_sum.to_csv("Results/Pgrow_sum_coupled.csv", index=None)
#df_Pgrow_sum.to_csv("Results/Pgrow_sum_tventconst.csv", index=None)

df_Zgrow_sum.to_csv("Results/Zgrow_sum_coupled.csv", index=None)
#df_Zgrow_sum.to_csv("Results/Zgrow_sum_tventconst.csv", index=None)

df_Dremin_sum.to_csv("Results/Dremin_sum_coupled.csv", index=None)
#df_Dremin_sum.to_csv("Results/Dremin_sum_tventconst.csv", index=None)

df_Pmix_sum.to_csv("Results/Pmix_sum_coupled.csv", index=None)
#df_Pmix_sum.to_csv("Results/Pmix_sum_tventconst.csv", index=None)

df_Zmix_sum.to_csv("Results/Zmix_sum_coupled.csv", index=None)
#df_Zmix_sum.to_csv("Results/Zmix_sum_tventconst.csv", index=None)

df_Nmix_sum.to_csv("Results/Nmix_sum_coupled.csv", index=None)
#df_Nmix_sum.to_csv("Results/Nmix_sum_tventconst.csv", index=None)

df_Dmix_sum.to_csv("Results/Dmix_sum_coupled.csv", index=None)
#df_Dmix_sum.to_csv("Results/Dmix_sum_tventconst.csv", index=None)

df_Dsink_sum.to_csv("Results/Dsink_sum_coupled.csv", index=None)
#df_Dsink_sum.to_csv("Results/Dsink_sum_tventconst.csv", index=None)

df_t_vent1_phys.to_csv("Results/t_vent1_phys_coupled.csv", index=None)

df_t_vent2_phys.to_csv("Results/t_vent2_phys_coupled.csv", index=None)

"""
GETTING END TIME

"""

ent = perf_counter()
elapsed_time = ent - st

print("TIME TO RUN = ", elapsed_time)