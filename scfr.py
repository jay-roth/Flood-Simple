# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 17:23:08 2020

@author: Jason.Roth
"""

import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt


def plot_hydrograph(t,q):
    plt.plot(t, q)


def resample(delta, xvals, yvals):
    """
    resamples a function of x and y values at a new delta x via linear
    interpolation
    """
    nu_x = np.zeros(1)
    nu_y = np.zeros(1)
    x = 0
    # interpolate out just a bit longer than the original distribution
    while x < xvals[-1]:
        x+=delta
        nu_x = np.append(nu_x, x)
        y = linterp(x, xvals, yvals)
        nu_y = np.append(nu_y, y)
    return nu_x, nu_y

def linterp(x, xvals, yvals):
    """
    Linearly interpotates the y value given a value x and 
    vectors of x and y for the function range.  xvals and yvals must be same 
    dimensions
    
    Inputs:
        x, (float) - known value of x for which corresponding unknown value
                    of y is needed
        xvals, (vector, floats) - vector of x-values for which corresponding 
                    y-values are known.
        yvals, (vector, floats) - vector of y-values for which corresponding
                    x- values are given.
    Outputs:
        y (float) - determined value of y corresponding to x
    """
    # test for dimensional consistency
    if xvals.shape == yvals.shape: 
        # make sure x is in the range of the function
        # check to see if this x value is explicit
        if x in xvals:
            i = np.where(xvals==x)[0][0]
            y = yvals[i]
        elif x >= xvals[-1]:
            y = yvals[-1] +\
                (yvals[-1]-yvals[-2])/(xvals[-1]-xvals[-2])*(x - xvals[-1])
        elif x <= xvals[0]:
            y = yvals[0]
        else:
            i = 0
            # get last xval just smaller than desired location    
            while x >= xvals[i] and i < xvals.shape[0]-1:
                i+=1
            emm = (yvals[i] - yvals[i-1])/(xvals[i] - xvals[i-1])
            y = yvals[i-1] + (x - xvals[i-1])*emm
    else:
        print("x and y ranges do not agree")
        y = 0
    return y

def make_sfunc(dep, q_out, p_dep, p_vol, delT):
    """
    makes the 2S/delT + O function
    
    """
    m = dep.shape[0]
    s_vals = np.zeros(m)
    for i in range(m):
        s_vals[i] = linterp(dep[i],p_dep, p_vol)
    sfunc = s_vals*2.0/delT + q_out
    return sfunc

def sto_ind_meth(p_dep, p_vol, q_in, dep, q_out, delT, vol_init):
    delT*=3600.
    sfunc = make_sfunc(dep, q_out, p_dep, p_vol, delT)
    ## what are the initial conditions
    do = linterp(vol_init, p_vol, p_dep)
    qo = linterp(do, dep, q_out)
    t = 0
    
    i = 1
    q = np.ones(1)*qo
    t = np.zeros(1)
    s = np.ones(1)*vol_init
    h = np.ones(1)*do
    #sf = np.zeros(1)
    # route while outflow is greater than tol or throughout the whole storm
    while q[i-1] > 0.1 or i <= q_in.shape[0]-1:
        # no inflow if beyond hydrograph period
        lhs = 2.0*s[i-1]/delT - q[i-1]
        if lhs < 0:
            print('halt')
        if i <= q_in.shape[0]-1:
            lhs += q_in[i]+q_in[i-1]
        qt = linterp(lhs, sfunc, q_out) 
        q = np.append(q, qt)
        t = np.append(t, t[i-1]+delT)
        st = ((lhs - qt)*delT)/2.
        s = np.append(s, st)
        h = np.append(h,linterp(st, p_vol, p_dep))
        i+=1
    return q, t, s, h

def calc_manning_flow(dep, d, s, n, uni='us'):
    r = d/2.
    pi=np.pi
    if dep > d:
        print ("Warning: depth {0} ft is greater than".format(dep)+\
                   "diameter of pipe {0}".format(d))
        dep = d
    if dep > r:
        h = (2.*r-dep)
        th = 2.*pi-2.*np.acos((r-h)/r)
        A = r**2*th/2.+r**2/2.*np.sin(2.*pi-th)
    else:
        h = dep
        th = 2.*np.acos((r-h)/r)
        A = r**2*(th-np.sin(th))/2.
    P = r*th
    q = A*(A/P)**(2./3.)*s**(1./2.)/n
    if uni == "us":
        q*=1.49
    return q

def calc_gamma(ropf, inc,  tol=1e-7, min_pts=100):
    """
    calculates runoff hydrograph at an interval 
    m-vals across the range of 100-600 can be approximated
    using prf and the equation below. Relationship was constructed using
    table 630-16.5 from NEH
    
    Inputs:

    Outputs:

    """
    ## calculate m factor for this runoff peaking factor (ropf)
    m = round(1.46216e-5*ropf**2+4.36124e-4*ropf+6.7188e-2,2)
    
    i = 1
    q = tol+1
    t_rat = np.zeros(1)
    q_rat = np.zeros(1)
    
    while ((q_rat[i-1] > tol) or (i < min_pts)):
        t_rat =np.append(t_rat, t_rat[i-1]+inc)
        q = np.exp(m)*t_rat[i]**m*np.exp((-m*t_rat[i]))
        if q > tol:
            q_rat = np.append(q_rat, q)
        else:
            q_rat = np.append(q_rat, 0)
        i+=1
    return t_rat, q_rat

def calc_tc(a, y, cn, l):
    """
    calculates tc, tl based on MN guidance
    https://www.nrcs.usda.gov/Internet/FSE_DOCUMENTS/stelprdb1270339.pdf
    
    
    Inputs:
        a , float- area of watershed in acres
        y , float- slope of watershed in ft/ft
        cn - curve number for watershe
        l - flowpath length in feet
    
    Outputs:
        tc, float- time of concentration in hrs
    
    """
    # find storage of watershed (in) 
    # NEH630 Ch10 Eqn 10-12
    s = 1000.0/cn-10.0  
    
    # convert slope to pct
    y=y * 100.0
    
    
    if a <= 30:
            # NEH630 Ch15 Eqn 15-4b
            tc = (l**0.8*(s+1.)**0.7)/(1140.*y**0.5) 
    else:
        if a > 1500:
            if y < 2:
                # NEH630 Ch15 Eqn 15-4b
                tc = 1.65 * (l**0.8*(s+1.)**0.7)/(1140.*y**0.5)
            else:
                # Folmar, N.D., and A.C. Miller. 2008. 
                tc = l**0.65/108.3
        else:
            if y < 2:
                # NEH630 Ch15 Eqn 15-4b
                tc = 1.65 *(l**0.8*(s+1.)**0.7)/(1140.*y**0.5)
            else:
                # Folmar, N.D., and A.C. Miller. 2008.
                tc = l**0.65/108.3
    tc = round(tc, 1)            
    return tc

def calc_cn_runoff(cn, p, ini_rat=0.05):
    """
    Calculates total and incremental runoff for a given storm 
    """
    s = max(1000./cn - 10.,0)
    ia = s*ini_rat
    inc_ro = np.zeros(p.shape[0])
    tot_ro = np.zeros(p.shape[0])
    for i in range(p.shape[0]):
        if p[i] > ia:
            tot_ro[i] = ((p[i]-ia)**2/(p[i]+s-ia)) 
            inc_ro[i] = tot_ro[i] - tot_ro[i-1]
    return tot_ro, inc_ro

def make_hydro(inc_ro, uh_q):
    """
    creates a runoff hydrograph given a incremental runoff and 
    a unit hydrograph. Time intervals for both inputs must agree.
    
    """
    ## now need to 
    # reverse sort incremental runoff
    
    ## x is the runoff excess from the design storm hyetograph and 
    ## curve number 
    
    ## y is unit runoff hydrograph using the design hydrograph and watershed
    ## attributes
    
    ## reversed incremental runoff matrix has m rows, 2 times the length
    ## of the incremental runoff vector
    
    
    ## reversed incremental runoff matrix has n rows, the max of the length
    ## of the incremental runoff or the hydrograph vectors
    ## take longest vector and make the other one same length padding
    ## 0's on the end
    if inc_ro.shape[0] > uh_q.shape[0]:
        n = inc_ro.shape[0]
        q = np.zeros(n)
        q[0:uh_q.shape[0]] = uh_q
        ro = inc_ro
    else:
        n = uh_q.shape[0]
        ro = np.zeros(n)
        ro[0:inc_ro.shape[0]] = inc_ro
        q = uh_q
    m= 2*ro.shape[0]-1    
    ro_mat = np.zeros((m,n))
    ro = ro[::-1]

    for i in range(m):
        if i < n:
            ro_mat[i,0:i+1] = ro[n-i-1:n]
        else:
            ro_mat[i, i-n+1:] = ro[0:m-i]

    
    q = np.matmul(ro_mat,q)    
    return q

    
def make_stg_vol(dep, area):
    """
    creates stage volume from stage area data
    
    """
    vol = np.zeros(dep.shape[0])
    for i in range(1,dep.shape[0]):
        vol[i] = vol[i-1] + (dep[i]-dep[i-1])*(area[i]+area[i-1])/2.0
    return vol

def calc_wier_C(r, h, typ="circ"):
    """
    calculate wier coefficient for depending on stage
    
    https://www.greensboro-nc.gov/home/showdocument?id=3710
    
    """
    if typ == "circ":
        if h/r < 0.5:
            Cw = 3.4-0.5*(h/r)
        else:
            Cw = 3.15-2.3*(min(h/r,1.0)-0.5)
    else:
        if h/r< 0.5:
            Cw = 2.65
        elif h/r < 1.8:
            Cw = 2.65+0.5*(h/r-0.5)
        else:
            Cw = 3.3
    return Cw

def calc_riser_q(r, dep):
    """
    r, dict - {'typ':'riser', 'elev':2, 'shp':'circ', 
                                               'dim':8, 'Cd':0.6, 'Cw':3.3}
    dep, array - depths to calculate flow for
    
    """
    g = 32.2
    q = np.zeros(dep.shape[0])
    e = r['elev']
    cd = r['Cd']
    cw = r['Cw']
    rad = r['dim']/2.0
    # calculate the geometry
    if r['shp'] == 'circ':
        a = np.pi*r['dim']**2/4.
        l = np.pi*r['dim']
    elif r['shp'] == 'sqr':
        a = r['dim']**2
        l = 4.*r['dim']
    for i in range(dep.shape[0]):
        d = dep[i]
        if d > e:
            ## NEH 650 Ch 3 Equation 3- 25. 
            ## Cw  typical vals 3.2 – 3.3 
            cw = calc_wier_C(rad, d-e, typ="circ")
            qw = cw*l*(d-e)**(3./2.)
            ## NEH 650 Ch 3 Equation 3- 33.
            qo = a*cd*(2.*g*(d-e))**0.5
            q[i] = min(qw, qo)
        else:
            q[i] = 0 
    return q

def calc_barrel_q(b, dep, elev_rise):
    """

    """
    g = 32.2
    q = np.zeros(dep.shape[0])
    e1 = elev_rise
    e2 = b['elev_out']
    d = b['diam']
    l = b['length']
    n = b['n']
    twd = b['tw']
    #Km = Kent + Kben + Kext
    km = b['Cd']
    a = np.pi*b['diam']**2/4.
    ## Equation 3- 7. 
    kp = 29.164 *n**2/(d/2.)**(4./3.)
    for i in range(dep.shape[0]):
        d = dep[i]
        if d > e1:
            ## NEH 650 Ch 3 Equation 3- 12.. 
            q[i] = a*((2.*g*(d-(e2+twd)))/(1 + km + (kp*l)))**0.5
        else:
            q[i] = 0
    return q

def calc_bcw_q(bcw, dep):
    """
    calculates a stage dicharge relationship for a broadcrested wier
    for all depths in dep
    """
    #Broad crested 2.6 – 3.1 
    q = np.zeros(dep.shape[0])
    e = bcw['elev']
    cw = bcw['Cw']
    l = bcw['length']
    for i in range(dep.shape[0]):
        d = dep[i]
        if d > e:
            ## NEH 650 Ch 3 Equation 3- 25.
            q[i] = cw*l*(d-e)**(3./2.)
        else:
            q[i] = 0
    return q

def make_stg_disch(outlet, dep):
    """
    delegates the creation of and then merges stage discharge relationships
    predefined outlet types 
    
    """

    # make a container for the discharge values, make super high so they are 
    # replaced with limiting values from outlets, more than the mississippi
    q_out = np.ones(dep.shape[0]) * 1000000.0
    
    ## check for type of outlet
    if outlet['typ'] == "riser":
        ## first calculate riser flow
        q_rise = calc_riser_q(outlet, dep)
        q_out = np.where(q_rise < q_out, q_rise, q_out)
        ## check for a barrel entry
        if "barrel" in outlet.keys():
            b = outlet['barrel']
            q_barr = calc_barrel_q(b, dep, outlet['elev'])
            q_out = np.where(q_barr < q_out, q_barr, q_out)
            
    elif outlet['typ'] == 'culvert':
        print('under development')
    elif outlet['typ'] == "hooded":
        print('under development')
    elif outlet['typ'] == "bcw":
        q_bcw = calc_bcw_q(outlet, dep)
        q_out = np.where(q_bcw < q_out, q_bcw, q_out)
    return q_out

###############################################################################
## BEGIN USER INPUT ###########################################################
###############################################################################
## wshd area in acres
area=50.0

# wshd curve number
cnum=88

## optional, flow path length, in feet, will calculate if unknown
flen= ""

## wshd flow path
slp = 0.001

## intial abstraction value, 0.05 or 0.2 most common
init=0.05 

## county for precip
cty="CHIPPEWA"

## recurrence interval for storm
rec_int=25

## precip distribution to use MSE1-4 or SCS1-3
dist = "MSE3"

## runoff peak factor to use.
ropf = 400.0

## number of increments to divide each timestep (delD) into for routing
#t_inc = 5.0
delT = 0.2 # 0.2
## initial pond depth
dep_init = 0.0

## sum of losses for barrel flow
#k_b = k_entrance + k_bend + k_exit 
k_b = 0.5+1.0+1.0

## outlet characteristics
prim = {'typ':'riser', 'elev':0., 'shp':'circ', 'dim':2.0, 'Cd':0.7, 
        'Cw':2.6, "barrel":{'elev_in':-2., 'elev_out':-2.5, 'diam':0.75, 
                            'length':40., 'Cd':k_b, 'n':0.013, 'tw':0.6*0.75}}

aux= {'typ':'bcw', 'elev':4, 'length':60, 'Cw':2.6}
###############################################################################
## END USER INPUT #############################################################
###############################################################################

ac2mi = 1.0/640
### read in supporting data sets, hyetographs, precip data, stage vol 
cwd = os.getcwd()
dat_dir = os.path.join(cwd, "data")
## read in precipitation amounts
prec_df = pd.read_csv(os.path.join(dat_dir,"precipitation.csv"),
                      index_col="County")
prec = round(prec_df.loc[cty, str(rec_int)],2)

## read in precipitation distributions
p_dist_df = pd.read_csv(os.path.join(dat_dir,"hyetographs.csv"),
                      index_col="Time-Hrs")
p_dist = p_dist_df[dist] 
## house cleaning
del p_dist_df, prec_df

# calculate flow length if not specified
if flen == "":
    # eq 630 15-5
    flen = round(209*area**0.6,0)
    
# calculate time of concentration
Tc = calc_tc(area, slp, cnum, flen)

# calculate delta D for further computations.
#16A–13
delD = 0.133*Tc 
if delD > 0.1:
    delD = round(delD, 1)
else:
    delD = round(delD, 1)

if delT == "":
    delT = delD
#delT = round(delD/t_inc, 2)  
## resample precip distro to delD interval
px, py = resample(delT, p_dist.index.values, p_dist.values)

## calculate incremental runoff for the storm at delD intervals
tot_roff, inc_roff = calc_cn_runoff(cnum, py*prec, ini_rat=init)
## house cleaning
del p_dist, py

## determine hydrograph that is being used and resample
## if using SCS runoff hydrograph read it in, else compute with gamma f(x)
if ropf == 484:
    duh_df = pd.read_csv(os.path.join(dat_dir,"hydrographs.csv"),
                          index_col="index")
    duh_t = duh_df['t_484'].values()
    duh_q = duh_df['q_484'].values()
    del duh_df
    ## neeed to resample the input duh to delD
else:
# calculate the duh for the number of points and time increment 
    duh_t, duh_q = calc_gamma(ropf, 0.1)

## calculation unit hydrograph parameters
#Eq 630-16A7
Tp = round(delD/2 + 0.6*Tc,1) 
#Eq 630-16A6
Qp = round(ropf*(area*ac2mi)*1.0/Tp,1) 

uh_q = duh_q*Qp
uh_t = duh_t*Tp
## resample the unit hydrographs to delD time step
uh_t, uh_q = resample(delT, uh_t, uh_q)

# check the volume under the unit hydro to make sure it agrees with geometry
u_ro = 645.33*area*ac2mi*1.0
if abs(uh_q.sum()*delT - u_ro)/u_ro < 0.02:
    q_in = make_hydro(inc_roff, uh_q)
    ## should check in_hydro volume
else:
    print('unit hydrograph volume descrepancy too large to continue')
    q_in = uh_q * 0.0 

## read in stage area information
stg_area = pd.read_csv(os.path.join(dat_dir,"stg_area.csv"))
p_dep = stg_area['stg_ft'].values
p_area = stg_area['area_sft'].values
p_vol = make_stg_vol(p_dep, p_area)

# calculate initial pond volume
vol_init = linterp(dep_init, p_dep, p_vol)

# make a range of intervals to calculate discharge over
dep_inc = 0.01
dep = np.arange(p_dep.min(), p_dep.max()+3.0, dep_inc)
q_prim = make_stg_disch(prim, dep)
q_aux = make_stg_disch(aux, dep)
q_tot = q_prim + q_aux

q_out, t, s, h = sto_ind_meth(p_dep, p_vol, q_in, dep, q_tot, delT, vol_init)
    

t/=3600.00


##fig, ax = plt.subplots()

p = plot_hydrograph(t,q_in)
p = plot_hydrograph(t,q_out)


