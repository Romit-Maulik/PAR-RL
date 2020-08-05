#------------------------------------------------------------------------------+
#
#   Nathan A. Rooy
#   Functions for parameterizing 2D airfoils via Bezier Curves
#   2020-01-24
#
#------------------------------------------------------------------------------+

#--- IMPORT DEPENDENCIES ------------------------------------------------------+

import numpy as np
from math import pi
from math import sqrt

#--- MAIN ---------------------------------------------------------------------+

def get_t(x, ctlPts):

    # given x, solve for t
    a = ctlPts[0][0]
    b = ctlPts[1][0]
    c = ctlPts[2][0]
    
    d = abs(c*x + x*a - 2*x*b + b**2 - c*a)
    t_0 = (a - b + sqrt(d)) / (c + a - 2*b) 
    t_1 = (a - b - sqrt(d)) / (c + a - 2*b)
    
    # determine which root to return
    roots = np.array([t_0, t_1])
    t = roots[np.where((np.abs(roots) <= 1.0001) & (np.abs(roots) >= 0))][0]
    
    # rounding error is annoying... 
    if t > 1: t = 1  
    return t
    
    
def quadratic_bezier(t, ctlPts):
    y =(1-t)*((1-t)*ctlPts[0][1]+t*ctlPts[1][1])+t*((1-t)*ctlPts[1][1]+t*ctlPts[2][1])
    return y
    


def bezier_airfoil(x_coords, ctlPts):

    # cycle through control points
    curve=[]
    total_pts = len(ctlPts)
    for i in range(0, len(ctlPts)-1):

        # calculate first bezier curve
        if i==0:
            x_mid = (ctlPts[1][0]+ctlPts[2][0])/2
            y_mid = (ctlPts[1][1]+ctlPts[2][1])/2
            x_clip = sorted(x_coords[np.where(x_coords > x_mid)], reverse=True)
            x_bezier = [get_t(item, [ctlPts[0],ctlPts[1],[x_mid, y_mid]]) for item in x_clip]
            y = quadratic_bezier(np.asarray(x_bezier), [ctlPts[0],ctlPts[1],[x_mid,y_mid]])
            curve.append(ctlPts[0])     # append top trailing edge point
            curve=curve+list(zip(x_clip ,y))
            
        # calculate middle bezier curves
        if i>=1 and i<total_pts-3:
            x_mid1=(ctlPts[i][0]+ctlPts[i+1][0])/2
            y_mid1=(ctlPts[i][1]+ctlPts[i+1][1])/2
            x_mid2=(ctlPts[i+1][0]+ctlPts[i+2][0])/2
            y_mid2=(ctlPts[i+1][1]+ctlPts[i+2][1])/2
            
            # get points within current control point bounds
            if x_mid1 > x_mid2: x_clip = sorted(x_coords[(x_coords > x_mid2) & (x_coords <= x_mid1)], reverse=True)
            if x_mid1 < x_mid2: x_clip = sorted(x_coords[(x_coords < x_mid2) & (x_coords >= x_mid1)], reverse=False)
                
            x_bezier = [get_t(item, [[x_mid1, y_mid1], ctlPts[i+1], [x_mid2, y_mid2]]) for item in x_clip]
            y = quadratic_bezier(np.asarray(x_bezier), [[x_mid1,y_mid1],ctlPts[i+1],[x_mid2,y_mid2]])
            curve=curve+list(zip(x_clip, y))
            
        # calculate last bezier curve
        if i==total_pts-2:
            x_mid=(ctlPts[-3][0]+ctlPts[-2][0])/2
            y_mid=(ctlPts[-3][1]+ctlPts[-2][1])/2
            x_clip = sorted(x_coords[x_coords > x_mid], reverse=False)
            x_bezier = [get_t(item, [[x_mid, y_mid],ctlPts[-2],ctlPts[-1]]) for item in x_clip]
            y = quadratic_bezier(np.asarray(x_bezier), [[x_mid, y_mid],ctlPts[-2],ctlPts[-1]])
            curve=curve+list(zip(x_clip, y))
            curve.append(ctlPts[-1])    # append bottom trailing edge point

    # geo check
    # 1) make sure leading edge is [0,0]
    return curve
    
    
def munge_ctlpts(cps, n_top, n_bot):
    '''
    Munge the outputs from a given optimizer into a form that's suitable
    for the bezier function
    
    x = [le_radius, px1, py1, px2, py2,...,pxn, pyn]
    '''
    assert len(cps) % 2 == 1, 'The number of inputs must be odd because each control point has 2 points associated with it, plus a leading edge radius'
    
    te_half_thickness = 0.00147
    te_top = [1, te_half_thickness]
    te_bot = [1,-te_half_thickness]
    le_x = 0
    
    le_radius = cps[0]
    
    ctlpts = []
    ctlpts.append(te_top)
    for i in range(1, n_top * 2 + 1, 2):
        ctlpts.append([cps[i], cps[i+1]])
        
    ctlpts.append([le_x,  le_radius])
    ctlpts.append([le_x, -le_radius])
    
    for i in range(n_top * 2 + 1, len(cps), 2):
        ctlpts.append([cps[i], cps[i+1]])

    ctlpts.append(te_bot)
        
    return ctlpts

