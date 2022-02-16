#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 11:06:39 2022

@author: jstout
"""

import numpy as np

'''Pulled from: https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion'''
def appendSpherical_np(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew

tmp_ = appendSpherical_np()
[TH, PHI, R] = 

#Make circular plot from brainstorm 
% Spherical coordinates
[TH,PHI,R] = cart2sph(x, y, z);
% Flat projection
R = 1 - PHI ./ pi*2;
% Convert back to cartesian coordinates
[X,Y] = pol2cart(TH,R);

% Convert back to cartesian coordinates
[TH,R] = cart2pol(X,Y);
% Convex hull
facesBorder = convhull(X,Y);
iBorder = unique(facesBorder);


% Deformation field in radius computed from the border sensors, projected onto the circle
Rcircle = 1;
Dborder = Rcircle ./ R(iBorder);
% Compute the radius deformation to apply to all the sensors
funcTh = [TH(iBorder)-2*pi; TH(iBorder); TH(iBorder)+2*pi];
funcD  = [Dborder; Dborder; Dborder];
D = interp1(funcTh, funcD, TH, 'linear', 0);

% Remove the possible zero values
D(D == 0) = 1;
% Compute new radius: the closer to the center, the less transformed
R = min(R .* D, Rcircle);
% Convert back to cartesian coordinates
[X,Y] = pol2cart(TH, R);

