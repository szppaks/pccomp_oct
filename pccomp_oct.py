#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###################################################################
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

copyright (c) 2020, Peter Szutor

@author: Peter Szutor, Hungary, szppaks@gmail.com
Created on Wed Feb 26 17:23:24 2020
###################################################################



Octree-based lossy point-cloud compression with open3d and numpy
Average compressing rate (depends on octreee depth setting parameter): 0.012 - 0.1   


Input formats: You can get a list of supported formats from : http://www.open3d.org/docs/release/tutorial/Basic/file_io.html#point-cloud
               (xyz,pts,ply,pcd)

Usage:
    
Dependencies: Open3D, Numpy  (You can install theese modules:  pip install open3d, pip install numpy)    
    
Compress a point cloud:
    
octreezip(<filename>,<depth>) -> <result>
<filename>: (str) Point Cloud file name. Saved file name: [filename without ext]_ocz.npz  (Yes, it's a numpy array file)
<depth>   : (str) Octree depth. You can try 11-16 for best result. Bigger depht results higher precision and bigger compressed file size.
<result>  : (str) If the compressing was success you get: "Compressed into:[comp.file name] | Storing resolution:0.003445". Storing resolution means the precision.
                  The PC file is missing or bad: "PC is empty, bad, or missing"
                  Other error: "Error: [error message]"
                  

Uncompressing:
octreeunzip(<filename>) -> <result>
<filename>: (str) Zipped Point Cloud file name (npz). Saved file name: [filename].xyz  (standard XYZ text file)
<result>  : (str) If the compressing was success you get: "Saved: [filename].xyz"
                  Other error: "Error: [error message]"
                  


"""
import numpy as np
import os
import open3d as o3d

def d1halfing_fast(pmin,pmax,pdepht):
    return np.linspace(pmin,pmax,2**int(pdepht)+1)
                       
def octreecodes(ppoints,pdepht):
    minx=np.amin(ppoints[:,0])
    maxx=np.amax(ppoints[:,0])
    miny=np.amin(ppoints[:,1])
    maxy=np.amax(ppoints[:,1])
    minz=np.amin(ppoints[:,2])
    maxz=np.amax(ppoints[:,2])
    xletra=d1halfing_fast(minx,maxx,pdepht)
    yletra=d1halfing_fast(miny,maxy,pdepht)
    zletra=d1halfing_fast(minz,maxz,pdepht)
    otcodex=np.searchsorted(xletra,ppoints[:,0],side='right')-1
    otcodey=np.searchsorted(yletra,ppoints[:,1],side='right')-1
    otcodez=np.searchsorted(zletra,ppoints[:,2],side='right')-1
    ki=otcodex*(2**(pdepht*2))+otcodey*(2**pdepht)+otcodez
    return (ki,minx,maxx,miny,maxy,minz,maxz)

def octreezip(pfilename,pdepht):
    try:
        pcd = o3d.io.read_point_cloud(pfilename,format='auto')
        ppoints=np.asarray(pcd.points)
        if len(ppoints)>0:
            occ=octreecodes(ppoints,pdepht)
            occsorted=np.sort(occ[0])
            prec=np.amax(np.asarray([occ[2]-occ[1],occ[4]-occ[3],occ[6]-occ[5]])/(2**pdepht))
            paramarr=np.asarray([pdepht,occ[1],occ[2],occ[3],occ[4],occ[5],occ[6]]) #depth and boundary
            np.savez_compressed(os.path.splitext(pfilename)[0]+'_ocz',points=occsorted,params=paramarr)
            retmessage='Compressed into:'+str(os.path.splitext(pfilename)[0])+'.ocz | Storing resolution:'+str(prec)
        else:
            retmessage='PC is empty, bad, or missing'
    except Exception as e:
        retmessage='Error:'+str(e)
    return retmessage

def octreeunzip(pfilename):
    try:
        pc=np.load(pfilename)
        pcpoints=pc['points']
        pcparams=pc['params']
        pdepht=(pcparams[0])
        minx=(pcparams[1])
        maxx=(pcparams[2])
        miny=(pcparams[3])
        maxy=(pcparams[4])
        minz=(pcparams[5])
        maxz=(pcparams[6])
        xletra=d1halfing_fast(minx,maxx,pdepht)
        yletra=d1halfing_fast(miny,maxy,pdepht)
        zletra=d1halfing_fast(minz,maxz,pdepht)    
        occodex=(pcpoints/(2**(pdepht*2))).astype(int)
        occodey=((pcpoints-occodex*(2**(pdepht*2)))/(2**pdepht)).astype(int)
        occodez=(pcpoints-occodex*(2**(pdepht*2))-occodey*(2**pdepht)).astype(int)
        koorx=xletra[occodex]
        koory=yletra[occodey]
        koorz=zletra[occodez]
        points=np.array([koorx,koory,koorz]).T
        np.savetxt(os.path.splitext(pfilename)[0]+'.xyz',points,fmt='%.4f')
        retmessage='Saved:'+os.path.splitext(pfilename)[0]+'.xyz'
    except Exception as e:
        retmessage='Error:'+str(e)
    return retmessage



