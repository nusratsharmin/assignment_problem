# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:08:11 2015

@author: nusrat
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 10:13:16 2015

@author: nusrat
"""

import numpy as np
import nibabel as nib
from dipy.tracking.vox2track import streamline_mapping
from dipy.tracking import utils
from dissimilarity_common import compute_disimilarity
from dipy.tracking.distances import bundles_distances_mam 
from sklearn.cluster import MiniBatchKMeans,KMeans
from sklearn import metrics
from toy_example import load_tractome_tractography, apply_affine
from dipy.viz import fvtk
from sklearn.metrics import pairwise_distances
from mapping_test import *
from transforms3d import euler, affines
from common_functions import load_tract, Jac_BFN, visualize_tract
from fastPFP import fastPFP_faster, greedy_assignment
from dipy.tracking.metrics import length
from scipy.optimize import fmin_powell
from optimal_cost1 import *
from optimal_cost_max import *
from time import time
from munkres import Munkres, print_matrix
from sklearn.utils import linear_assignment_


def voxel_measure(static_center, moving_center, show=False, vol_size=(364, 436, 364)):
    
    vol_A = np.zeros(vol_size)

    #ci, cj, ck = vol_size[0] / 2, vol_size[1] / 2, vol_size[2] / 2
    ci, cj, ck = vol_size[0]/2 , vol_size[1]/2 , vol_size[2]/2 
    spts = np.concatenate(static_center, axis=0)
    spts = np.round(spts).astype(np.int) + np.array([ci, cj, ck])

    mpts = np.concatenate(moving_center, axis=0)
    mpts = np.round(mpts).astype(np.int) + np.array([ci, cj, ck])

    for index in spts:
        i, j, k = index
        vol_A[i, j, k] = 1

    vol_B = np.zeros(vol_size)
    
    for index in mpts:
        i, j, k = index
        vol_B[i, j, k] = 1

    vol_and = np.logical_and(vol_A, vol_B)
    
    return   np.divide(np.sum(vol_and),np.sum(vol_B))

def pad_to_square(a, pad_value=0):
  m = a.reshape((a.shape[0], -1))
  padded = pad_value * np.ones(2 * [max(m.shape)], dtype=m.dtype)
  padded[0:m.shape[0], 0:m.shape[1]] = m
  return padded   
target_ids=[204]
source_ids = [205]
source = str(source_ids[0])
target = str(target_ids[0])
    # for source 

filename_tractography = '/home/nusrat//targetDirectory/HCP/MICCAI2015_DTI_EUDX/100307_1M_wmql/wmql_cst.right.trk'

arg1 = '/home/nusrat/Samples_for_Prob_Map/' + str(source)+'/' + 'dti_linear.trk'
arg2 = '/home/nusrat/Samples_for_Prob_Map/' + str(target)+'/' + 'dti_linear.trk'    
    # for source CST+50 SFF
#arg2 = '/home/nusrat/Samples_for_Prob_Map/50_SFF_in_ext/ROI_seg_native/' + source + '_cst_R_tvis_sff_in_ext.pkl'
    
arg3='/home/nusrat/Samples_for_Prob_Map/ROI_seg_tvis/ROI_seg_tvis_native/' + source + '_corticospinal_R_tvis.pkl'    
#arg3='/home/nusrat/Samples_for_Prob_Map/ROI_seg_tvis/ROI_seg_tvis_native/' + source + '_corticospinal_R_tvis.pkl'    
    # for source CST+ ext
#arg3 = '/home/nusrat/Samples_for_Prob_Map/50_SFF_in_ext/ROI_seg_native/' + source + '_cst_R_tvis_ext.pkl'
arg4 = '/home/nusrat/Samples_for_Prob_Map/50_SFF_in_ext/ROI_seg_native/' + target+ '_cst_R_tvis_ext.pkl'    

s_cst_idx = '/home/nusrat/Samples_for_Prob_Map/ROI_seg_tvis/ROI_seg_tvis_native/' + source+ '_corticospinal_R_tvis.pkl'

t_cst_idx = '/home/nusrat/Samples_for_Prob_Map/ROI_seg_tvis/ROI_seg_tvis_native/' + target + '_corticospinal_R_tvis.pkl'

s_cst = load_tract( arg1, s_cst_idx)
t_cst = load_tract( arg2, t_cst_idx)
s_cst_ext = load_tract( arg1, arg3)
t_cst_ext = load_tract( arg2, arg4)
#SA=s_cst[:len(t_cst)]
SA=s_cst
SB=t_cst_ext
dmA = bundles_distances_mam(SA, SA)
dmB = bundles_distances_mam(SB, SB)
sigma = np.mean([np.median(dmA), np.median(dmB)])
A = np.exp(- dmA / sigma)
B = np.exp(- dmB / sigma)
#dm1=np.array(A.max(axis=0))
#dm2=np.array(B.max(axis=0))

lengths1 = [length( S ) for S in SA] 
matrix_lenA=np.array(lengths1)

lengths2 = [length( S ) for S in SB] 
matrix_lenB=np.array(lengths2)
np.fill_diagonal(dmA,10000)
np.fill_diagonal(dmB,10000)
 
dm1=np.array(dmA.max(axis=0))
dm2=np.array(dmB.max(axis=0))
dm1=np.array(dmA.max(axis=0))
dm2=np.array(dmB.max(axis=0))
dij = np.zeros(( matrix_lenA.shape[0],  matrix_lenB.shape[0]))

for i in xrange(0,matrix_lenA.shape[0]):
     a=np.divide(dm2,dm1[i])
     dij[i]=a
#print dij
'''     
lij = np.zeros(( matrix_lenA.shape[0],  matrix_lenB.shape[0]))
for i in xrange(0,matrix_lenA.shape[0]):
     b=np.divide(matrix_lenB,matrix_lenA[i])
     lij[i]=b
'''     
lij=[np.divide(matrix_lenB,matrix_lenA[i]) for i in xrange(0,matrix_lenA.shape[0])]
#print lij
    #cij=np.add(lij,dij)   
#cij=np.add(lij,dij)
cij=lij*dij
if np.sum(cij)>0:
    
    print np.sqrt(np.sum(cij))
    
dmAB = bundles_distances_mam(SA, SB)    
cij_new=lij*dmAB

    

#map_all=optimal_cost (cij)      
#cij=pad_to_square(cij, -10000)

#m = Munkres()
#indexes = m.compute(cij_new)
'''
dmA = bundles_distances_mam(SA, SA)
dmB = bundles_distances_mam(SB, SB)
sigma = np.mean([np.median(dmA), np.median(dmB)])
A = np.exp(- dmA / sigma)
B = np.exp(- dmB / sigma)

X = fastPFP_faster(B, A, alpha=0.9, threshold1=1.0e-4, threshold2=1.0e-4, max_iter1=100, max_iter2=100)
'''

#print cij_new
print "min" 
map_all=optimal_cost (cij)   
map_all=np.array(map_all,dtype=int)   
map_list=SB[map_all]
print voxel_measure(t_cst,map_list)

'''

print "max" 
map_all=optimal_cost_max (cij_new)   
map_all=np.array(map_all,dtype=int)   
map_list=SB[map_all]

#print voxel_measure(map_list,SB[0:len(SA)])

print voxel_measure(t_cst,map_list)


map_sklearn=linear_assignment_.linear_assignment(cij)
map_list=np.array([map_sklearn[i][1] for i in xrange(0,len(map_sklearn))])
mapping=SB[map_list]
print voxel_measure(t_cst,mapping)

ren = fvtk.ren() 
fvtk.add(ren, fvtk.line(t_cst.tolist(), fvtk.colors.red, linewidth=2, opacity=0.5))
fvtk.add(ren, fvtk.line(s_cst.tolist(), fvtk.colors.green, linewidth=2, opacity=0.5))
fvtk.add(ren, fvtk.line(mapping.tolist(), fvtk.colors.blue, linewidth=2, opacity=1))
fvtk.show(ren)

'''
