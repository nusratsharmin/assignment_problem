# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 13:14:09 2014

@author: nusrat
"""

import numpy as np
import pdb

def optimal_cost(C):
 #C= np.random.random( (3,3) )
 #C = np.array([[10, 2, 3],[ 4 ,50 ,6],[ 17, 8, 10]])

 #pdb.set_trace()
 nOfRows    = C.shape[0];
 assignment = np.zeros(nOfRows);
 cost       = 0;
 #print nOfRows
 for i in xrange(0,nOfRows ):
  #[minDist, index1] = min(distMatrix, [], 1);
  index1=C.argmin(axis=0)
  minDist=C.min(axis=0)
  #print index1,minDist
 # [minDist, index2] = min(minDist);
  index2=minDist.argmin()
  minDist=minDist.min()
  #print minDist,index1,index2
  row = index1[index2];
  col = index2;

  if ( minDist):
    assignment[row] = col;
    cost = cost + minDist;
    

    C[row, :] = 100000;
    C[:, col] = 100000;
    #print C
  else:
    break
 return assignment
