# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 23:41:04 2019

@author: krish
"""

import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import itertools
import numpy as np
from operator import itemgetter 
from itertools import repeat



#program batch_beam_size in the init class


def get_start_end_indices(self):
    
   start2end_id=[0]
   
   start2end_id=start2end_id+self.batch_beam_size
   
   start2end_id=np.cumsum(start2end_id,0)
   
   self.ind = start2end_id


def get_results(self):
    
    gen_cap = torch.LongTensor(np.zeros((self.maxlen, self.batch_size),dtype=int))
    
    if len(self.comp_beams[i]) == 0:
        self.comp_beams[i] = self.beams[i]
    self.comp_beams[i] = sorted(self.comp_beams[i],
        key=itemgetter('total_logprob'), reverse=True)
    
    best_beam = self.comp_beam[i][0]['words'][1:]
    
    best_beam.extend(repeat(0, self.maxlen - len(best_beam))) 
    #Above statement doesn't do anything if the best_beam>maxlen
    best_beam = best_beam[:self.maxlen]
    gen_cap[:, i] = torch.LongTensor(best_beam)
 return gen_cap


    
    
