import sys
from tqdm import tqdm
from quality_sampling import *

import os
import json
import numpy as np
from dppy.finite_dpps import FiniteDPP
import random
import numpy as np
import time

def submatrix_norm(rule_indices, rating_results_path):#using ||S-I||_F
    rule_vector_ls = []
    for rule_index in rule_indices:
        rule_vector = []
        rule_folder = os.path.join(rating_results_path, f'rule{rule_index}')
        for i in range(1000):
            file_path = os.path.join(rule_folder, f'batch{i}_float.json')
            with open(file_path, 'r') as file:
                scores = json.load(file)
                rule_vector.extend(scores)
        rule_vector_ls.append(rule_vector)

    S = np.corrcoef(np.array(rule_vector_ls), rowvar=True)
    # Extract the submatrix for the selected indices
    r = len(rule_indices)
    I =  np.eye(len(rule_indices))
    frobenius_norm = np.linalg.norm(S-I, 'fro') / r
    return frobenius_norm
