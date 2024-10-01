import os
import json
import numpy as np
from tqdm import tqdm
import sys
from dppy.finite_dpps import FiniteDPP
import random
import numpy as np
import time


start_time = time.time()
root_dir = '/DataSelection/experiments_Math/rating/rule_rating_results'
quality_vector_ls = [] 

for rule_idx in tqdm(range(50), desc="Get 1M-dim quality vector"):
    quality_vector = []
    rule_folder = os.path.join(root_dir, f'rule{rule_idx}')
    # Loop through each JSON file in the rule folder
    for batch_idx in range(1000):
        input_file = os.path.join(rule_folder, f'batch{batch_idx}_float.json')
        with open(input_file, 'r') as f:
            num_list = json.load(f)
        quality_vector.extend(num_list)        
    quality_vector_ls.append(quality_vector)

quality_matrix = np.array(quality_vector_ls)
print("quality_matrix.shape = ", quality_matrix.shape)
print(f"Done! Reading quality vectors took {(time.time()-start_time)/60 :0.2f} min."); sys.stdout.flush()


start_time = time.time()
# Initialize a Determinantal Point Process with the kernel matrix
kernel_matrix = np.dot(quality_matrix, quality_matrix.T)
print("kernel_matrix.shape = ", kernel_matrix.shape) #DPP selects the rows of quality_matrix here (each row corresponds to a rule)
print(f"Done! Calculating kernel_matrix took {(time.time()-start_time)/60 :0.2f} min."); sys.stdout.flush()


start_time = time.time()
dpp = FiniteDPP(kernel_type='likelihood', L=kernel_matrix)
def dpp_sample(k):
    dpp.sample_exact_k_dpp(size=k)
    selected_indices = dpp.list_of_samples[-1]
    return selected_indices
r = 10
dpp_indices = dpp_sample(r)
print("dpp_indices = ", sorted(dpp_indices))
print(f"Done! DPP took {(time.time()-start_time)/60 :0.2f} min."); sys.stdout.flush()

