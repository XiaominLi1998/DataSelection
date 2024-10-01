import sys
import time
from tqdm import tqdm
sys.path.append("DataSelection/experiments_Math/")
from quality_sampling import *


# Step 1: Read data SlimPajama6B_short_1M.jsonl
data_file = 'DataSelection/SlimPajama_data/SlimPajama6B_short_1M.jsonl'
start_time = time.time()
with open(data_file, 'r') as f:
    lines = f.readlines()
print(f"Loading {data_file} took {(time.time() - start_time):.2f} seconds.")
output_dir = 'selected_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Step 2: Select SlimPajama6B_short_1M.jsonl (using rule quality scores) and save
dpp_indices = [0, 4, 13, 26, 27, 31, 33, 38, 44, 45] #Replace with generated dpp indices
rule_indices = dpp_indices
rating_results_path = "DataSelection/experiments_Math/rating/rule_rating_results"
scores = average_scores(rule_indices, rating_results_path)
all_indices = quality_sampling(scores, tau=1) #this is essentially sampling 100%

percent_ls = [1, 2, 5, 10, 20]
for percent in tqdm(percent_ls, desc="percent_ls"):
    size = int(len(all_indices) * percent / 100)
    selected_indices = all_indices[:size]
    selected_data = [lines[idx] for idx in selected_indices]

    # Save the subset to a new file
    output_file = f'{output_dir}/SlimPajama6B_short_1M_{percent}percent.jsonl'
    with open(output_file, 'w') as out_f:
        out_f.writelines(selected_data)

    print(f"Saved {percent}% subset to {output_file}")