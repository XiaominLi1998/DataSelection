import os
import json
import numpy as np
from tqdm import tqdm

# Root directory containing the rule folders
root_dir = 'rule_rating_results'


# # Sanity check: check each rule folder indeed has batchfile0 ~ batchfile 999 (1000 files)
# for rule_idx in tqdm(range(50), desc="Check rule folder has 1000 files"):
#     rule_folder = os.path.join(root_dir, f'rule{rule_idx}')
#     batch_size = 1000
#     for batch_idx in tqdm(range(batch_size), desc="batch:"):
#         input_file = os.path.join(rule_folder, f'batch{batch_idx}.json')
#         if os.path.exists(input_file):
#             pass  # Replace with actual processing code
#         else:
#             print(f"File not found: {input_file}")

# Function to process each list of strings
def process_list(string_list):
    float_list = []
    for s in string_list:
        try:
            float_list.append(float(s))
        except ValueError:
            float_list.append(None)  # Placeholder for non-float values
    
    # Calculate the mean of the valid floats
    valid_floats = [f for f in float_list if f is not None]
    mean_value = np.mean(valid_floats)
    
    # Replace None values with the mean value
    processed_list = [f if f is not None else mean_value for f in float_list]
    
    return processed_list


# Loop through each rule folder and process strings responses to float ratings.
for rule_idx in tqdm(range(50), desc="Processing rule folder"):
    rule_folder = os.path.join(root_dir, f'rule{rule_idx}')
    batch_size = 1000
    # Loop through each JSON file in the rule folder
    for batch_idx in tqdm(range(batch_size), desc="batch:"):
        input_file = os.path.join(rule_folder, f'batch{batch_idx}.json')
        output_file = os.path.join(rule_folder, f'batch{batch_idx}_float.json')
        
        if os.path.exists(output_file):
            continue
        if not os.path.exists(input_file):
            print(f"Input file {input_file} does not exist!")
            continue
        # Read the JSON file
        with open(input_file, 'r') as f:
            string_list = json.load(f)
        
        # Process the list
        processed_list = process_list(string_list)
        
        # Save the processed list to a new JSON file
        with open(output_file, 'w') as f:
            json.dump(processed_list, f)

print("Processing completed.")
