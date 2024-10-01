import os
import json
from tqdm import tqdm
import numpy as np

def average_scores(rule_indices, rating_results_path):
    cumulative_sum = np.zeros(1000000)

    for rule_index in tqdm(rule_indices, desc=f"Average scores for {len(rule_indices)} rules:"):
        rule_vector = []
        rule_folder = os.path.join(rating_results_path, f'rule{rule_index}')
        for i in range(1000):
            file_path = os.path.join(rule_folder, f'batch{i}_float.json')
            with open(file_path, 'r') as file:
                scores = json.load(file)
                rule_vector.extend(scores)
        cumulative_sum += np.array(rule_vector)

    # Calculate the element-wise average
    average_vector = cumulative_sum / len(rule_indices)
    return average_vector.tolist()

def quality_sampling(quality_scores, tau=1):
    """
    Perform quality sampling using the Gumbel top-k trick.

    Parameters:
    quality_scores (list or numpy array): List of quality scores.
    tau (float): Temperature parameter.
    k (int): Number of top elements to sample.

    Returns:
    list: Indices of the top-k sampled elements.
    """
    quality_scores = np.array(quality_scores)
    log_probs = quality_scores / tau
    
    # Add Gumbel noise to the log-probabilities
    noisy_scores = log_probs + np.random.gumbel(loc=0, scale=1, size=log_probs.shape)

    sorted_indices = np.argsort(noisy_scores)[::-1]
    return sorted_indices.tolist()
