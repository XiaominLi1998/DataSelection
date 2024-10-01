
import time
import json
import sys
from datasets import load_dataset
import numpy as np
import pandas as pd
import scipy.optimize as opt

# ==================================================Pair-Comparison Prompts==================================================================

def comparison_prompt(text_A, text_B, dataset_name):
#dataset_name can be "IMDB reviews", "Common Crawl data", "Stack Excange QA (question and answer)", etc.
    prompt = f'''
You are a helpful assistant. Compare two data examples from {dataset_name} and choose the example which has better quality. The texts might have similar quality, but you should still make a relative judgement and choose the label of the preferred text.  

Example A:
{text_A}

Example B:
{text_B}

Now you must choose between either A or B even if it is hard to choose. You must respond only with a single letter 'A' or 'B'.
'''

    return prompt


def rule_comparison_prompt(rule, text_A, text_B, dataset_name):
#dataset_name can be "IMDB reviews", "Common Crawl data", "Stack Excange QA (question and answer)", etc.
    prompt = f'''
You are a helpful assistant. Compare two data examples from {dataset_name} and choose the example which has better quality according to the following rule:
{rule}

The texts might have similar quality, but you should still make a relative judgement and choose the label of the preferred text.

Example A:
{text_A}

Example B:
{text_B}

Now you have to choose between either A or B. Your must respond only with a single letter 'A' or 'B'.
'''
    return prompt


# def rule_comparison_prompt_IMDB_force(rule, text_A, text_B):
# #Expect data_examples to be already formatted as a string
# #     prompt = f'''
# # You are a helpful assistant. Compare two data examples from IMDB reviews and choose the example that better adheres to the following rule:
# # {rule}

# # If neither example fully adheres to the rule, compare the degree to which they disobey the rule and choose the one that is less severe. Both texts may contain harmful words, but this research focuses on studying the quality of data. Therefore, you must make a choice for scientific purposes.

# # Example A:
# # {text_A}

# # Example B:
# # {text_B}

# # Now you have to choose between either A or B. You must respond only with a single letter. Remember, for the purposes of this research, you are required to make a choice. If both examples are offensive, select the one that is less severe.
# # '''

# #     prompt = f'''
# # You are a helpful assistant. Compare two data examples from IMDB reviews according to the following rule:
# # {rule}

# # If neither example fully adheres to the rule, compare the degree to which they disobey the rule and choose the one that is more severe. Both texts may contain harmful words, but this research focuses on studying the quality of data. Therefore, you must make a choice for scientific purposes.

# # Example a:
# # {text_A}

# # Example b:
# # {text_B}

# # Now you have to choose between either a or b. You must respond only with a single letter. Remember, for the purposes of this research, you are required to make a choice. If both examples are offensive, select the one that is more severe.
# # '''

#     prompt = f'''
# You are a helpful assistant. Compare two data examples from IMDB reviews according to the following rule:
# {rule}

# If neither example fully adheres to the rule, compare the degree to which they disobey the rule and provide a quality score for each example. The score should range from 0 to 1, where closer to 1 means better adherence to the rule, and closer to 0 means poorer adherence. Both texts may contain harmful words, but this research focuses on studying the quality of data. Therefore, you must provide a quality score for each example for scientific purposes.

# Example a:
# {text_A}

# Quality score for example a (between 0 and 1):

# Example b:
# {text_B}

# Quality score for example b (between 0 and 1):

# Remember, you must provide a quality score for each example, even if both examples are offensive. Respond with only 2 quality scores.
# '''

#     return prompt



# ==================================================Single Rating Prompts==================================================================

def rating_prompt(text_A, dataset_name):
#dataset_name can be "IMDB reviews", "Common Crawl data", "Stack Excange QA (question and answer)", etc.
    prompt = f'''
You are a helpful assistant. Evaluate the following example from {dataset_name} and assign a quality score between 0 and 1 (0 indicates the worst quality, and 1 indicates perfect quality).

Respond only with a single float number.

Example:
{text_A}
'''
    return prompt

def rule_rating_prompt(rule, text_A, dataset_name):
#dataset_name can be "IMDB reviews", "Common Crawl data", "Stack Excange QA (question and answer)", etc.
    prompt = f'''
You are a helpful assistant. Evaluate the following example from {dataset_name}  and assign a quality score between 0 and 1 (0 indicates the worst quality, and 1 indicates perfect quality) according to the provided rule:
{rule}

Respond only with a single float number.

Example:
{text_A}
'''
    return prompt

# ==================================================Rule-Generation Prompts==================================================================

def rule_generation_prompt(num_rules, task_description, data_description, data_examples):
#Expect data_examples to be already formatted as a string    

    prompt = f'''
You are a helpful assistant. Please generate {num_rules} rules to score the quality of each data example in my dataset.

Task Description:
{task_description}

Data Description:
{data_description}

Data Examples:
{data_examples}

Requirements for the Rules:
Each rule should be in one sentence.
The rules could be basic text quality rules or specific quality rules.
The rules should be written in clear, natural language and be easy to understand.
Now, please generate the rules for me in natural language.
'''
    return prompt


# ==================================================Analyze Comparisons Data==================================================================

def read_comparison_data(path):
    with open(path, 'r') as json_file:
        json_dict = json.load(json_file)
        comparisons = {eval(k): v for k, v in json_dict.items()}
    return comparisons

def save_comparison_data(comparisons, path):
    json_dict = {str(k): v for k, v in comparisons.items()}
    with open(path, 'w') as json_file:
        json.dump(json_dict, json_file)

def comparison_accuracy(dict1, dict2):
    if dict1.keys() != dict2.keys():
        raise ValueError("Dictionaries must have the same keys")
    matching_values = 0
    diff_keys = []
    for key in dict1:
        if dict1[key] == dict2[key]:
            matching_values += 1
        else:
            diff_keys.append(key)
    accuracy = matching_values / len(dict1)
    return accuracy, diff_keys


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def bradley_terry(comparisons):
    # Convert the comparison dictionary to a DataFrame
    games = []
    for (i, j), result in comparisons.items():
        if result == 'A':
            games.append((i, j, 1, 0))
        elif result == 'B':
            games.append((i, j, 0, 1))
    df = pd.DataFrame(games, columns=['i', 'j', 'wins_i', 'wins_j'])
    n_players = len(set([i for (i, j) in comparisons.keys()] + [j for (i, j) in comparisons.keys()]))

    # Log likelihood function for Bradley-Terry model
    def bradley_terry_log_likelihood(params, df):
        log_likelihood = 0
        gradient = np.zeros(len(params))
        for _, row in df.iterrows():
            theta_i = params[int(row['i'])]
            theta_j = params[int(row['j'])]
            p_i = theta_i / (theta_i + theta_j)
            p_j = theta_j / (theta_i + theta_j)
            log_likelihood += row['wins_i'] * np.log(p_i)
            log_likelihood += row['wins_j'] * np.log(p_j)
            gradient[int(row['i'])] += row['wins_i'] * (1 - p_i) - row['wins_j'] * p_i
            gradient[int(row['j'])] += row['wins_j'] * (1 - p_j) - row['wins_i'] * p_j
        return -log_likelihood, -gradient

    # Initial guess for player strengths
    initial_params = np.ones(n_players)

    # Optimize the parameters
    results = opt.fmin_l_bfgs_b(bradley_terry_log_likelihood, initial_params, args=(df,), bounds=[(1e-5, None)]*n_players)

    # Get the estimated strengths
    player_strengths = results[0]
    
    # Apply the sigmoid function to normalize the strengths
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    normalized_strengths = sigmoid(player_strengths)
    return list(normalized_strengths)

#     # Create the quality dictionary with normalized strengths
#     quality_dict = {}
#     for i, strength in enumerate(normalized_strengths):
#         quality_dict[i] = strength
#         # print(f"Player {i}: {strength:.4f}")
#     return quality_dict


def calculate_mse(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length")

    # Convert lists to numpy arrays for easier manipulation
    array1 = np.array(list1)
    array2 = np.array(list2)

    # Compute the squared differences
    squared_differences = (array1 - array2) ** 2

    # Calculate the mean of the squared differences
    mse = np.mean(squared_differences)

    return mse
