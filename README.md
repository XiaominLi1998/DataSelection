# Pipeline for Rule Generation and Data Selection

This repository provides the necessary code for running a pipeline that includes generating rules, rating data, selecting rules, and sampling data. Note that while only the Math domain is included as an example, the procedure and code for other domains follow the same structure.

## Download Data
Download the SlimPajama-6B data from Hugging Face at the repository [DKYoon/SlimPajama-6B](https://huggingface.co/datasets/DKYoon/SlimPajama-6B) and save the first 1M samples.

## Generate Rules
Prompts and rules can be found in the following script:
`DataSelection/experiments_Math/rating/rating_prompts.py`

## Data Rating
The script below runs the training code located at `DataSelection/experiments_Math/rating/rating_with_50rules.py`
```bash
   ./DataSelection/experiments_Math/rating/run_50rating_jobs.sh
```

This will prompt Llama3-8B-Instruct to rate the data and record the answers. The results will be stored in the folder: `rating/rule_rating_results`

## Select Rules and Data

1. **Generate DPP indices to select rules:**
    ```bash
    python dpp_select_rules.py
    ```
2. **Use the DPP rule indices to sample data:**
    ```bash
    python select_data.py
    ```

## Training and Benchmarking

Create a folder named "benchmark" and download `bigcode-evaluation-harness` and `lm-evaluation-harness` into it.

The script below runs the training code located at `DataSelection/train_benchmark/ft_rule_rating_selected.py`:
```bash
   ./DataSelection/train_benchmark/run_ft_rule_rating_selected.sh
```

The trained models and results will be saved to `DataSelection/train_benchmark`.
