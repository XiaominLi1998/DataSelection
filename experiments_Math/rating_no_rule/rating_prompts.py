
def rating_prompt(text_A):
    prompt = f'''
You are a helpful assistant. We are training a language model using the SlimPajama dataset to improve performance on Mathematics reasoning task. Evaluate the following example from SlimPajama dataset and assign a quality score between 0 and 1 (0 indicates the worst quality, and 1 indicates perfect quality).

Respond only with a single float number.

Example:
{text_A}
'''
    return prompt
