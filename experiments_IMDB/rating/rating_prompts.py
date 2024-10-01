rule_generation_prompt = """
Generate 50 specific rules for rating data from the training dataset (SlimPajama), in order to select a high-quality subset to train large language models (LLMs) that will improve their performance on the IMDB sentiment analysis task. The descriptions of the training data and downstream task are provided below. The rules should focus on various aspects such as data quality, relevance, diversity, and other characteristics that would be beneficial for sentiment analysis.

Description of training data:
The SlimPajama dataset is a large-scale dataset. It is designed to be a compact, high-quality dataset curated for pre-training large language models. The dataset includes a diverse range of texts, sourced from various domains such as web pages, books, and academic articles, providing a rich and varied training corpus for developing robust and versatile language models.


Description of downstream task:
The IMDB review dataset, created by StanfordNLP, is a widely used dataset for sentiment analysis. It contains 50,000 highly polar movie reviews. Each review is labeled as either positive or negative, making it an ideal dataset for binary sentiment classification tasks. The dataset provides a challenging benchmark for evaluating the performance of sentiment analysis models.


Requirements for the Rules:
Each rule should be concise and specific.
The rules could be basic text quality rules or task-relaated quality rules.
The rules should be written in clear, natural language and be easy to understand.
Now, please generate the rules for me in natural language.
"""

rules = [
    "Text Length: Be between 100 and 1000 words to match the typical length of IMDB reviews.",
    "Sentiment Clarity: Clearly express either positive or negative sentiments.",
    "Language Quality: Have fewer than 2 spelling or grammatical errors per 100 words.",
    "Language Focus: Be in English to maintain focus on the language of the target dataset.",
    "Source Diversity: Be sourced evenly from web pages, books, and academic articles.",
    "Tone Appropriateness: Minimize neutral tones as they are less useful for binary sentiment analysis.",
    "Cultural Relevance: Discuss culturally significant topics relevant to a global English-speaking audience.",
    "Language Style: Use informal, conversational language.",
    "Sarcasm Avoidance: Avoid sarcasm to prevent misinterpretation by sentiment analysis models.",
    "Subjectivity: Express opinions rather than just stating facts.",
    "Emotional Expression: Express emotions to aid in sentiment understanding.",
    "Redundancy Avoidance: Avoid redundancy and excessive similarity to other texts in the dataset.",
    "Contemporary Relevance: Be from the past decade to ensure relevance.",
    "Industry Relevance: Include mentions of movies, actors, or film industry terms.",
    "Sentiment Indicators: Contain explicit sentiment indicators.",
    "Sentence Complexity: Feature complex sentence structures.",
    "Figurative Language: Use metaphors and similes.",
    "Contextual Richness: Provide enough context to understand the sentiment on their own.",
    "Jargon Avoidance: Avoid heavy use of irrelevant technical jargon.",
    "Format Appropriateness: Avoid non-continuous formats like lists and tables.",
    "Persuasiveness: Be persuasive, reflecting the tone often found in positive or negative reviews.",
    "Genre Balance: Represent a balanced variety of genres (e.g., fiction, non-fiction, journalism).",
    "Citation Minimization: Avoid being predominantly composed of citations or quotes.",
    "Interactive Media Handling: Exclude interactive media texts unless they provide narrative value.",
    "Structural Cohesion: Be cohesive and well-structured.",
    "Offensive Content Avoidance: Avoid containing hate speech, excessive violence, or other offensive content.",
    "Demographic Inclusivity: Discuss or be relevant to a variety of demographic groups.",
    "Sentiment Extremity: Express strong sentiments, either positive or negative.",
    "Colloquial Language: Mimic spoken language, as often found in movie reviews.",
    "Descriptive Nature: Avoid being purely descriptive and lack subjective opinions.",
    "Historical Context: Include historical references only if they enhance the sentiment or narrative.",
    "Plagiarism Avoidance: Be free from plagiarism.",
    "Domain-Specific Language: Contain relevant film and media terms.",
    "User-Generated Content: Include user-generated content such as blogs and user reviews.",
    "Narrative Emphasis: Be narrative-driven, resembling the storytelling found in reviews.",
    "Error Avoidance: Avoid formatting or data errors.",
    "Topical Relevance: Discuss topics commonly found in movie reviews such as plot, acting, and direction.",
    "Satire Handling: Avoid satire unless it is clearly marked or well-known.",
    "Subject Line Clarity: Have moderate and descriptive subject lines.",
    "Outdated Content Avoidance: Avoid containing outdated societal views or terminologies.",
    "Regional Representation: Represent various English dialects and regional variations.",
    "Emotional Variability: Exhibit a range of emotions from joy to sadness, to anger.",
    "Controversial Topic Inclusion: Include discussions on controversial topics if they enhance sentiment understanding.",
    "Generalization Avoidance: Avoid making broad generalizations without substantiation.",
    "Source Reliability: Be from reliable and reputable sources.",
    "Uniqueness: Be unique with no duplicates in the dataset.",
    "Formality Variance: Include a variety of formality levels, particularly matching the informal style of many movie reviews.",
    "Impactful Sentences: Contain emotionally resonant sentences critical for sentiment analysis.",
    "Engagement: Be engaging and likely to provoke reader reactions.",
    "Visual Storytelling: Include vivid descriptions akin to visual storytelling in movies."
]


def rating_prompt(text_A):
    prompt = f'''
You are a helpful assistant. We are training a language model using the SlimPajama dataset for the IMDB sentiment analysis task. Evaluate the following example from SlimPajama dataset and assign a quality score between 0 and 1 (0 indicates the worst quality, and 1 indicates perfect quality).

Respond only with a single float number.

Example:
{text_A}
'''
    return prompt

def rule_rating_prompt(rule, text_A):
    prompt = f'''
You are a helpful assistant. Evaluate the following example from SlimPajama dataset and assign a quality score between 0 and 1 (0 indicates the worst quality, and 1 indicates perfect quality) according to the provided rule:
{rule}

Respond only with a single float number.

Example:
{text_A}
'''
    return prompt