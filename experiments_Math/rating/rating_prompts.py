rule_generation_prompt = """
Generate 50 specific rules for rating data from the training dataset (SlimPajama), in order to select a high-quality subset to train large language models (LLMs) that will improve their performance on Mathematics tasks. The descriptions of the training data and downstream task are provided below. The rules should focus on various aspects such as data quality, relevance, diversity, and other characteristics that would be beneficial for Mathematical reasoning and analysis.

Description of training data:
The SlimPajama dataset is a large-scale dataset. It is designed to be a compact, high-quality dataset curated for pre-training large language models. The dataset includes a diverse range of texts, sourced from various domains such as web pages, books, and academic articles, providing a rich and varied training corpus for developing robust and versatile language models.


Description of downstream task:
The MMLU (Massive Multitask Language Understanding) includes a range of subsets designed to evaluate language models across various academic subjects, including mathematics. The Math subsets specifically assess a model's capability to understand and solve mathematical problems. These are categorized into multiple difficulty levels—from elementary mathematics to college-level topics like abstract algebra. Each subset consists of multiple-choice questions that test different areas of mathematical knowledge, aiming to measure both basic arithmetic skills and more complex mathematical reasoning. This structure allows researchers to gauge a model's proficiency in mathematical logic and its application to solve real-world problems.

Requirements for the Rules:
Each rule should be concise and specific.
The rules could be basic text quality rules or task-relaated quality rules.
The rules should be written in clear, natural language and be easy to understand.
Now, please generate the rules for me in natural language.
"""

rules = [
    "Mathematical Keywords: Prioritize texts containing keywords related to mathematics such as 'algebra', 'calculus', 'geometry', 'equations', 'theorems', etc.",
    "Problem Statements: Include examples that present mathematical problems or puzzles.",
    "Solution Explanations: Select texts that not only present problems but also explain solutions step-by-step.",
    "High-Quality Sources: Favor texts sourced from academic articles, educational websites, and textbooks over general web pages.",
    "Symbolic Representation: Ensure the presence of mathematical symbols and expressions formatted in LaTeX or similar markup languages.",
    "Advanced Topics Coverage: Include texts that cover advanced mathematical topics such as differential equations, statistics, and abstract algebra.",
    "Logical Structuring: Texts should demonstrate clear logical structuring, particularly in argumentation and problem-solving.",
    "Historical Context: Include content that provides historical context or development of mathematical theories and applications.",
    "Data Sets and Examples: Prioritize texts that include real-world data sets or examples where mathematical principles are applied.",
    "No Misconceptions: Exclude texts containing mathematical misconceptions or common errors unless they are being corrected.",
    "Illustrations and Diagrams: Include texts with clear diagrams, graphs, and illustrations that aid mathematical understanding.",
    "Proofs and Theorems: Include detailed explanations of proofs and discussions of theorems.",
    "Mathematics in Technology: Include examples that link mathematics with its applications in technology and engineering.",
    "Interdisciplinary Links: Select texts that illustrate the application of mathematics in other scientific disciplines like physics and chemistry.",
    "Question and Answer Format: Include texts that follow a question and answer format, especially for complex mathematical concepts.",
    "Exclusion of Irrelevant Content: Exclude texts that are primarily non-mathematical in nature, such as pure narrative or opinion pieces.",
    "Mathematical Definitions: Include texts that provide clear definitions of mathematical terms and concepts.",
    "Tutorial Style: Select tutorial-style texts that are aimed at teaching or explaining mathematical concepts.",
    "Accuracy of Content: Exclude any text with factual inaccuracies related to mathematics.",
    "Age-Appropriate Content: Select content that is appropriate for the educational level, from elementary to college-level mathematics.",
    "Challenge Level: Include texts with varying levels of difficulty to ensure a range of challenges in problem-solving.",
    "Language Clarity: Ensure the text uses clear and precise language appropriate for teaching or explaining mathematics.",
    "Cultural Diversity: Include mathematical content from diverse cultural backgrounds to promote inclusivity.",
    "Recency of Content: Prioritize recent texts that reflect the current state of mathematical education and theory.",
    "Real-World Applications: Select texts that discuss the application of mathematical concepts in real-world scenarios.",
    "Peer-Reviewed Sources: Favor texts extracted from peer-reviewed academic journals and conferences.",
    "Multiple Perspectives: Include texts that present multiple perspectives or methods for solving a single mathematical problem.",
    "Step-by-Step Guides: Prioritize texts that provide step-by-step guides to solving mathematical problems.",
    "Integration of Tools: Include texts that discuss or utilize mathematical tools and software.",
    "Variety of Formats: Include a variety of text formats such as articles, essays, and problem sets.",
    "Consistency in Terminology: Ensure consistency in mathematical terminology across the selected texts.",
    "Explanatory Footnotes: Include texts that make use of footnotes or side-notes to explain complex terms or provide additional context.",
    "Interactive Elements: Select texts that include or suggest interactive elements like quizzes or interactive diagrams.",
    "Avoid Redundancy: Avoid texts that are redundant in content, especially if they do not add new information or perspective.",
    "Mathematical Puzzles: Include texts that feature mathematical puzzles and games to enhance problem-solving skills.",
    "Comparative Analyses: Select texts that involve comparative analyses of different mathematical methods or theories.",
    "Language Models and Mathematics: Include texts discussing the intersection of language processing models and mathematics.",
    "Excerpts from Lectures: Include transcribed excerpts from academic lectures on mathematics.",
    "Mathematical Narratives: Include narratives that weave mathematical concepts into broader storylines or real-life applications.",
    "Authoritative Authors: Prioritize texts authored by well-regarded mathematicians or educators.",
    "Exclusion of Vague Language: Avoid texts that use vague or ambiguous language when explaining mathematical concepts.",
    "Feedback Loops: Include texts that describe the importance of feedback loops in mathematical learning.",
    "Error Analysis: Include texts that focus on error analysis in mathematical calculations or theories.",
    "Cross-Referencing: Favor texts that cross-reference other works or theories effectively.",
    "Mathematical Software Tutorials: Include tutorials or guides on using mathematical software.",
    "Engagement Metrics: Favor texts that have historically engaged readers or viewers, indicating quality and interest.",
    "Student Contributions: Include texts written by students, which can provide fresh perspectives and innovative approaches.",
    "Reviews and Critiques: Select texts that review or critique mathematical theories or textbooks.",
    "Accessibility Features: Include texts that are accessible to people with disabilities, such as those formatted for screen readers.",
    "Alignment with Curriculum: Ensure that the content aligns well with standard mathematical curriculums at various educational levels."
]



def rule_rating_prompt(rule, text_A):
    prompt = f'''
You are a helpful assistant. We are training a language model using the SlimPajama dataset to improve performance on Mathematics reasoning task. Evaluate the following example from SlimPajama dataset and assign a quality score between 0 and 1 (0 indicates the worst quality, and 1 indicates perfect quality) according to the provided rule:
{rule}

Respond only with a single float number.

Example:
{text_A}
'''
    return prompt