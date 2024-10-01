rule_generation_prompt = """
Generate 50 specific rules for rating data from the training dataset (SlimPajama), in order to select a high-quality subset to train large language models (LLMs) that will improve their performance on Medical tasks. The descriptions of the training data and downstream task are provided below. The rules should focus on various aspects such as data quality, relevance, diversity, and other characteristics that would be beneficial for Medical analysis.

Description of training data:
The SlimPajama dataset is a large-scale dataset. It is designed to be a compact, high-quality dataset curated for pre-training large language models. The dataset includes a diverse range of texts, sourced from various domains such as web pages, books, and academic articles, providing a rich and varied training corpus for developing robust and versatile language models.


Description of downstream task:
The MMLU (Massive Multitask Language Understanding) includes three medical-related subsets: mmlu_college_medicine, mmlu_medical_genetics, and mmlu_professional_medicine. These subsets test a language model's understanding of general medical knowledge, genetic concepts, and advanced professional medical practices, respectively, through multiple-choice questions tailored to assess both foundational and specialized medical expertise.

Requirements for the Rules:
Each rule should be concise and specific.
The rules could be basic text quality rules or task-related quality rules.
The rules should be written in clear, natural language and be easy to understand.
Now, please generate the rules for me in natural language.
"""

rules = [
    "Relevance to Medical Topics: Include texts that contain medical terminology or discuss medical topics.",
    "Exclusion of Non-Medical Content: Exclude texts that do not pertain to health, medicine, or biological sciences.",
    "Clarity of Medical Information: Select texts where medical information is clearly explained and easy to understand.",
    "Accuracy of Medical Content: Ensure texts contain medically accurate information, verified against reputable medical sources.",
    "Diversity of Medical Subfields: Include texts covering a range of medical fields such as genetics, anatomy, pharmacology, and pathology.",
    "Contemporary Relevance: Prefer texts discussing current medical practices and technologies over outdated treatments.",
    "Technical Depth: Include texts with a deep, technical discussion of medical topics suitable for professional medicine.",
    "Exclusion of Ambiguous Content: Avoid texts with ambiguous or unclear medical claims or data.",
    "Citation of Sources: Select texts that cite reputable medical journals or textbooks.",
    "Grammar and Spelling: Ensure texts are free from grammatical errors and spelling mistakes.",
    "Use of Professional Language: Prefer texts that utilize professional medical jargon correctly.",
    "Inclusion of Case Studies: Include texts that discuss medical case studies or clinical trials.",
    "Representation of Rare Diseases: Ensure inclusion of texts discussing rare or less common diseases.",
    "Coverage of Ethical Considerations: Include texts discussing ethical considerations in medical practice and research.",
    "Language Diversity: Include texts in multiple languages relevant to global medical practice.",
    "Patient Education Focus: Include texts aimed at patient education that explain medical conditions and treatments clearly.",
    "Statistical Data Presentation: Prefer texts that present medical data and statistics clearly.",
    "Illustration of Medical Procedures: Include texts with detailed descriptions or illustrations of medical procedures.",
    "Pharmacological Content: Include texts discussing drug mechanisms, interactions, side effects, and benefits.",
    "Genetic Concepts Coverage: Ensure texts covering genetic concepts are detailed and accurate.",
    "Medical Research Updates: Include texts with the latest research findings in the medical field.",
    "Interdisciplinary Approach: Select texts that integrate medical knowledge with other sciences like biochemistry or physics.",
    "Historical Medical Milestones: Include texts discussing historical advancements in medicine.",
    "Medical Guidelines and Protocols: Include texts that detail medical guidelines, protocols, or standard operating procedures.",
    "Interviews with Medical Professionals: Include interviews or discussions with recognized experts in the medical field.",
    "Patient Case Confidentiality: Exclude texts that potentially breach patient confidentiality or privacy.",
    "Texts from Medical Conferences: Include content from recent medical conferences or symposiums.",
    "Exclusion of Pseudoscience: Strictly exclude texts promoting unverified or pseudoscientific claims.",
    "Clinical Pathway Discussions: Include texts discussing clinical decision-making processes and pathways.",
    "Medical Device Descriptions: Include texts that describe the use and innovation of medical devices.",
    "Nutritional and Lifestyle Medicine: Include texts discussing the impact of nutrition and lifestyle on health.",
    "Pediatric Medicine Coverage: Ensure texts covering pediatric medicine are included.",
    "Mental Health Discussions: Include texts that address various aspects of mental health care.",
    "Healthcare Policy Analysis: Include texts analyzing healthcare policies and their implications.",
    "Disease Prevention Focus: Include texts focused on disease prevention strategies and methods.",
    "Surgical Techniques Description: Prefer texts that detail surgical procedures and techniques.",
    "Medical Training and Education: Include texts related to medical training and education methods.",
    "Veterinary Medicine: Include texts on veterinary medicine where relevant to comparative medicine.",
    "Environmental Health Issues: Include texts discussing the impact of environmental factors on health.",
    "Bioinformatics Data Handling: Include texts discussing the handling and analysis of bioinformatics data.",
    "Medical Imaging Techniques: Include texts discussing modern medical imaging techniques and their applications.",
    "Cultural Competence in Healthcare: Include texts that discuss cultural considerations in healthcare provision.",
    "Global Health Challenges: Include texts discussing global health issues and strategies.",
    "Emergency Medicine Protocols: Include texts detailing protocols and procedures in emergency medicine.",
    "Health Insurance Systems: Include texts discussing different health insurance systems and policies.",
    "Medical Ethics Case Studies: Include case studies discussing medical ethics dilemmas and resolutions.",
    "Integrative Medicine Approaches: Include texts on integrative approaches combining traditional and modern medicine.",
    "AI and Machine Learning in Medicine: Include discussions on the application of AI and machine learning in medical contexts.",
    "Telemedicine and Remote Care: Include texts on the advancements and challenges in telemedicine.",
    "Healthcare Accessibility and Equity: Include texts discussing issues of accessibility and equity in healthcare."
]


def rule_rating_prompt(rule, text_A):
    prompt = f'''
You are a helpful assistant. We are training a language model using the SlimPajama dataset to improve performance on Medical tasks. Evaluate the following example from SlimPajama dataset and assign a quality score between 0 and 1 (0 indicates the worst quality, and 1 indicates perfect quality) according to the provided rule:
{rule}

Respond only with a single float number.

Example:
{text_A}
'''
    return prompt