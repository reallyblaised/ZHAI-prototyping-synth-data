"""
Prompt templates for the food insecurity synthetic dataset generator.

This file contains the prompts used for:
1. Generating knowledge triplets
2. Creating articles from structured truth data
3. Generating non-food-related articles
"""

# Prompt to generate knowledge triplets for a specific category and country
TRIPLETS_GENERATION_PROMPT = """
Generate between 1 and 5 realistic cause-effect relationships (knowledge triplets) about food insecurity 
in {country}, specifically related to {category}. Make sure to randomly select the number of triplets.

Each triplet should have:
- subject: A specific cause or factor
- relation: The type of relationship
- effect: The impact on food security

Format as valid JSON like this:
{{
  "knowledge_triplets": [
    {{
      "subject": "cause or factor",
      "relation": "type of relationship",
      "effect": "effect on food security"
    }},
    ...
  ]
}}

Make the triplets realistic, detailed, and specific to {country} and {category} and geopolitical, social, and economic context of the country in the last 10 years. Reference true events and data.
Output ONLY the JSON/
"""

# Prompt to generate an article from structured truth data
ARTICLE_FROM_TRUTH_PROMPT = """
Write a realistic news article in {language} about food insecurity in {district}, {region}, {country}.

The article should focus on {category} as the main cause of food insecurity.

Include these specific cause-effect relationships in your article:
{triplets_text}

Format the article as a professional news piece with:
- A headline
- Dateline ({news_source}, {date})
- 300-{max_words} words of content
- Include quotes from officials or affected people
- Make the article realistic and factual in tone

Write ONLY the article text as it would appear in a news publication.

Start the article with the special characters: <start_of_article>. End the article with the special characters: <end_of_article>.
"""

# Prompt to generate non-food-related articles
NON_FOOD_ARTICLE_PROMPT = """
Write a realistic news article in {language} about {topic} (not related to food insecurity) in {district}, {region}, {country}.

Format the article as a professional news piece with:
- A headline
- Dateline ({news_source}, {date})
- 300-{max_words} words of content
- Include quotes from relevant people
- Make the article realistic and factual in tone

The article should be completely unrelated to food insecurity, hunger, famine, or agricultural issues.

Write ONLY the article text as it would appear in a news publication.
"""