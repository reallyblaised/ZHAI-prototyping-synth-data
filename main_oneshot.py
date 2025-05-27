#!/usr/bin/env python3
"""
Food Insecurity Synthetic Dataset Generator - Truth-First Approach

This script generates a synthetic dataset of news articles about food insecurity,
using a truth-first approach where structured data is generated before the article.
This ensures perfect ground truth data and guaranteed inclusion of desired elements.
"""

import os
import json
import yaml
import random
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from enum import Enum
from pydantic import BaseModel, Field

# Import prompt templates
from proposer_prompt import (
    TRIPLETS_GENERATION_PROMPT, 
    ARTICLE_FROM_TRUTH_PROMPT, 
    NON_FOOD_ARTICLE_PROMPT
)

# Define the schema for food insecurity knowledge triplets
class FoodInsecurityCategory(str, Enum):
    conflict = "conflict and violence"
    political = "political instability"
    humanitarian = "humanitarian aid"
    economic = "economic issues"
    production = "production shortage"
    weather = "weather conditions"
    crisis = "food crisis"
    land = "land-related issues"
    pests = "pests and disease"
    displacement = "forced displacement"
    environmental = "environmental issues"
    other = "other"

# Define reputable news sources by language
NEWS_SOURCES = {
    "EN": [
        "BBC", "Financial Times", "Wall Street Journal", "CNN", "The Guardian",
        "The Washington Post", "The Times of Israel", "Premium Times", "Mail & Guardian",
        "The Nation", "Daily Nation", "The National"
    ],
    "FR": [
        "Le Monde", "Le Figaro", "L'Express", "Le Point", "Libération"
    ],
    "AR": [
        "Al Jazeera", "Arab News", "Asharq Al-Awsat", "Al-Ahram", "Al-Arabiya"
    ]
}

class KnowledgeTriplet(BaseModel):
    subject: str = Field(..., description="A specific cause or factor related to food insecurity")
    relation: str = Field(..., description="The type of relationship between the cause and effect")
    effect: str = Field(..., description="The impact on food security")

class FoodInsecurityKnowledge(BaseModel):
    knowledge_triplets: List[KnowledgeTriplet] = Field(
        ..., 
        description="List of knowledge triplets describing causes and effects of food insecurity",
        min_items=1,
        max_items=5
    )

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def init_model(config: Dict[str, Any]) -> LLM:
    """Initialize vLLM model with proper configuration for Qwen3."""
    print(f"Initializing vLLM with model: {config['model']['name']}")
    
    # Initialize the model with optimal settings for Qwen3
    model = LLM(
        model=config["model"]["name"],
        tensor_parallel_size=config.get("model", {}).get("tensor_parallel_size", 2),
        trust_remote_code=True,  # Required for Qwen models
        enable_reasoning=False,   # Required for Qwen3 thinking mode
        # reasoning_parser="deepseek_r1",  # Required parser for Qwen3
        download_dir=config.get("model", {}).get("download_dir", "/ceph/submit/data/user/b/blaised/cache/vllm")
    )
    
    print(f"Model {config['model']['name']} loaded successfully")
    return model


def load_admin_locations(file_path: str = "data/admin_lvl1.json") -> Dict[str, List[Dict]]:
    """
    Load administrative locations from JSON file and organize by country.
    
    Args:
        file_path: Path to the admin_lvl1.json file
        
    Returns:
        Dictionary mapping country names to lists of admin level 1 locations
    """
    try:
        with open(file_path, "r") as f:
            locations = json.load(f)
        
        # Organize locations by country name (location_name)
        country_to_locations = {}
        for location in locations:
            country = location.get("location_name")
            if country:
                if country not in country_to_locations:
                    country_to_locations[country] = []
                country_to_locations[country].append(location)
        
        if not country_to_locations:
            raise ValueError("No valid country locations found in the input data")
            
        return country_to_locations
    except Exception as e:
        print(f"Error loading admin locations: {str(e)}")
        return {}


def select_real_location(config: Dict[str, Any], admin_locations: Dict[str, List[Dict]]) -> Tuple[str, str, Optional[str]]:
    """
    Select a random country from config and a random admin level 1 location from that country.
    
    Args:
        config: Configuration dictionary
        admin_locations: Dictionary mapping country names to lists of admin level 1 locations
        
    Returns:
        Tuple of (country name, region name, district name or None if no match)
    """
    # Get all available countries from config
    all_countries = []
    for region_data in config["regions"]:
        all_countries.extend(region_data["countries"])
    
    # Shuffle the list of countries to pick randomly
    random.shuffle(all_countries)
    
    # Find a country that exists in both config and admin_locations
    found_country = None
    found_region = None
    
    for country in all_countries:
        # Look for exact match
        if country in admin_locations and admin_locations[country]:
            found_country = country
            # Get the region from config that contains this country
            for region_data in config["regions"]:
                if country in region_data["countries"]:
                    found_region = region_data["name"]
                    break
            break
            
        # Try alternative matches (e.g. "Syrian Arab Republic" for "Syria")
        for admin_country in admin_locations:
            if country.lower() in admin_country.lower() or admin_country.lower() in country.lower():
                found_country = admin_country
                # Get the region from config that contains this country
                for region_data in config["regions"]:
                    if country in region_data["countries"]:
                        found_region = region_data["name"]
                        break
                break
    
    # If no match found, return None for district
    if not found_country or not found_region:
        print(f"Warning: No admin location found for any country in config. Using fallback.")
        region_data = random.choice(config["regions"])
        found_region = region_data["name"]
        found_country = random.choice(region_data["countries"])
        return found_country, found_region, None
    
    # Select a random admin level 1 location from the chosen country
    district_data = random.choice(admin_locations[found_country])
    district_name = district_data.get("name")
    
    return found_country, found_region, district_name


def generate_structured_data(config: Dict[str, Any], language: str, admin_locations: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    Generate structured food insecurity data as the ground truth.
    
    This is the first step in the truth-first approach, establishing
    the metadata and category before generating content.
    
    Args:
        config: Configuration dictionary
        language: Target language for the article
        admin_locations: Dictionary mapping country names to lists of admin level 1 locations
        
    Returns:
        Dictionary containing structured data (metadata, category, etc.)
    """
    # Select random region, country, and district using real admin locations
    country, region, district = select_real_location(config, admin_locations)
    
    # If no district was found, use a fallback
    if district is None:
        district = f"District-{random.randint(1, 10)}"
    
    # Generate random publication date within last 2 years
    days_ago = random.randint(0, 365*10)  # 10 years
    pub_date = datetime.now() - timedelta(days=days_ago)
    
    # Set realistic business hours (8 AM to 8 PM)
    hour = random.randint(8, 20)
    minute = random.randint(0, 59)
    pub_date = pub_date.replace(hour=hour, minute=minute, second=random.randint(0, 59), microsecond=0)
    
    # Generate a random news source from the appropriate language list
    base_sources = NEWS_SOURCES.get(language, NEWS_SOURCES["EN"])  # Fallback to English if language not found
    news_source = f"Synthetic {random.choice(base_sources)}"
    
    # Select category
    category = random.choice(config["categories"])
    if category == "other":
        category = "other food insecurity issue"
    
    # Generate article ID
    article_id = str(uuid.uuid4())
    
    # Create structure to hold truth data
    structured_data = {
        "article_id": article_id,
        "language": language,
        "metadata": {
            "publication_date": pub_date,
            "country": country,
            "region": region,
            "district": district,
            "source": news_source
        },
        "food_insecurity": {
            "is_relevant": True,
            "category": category,
        },
        "knowledge_graph": {
            "triplets": []
        }
    }
    
    return structured_data


def generate_knowledge_triplets(model: LLM, config: Dict[str, Any], structured_data: Dict[str, Any]) -> Dict[str, Any]:
    """ 
    Generate knowledge triplets for the given category and context using guided JSON.
    
    This is the second step in our truth-first approach, creating
    the causal relationships that will be incorporated into the article.
    
    Args:
        model: Initialized language model
        structured_data: Dictionary containing structured data
        
    Returns:
        Updated structured data with knowledge triplets
    """
    # Get Pydantic schema for guided JSON generation
    json_schema = FoodInsecurityKnowledge.model_json_schema()
    
    # Create an enhanced prompt that includes the schema
    enhanced_prompt = TRIPLETS_GENERATION_PROMPT.format(
        country=structured_data['metadata']['country'],
        category=structured_data['food_insecurity']['category']
    )
    
    # Add JSON schema information to the prompt
    schema_description = f"""
Please generate the response according to this exact JSON schema:
```json
{json.dumps(json_schema, indent=2)}
```

The response should be a valid JSON object with a 'knowledge_triplets' array containing between 1 and 5 triplets,
each with 'subject', 'relation', and 'effect' fields.
"""
    
    enhanced_prompt = enhanced_prompt + "\n" + schema_description # NOTE: `vllm` tip: "in the prompt the JSON schema and how the fields should be populated"
    
    # Sample parameters for knowledge triplet generation
    guided_decoding_params_json = GuidedDecodingParams(json=json_schema)
    sampling_params = SamplingParams(
        temperature=config.get("model", {}).get("temperature", 0.6), # https://huggingface.co/Qwen/Qwen3-8B/blob/main/generation_config.json
        max_tokens=config.get("model", {}).get("max_tokens", 2048),
        guided_decoding=guided_decoding_params_json
    )

    try:
        # Generate triplets
        outputs = model.generate([enhanced_prompt], sampling_params)
        response = outputs[0].outputs[0]
        response_text = response.text.strip()
        
        # Extract JSON from the response
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        
        if json_start >= 0 and json_end > 0:
            json_str = response_text[json_start:json_end]
            try:
                # Parse the JSON and validate against our schema
                triplets_data = json.loads(json_str)
                validated_data = FoodInsecurityKnowledge.model_validate(triplets_data)
                
                # Update the structured data with validated triplets
                structured_data["knowledge_graph"]["triplets"] = validated_data.knowledge_triplets
                print(f"Generated {len(validated_data.knowledge_triplets)} knowledge triplets")
            except Exception as parse_error:
                raise Exception(f"Error parsing response as valid JSON: {parse_error}")
        else:
            raise Exception("No valid JSON found in the response")
            
    except Exception as e:
        raise Exception(f"Error generating knowledge triplets: {str(e)}")


    return structured_data


def generate_article_from_truth(model: LLM, config: Dict[str, Any], structured_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate an article based on the structured truth data.
    
    This is the final step in our truth-first approach, creating content
    that explicitly incorporates the predefined metadata and knowledge triplets.
    
    Args:
        model: Initialized language model
        structured_data: Dictionary containing structured data with triplets
        config: Configuration dictionary
        
    Returns:
        Updated structured data with generated article text
    """
    # Extract data for the prompt
    language = structured_data["language"]
    country = structured_data["metadata"]["country"]
    region = structured_data["metadata"]["region"]
    district = structured_data["metadata"]["district"]
    category = structured_data["food_insecurity"]["category"]
    news_source = structured_data["metadata"]["source"]
    pub_date = structured_data["metadata"]["publication_date"].strftime("%Y-%m-%d")
    
    # Format knowledge triplets for inclusion in the prompt
    triplets_text = ""
    for triplet in structured_data["knowledge_graph"]["triplets"]:
        if hasattr(triplet, "subject"):  # Pydantic model
            triplets_text += f"- {triplet.subject} {triplet.relation} {triplet.effect}\n"
        else:  # Dictionary
            triplets_text += f"- {triplet['subject']} {triplet['relation']} {triplet['effect']}\n"
    
    # Create a prompt that incorporates all the structured data
    prompt = ARTICLE_FROM_TRUTH_PROMPT.format(
        language=language,
        country=country,
        region=region,
        district=district,
        category=category,
        news_source=news_source,
        date=pub_date,
        triplets_text=triplets_text,
        max_words=config["dataset"]["max_words"]
    )

    # Generation parameters
    sampling_params = SamplingParams(
        temperature=config.get("model", {}).get("temperature", 0.6),
        max_tokens=config.get("model", {}).get("max_tokens", 2048),
    )
    
    try:
    # Generate article
        outputs = model.generate([prompt], sampling_params)
        
        # extract the article text between the start and end markers (set in the prompt), avoiding the reasoning trace
        raw_text = outputs[0].outputs[0].text.strip()

        start_marker = "<<!--START OF ARTICLE-->>"
        end_marker = "<<!--END OF ARTICLE-->>"
        
        start_idx = raw_text.find(start_marker)
        end_idx = raw_text.find(end_marker)
        
        if start_idx >= 0 and end_idx >= 0:
            article_text = raw_text[start_idx + len(start_marker):end_idx].strip()
        else:
            article_text = raw_text

        # Update structured data with the generated article
        structured_data["text"] = article_text
        structured_data["word_count"] = len(article_text.split())
        
    except Exception as e:
        print(f"Error generating article: {str(e)}")
        structured_data["text"] = f"Error generating article: {str(e)}"
        structured_data["word_count"] = 0
    
    return structured_data


def generate_non_food_article(model: LLM, config: Dict[str, Any], language: str, admin_locations: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    Generate a non-food-related article for the noise portion of the dataset.
    
    Args:
        model: Initialized language model
        config: Configuration dictionary
        language: Target language for the article
        admin_locations: Dictionary mapping country names to lists of admin level 1 locations
        
    Returns:
        Dictionary containing article data
    """
    # Select random region, country, and district using real admin locations
    country, region, district = select_real_location(config, admin_locations)
    
    # If no district was found, use a fallback
    if district is None:
        district = f"District-{random.randint(1, 10)}"
    
    # Generate random publication date within last 2 years
    days_ago = random.randint(0, 730)  # 2 years
    pub_date = datetime.now() - timedelta(days=days_ago)
    
    # Set realistic business hours (8 AM to 8 PM)
    hour = random.randint(8, 20)
    minute = random.randint(0, 59)
    pub_date = pub_date.replace(hour=hour, minute=minute, second=random.randint(0, 59), microsecond=0)
    
    # Generate a random news source from the appropriate language list
    base_sources = NEWS_SOURCES.get(language, NEWS_SOURCES["EN"])  # Fallback to English if language not found
    news_source = f"Synthetic {random.choice(base_sources)}"
    
    # Generate article ID
    article_id = str(uuid.uuid4())
    
    # Random topic for non-food article
    topics = ["politics", "sports", "culture", "technology", "business", "education", "health", "entertainment"]
    topic = random.choice(topics)
    
    # Create prompt for non-food article
    prompt = NON_FOOD_ARTICLE_PROMPT.format(
        language=language,
        topic=topic,
        country=country,
        region=region,
        district=district,
        news_source=news_source,
        date=pub_date.strftime("%Y-%m-%d"),
        max_words=config["dataset"]["max_words"]
    )
    
    # Generation parameters
    sampling_params = SamplingParams(
        temperature=config.get("model", {}).get("temperature", 0.6), # https://huggingface.co/Qwen/Qwen3-8B/blob/main/generation_config.json
        top_p=config.get("model", {}).get("top_p", 0.95),
        max_tokens=config.get("model", {}).get("max_tokens", 2048),
    )
    
    try:
        # Generate article
        outputs = model.generate([prompt], sampling_params)
        
        # extract the article text between the start and end markers (set in the prompt), avoiding the reasoning trace
        raw_text = outputs[0].outputs[0].text.strip()
        start_marker = "<start_of_article>"
        end_marker = "<end_of_article>"
        
        start_idx = raw_text.find(start_marker)
        end_idx = raw_text.find(end_marker)
        
        if start_idx >= 0 and end_idx >= 0:
            article_text = raw_text[start_idx + len(start_marker):end_idx].strip()
        else:
            article_text = raw_text
        
        # Create article data structure
        article_data = {
            "article_id": article_id,
            "text": article_text,
            "language": language,
            "word_count": len(article_text.split()),
            "metadata": {
                "publication_date": pub_date,
                "country": country,
                "region": region,
                "district": district,
                "source": news_source
            },
            "food_insecurity": {
                "is_relevant": False,
                "category": "NA",
            },
            "knowledge_graph": {
                "triplets": []
            }
        }
        
        return article_data
   
    except Exception as e:
        print(f"Error generating non-food article: {str(e)}")
        # Return a minimal valid structure on error
        return {
            "article_id": article_id,
            "text": f"Error generating article: {str(e)}",
            "language": language,
            "word_count": 0,
            "metadata": {
                "publication_date": pub_date,
                "country": country,
                "region": region,
                "district": district,
                "source": news_source
            },
            "food_insecurity": {
                "is_relevant": False,
                "category": "NA",
            },
            "knowledge_graph": {
                "triplets": []
            }
        }


def create_parquet_schema():
    """Create the PyArrow schema for the Parquet file."""
    return pa.schema([
        pa.field('article_id', pa.string()),
        pa.field('text', pa.string()),
        pa.field('language', pa.string()),
        pa.field('word_count', pa.int32()),
        pa.field('metadata', pa.struct([
            pa.field('publication_date', pa.timestamp('s')),
            pa.field('country', pa.string()),
            pa.field('region', pa.string()),
            pa.field('district', pa.string()),
            pa.field('source', pa.string())
        ])),
        pa.field('food_insecurity', pa.struct([
            pa.field('is_relevant', pa.bool_()),
            pa.field('category', pa.string()),
        ])),
        pa.field('knowledge_graph', pa.struct([
            pa.field('triplets', pa.list_(pa.struct([
                pa.field('subject', pa.string()),
                pa.field('relation', pa.string()),
                pa.field('effect', pa.string())
            ])))
        ]))
    ])


def save_to_parquet(dataset: List[Dict], output_path: str = "data/food_insecurity_articles.parquet"):
    """Save dataset to Parquet file."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert datetime objects to timestamp objects for Parquet
    # And convert any Pydantic models to dictionaries
    processed_dataset = []
    for article in dataset:
        # Deep copy to avoid modifying the original
        article_copy = article.copy()
        
        # Convert datetime
        article_copy["metadata"]["publication_date"] = article["metadata"]["publication_date"].replace(tzinfo=None)
        
        # Convert Pydantic triplets to dictionaries if needed
        if "knowledge_graph" in article_copy and "triplets" in article_copy["knowledge_graph"]:
            triplets = []
            for triplet in article_copy["knowledge_graph"]["triplets"]:
                if hasattr(triplet, "model_dump"):  # Check if it's a Pydantic model
                    triplets.append(triplet.model_dump())
                elif isinstance(triplet, dict):
                    triplets.append(triplet)
                else:
                    # Try to convert to dict as fallback
                    triplets.append({
                        "subject": str(getattr(triplet, "subject", "Unknown cause")),
                        "relation": str(getattr(triplet, "relation", "leads to")),
                        "effect": str(getattr(triplet, "effect", "Unknown effect"))
                    })
            article_copy["knowledge_graph"]["triplets"] = triplets
        
        processed_dataset.append(article_copy)
    
    # Create PyArrow table
    schema = create_parquet_schema()
    try:
        table = pa.Table.from_pylist(processed_dataset, schema=schema)
        
        # Save to Parquet file
        pq.write_table(table, output_path)
        print(f"Dataset saved to: {output_path}")
    except Exception as e:
        print(f"Error saving to Parquet: {str(e)}")
        # Save as JSON as fallback
        fallback_path = output_path.replace(".parquet", ".json")
        with open(fallback_path, "w") as f:
            json.dump(processed_dataset, f, default=str)
        print(f"Dataset saved as JSON fallback to: {fallback_path}")


def print_dataset_stats(dataset: List[Dict]):
    """Print statistics about the generated dataset."""
    food_articles = sum(1 for a in dataset if a["food_insecurity"]["is_relevant"])
    
    # Count articles by language
    lang_counts = {}
    for article in dataset:
        lang = article["language"]
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    
    # Count articles by category
    cat_counts = {}
    for article in dataset:
        if article["food_insecurity"]["is_relevant"]:
            cat = article["food_insecurity"]["category"]
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
    
    # Average triplets per food article
    triplet_counts = [len(a["knowledge_graph"]["triplets"]) for a in dataset if a["food_insecurity"]["is_relevant"]]
    avg_triplets = sum(triplet_counts) / max(1, len(triplet_counts))
    
    # Average word count
    avg_words = sum(a["word_count"] for a in dataset) / len(dataset)
    
    print("\n===== DATASET STATISTICS =====")
    print(f"Total articles: {len(dataset)}")
    print(f"Food insecurity articles: {food_articles} ({food_articles/len(dataset)*100:.1f}%)")
    print(f"Average word count: {avg_words:.1f}")
    print(f"Articles by language: {lang_counts}")
    print(f"Food insecurity categories: {cat_counts}")
    print(f"Average triplets per food article: {avg_triplets:.2f}")
    print("=============================\n")


def main():
    """
    Generate the food insecurity synthetic dataset using the truth-first approach.
    
    This follows a three-step process for food insecurity articles:
    1. Generate structured data (metadata, categories)
    2. Generate knowledge triplets about food insecurity
    3. Generate articles that incorporate the structured data
    
    Non-food articles are generated in a simpler one-step process.
    """
    start_time = time.time()
    
    # Load configuration
    config = load_config()
    output_path = "scratch/food_insecurity_articles.parquet"
    
    # Load administrative locations
    admin_locations = load_admin_locations()
    print(f"Loaded administrative locations for {len(admin_locations)} countries")
    
    # Initialize model
    model = init_model(config)

    # Calculate distribution of articles
    total_articles = config["dataset"]["size"]
    languages = config["dataset"]["languages"]
    articles_per_language = total_articles // len(languages)
    noise_percentage = config["dataset"]["noise_percentage"] / 100.0
    
    noise_articles_per_language = int(articles_per_language * noise_percentage)
    food_articles_per_language = articles_per_language - noise_articles_per_language
    
    # Generate articles
    dataset = []
    
    # Generate food insecurity articles using truth-first approach
    for language in languages:
        print(f"\nGenerating {food_articles_per_language} food insecurity articles in {language}...")
        for _ in tqdm(range(food_articles_per_language)):
            # Step 1: Generate structured data first
            structured_data = generate_structured_data(config, language, admin_locations)

            # Step 2: Generate knowledge triplets
            structured_data = generate_knowledge_triplets(model, config, structured_data)

            # Step 3: Generate article from structured data
            article_data = generate_article_from_truth(model, config, structured_data)
            
            dataset.append(article_data)
        
        # Generate non-food articles
        print(f"Generating {noise_articles_per_language} non-food articles in {language}...")
        for _ in tqdm(range(noise_articles_per_language)):
            article = generate_non_food_article(model, config, language, admin_locations)
            dataset.append(article)
    
    # Print statistics about the generated dataset
    print_dataset_stats(dataset)
    
    # Save to Parquet file
    save_to_parquet(dataset, output_path)
    
    # Calculate total time
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    
    print(f"Dataset generation complete in {int(minutes)}m {int(seconds)}s")


if __name__ == "__main__":
    # Uncomment to run the main dataset generation
    main()