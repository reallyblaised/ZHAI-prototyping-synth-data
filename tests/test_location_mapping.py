import os
import random
import pytest
import json
from main import load_admin_locations, select_real_location, load_config


def test_load_admin_locations(admin_data_path):
    """Test that admin locations are loaded correctly."""
    # Ensure the data file exists
    assert os.path.exists(admin_data_path), f"Test data file {admin_data_path} does not exist"
    
    # Load locations
    locations = load_admin_locations(admin_data_path)
    
    # Verify we have data
    assert locations, "No locations were loaded"
    assert isinstance(locations, dict), "Locations should be a dictionary"
    assert len(locations) > 0, "No country entries found in locations"
    
    # Verify structure of first entry
    first_country = next(iter(locations))
    assert first_country, "No country name found"
    assert isinstance(locations[first_country], list), f"Locations for {first_country} should be a list"
    assert len(locations[first_country]) > 0, f"No locations found for {first_country}"
    
    # Verify a location entry has the expected structure
    first_location = locations[first_country][0]
    assert "name" in first_location, "Location missing 'name' field"
    assert "location_name" in first_location, "Location missing 'location_name' field"
    assert first_location["location_name"] == first_country, "Location country mismatch"


def test_specific_countries(config_path, admin_data_path):
    """Test that specific countries from config.yaml can be found in the admin locations."""
    # Load config and admin locations
    config = load_config(config_path)
    locations = load_admin_locations(admin_data_path)
    
    # Create a set of all countries in the admin locations
    admin_countries = set(locations.keys())
    
    # Create a list of test countries from config
    config_countries = []
    for region in config["regions"]:
        config_countries.extend(region["countries"])
    
    # Test finding each country with exact matches and partial matches
    found_countries = []
    not_found_countries = []
    
    for country in config_countries:
        # Check for exact match
        if country in admin_countries:
            found_countries.append((country, country, "exact"))
            continue
            
        # Check for partial match
        found = False
        for admin_country in admin_countries:
            if country.lower() in admin_country.lower() or admin_country.lower() in country.lower():
                found_countries.append((country, admin_country, "partial"))
                found = True
                break
                
        if not found:
            not_found_countries.append(country)
    
    # Print results for debugging
    print(f"\nFound matches for {len(found_countries)} countries:")
    for config_country, admin_country, match_type in found_countries[:5]:  # Show first 5
        print(f"  Config: {config_country} -> Admin: {admin_country} ({match_type})")
    
    if not_found_countries:
        print(f"\nCould not find matches for {len(not_found_countries)} countries:")
        print(f"  {not_found_countries}")
    
    # We should at least find some matches
    assert len(found_countries) > 0, "No countries from config found in admin locations"


def test_select_real_location(config_path, admin_data_path):
    """Test the select_real_location function."""
    # Load config and admin locations
    config = load_config(config_path)
    locations = load_admin_locations(admin_data_path)
    
    # Test multiple random selections
    for _ in range(5):
        country, region, district = select_real_location(config, locations)
        
        # Verify the results
        assert country, "No country selected"
        assert region, "No region selected"
        
        # District might be None if no match found, but if not None it should be a string
        if district is not None:
            assert isinstance(district, str), "District should be a string when present"
            
            # Verify this district actually exists in the admin data for this country
            country_districts = [loc["name"] for loc in locations.get(country, [])]
            assert district in country_districts, f"District {district} not found in {country}"


def test_syria_example(admin_data_path):
    """Test the specific Syria example mentioned in the requirements."""
    # Load admin locations
    locations = load_admin_locations(admin_data_path)
    
    # Check for Syrian Arab Republic
    assert "Syrian Arab Republic" in locations, "Syrian Arab Republic not found in admin locations"
    
    # Get districts for Syria
    syria_districts = locations["Syrian Arab Republic"]
    
    # Extract district names
    district_names = [district["name"] for district in syria_districts]
    
    # Check that Idleb and Lattakia are in the districts
    assert "Idleb" in district_names, "Idleb not found in Syrian districts"
    assert "Lattakia" in district_names, "Lattakia not found in Syrian districts"
    
    # Test selection of a random district
    random_district = random.choice(syria_districts)
    assert random_district["name"] in district_names, f"Randomly selected district {random_district['name']} not in Syrian districts"
    
    # Verify it has the expected structure
    assert random_district["location_name"] == "Syrian Arab Republic", "District's location_name doesn't match Syria"
    assert "code" in random_district, "District missing code field" 