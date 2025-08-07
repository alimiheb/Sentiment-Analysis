import json
from datetime import datetime
import os

def merge_review_files():
    """
    Merge Glassdoor.json and Indeed.json files into a unified format
    """
    
    # File paths
    glassdoor_file = "Glassdoor.json"
    indeed_file = "Indeed.json"
    merged_file = "merged_reviews.json"
    
    # Check if files exist
    if not os.path.exists(glassdoor_file):
        print(f"Error: {glassdoor_file} not found")
        return
    
    if not os.path.exists(indeed_file):
        print(f"Error: {indeed_file} not found")
        return
    
    # Load data from both files
    try:
        with open(glassdoor_file, 'r', encoding='utf-8') as f:
            glassdoor_data = json.load(f)
        print(f"Loaded {len(glassdoor_data)} reviews from Glassdoor")
        
        with open(indeed_file, 'r', encoding='utf-8') as f:
            indeed_data = json.load(f)
        print(f"Loaded {len(indeed_data)} reviews from Indeed")
        
    except json.JSONDecodeError as e:
        print(f"Error reading JSON files: {e}")
        return
    except Exception as e:
        print(f"Error loading files: {e}")
        return
    
    # Merge data into unified format
    merged_reviews = []
    
    # Process Glassdoor reviews
    for review in glassdoor_data:
        unified_review = {
            "source": "glassdoor",
            "review_id": review.get("review_id"),
            "title": review.get("summary", ""),
            "text": {
                "pros": review.get("pros", ""),
                "cons": review.get("cons", ""),
                "advice": review.get("advice", "")
            },
            "overall_rating": review.get("rating_overall", 0),
            "detailed_ratings": {
                "career_opportunities": review.get("rating_career_opportunities", 0),
                "compensation_and_benefits": review.get("rating_compensation_and_benefits", 0),
                "culture_and_values": review.get("rating_culture_and_values", 0),
                "diversity_and_inclusion": review.get("rating_diversity_and_inclusion", 0),
                "senior_leadership": review.get("rating_senior_leadership", 0),
                "work_life_balance": review.get("rating_work_life_balance", 0)
            },
            "job_title": review.get("job_title", ""),
            "location": review.get("location", ""),
            "current_employee": review.get("is_current_job", None),
            "employment_length": review.get("length_of_employment", 0),
            "submission_date": review.get("review_date_time", ""),
            "recommend_to_friend": review.get("rating_recommend_to_friend", ""),
            "ceo_approval": review.get("rating_ceo", ""),
            "business_outlook": review.get("rating_business_outlook", ""),
            "company": review.get("employer_short_name", "Talan"),
            "helpful_count": review.get("count_helpful", 0),
            "has_employer_response": review.get("has_employer_response", False)
        }
        merged_reviews.append(unified_review)
    
    # Process Indeed reviews
    for review in indeed_data:
        # Extract text content
        review_text = ""
        if isinstance(review.get("text"), dict):
            review_text = review["text"].get("text", "")
        elif isinstance(review.get("text"), str):
            review_text = review["text"]
        
        # Extract title
        title_text = ""
        if isinstance(review.get("title"), dict):
            title_text = review["title"].get("text", "")
        elif isinstance(review.get("title"), str):
            title_text = review["title"]
        
        # Extract pros and cons if available
        pros_text = review.get("pros", {})
        cons_text = review.get("cons", {})
        if isinstance(pros_text, dict):
            pros_text = pros_text.get("text", "")
        if isinstance(cons_text, dict):
            cons_text = cons_text.get("text", "")
        
        unified_review = {
            "source": "indeed",
            "review_id": review.get("encryptedReviewId", ""),
            "title": title_text,
            "text": {
                "full_review": review_text,
                "pros": pros_text,
                "cons": cons_text,
                "advice": ""
            },
            "overall_rating": review.get("overallRating", 0),
            "detailed_ratings": {
                "compensation_and_benefits": review.get("compensationAndBenefitsRating", {}).get("rating", 0),
                "culture_and_values": review.get("cultureAndValuesRating", {}).get("rating", 0),
                "job_security_and_advancement": review.get("jobSecurityAndAdvancementRating", {}).get("rating", 0),
                "management": review.get("managementRating", {}).get("rating", 0),
                "work_life_balance": review.get("workAndLifeBalanceRating", {}).get("rating", 0)
            },
            "job_title": review.get("jobTitle", ""),
            "location": review.get("location", ""),
            "current_employee": review.get("currentEmployee", None),
            "employment_length": None,  # Not available in Indeed data
            "submission_date": review.get("submissionDate", ""),
            "recommend_to_friend": "",  # Not directly available
            "ceo_approval": "",  # Not available
            "business_outlook": "",  # Not available
            "company": "Talan",  # Assuming from context
            "helpful_count": review.get("helpful", 0),
            "unhelpful_count": review.get("unhelpful", 0),
            "review_status": review.get("reviewStatus", ""),
            "country": review.get("country", ""),
            "indexable": review.get("indexable", False)
        }
        merged_reviews.append(unified_review)
    
    # Add metadata
    merged_data = {
        "metadata": {
            "total_reviews": len(merged_reviews),
            "glassdoor_reviews": len(glassdoor_data),
            "indeed_reviews": len(indeed_data),
            "merge_timestamp": datetime.now().isoformat(),
            "description": "Merged employee reviews from Glassdoor and Indeed for Talan company"
        },
        "reviews": merged_reviews
    }
    
    # Save merged data
    try:
        with open(merged_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nMerge completed successfully!")
        print(f"Total reviews merged: {len(merged_reviews)}")
        print(f"Output file: {merged_file}")
        print(f"File size: {os.path.getsize(merged_file)} bytes")
        
    except Exception as e:
        print(f"Error saving merged file: {e}")
        return
    
    # Print summary statistics
    print("\n--- Merge Summary ---")
    print(f"Glassdoor reviews: {len(glassdoor_data)}")
    print(f"Indeed reviews: {len(indeed_data)}")
    print(f"Total merged reviews: {len(merged_reviews)}")
    
    # Count ratings distribution
    ratings_count = {}
    for review in merged_reviews:
        rating = review["overall_rating"]
        ratings_count[rating] = ratings_count.get(rating, 0) + 1
    
    print("\nRatings distribution:")
    for rating in sorted(ratings_count.keys()):
        print(f"  {rating} stars: {ratings_count[rating]} reviews")

if __name__ == "__main__":
    merge_review_files()
