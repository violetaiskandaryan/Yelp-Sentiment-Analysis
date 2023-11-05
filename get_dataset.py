import json
import pandas as pd

# Load data from JSON files
def load_json(file_path):
    json_objects = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'text' in data:
                    data['text'] = data['text'].replace('\n', ' ')
                json_objects.append(data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return json_objects

# Read business.json and create a DataFrame
business_data = load_json('yelp_academic_dataset_business.json')
business_df = pd.DataFrame(business_data)

# Read review.json and create a DataFrame
review_data = load_json('yelp_academic_dataset_review.json')
review_df = pd.DataFrame(review_data)

# Read user.json and create a DataFrame
user_data = load_json('yelp_academic_dataset_user.json')
user_df = pd.DataFrame(user_data)

# Data cleaning and preprocessing (you can add more steps here)

# Merge data from different DataFrames based on common identifiers
# For example, merge review_df and business_df on 'business_id'
merged_df = pd.merge(review_df, business_df, on='business_id', how='inner')

# Merge additional information from user_df based on 'user_id'
merged_df = pd.merge(merged_df, user_df, on='user_id', how='inner')


merged_df.rename(columns={'review_count_x': 'business_review_count', 'stars_y': 'business_stars',
                          'review_count_y': 'user_review_count', 'useful_y': 'useful', 'funny_y': 'funny', 'cool_y': 'cool', 'stars_x': 'review_stars'}, inplace=True)

merged_df = merged_df.dropna()
# Create the final dataset for star rating prediction
dataset = merged_df[['latitude', 'longitude', 'business_review_count', 'categories', 'hours', 'business_stars',
                     'date', 'user_review_count', 'yelping_since', 'useful', 'funny', 'cool', 'average_stars',
                     'text', 'review_stars']]



# Save the dataset to a CSV file
dataset.to_csv('star_rating_dataset.csv', sep='|', index=False)
