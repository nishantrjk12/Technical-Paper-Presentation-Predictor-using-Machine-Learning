
# Step 1: Generate a synthetic dataset for the Technical Paper Presentation Predictor
# This script creates a CSV file with realistic student presentation data.
# Dataset is balanced across Low / Medium / High categories so the model learns all 3 classes.

import pandas as pd
import numpy as np

np.random.seed(42)

def generate_group(n, score_min, score_max, category):
    """Generate n students whose weighted scores fall within [score_min, score_max]."""
    rows = []
    attempts = 0
    while len(rows) < n and attempts < n * 20:
        attempts += 1

        # Random scores per feature
        pq  = np.random.randint(1, 11)
        ps  = np.random.randint(1, 11)
        ppd = np.random.randint(1, 11)
        cc  = np.random.randint(1, 11)
        td  = np.random.randint(1, 11)
        qa  = np.random.randint(1, 11)
        tm  = np.random.randint(1, 11)
        cl  = np.random.randint(1, 11)

        # Weighted score + noise
        score = (pq*1.5 + ps*1.5 + ppd*1.0 + cc*1.5 + td*1.5 + qa*1.0 + tm*0.5 + cl*1.5)
        noise = np.random.normal(0, 3)
        score = np.clip(score + noise, 10, 100)

        if score_min <= score <= score_max:
            rows.append({
                'Paper_Quality':       pq,
                'Presentation_Skills': ps,
                'PPT_Design':          ppd,
                'Content_Clarity':     cc,
                'Technical_Depth':     td,
                'QA_Handling':         qa,
                'Time_Management':     tm,
                'Confidence_Level':    cl,
                'Category':            category
            })
    return rows

# ---- Generate roughly equal numbers per category ----
low_rows    = generate_group(130, 10,  39,  'Low')
medium_rows = generate_group(150, 40,  70,  'Medium')
high_rows   = generate_group(130, 71,  100, 'High')

all_rows = low_rows + medium_rows + high_rows

# Shuffle so categories are not in order
df = pd.DataFrame(all_rows).sample(frac=1, random_state=42).reset_index(drop=True)

# ---- Save to CSV ----
df.to_csv('dataset.csv', index=False)

print("Dataset created successfully!")
print("Shape:", df.shape)
print("\nCategory Distribution:")
print(df['Category'].value_counts())
print("\nFirst 5 rows:")
print(df.head())
