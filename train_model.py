import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

print("Loading dataset...")
df = pd.read_csv('flight_dataset/Clean_Dataset.csv')

print("Preprocessing dataset...")
ROUTE_MAP = {
    ("Delhi",     "Mumbai"):     0,
    ("Delhi",     "Chennai"):    1,
    ("Delhi",     "Kolkata"):    2,
    ("Delhi",     "Hyderabad"):  3,
    ("Delhi",     "Bangalore"):  4,
    ("Mumbai",    "Delhi"):      5,
    ("Mumbai",    "Chennai"):    6,
    ("Mumbai",    "Kolkata"):    7,
    ("Mumbai",    "Hyderabad"):  8,
    ("Mumbai",    "Bangalore"):  9,
    ("Chennai",   "Delhi"):     10,
    ("Chennai",   "Mumbai"):    11,
    ("Chennai",   "Kolkata"):   12,
    ("Chennai",   "Hyderabad"): 13,
    ("Chennai",   "Bangalore"): 14,
    ("Kolkata",   "Delhi"):     15,
    ("Kolkata",   "Mumbai"):    16,
    ("Kolkata",   "Chennai"):   17,
    ("Kolkata",   "Hyderabad"): 18,
    ("Kolkata",   "Bangalore"): 19,
    ("Hyderabad", "Delhi"):     20,
    ("Hyderabad", "Mumbai"):    21,
    ("Hyderabad", "Chennai"):   22,
    ("Hyderabad", "Kolkata"):   23,
    ("Hyderabad", "Bangalore"): 24,
}

def encode_route(row):
    return ROUTE_MAP.get((row['source_city'], row['destination_city']), 0)

df['route'] = df.apply(encode_route, axis=1)

df['duration_tmp'] = df['duration'].astype(int)
df['duration_mins'] = (df['duration'] * 60).astype(int)

# Use original exact feature names
features_order = [
    'duration', 'days_left', 'duration_mins', 'route', 
    'airline_Air_India', 'airline_GO_FIRST', 'airline_Indigo', 'airline_SpiceJet', 'airline_Vistara', 
    'source_city_Chennai', 'source_city_Delhi', 'source_city_Hyderabad', 'source_city_Kolkata', 'source_city_Mumbai', 
    'destination_city_Chennai', 'destination_city_Delhi', 'destination_city_Hyderabad', 'destination_city_Kolkata', 'destination_city_Mumbai', 
    'departure_time_Early_Morning', 'departure_time_Evening', 'departure_time_Late_Night', 'departure_time_Morning', 'departure_time_Night', 
    'arrival_time_Early_Morning', 'arrival_time_Evening', 'arrival_time_Late_Night', 'arrival_time_Morning', 'arrival_time_Night', 
    'stops_two_or_more', 'stops_zero', 'class_Economy'
]

# Rename duration_tmp to duration to match expected names
df.rename(columns={'duration': 'raw_duration', 'duration_tmp': 'duration'}, inplace=True)

df['class_Economy'] = (df['class'] == 'Economy').astype(int)

df['airline'] = df['airline'].str.replace(' ', '_')
for a in ['Air_India', 'GO_FIRST', 'Indigo', 'SpiceJet', 'Vistara']:
    df[f'airline_{a}'] = (df['airline'] == a).astype(int)

for c in ['Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai']:
    df[f'source_city_{c}'] = (df['source_city'] == c).astype(int)

for c in ['Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai']:
    df[f'destination_city_{c}'] = (df['destination_city'] == c).astype(int)

df['departure_time'] = df['departure_time'].str.replace(' ', '_')
for t in ['Early_Morning', 'Evening', 'Late_Night', 'Morning', 'Night']:
    df[f'departure_time_{t}'] = (df['departure_time'] == t).astype(int)

df['arrival_time'] = df['arrival_time'].str.replace(' ', '_')
for t in ['Early_Morning', 'Evening', 'Late_Night', 'Morning', 'Night']:
    df[f'arrival_time_{t}'] = (df['arrival_time'] == t).astype(int)

df['stops_two_or_more'] = (df['stops'] == 'two_or_more').astype(int)
df['stops_zero'] = (df['stops'] == 'zero').astype(int)

X = df[features_order]
y = df['price']

print(f"X shape: {X.shape}, y shape: {y.shape}")

print("Training model with constraints (n_estimators=100, max_depth=12)...")
model = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_split=5, n_jobs=-1, random_state=42)
model.fit(X, y)

print("Training R2 score:", model.score(X, y))

# Save the model
out_file = 'flight_price_model.pkl'
with open(out_file, 'wb') as f:
    pickle.dump(model, f)

size_mb = os.path.getsize(out_file) / (1024 * 1024)
print(f"Model saved completely to {out_file}.")
print(f"Model file size: {size_mb:.2f} MB")
