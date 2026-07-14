"""
generate_data.py
-----------------
Generates a synthetic but realistic e-commerce dataset:
  - products.csv  : product catalog with category, brand, description, price
  - ratings.csv   : user-product interactions (explicit ratings 1-5)
  - users.csv     : user profiles

Run:
    python data/generate_data.py
"""

import numpy as np
import pandas as pd
import random

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

N_USERS = 300
N_PRODUCTS = 150
N_RATINGS = 6000  # interactions

# ---------------------------------------------------------------------------
# 1. Product catalog
# ---------------------------------------------------------------------------
CATEGORIES = {
    "Electronics": ["Smartphone", "Laptop", "Headphones", "Smartwatch", "Camera",
                    "Tablet", "Bluetooth Speaker", "Gaming Console", "Monitor", "Router"],
    "Fashion": ["Running Shoes", "Denim Jacket", "T-Shirt", "Sneakers", "Handbag",
                "Sunglasses", "Wrist Watch", "Backpack", "Formal Shirt", "Winter Coat"],
    "Home & Kitchen": ["Air Fryer", "Coffee Maker", "Blender", "Vacuum Cleaner",
                       "Cookware Set", "Bedsheet", "Table Lamp", "Microwave Oven",
                       "Dinner Set", "Water Purifier"],
    "Books": ["Mystery Novel", "Sci-Fi Novel", "Self-Help Book", "Cookbook",
              "Biography", "History Book", "Fantasy Novel", "Programming Guide",
              "Poetry Collection", "Business Book"],
    "Sports & Fitness": ["Yoga Mat", "Dumbbell Set", "Cricket Bat", "Football",
                         "Treadmill", "Cycling Helmet", "Resistance Bands",
                         "Badminton Racket", "Skipping Rope", "Gym Bag"],
    "Beauty": ["Face Moisturizer", "Shampoo", "Perfume", "Lipstick", "Sunscreen",
               "Hair Dryer", "Face Wash", "Nail Polish Set", "Beard Trimmer", "Body Lotion"],
}

BRANDS = ["Zenith", "Nova", "Urban", "Pulse", "Everest", "Aria", "Crest",
          "Nimbus", "Orbit", "Falcon", "Lumen", "Vertex", "Solace", "Kairo"]

ADJECTIVES = ["Premium", "Compact", "Wireless", "Eco-Friendly", "Lightweight",
              "Durable", "Portable", "Smart", "Classic", "Professional",
              "Ergonomic", "High-Performance", "Modern", "Budget-Friendly"]

DESC_TEMPLATES = [
    "A {adj} {product} designed for everyday use with excellent build quality.",
    "Experience the best {product} in the {cat} category, built to be {adj}.",
    "This {adj} {product} from {brand} combines style, comfort, and performance.",
    "Top-rated {product} offering {adj} performance and long-lasting durability.",
    "Upgrade your lifestyle with this {adj} {product}, a favorite in {cat}.",
]

rows = []
pid = 1
for category, products in CATEGORIES.items():
    for product in products:
        # create ~2-3 variants per base product for richer catalog
        n_variants = N_PRODUCTS // (len(CATEGORIES) * 10) + 1
        for _ in range(n_variants):
            brand = random.choice(BRANDS)
            adj = random.choice(ADJECTIVES)
            template = random.choice(DESC_TEMPLATES)
            description = template.format(adj=adj, product=product, cat=category, brand=brand)
            base_price = {
                "Electronics": (2000, 60000),
                "Fashion": (500, 8000),
                "Home & Kitchen": (800, 15000),
                "Books": (150, 1200),
                "Sports & Fitness": (300, 10000),
                "Beauty": (150, 3000),
            }[category]
            price = round(np.random.uniform(*base_price), 2)
            rows.append({
                "product_id": pid,
                "product_name": f"{brand} {adj} {product}",
                "category": category,
                "brand": brand,
                "description": description,
                "price": price,
            })
            pid += 1

products_df = pd.DataFrame(rows)
products_df = products_df.sample(frac=1, random_state=RNG_SEED).reset_index(drop=True)
products_df = products_df.iloc[:N_PRODUCTS].copy()
products_df["product_id"] = range(1, len(products_df) + 1)
products_df.to_csv("data/products.csv", index=False)
print(f"Generated {len(products_df)} products -> data/products.csv")

# ---------------------------------------------------------------------------
# 2. Users
# ---------------------------------------------------------------------------
FIRST_NAMES = ["Aarav", "Riya", "Kabir", "Ananya", "Vihaan", "Ishaan", "Diya",
               "Aditi", "Rohan", "Meera", "Sara", "Arjun", "Nisha", "Kunal",
               "Priya", "Dev", "Tara", "Farhan", "Simran", "Yash"]
LAST_NAMES = ["Sharma", "Verma", "Iyer", "Khan", "Patel", "Gupta", "Nair",
              "Reddy", "Chopra", "Malhotra", "Bose", "Kapoor"]

user_rows = []
for uid in range(1, N_USERS + 1):
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    # each user has a preferred category (drives realistic rating patterns)
    preferred_category = random.choice(list(CATEGORIES.keys()))
    user_rows.append({"user_id": uid, "user_name": name, "preferred_category": preferred_category})

users_df = pd.DataFrame(user_rows)
users_df.to_csv("data/users.csv", index=False)
print(f"Generated {len(users_df)} users -> data/users.csv")

# ---------------------------------------------------------------------------
# 3. Ratings (interactions) — biased towards each user's preferred category
#    to create realistic, learnable patterns for both CF and content-based models.
# ---------------------------------------------------------------------------
rating_rows = []
seen_pairs = set()

products_by_category = products_df.groupby("category")["product_id"].apply(list).to_dict()
all_product_ids = products_df["product_id"].tolist()

attempts = 0
while len(rating_rows) < N_RATINGS and attempts < N_RATINGS * 20:
    attempts += 1
    user = users_df.sample(1).iloc[0]
    uid = user["user_id"]
    pref_cat = user["preferred_category"]

    # 70% chance rate a product from preferred category, 30% random exploration
    if random.random() < 0.7 and products_by_category.get(pref_cat):
        pid = random.choice(products_by_category[pref_cat])
    else:
        pid = random.choice(all_product_ids)

    if (uid, pid) in seen_pairs:
        continue
    seen_pairs.add((uid, pid))

    # rating distribution: higher ratings for preferred-category items
    prod_cat = products_df.loc[products_df.product_id == pid, "category"].values[0]
    if prod_cat == pref_cat:
        rating = np.random.choice([3, 4, 5], p=[0.15, 0.35, 0.50])
    else:
        rating = np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.20, 0.30, 0.20, 0.15])

    rating_rows.append({"user_id": uid, "product_id": pid, "rating": int(rating)})

ratings_df = pd.DataFrame(rating_rows)
ratings_df.to_csv("data/ratings.csv", index=False)
print(f"Generated {len(ratings_df)} ratings -> data/ratings.csv")

print("\nDataset generation complete.")
print(ratings_df["rating"].value_counts().sort_index())
