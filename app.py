"""
app.py
------
Interactive Streamlit demo for the Product Recommendation Engine.

Run:
    streamlit run app.py
"""

import pickle
import os
import pandas as pd
import streamlit as st

from src.hybrid import HybridRecommender

st.set_page_config(page_title="Product Recommendation Engine", page_icon="🛍️", layout="wide")

# ---------------------------------------------------------------------------
# PREMIUM THEME — dark glass UI, cyan → violet signature gradient
# ---------------------------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
:root {
    --bg-primary: #0B1220;
    --bg-secondary: #111827;
    --bg-tertiary: #0F172A;
    --accent-cyan: #22D3EE;
    --accent-blue: #3B82F6;
    --accent-purple: #A855F7;
    --accent-indigo: #6366F1;
    --success: #10B981;
    --warning: #F59E0B;
    --error: #F43F5E;
    --text-primary: #E7ECF3;
    --text-secondary: #8FA1B8;
    --glass-bg: rgba(255,255,255,0.045);
    --glass-bg-hover: rgba(255,255,255,0.075);
    --glass-border: rgba(255,255,255,0.09);
    --glow-cyan: rgba(34,211,238,0.35);
    --glow-purple: rgba(168,85,247,0.35);
}

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(ellipse 900px 500px at 12% -8%, rgba(34,211,238,0.14), transparent 60%),
        radial-gradient(ellipse 800px 550px at 88% 8%, rgba(168,85,247,0.13), transparent 60%),
        radial-gradient(ellipse 700px 500px at 50% 105%, rgba(99,102,241,0.10), transparent 60%),
        linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-tertiary) 100%);
    background-attachment: fixed;
    color: var(--text-primary);
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    opacity: 0.5;
    background:
        radial-gradient(circle 500px at 20% 20%, rgba(34,211,238,0.06), transparent 70%),
        radial-gradient(circle 600px at 80% 30%, rgba(168,85,247,0.05), transparent 70%);
    animation: auroraDrift 22s ease-in-out infinite alternate;
}

@keyframes auroraDrift {
    0%   { transform: translate(0px, 0px) scale(1); }
    100% { transform: translate(30px, -20px) scale(1.06); }
}

section.main > div { position: relative; z-index: 1; }

.hero-wrap {
    position: relative;
    padding: 2.2rem 2rem 1.6rem 2rem;
    margin-bottom: 1.2rem;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(34,211,238,0.07), rgba(168,85,247,0.07));
    border: 1px solid var(--glass-border);
    overflow: hidden;
}

.hero-ring {
    position: absolute;
    top: -140px;
    left: -100px;
    width: 320px;
    height: 320px;
    border-radius: 50%;
    background: conic-gradient(from 0deg, var(--accent-cyan), var(--accent-purple), var(--accent-indigo), var(--accent-cyan));
    filter: blur(70px);
    opacity: 0.28;
    animation: ringSpin 18s linear infinite;
    z-index: 0;
}
@keyframes ringSpin { from { transform: rotate(0deg);} to { transform: rotate(360deg);} }

.hero-content { position: relative; z-index: 1; }

.hero-title {
    font-family: 'Sora', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    letter-spacing: -0.02em;
    margin: 0;
    background: linear-gradient(90deg, #F1F5F9 10%, var(--accent-cyan) 55%, var(--accent-purple) 95%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-family: 'Inter', sans-serif;
    color: var(--text-secondary);
    font-size: 0.98rem;
    margin-top: 0.55rem;
    max-width: 760px;
    line-height: 1.55;
}
.hero-sub b { color: var(--text-primary); font-weight: 600; }

.badge-row { margin-top: 1rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
.badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    padding: 0.28rem 0.65rem;
    border-radius: 999px;
    border: 1px solid var(--glass-border);
    background: var(--glass-bg);
    color: var(--text-secondary);
    letter-spacing: 0.02em;
}

h2, h3 {
    font-family: 'Sora', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
    color: var(--text-primary) !important;
}

[data-baseweb="tab-list"] {
    gap: 6px;
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: 14px;
    padding: 6px;
}
button[data-baseweb="tab"] {
    border-radius: 10px !important;
    color: var(--text-secondary) !important;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    transition: all 0.25s ease;
}
button[data-baseweb="tab"]:hover {
    background: var(--glass-bg-hover) !important;
    color: var(--text-primary) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(90deg, rgba(34,211,238,0.18), rgba(168,85,247,0.18)) !important;
    color: var(--text-primary) !important;
    box-shadow: inset 0 0 0 1px var(--glow-cyan);
}
[data-baseweb="tab-highlight"] { background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple)) !important; }

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 18px !important;
    backdrop-filter: blur(14px);
    transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-3px);
    border-color: rgba(34,211,238,0.4) !important;
    box-shadow: 0 10px 30px -12px var(--glow-cyan);
}

div[data-testid="stMetric"] {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 0.9rem 1.1rem;
    transition: box-shadow 0.25s ease, transform 0.25s ease;
}
div[data-testid="stMetric"]:hover {
    box-shadow: 0 8px 24px -10px var(--glow-purple);
    transform: translateY(-2px);
}
div[data-testid="stMetricValue"] {
    font-family: 'Sora', sans-serif !important;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
div[data-testid="stMetricLabel"] { color: var(--text-secondary) !important; }

div[data-baseweb="select"] > div {
    background: var(--glass-bg) !important;
    border-color: var(--glass-border) !important;
    border-radius: 12px !important;
}
div[data-baseweb="select"] > div:hover { border-color: var(--accent-cyan) !important; }

.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)) !important;
    box-shadow: 0 0 12px var(--glow-cyan);
}
.stSlider div[data-testid="stTickBarMin"], .stSlider div[data-testid="stTickBarMax"] { color: var(--text-secondary); }

div[role="radiogroup"] label {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    padding: 0.45rem 0.8rem;
    margin-bottom: 0.35rem;
    transition: all 0.2s ease;
}
div[role="radiogroup"] label:hover { border-color: var(--accent-cyan); background: var(--glass-bg-hover); }

div[data-testid="stDataFrame"] {
    border: 1px solid var(--glass-border);
    border-radius: 14px;
    overflow: hidden;
}

div[data-testid="stAlert"] {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: 14px;
    backdrop-filter: blur(10px);
}

.stCaption, [data-testid="stCaptionContainer"] { color: var(--text-secondary) !important; }

hr { border-color: var(--glass-border) !important; }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: linear-gradient(var(--accent-cyan), var(--accent-purple)); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    products_df = pd.read_csv("data/products.csv")
    ratings_df = pd.read_csv("data/ratings.csv")
    users_df = pd.read_csv("data/users.csv")
    return products_df, ratings_df, users_df


@st.cache_resource
def load_model(_products_df, _ratings_df):
    model_path = "models/hybrid_recommender.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    model = HybridRecommender(_products_df, _ratings_df)
    os.makedirs("models", exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return model


products_df, ratings_df, users_df = load_data()
model = load_model(products_df, ratings_df)

# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-ring"></div>
    <div class="hero-content">
        <div class="hero-title">🛍️ Product Recommendation Engine</div>
        <div class="hero-sub">
            A hybrid recommender combining <b>Content-Based Filtering</b> (TF-IDF + cosine similarity)
            and <b>Collaborative Filtering</b> (item-based CF + SVD matrix factorization) —
            with automatic cold-start handling for new users.
        </div>
        <div class="badge-row">
            <span class="badge">{len(products_df)} PRODUCTS</span>
            <span class="badge">{len(users_df)} USERS</span>
            <span class="badge">{len(ratings_df):,} RATINGS</span>
            <span class="badge">SVD + TF-IDF</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎯  Get Recommendations", "📦  Product Explorer", "📊  Dataset Overview"])

# ---------------------------------------------------------------------------
# TAB 1: Recommendations
# ---------------------------------------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Choose a user")
        user_options = users_df.apply(
            lambda r: f"{r.user_id} — {r.user_name} ({r.preferred_category})", axis=1
        ).tolist()
        selected = st.selectbox("User", user_options, index=0)
        user_id = int(selected.split(" — ")[0])

        top_n = st.slider("Number of recommendations", 3, 20, 10)
        mode = st.radio("Strategy", ["Hybrid (recommended)", "Content-Based only", "Collaborative only"])

        st.markdown("---")
        st.markdown("**This user's rating history:**")
        user_hist = ratings_df[ratings_df.user_id == user_id].merge(products_df, on="product_id")
        if user_hist.empty:
            st.info("This user has no ratings yet (cold-start case). "
                    "Recommendations will fall back to popularity-based suggestions.")
        else:
            st.dataframe(
                user_hist[["product_name", "category", "rating"]]
                .sort_values("rating", ascending=False),
                hide_index=True, use_container_width=True
            )

    with col2:
        st.subheader(f"Recommendations for user {user_id}")

        if mode == "Hybrid (recommended)":
            recs = model.recommend(user_id, top_n=top_n)
            display_cols = ["product_name", "category", "brand", "price", "score", "source"]
        elif mode == "Content-Based only":
            recs = model.cb.recommend_for_user(user_id, ratings_df, top_n=top_n)
            display_cols = ["product_name", "category", "brand", "price", "score"]
        else:
            raw = model.cf.recommend_svd(user_id, top_n=top_n)
            pids = [pid for pid, _ in raw]
            recs = products_df[products_df.product_id.isin(pids)].copy()
            score_map = dict(raw)
            recs["score"] = recs.product_id.map(score_map)
            recs = recs.sort_values("score", ascending=False)
            display_cols = ["product_name", "category", "brand", "price", "score"]

        if recs.empty:
            st.warning("No recommendations could be generated (try another user).")
        else:
            display_cols = [c for c in display_cols if c in recs.columns]
            st.dataframe(recs[display_cols].reset_index(drop=True),
                         hide_index=True, use_container_width=True)

            card_cols = st.columns(2)
            for i, (_, row) in enumerate(recs.head(6).iterrows()):
                with card_cols[i % 2]:
                    with st.container(border=True):
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.markdown(f"**{row.product_name}**")
                            st.caption(f"{row.category} · {row.brand}")
                        with c2:
                            st.metric("Price", f"₹{row.price:,.0f}")

# ---------------------------------------------------------------------------
# TAB 2: Product Explorer (find similar items)
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Find products similar to a given item")
    product_options = products_df.apply(lambda r: f"{r.product_id} — {r.product_name}", axis=1).tolist()
    selected_product = st.selectbox("Pick a product", product_options)
    pid = int(selected_product.split(" — ")[0])

    similar = model.cb.similar_products(pid, top_n=8)
    st.markdown(f"**Products similar to:** {products_df[products_df.product_id==pid].product_name.values[0]}")
    st.dataframe(
        similar[["product_name", "category", "brand", "price", "similarity_score"]],
        hide_index=True, use_container_width=True
    )

# ---------------------------------------------------------------------------
# TAB 3: Dataset Overview
# ---------------------------------------------------------------------------
with tab3:
    c1, c2, c3 = st.columns(3)
    c1.metric("Products", len(products_df))
    c2.metric("Users", len(users_df))
    c3.metric("Ratings (interactions)", len(ratings_df))

    st.subheader("Ratings by category")
    cat_ratings = ratings_df.merge(products_df, on="product_id").groupby("category").size()
    st.bar_chart(cat_ratings)

    st.subheader("Rating distribution")
    st.bar_chart(ratings_df.rating.value_counts().sort_index())

    st.subheader("Sample product catalog")
    st.dataframe(products_df.head(20), hide_index=True, use_container_width=True)
