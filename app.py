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

st.title("🛍️ Product Recommendation Engine")
st.caption("Hybrid recommender combining **Content-Based Filtering** (TF-IDF + cosine similarity) "
           "and **Collaborative Filtering** (item-based CF + SVD matrix factorization).")

tab1, tab2, tab3 = st.tabs(["🎯 Get Recommendations", "📦 Product Explorer", "📊 Dataset Overview"])

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

            for _, row in recs.head(6).iterrows():
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
