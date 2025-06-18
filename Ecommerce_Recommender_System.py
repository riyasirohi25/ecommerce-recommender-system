#!/usr/bin/env python
# coding: utf-8

# # 🛍️ E-Commerce Recommender System
# This notebook contains both Content-Based and Collaborative Filtering implementations with Streamlit UI.

# In[1]:


import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split


# In[2]:


# Product data for content-based filtering
product_data = pd.DataFrame({
    'product_id': ['P1', 'P2', 'P3', 'P4', 'P5'],
    'product_name': ['Red Shirt', 'Blue Jeans', 'Black Shoes', 'White Shirt', 'Red Skirt'],
    'description': [
        'Red cotton shirt for men',
        'Blue denim jeans for casual wear',
        'Black leather shoes for formal use',
        'White cotton shirt, formal wear',
        'Red mini skirt, party wear'
    ]
})

# User-product ratings for collaborative filtering
rating_data = pd.DataFrame({
    'user_id': ['U1', 'U1', 'U2', 'U2', 'U3'],
    'product_id': ['P1', 'P2', 'P2', 'P3', 'P4'],
    'rating': [5, 4, 5, 3, 4]
})


# In[3]:


def content_based_recommend(product_name, top_n=3):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(product_data['description'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    idx = product_data[product_data['product_name'] == product_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    recommended = [product_data['product_name'][i[0]] for i in sim_scores]
    return recommended


# In[4]:


def train_collaborative_model():
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(rating_data[['user_id', 'product_id', 'rating']], reader)
    trainset, _ = train_test_split(data, test_size=0.2)
    model = SVD()
    model.fit(trainset)
    return model

model = train_collaborative_model()

def predict_rating(user_id, product_id):
    pred = model.predict(user_id, product_id)
    return round(pred.est, 2)


# In[5]:


st.set_page_config(page_title="E-Commerce Recommender", layout="centered")
st.title("🛍️ E-Commerce Recommender System")

tab1, tab2 = st.tabs(["📌 Content-Based", "📌 Collaborative Filtering"])

# ---------- Tab 1: Content-Based -----------
with tab1:
    st.header("Content-Based Recommendations")
    selected_product = st.selectbox("Choose a product you like:", product_data['product_name'].tolist())

    if selected_product:
        recs = content_based_recommend(selected_product)
        st.subheader("You may also like 👇")
        for r in recs:
            st.markdown(f"✅ {r}")

# ---------- Tab 2: Collaborative Filtering -----------
with tab2:
    st.header("Collaborative Filtering Predictions")
    selected_user = st.selectbox("Choose a user ID:", rating_data['user_id'].unique())
    selected_prod = st.selectbox("Choose a product ID:", product_data['product_id'].unique())

    if selected_user and selected_prod:
        rating = predict_rating(selected_user, selected_prod)
        st.subheader(f"🔮 Predicted Rating by {selected_user} for {selected_prod}: **{rating}⭐**")


# In[ ]:





# In[ ]:




