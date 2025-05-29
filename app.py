import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Datasetləri yüklə (full təmizlənmiş csv fayllar)
books_df = pd.read_csv("books_cleaned_full.csv")
ratings_df = pd.read_csv("ratings_cleaned_full.csv")
tags_df = pd.read_csv("tags_cleaned_full.csv")
book_tags_df = pd.read_csv("book_tags_cleaned_full.csv")

# SVD modelini yüklə və ya qur
model_path = "svd_model.pkl"
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        svd_model = pickle.load(f)
else:
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(ratings_df[['user_id', 'book_id', 'rating']], reader)
    trainset, _ = train_test_split(data, test_size=0.2, random_state=42)
    svd_model = SVD()
    svd_model.fit(trainset)
    with open(model_path, "wb") as f:
        pickle.dump(svd_model, f)

# Tag similarity matrisini hazırla
book_tags_merged = book_tags_df.merge(tags_df, on="tag_id")
popular_tags = book_tags_merged[book_tags_merged['count'] > 5]
book_tag_matrix = popular_tags.pivot_table(
    index='best_book_id',
    columns='tag_name',
    values='count',
    fill_value=0
).astype(np.float32)
similarity_matrix = cosine_similarity(book_tag_matrix)

# Tag similarity hesablama
def get_tag_scores(book_id):
    row = books_df[books_df['book_id'] == book_id]
    if row.empty:
        return {}
    best_book_id = row['best_book_id'].values[0]
    if best_book_id not in book_tag_matrix.index:
        return {}
    idx = book_tag_matrix.index.get_loc(best_book_id)
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    top_scores = {book_tag_matrix.index[i]: score for i, score in similarity_scores if i != idx}
    return top_scores

# Content-based tövsiyə funksiyası
def content_based_only(tag_scores, top_n=5):
    results = []
    for b_id, score in sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)[:top_n*5]:
        book_info = books_df[books_df['best_book_id'] == b_id][['book_id', 'title', 'authors', 'image_url']]
        if not book_info.empty:
            info = book_info.iloc[0]
            results.append({
                'book_id': info['book_id'],
                'title': info['title'],
                'authors': info['authors'],
                'image_url': info['image_url'],
                'score': round(score, 2)
            })
    return pd.DataFrame(results)

# Hibrid tövsiyə funksiyası
def hybrid_recommend(user_id, tag_scores, top_n=5):
    scored_books = []
    for b_id, tag_score in tag_scores.items():
        book_row = books_df[books_df['best_book_id'] == b_id]
        if not book_row.empty:
            book_id = book_row['book_id'].values[0]
            try:
                svd_score = svd_model.predict(user_id, book_id).est
                final_score = 0.6 * svd_score + 0.4 * tag_score
                scored_books.append((book_id, final_score))
            except:
                continue
    top_books = sorted(scored_books, key=lambda x: x[1], reverse=True)[:top_n]
    top_df = pd.DataFrame(top_books, columns=['book_id', 'score'])
    merged = top_df.merge(books_df[['book_id', 'title', 'authors', 'image_url']], on='book_id')
    return merged[['book_id', 'title', 'authors', 'image_url', 'score']]

# Ən populyar kitabları hesablamaq üçün: ratings_df-dəki ən çox reytinq verilmiş kitablar
def get_most_popular_books(n=5):
    pop_counts = ratings_df['book_id'].value_counts().reset_index()
    pop_counts.columns = ['book_id', 'rating_count']
    pop_books = books_df.merge(pop_counts, on='book_id').sort_values(by='rating_count', ascending=False).head(n)
    return pop_books

# Streamlit UI
st.title("📚 Hibrid Kitab Tövsiyə Sistemi")

st.sidebar.header("🔍 Filtrlər")
min_score = st.sidebar.slider("Minimum skor:", 0.0, 5.0, 2.0, 0.1)
st.sidebar.markdown("💡 Tövsiyə sayı azdırsa, minimum skoru 0.0-a endirin.")
authors_list = ["Hamısı"] + sorted(books_df['authors'].dropna().unique().tolist())
selected_author = st.sidebar.selectbox("Müəllif seç:", authors_list)

user_id_input = st.text_input("İstifadəçi ID daxil edin (istəyə bağlı):")
book_title = st.selectbox("Oxuduğunuz kitabı seçin:", books_df['title'].unique())
book_id = books_df[books_df['title'] == book_title]['book_id'].values[0]
top_n = st.slider("Tövsiyə sayı:", 1, 20, 5)

if st.button("Tövsiyə al"):
    tag_scores = get_tag_scores(book_id)
    töv_type = ""
    if not tag_scores:
        st.warning("Bu kitab üçün tag məlumatı kifayət qədər deyil və ya kitab tapılmadı. Başqa kitab seçin.")
        results = pd.DataFrame()
    elif not user_id_input:
        results = content_based_only(tag_scores, top_n)
        töv_type = "Məzmuna əsaslanaraq"
    else:
        try:
            user_id = int(user_id_input)
            if not (ratings_df['user_id'] == user_id).any():
                results = content_based_only(tag_scores, top_n)
                töv_type = "Məzmuna əsaslanaraq"
            else:
                results = hybrid_recommend(user_id, tag_scores, top_n)
                töv_type = "Hibrid modelə əsaslanaraq"
        except:
            results = content_based_only(tag_scores, top_n)
            töv_type = "Məzmuna əsaslanaraq"

    # Əsas nəticələr (və fallback populyar kitablar)
    if not results.empty:
        if 'score' in results.columns:
            results = results[results['score'] >= min_score]
        if 'authors' in results.columns and selected_author != "Hamısı":
            results = results[results['authors'] == selected_author]

        if not results.empty:
            st.write(f"### {töv_type} tövsiyə edilən kitablar:")
            for _, row in results.iterrows():
                col1, col2 = st.columns([1, 4])
                with col1:
                    if pd.notnull(row['image_url']):
                        st.image(row['image_url'], width=100)
                with col2:
                    st.markdown(f"**{row['title']}**")
                    st.markdown(f"Müəllif: *{row['authors']}*")
                    st.markdown(f"Skor: **{row['score']}**")
                st.markdown("---")
        else:
            st.warning("Filtrlərə uyğun nəticə tapılmadı.")
            st.info("Buna baxmayaraq, ən çox reytinq verilmiş 5 kitab:")
            pop_books = get_most_popular_books(5)
            for _, row in pop_books.iterrows():
                col1, col2 = st.columns([1, 4])
                with col1:
                    if pd.notnull(row['image_url']):
                        st.image(row['image_url'], width=100)
                with col2:
                    st.markdown(f"**{row['title']}**")
                    st.markdown(f"Müəllif: *{row['authors']}*")
                    st.markdown(f"Reytinq sayı: **{row['rating_count']}**")
                st.markdown("---")
    else:
        st.warning("Tövsiyə verilə bilmədi.")
        st.info("Ən çox reytinq verilmiş 5 kitab:")
        pop_books = get_most_popular_books(5)
        for _, row in pop_books.iterrows():
            col1, col2 = st.columns([1, 4])
            with col1:
                if pd.notnull(row['image_url']):
                    st.image(row['image_url'], width=100)
            with col2:
                st.markdown(f"**{row['title']}**")
                st.markdown(f"Müəllif: *{row['authors']}*")
                st.markdown(f"Reytinq sayı: **{row['rating_count']}**")
            st.markdown("---")