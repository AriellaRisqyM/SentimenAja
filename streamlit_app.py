import streamlit as st
import pandas as pd
import numpy as np
import re
import html
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Analisis Sentimen IndoBerTweet", layout="wide")

# ==========================================
# FUNGSI BANTUAN (HELPER FUNCTIONS)
# ==========================================
def clear_cache_if_new_file(uploaded_file):
    if uploaded_file is not None:
        if 'last_uploaded_file' not in st.session_state or st.session_state['last_uploaded_file'] != uploaded_file.name:
            st.session_state.clear()
            st.session_state['last_uploaded_file'] = uploaded_file.name

def display_paginated(df, key_prefix, page_size=5):
    total_pages = max(1, len(df) // page_size + (1 if len(df) % page_size > 0 else 0))
    page_number = st.number_input(f"Halaman (1 - {total_pages})", min_value=1, max_value=total_pages, value=1, step=1, key=f"page_{key_prefix}")
    start_idx = (page_number - 1) * page_size
    end_idx = start_idx + page_size
    st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True)

def advanced_clean_text(text):
    if not isinstance(text, str): return ""
    text = html.unescape(text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'http[s]?\:\/\/.[a-zA-Z0-9\.\/\_?=%&#\-\+!]+', '', text)
    text = re.sub(r'pic.twitter.com?.[a-zA-Z0-9\.\/\_?=%&#\-\+!]+', '', text)
    text = re.sub(r'\@([\w]+)', '', text)
    text = re.sub(r'\#([\w]+)', '', text)
    text = re.sub(r'[!\$%^&*@#()_+~={}\[\]%\-:";\'<>?,.\/]', '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'([a-zA-Z])\1\1+', r'\1', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

@st.cache_data
def load_kamus_alay():
    url_kamus = 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/kamus/nasalsabila_kamus-alay/_json_colloc'
    try:
        response = requests.get(url_kamus)
        return response.json()
    except:
        return {}

def normalize_text(text, kamus):
    if not isinstance(text, str) or not text.strip(): return ""
    words = text.split()
    normalized_words = [kamus.get(word, word) for word in words]
    return " ".join(normalized_words)

@st.cache_resource
def load_indobertweet():
    model_name = "Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis"
    return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

def label_indobertweet_biner(text, nlp_model):
    if not isinstance(text, str) or not text.strip(): return "Positif", 0.0
    try:
        hasil_semua = nlp_model(text[:512], truncation=True, top_k=None)
        if isinstance(hasil_semua[0], list): hasil_semua = hasil_semua[0]
        skor_pos, skor_neg = 0.0, 0.0
        for item in hasil_semua:
            lbl = item['label'].lower()
            if 'pos' in lbl or lbl == 'label_2': skor_pos = item['score']
            elif 'neg' in lbl or lbl == 'label_0': skor_neg = item['score']
        total = skor_pos + skor_neg
        if total == 0: return "Positif", 0.0
        if (skor_pos/total) > (skor_neg/total): return "Positif", (skor_pos/total)
        return "Negatif", (skor_neg/total)
    except:
        return "Positif", 0.0

# ==========================================
# UI APLIKASI UTAMA
# ==========================================
st.title("Aplikasi Analisis Sentimen IndoBerTweet")

tab1, tab2 = st.tabs(["Fitur 1: Analisis Dataset", "Fitur 2: Prediksi Teks Tunggal"])

with tab1:
    st.header("1. Upload Dataset")
    uploaded_file = st.file_uploader("Upload file CSV atau Excel", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        clear_cache_if_new_file(uploaded_file)
        
        # Load Data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.write("Preview Data Original:")
        st.dataframe(df.head(), use_container_width=True)
        
        st.header("2. Pilih Kolom Target")
        st.warning("⚠️ Peringatan: Pastikan kolom yang dipilih hanya memuat/memproses data teks (string).")
        kolom_teks = st.selectbox("Pilih kolom yang berisi teks:", df.columns)
        
        st.header("3. Konfigurasi Pemrosesan")
        keywords_input = st.text_input("Masukkan Keywords (pisahkan dengan koma):", "polri, polisi, polda, polres")
        tahun_input = st.text_input("Masukkan Range Tahun (pisahkan dengan koma):", "2022, 2023, 2024, 2025")
        kolom_tahun = st.selectbox("Pilih kolom tanggal/tahun (opsional):", ["Tidak Ada"] + list(df.columns))
        
        train_ratio = st.slider("Rasio Data Train (%)", min_value=50, max_value=90, value=80, step=5)
        test_ratio = 100 - train_ratio
        st.info(f"Pembagian Data: {train_ratio}% Train / {test_ratio}% Test")
        
        if st.button("Mulai Pemrosesan Data"):
            with st.spinner("Memproses data... Ini mungkin memakan waktu."):
                
                # 3.1 Cleaning Data
                df_clean = df.copy()
                df_clean.dropna(subset=[kolom_teks], inplace=True)
                df_clean['cleaned_text'] = df_clean[kolom_teks].astype(str).apply(advanced_clean_text)
                df_clean = df_clean[df_clean['cleaned_text'].str.strip().astype(bool)]
                df_clean.drop_duplicates(subset=['cleaned_text'], keep='first', inplace=True)
                st.session_state['df_clean'] = df_clean
                
                # 3.2 Case Folding
                df_clean['case_folded'] = df_clean['cleaned_text'].str.lower()
                st.session_state['df_case'] = df_clean
                
                # 3.3 Filtering
                keywords = [k.strip().lower() for k in keywords_input.split(',')]
                pattern = r'\b(?:' + '|'.join(map(re.escape, keywords)) + r')\b'
                mask_keyword = df_clean['case_folded'].str.contains(pattern, flags=re.IGNORECASE, na=False)
                df_filter = df_clean[mask_keyword].copy()
                
                if kolom_tahun != "Tidak Ada":
                    target_years = [int(y.strip()) for y in tahun_input.split(',')]
                    df_filter['created_at'] = pd.to_datetime(df_filter[kolom_tahun], errors='coerce')
                    df_filter['year_temp'] = df_filter['created_at'].dt.year
                    df_filter = df_filter[df_filter['year_temp'].isin(target_years)].copy()
                st.session_state['df_filter'] = df_filter
                
                # 3.4 Normalization
                kamus = load_kamus_alay()
                df_filter['normalized'] = df_filter['case_folded'].apply(lambda x: normalize_text(x, kamus))
                st.session_state['df_norm'] = df_filter
                
                # 4. Labeling
                nlp_model = load_indobertweet()
                results = df_filter['normalized'].apply(lambda x: label_indobertweet_biner(x, nlp_model))
                df_filter['label'] = results.apply(lambda x: x[0])
                df_filter['score'] = results.apply(lambda x: x[1])
                st.session_state['df_label'] = df_filter
                
                # 5. Data Split
                X = df_filter['normalized'].astype(str)
                y = df_filter['label']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(test_ratio/100.0), random_state=42, stratify=y)
                st.session_state['split_data'] = (X_train, X_test, y_train, y_test)
                
                # 6. TF-IDF
                tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
                X_train_tfidf = tfidf.fit_transform(X_train)
                X_test_tfidf = tfidf.transform(X_test)
                st.session_state['tfidf'] = tfidf
                st.session_state['tfidf_matrices'] = (X_train_tfidf, X_test_tfidf)
                
                # Document Frequency Plot
                doc_freq = (X_train_tfidf > 0).sum(axis=0).A1
                terms = tfidf.get_feature_names_out()
                df_df = pd.DataFrame({'Term': terms, 'DF': doc_freq})
                valid_keywords = [k for k in keywords if k in terms]
                df_kw_df = df_df[df_df['Term'].isin(valid_keywords)].sort_values('DF', ascending=False)
                st.session_state['df_kw_df'] = df_kw_df
                
                # 7. Modeling
                nb_model = MultinomialNB()
                nb_model.fit(X_train_tfidf, y_train)
                st.session_state['nb_model'] = nb_model
                
                svm_model = LinearSVC(random_state=42)
                svm_model.fit(X_train_tfidf, y_train)
                st.session_state['svm_model'] = svm_model
                
                st.success("Pemrosesan selesai! Silakan lihat hasil di bawah.")

        # ==========================================
        # RENDER HASIL PEMROSESAN (DENGAN PAGINATION)
        # ==========================================
        if 'df_clean' in st.session_state:
            st.subheader("3.1 Hasil Cleaning Data (Hapus Duplikat & Karakter)")
            display_paginated(st.session_state['df_clean'][[kolom_teks, 'cleaned_text']], "clean")
            
            st.subheader("3.2 Hasil Case Folding")
            display_paginated(st.session_state['df_case'][['cleaned_text', 'case_folded']], "case")
            
            st.subheader("3.3 Hasil Filtering Keyword & Tahun")
            display_paginated(st.session_state['df_filter'][['case_folded']], "filter")
            
            st.subheader("3.4 Hasil Normalisasi")
            display_paginated(st.session_state['df_norm'][['case_folded', 'normalized']], "norm")
            
            st.subheader("4. Hasil Pelabelan IndoBerTweet")
            display_paginated(st.session_state['df_label'][['normalized', 'label', 'score']], "label")
            
            st.subheader("5. Hasil Data Split")
            X_train, X_test, y_train, y_test = st.session_state['split_data']
            st.write(f"Total Data Latih (Train): **{len(X_train)}**")
            st.write(f"Total Data Uji (Test): **{len(X_test)}**")
            
            st.subheader("6. Distribusi TF-IDF (Document Frequency)")
            df_kw_df = st.session_state['df_kw_df']
            if not df_kw_df.empty:
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(x='DF', y='Term', data=df_kw_df, palette='Blues_r', ax=ax)
                ax.set_title("Document Frequency berdasarkan Keyword")
                st.pyplot(fig)
            else:
                st.write("Tidak ada keyword yang valid ditemukan dalam matriks TF-IDF.")
                
            st.subheader("7 & 8. Evaluasi Model & Confusion Matrix")
            X_train_tfidf, X_test_tfidf = st.session_state['tfidf_matrices']
            nb_pred = st.session_state['nb_model'].predict(X_test_tfidf)
            svm_pred = st.session_state['svm_model'].predict(X_test_tfidf)
            
            st.session_state['eval_nb'] = classification_report(y_test, nb_pred, output_dict=True)
            st.session_state['eval_svm'] = classification_report(y_test, svm_pred, output_dict=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Naive Bayes Performance**")
                st.text(classification_report(y_test, nb_pred))
                fig1, ax1 = plt.subplots(figsize=(5, 4))
                sns.heatmap(confusion_matrix(y_test, nb_pred, labels=["Positif", "Negatif"]), annot=True, fmt='d', cmap='Blues', xticklabels=["Pred Positif", "Pred Negatif"], yticklabels=["Asli Positif", "Asli Negatif"])
                st.pyplot(fig1)
                
            with col2:
                st.markdown("**SVM Performance**")
                st.text(classification_report(y_test, svm_pred))
                fig2, ax2 = plt.subplots(figsize=(5, 4))
                sns.heatmap(confusion_matrix(y_test, svm_pred, labels=["Positif", "Negatif"]), annot=True, fmt='d', cmap='Blues', xticklabels=["Pred Positif", "Pred Negatif"], yticklabels=["Asli Positif", "Asli Negatif"])
                st.pyplot(fig2)
                
            st.subheader("9. Wordcloud Sentimen")
            df_label = st.session_state['df_label']
            teks_pos = " ".join(df_label[df_label['label'] == 'Positif']['normalized'].dropna().astype(str))
            teks_neg = " ".join(df_label[df_label['label'] == 'Negatif']['normalized'].dropna().astype(str))
            
            col_wc1, col_wc2 = st.columns(2)
            with col_wc1:
                if teks_pos.strip():
                    wc_pos = WordCloud(width=400, height=300, background_color='white', colormap='Greens', max_words=100).generate(teks_pos)
                    fig_wc1, ax_wc1 = plt.subplots()
                    ax_wc1.imshow(wc_pos, interpolation='bilinear')
                    ax_wc1.axis('off')
                    ax_wc1.set_title("Sentimen Positif")
                    st.pyplot(fig_wc1)
            with col_wc2:
                if teks_neg.strip():
                    wc_neg = WordCloud(width=400, height=300, background_color='white', colormap='Reds', max_words=100).generate(teks_neg)
                    fig_wc2, ax_wc2 = plt.subplots()
                    ax_wc2.imshow(wc_neg, interpolation='bilinear')
                    ax_wc2.axis('off')
                    ax_wc2.set_title("Sentimen Negatif")
                    st.pyplot(fig_wc2)

with tab2:
    st.header("Analisis Prediksi Teks Tunggal")
    user_text = st.text_area("Masukkan teks untuk dianalisis:")
    
    if st.button("Analisis Teks"):
        if 'nb_model' not in st.session_state or 'svm_model' not in st.session_state:
            st.error("Silakan latih model di 'Fitur 1: Analisis Dataset' terlebih dahulu sebelum menggunakan fitur ini.")
        elif not user_text.strip():
            st.warning("Teks tidak boleh kosong.")
        else:
            # Tampilkan metrik evaluasi model (dari Tab 1)
            st.markdown("### Performa Model Keseluruhan")
            col_metrik1, col_metrik2 = st.columns(2)
            with col_metrik1:
                st.info("**Naive Bayes**\n"
                        f"- Akurasi: {st.session_state['eval_nb']['accuracy']:.2f}\n"
                        f"- Presisi (Macro): {st.session_state['eval_nb']['macro avg']['precision']:.2f}\n"
                        f"- Recall (Macro): {st.session_state['eval_nb']['macro avg']['recall']:.2f}\n"
                        f"- F1-Score (Macro): {st.session_state['eval_nb']['macro avg']['f1-score']:.2f}")
            with col_metrik2:
                st.info("**SVM (LinearSVC)**\n"
                        f"- Akurasi: {st.session_state['eval_svm']['accuracy']:.2f}\n"
                        f"- Presisi (Macro): {st.session_state['eval_svm']['macro avg']['precision']:.2f}\n"
                        f"- Recall (Macro): {st.session_state['eval_svm']['macro avg']['recall']:.2f}\n"
                        f"- F1-Score (Macro): {st.session_state['eval_svm']['macro avg']['f1-score']:.2f}")

            # Proses Teks
            kamus = load_kamus_alay()
            nlp_model = load_indobertweet()
            
            clean_t = advanced_clean_text(user_text)
            case_t = clean_t.lower()
            norm_t = normalize_text(case_t, kamus)
            
            # Pelabelan IndoBerTweet
            label, score = label_indobertweet_biner(norm_t, nlp_model)
            
            # Pelabelan NB & SVM
            tfidf = st.session_state['tfidf']
            text_tfidf = tfidf.transform([norm_t])
            nb_pred = st.session_state['nb_model'].predict(text_tfidf)[0]
            svm_pred = st.session_state['svm_model'].predict(text_tfidf)[0]
            
            st.markdown("### Hasil Prediksi Teks Anda")
            st.write(f"**Teks Normalisasi:** {norm_t}")
            
            # Render Tabel Skor
            res_df = pd.DataFrame({
                "Metode": ["IndoBerTweet", "Naive Bayes", "SVM"],
                "Label Sentimen": [label, nb_pred, svm_pred],
                "Confidence Score": [f"{score:.4f}", "-", "-"]
            })
            st.table(res_df)
