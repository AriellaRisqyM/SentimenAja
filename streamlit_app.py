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
# KONFIGURASI DAN FUNGSI PENDUKUNG
# ==========================================
st.set_page_config(page_title="Analisis Sentimen IndoBERTweet", layout="wide")

@st.cache_resource(show_spinner=False)
def load_indobertweet():
    # Cache model pipeline agar tidak download/load berulang kali
    return pipeline("sentiment-analysis", model="Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis")

@st.cache_resource(show_spinner=False)
def load_kamus_alay():
    url = 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/kamus/nasalsabila_kamus-alay/_json_colloc'
    try:
        response = requests.get(url)
        return response.json()
    except:
        return {}

def advanced_clean_text(text):
    if not isinstance(text, str): return ""
    text = html.unescape(text)
    text = re.sub(r'[^\x00-\x7F]+','', text)
    text = re.sub(r'http[s]?\:\/\/.[a-zA-Z0-9\.\/\_?=%&#\-\+!]+','', text)
    text = re.sub(r'pic\.twitter\.com?.[a-zA-Z0-9\.\/\_?=%&#\-\+!]+','', text)
    text = re.sub(r'\@([\w]+)', '', text)
    text = re.sub(r'\#([\w]+)', '', text)
    text = re.sub(r'[!\$%^&*@#()_+~={}\[\]%\-:";\'<>?,.\/]', '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'([a-zA-Z])\1\1+', r'\1', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def normalize_text(text, kamus_normalisasi):
    if not isinstance(text, str) or not text.strip(): return ""
    words = text.split()
    normalized_words = [kamus_normalisasi.get(word, word) for word in words]
    return " ".join(normalized_words)

def label_indobertweet_biner(text, nlp_model):
    if not isinstance(text, str) or not text.strip(): return "Positif", 0.0
    try:
        hasil = nlp_model(text[:512], truncation=True, top_k=None)
        if isinstance(hasil[0], list): hasil = hasil[0]
        skor_pos, skor_neg = 0.0, 0.0
        for item in hasil:
            label = item['label'].lower()
            if 'pos' in label or label == 'label_2': skor_pos = item['score']
            elif 'neg' in label or label == 'label_0': skor_neg = item['score']
        
        total = skor_pos + skor_neg
        if total == 0: return "Positif", 0.0
        
        skor_pos_norm = skor_pos / total
        skor_neg_norm = skor_neg / total
        if skor_pos_norm > skor_neg_norm: return "Positif", skor_pos_norm
        else: return "Negatif", skor_neg_norm
    except:
        return "Positif", 0.0

def display_paginated(df, key_prefix):
    # Fitur Pagination: Menampilkan 5 data per halaman
    page_size = 5
    total_pages = max(1, (len(df) - 1) // page_size + 1)
    page = st.number_input("Halaman", min_value=1, max_value=total_pages, step=1, key=f"{key_prefix}_page")
    start_idx = (page - 1) * page_size
    st.dataframe(df.iloc[start_idx : start_idx + page_size], use_container_width=True)

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=["Positif", "Negatif"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Prediksi Positif", "Prediksi Negatif"],
                yticklabels=["Asli Positif", "Asli Negatif"], ax=ax)
    ax.set_title(title, fontweight='bold')
    plt.tight_layout()
    return fig

def generate_wordcloud(text, colormap):
    wc = WordCloud(width=800, height=400, background_color='white', 
                   colormap=colormap, max_words=100).generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    return fig

# ==========================================
# UI APLIKASI
# ==========================================
st.title("Aplikasi Analisis Sentimen IndoBERTweet")

tab1, tab2 = st.tabs(["Fitur 1: Analisis Dataset", "Fitur 2: Prediksi Teks Tunggal"])

with tab1:
    st.header("Upload & Preprocessing Dataset")
    
    # 1. Upload File
    uploaded_file = st.file_uploader("Upload file dataset (CSV atau Excel)", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        # Hapus state lama jika upload file baru agar tidak ada cache tersisa
        if 'last_upload' not in st.session_state or st.session_state['last_upload'] != uploaded_file.name:
            st.session_state.clear()
            st.session_state['last_upload'] = uploaded_file.name
            
            if uploaded_file.name.endswith('.csv'):
                st.session_state['df_raw'] = pd.read_csv(uploaded_file)
            else:
                st.session_state['df_raw'] = pd.read_excel(uploaded_file)
                
        df = st.session_state['df_raw'].copy()
        
        st.subheader("Pengaturan Analisis")
        col1, col2 = st.columns(2)
        with col1:
            # 2. Pilih Kolom dengan peringatan
            text_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
            col_text = st.selectbox("Pilih kolom isi teks:", text_columns, help="Peringatan: Pastikan kolom yang dipilih hanya memuat/memproses data teks!")
            
            date_columns = ["None"] + list(df.columns)
            col_date = st.selectbox("Pilih kolom tanggal (Opsional):", date_columns)
            
        with col2:
            st.write("Range Tahun (jika ada kolom tanggal):")
            c_start, c_end = st.columns(2)
            start_year = c_start.number_input("Tahun Mulai", value=2022)
            end_year = c_end.number_input("Tahun Akhir", value=2025)
            
            keyword_input = st.text_input("Masukkan Keyword (pisahkan koma)", "polri, kepolisian, polisi")
            
        # 5. Data Split Setting
        train_ratio = st.slider("Persentase Data Latih (Train)", min_value=50, max_value=90, value=80, step=5)
        
        if st.button("🚀 Mulai Analisis & Pemrosesan"):
            st.session_state['process_done'] = True
            with st.spinner("Memproses data... Mohon tunggu!"):
                # Menyiapkan tools
                nlp_model = load_indobertweet()
                kamus_alay = load_kamus_alay()
                
                df.dropna(subset=[col_text], inplace=True)
                df.drop_duplicates(subset=[col_text], inplace=True)
                
                # 3.1 Cleaning
                df['cleaned_text'] = df[col_text].apply(advanced_clean_text)
                df = df[df['cleaned_text'].str.strip().astype(bool)].copy()
                st.session_state['df_clean'] = df[[col_text, 'cleaned_text']].copy()
                
                # 3.2 Case Folding
                df['case_folded_text'] = df['cleaned_text'].str.lower()
                st.session_state['df_case'] = df[['cleaned_text', 'case_folded_text']].copy()
                
                # 3.3 Keyword & Year Filtering
                if col_date != "None":
                    df['parsed_date'] = pd.to_datetime(df[col_date], errors='coerce')
                    df = df[df['parsed_date'].dt.year.between(start_year, end_year)]
                
                keywords = [k.strip().lower() for k in keyword_input.split(',') if k.strip()]
                if keywords:
                    pattern = r'\b(?:' + '|'.join(map(re.escape, keywords)) + r')\b'
                    mask = df['case_folded_text'].str.contains(pattern, flags=re.IGNORECASE, na=False)
                    df = df[mask].reset_index(drop=True)
                
                if df.empty:
                    st.error("Tidak ada data tersisa setelah difilter tahun & keyword!")
                    st.stop()
                    
                st.session_state['df_filter'] = df[['case_folded_text']].copy()
                
                # 3.4 Normalization
                df['normalized_text'] = df['case_folded_text'].apply(lambda x: normalize_text(x, kamus_alay))
                st.session_state['df_norm'] = df[['case_folded_text', 'normalized_text']].copy()
                
                # 4. Labeling IndoBERTweet
                df['label_info'] = df['normalized_text'].apply(lambda x: label_indobertweet_biner(x, nlp_model))
                df['label'] = df['label_info'].apply(lambda x: x[0])
                df['indobertweet_score'] = df['label_info'].apply(lambda x: x[1])
                st.session_state['df_labeled'] = df[['normalized_text', 'label', 'indobertweet_score']].copy()
                
                # Split Data
                X = df['normalized_text']
                y = df['label']
                test_size = 1.0 - (train_ratio / 100.0)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
                st.session_state['split_info'] = {"train": len(X_train), "test": len(X_test)}
                
                # 6. TF-IDF & Document Frequency
                tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
                X_train_tf = tfidf.fit_transform(X_train)
                X_test_tf = tfidf.transform(X_test)
                
                doc_freq = (X_train_tf > 0).sum(axis=0).A1
                feature_names = tfidf.get_feature_names_out()
                
                df_freq_list = []
                for kw in keywords:
                    if kw in feature_names:
                        idx = np.where(feature_names == kw)[0][0]
                        df_freq_list.append({"Keyword": kw, "Document Frequency": int(doc_freq[idx])})
                df_keyword_freq = pd.DataFrame(df_freq_list).sort_values("Document Frequency", ascending=False)
                st.session_state['df_freq'] = df_keyword_freq
                
                # 7 & 8. Modeling
                nb_model = MultinomialNB()
                nb_model.fit(X_train_tf, y_train)
                y_pred_nb = nb_model.predict(X_test_tf)
                
                svm_model = LinearSVC(random_state=42)
                svm_model.fit(X_train_tf, y_train)
                y_pred_svm = svm_model.predict(X_test_tf)
                
                st.session_state['nb_model'] = nb_model
                st.session_state['svm_model'] = svm_model
                st.session_state['tfidf'] = tfidf
                
                st.session_state['metrics'] = {
                    "nb_acc": accuracy_score(y_test, y_pred_nb),
                    "svm_acc": accuracy_score(y_test, y_pred_svm),
                    "nb_report": classification_report(y_test, y_pred_nb, zero_division=0),
                    "svm_report": classification_report(y_test, y_pred_svm, zero_division=0),
                    "y_test": y_test,
                    "y_pred_nb": y_pred_nb,
                    "y_pred_svm": y_pred_svm
                }
                
                # 9. Wordcloud
                pos_text = " ".join(df[df['label'] == 'Positif']['normalized_text'].dropna())
                neg_text = " ".join(df[df['label'] == 'Negatif']['normalized_text'].dropna())
                
                if pos_text: st.session_state['wc_pos'] = generate_wordcloud(pos_text, 'Greens')
                if neg_text: st.session_state['wc_neg'] = generate_wordcloud(neg_text, 'Reds')
                
        # Menampilkan Hasil setelah diproses (Di Luar scope Button)
        if st.session_state.get('process_done'):
            st.success("Proses Analisis Selesai!")
            
            with st.expander("3.1 Hasil Cleaning Data", expanded=True):
                display_paginated(st.session_state['df_clean'], "clean")
                
            with st.expander("3.2 Hasil Case Folding"):
                display_paginated(st.session_state['df_case'], "case")
                
            with st.expander("3.3 Hasil Filter Keyword & Tahun"):
                display_paginated(st.session_state['df_filter'], "filter")
                
            with st.expander("3.4 Hasil Normalisasi"):
                display_paginated(st.session_state['df_norm'], "norm")
                
            with st.expander("4. Hasil Pelabelan IndoBERTweet"):
                display_paginated(st.session_state['df_labeled'], "label")
                
            with st.expander("5. Info Split Data"):
                st.write(f"Distribusi Split Ratio -> Data Train: **{st.session_state['split_info']['train']}** baris | Data Test: **{st.session_state['split_info']['test']}** baris")
                
            with st.expander("6. TF-IDF & Keyword Document Frequency"):
                if not st.session_state['df_freq'].empty:
                    st.dataframe(st.session_state['df_freq'], use_container_width=True)
                else:
                    st.write("Tidak ada keyword filter yang terdaftar di top term TF-IDF.")
                    
            with st.expander("7. Hasil Evaluasi Model"):
                m = st.session_state['metrics']
                colA, colB = st.columns(2)
                with colA:
                    st.subheader(f"Naive Bayes (Akurasi: {m['nb_acc']:.4f})")
                    st.text(m['nb_report'])
                with colB:
                    st.subheader(f"SVM (Akurasi: {m['svm_acc']:.4f})")
                    st.text(m['svm_report'])
                    
            with st.expander("8. Confusion Matrix"):
                m = st.session_state['metrics']
                colA, colB = st.columns(2)
                with colA: st.pyplot(plot_confusion_matrix(m['y_test'], m['y_pred_nb'], "Naive Bayes"))
                with colB: st.pyplot(plot_confusion_matrix(m['y_test'], m['y_pred_svm'], "SVM"))
                    
            with st.expander("9. Wordcloud"):
                colA, colB = st.columns(2)
                if 'wc_pos' in st.session_state:
                    with colA:
                        st.subheader("Sentimen Positif")
                        st.pyplot(st.session_state['wc_pos'])
                if 'wc_neg' in st.session_state:
                    with colB:
                        st.subheader("Sentimen Negatif")
                        st.pyplot(st.session_state['wc_neg'])


with tab2:
    st.header("Prediksi Teks Tunggal")
    st.write("Coba masukkan teks secara manual. Sistem akan memproses dan mengklasifikasikan menggunakan model yang sudah dilatih di Fitur 1.")
    
    user_input = st.text_area("Masukkan teks:")
    
    if st.button("🔍 Analisis Teks"):
        if 'nb_model' not in st.session_state or 'svm_model' not in st.session_state:
            st.warning("⚠️ Silakan proses dan latih dataset di **Fitur 1** terlebih dahulu!")
        elif not user_input.strip():
            st.error("Teks tidak boleh kosong!")
        else:
            # Pipeline preprocessing manual
            cl_text = advanced_clean_text(user_input)
            cf_text = cl_text.lower()
            kamus_alay = load_kamus_alay()
            norm_text = normalize_text(cf_text, kamus_alay)
            
            # Prediksi
            nlp = load_indobertweet()
            indo_label, indo_score = label_indobertweet_biner(norm_text, nlp)
            
            vec = st.session_state['tfidf'].transform([norm_text])
            nb_pred = st.session_state['nb_model'].predict(vec)[0]
            svm_pred = st.session_state['svm_model'].predict(vec)[0]
            
            st.success("Analisis Berhasil!")
            st.markdown(f"**Teks Setelah Preprocessing:** `{norm_text}`")
            
            # Layout hasil tabel
            res_df = pd.DataFrame({
                "Model": ["IndoBERTweet", "Naive Bayes", "SVM (LinearSVC)"],
                "Prediksi Label": [indo_label, nb_pred, svm_pred],
                "Confidence/Score": [f"{indo_score:.4f}", "-", "-"]
            })
            st.table(res_df)
            
            st.divider()
            st.subheader("Metrik Performa Model (Berdasarkan Data Latih Saat Ini)")
            m = st.session_state['metrics']
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Naive Bayes (Akurasi: {m['nb_acc']:.2%})**")
                st.text(m['nb_report'])
            with c2:
                st.markdown(f"**SVM (Akurasi: {m['svm_acc']:.2%})**")
                st.text(m['svm_report'])
