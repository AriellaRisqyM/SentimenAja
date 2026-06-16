"""
Analisis Sentimen Teks Twitter terkait Polri
IndoBERTweet + TF-IDF + Naive Bayes + SVM
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import html
import warnings
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

import nltk

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis Sentimen Polri",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
h1 { font-size: 1.9rem !important; }
h2 { font-size: 1.4rem !important; margin-top: 1rem !important; }
.step-note { color: #555; font-size: 0.9rem; margin-bottom: 0.5rem; }
div[data-testid="stExpander"] > div > div { padding: 0.6rem 1rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# NLTK SETUP
# ─────────────────────────────────────────────────────────────
for _res in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_res}")
    except LookupError:
        try:
            nltk.download(_res, quiet=True)
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────
# CONSTANTS – DEFAULT POLRI KEYWORDS
# ─────────────────────────────────────────────────────────────
DEFAULT_KEYWORDS: list[str] = [
    "polri", "kepolisian", "mabes polri", "polda", "polres", "polsek",
    "polrestabes", "polresta", "brimob", "korbrimob", "gegana", "pelopor",
    "bareskrim", "ditreskrimum", "ditreskrimsus", "ditresnarkoba", "korlantas",
    "ditlantas", "satlantas", "intelkam", "satintelkam", "densus", "densus 88",
    "propam", "divpropam", "paminal", "wabprof", "provos", "polairud",
    "korpolairud", "sabhara", "samapta", "ditsamapta", "satsamapta", "binmas",
    "satbinmas", "bhabinkamtibmas", "polwan", "polisi", "kapolri", "wakapolri",
    "kapolda", "wakapolda", "kapolres", "wakapolres", "kapolsek", "wakapolsek",
    "penyidik", "reskrim", "kasat", "kanit", "jenderal polisi", "komjen", "irjen",
    "brigjen", "kombes", "akbp", "kompol", "akp", "iptu", "ipda", "aiptu",
    "aipda", "bripka", "brigpol", "brigadir", "briptu", "bripda", "bharada",
    "bharatu", "bharaka",
]

# ─────────────────────────────────────────────────────────────
# CACHED RESOURCES (persists across reruns, NOT across new sessions)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_indobertweet_model():
    """Load IndoBERTweet once. Cached at process level, never written to disk."""
    from transformers import pipeline as hf_pipeline  # lazy import

    model_name = "Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis"
    try:
        mdl = hf_pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=0,
        )
        return mdl, "GPU"
    except Exception:
        mdl = hf_pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
        )
        return mdl, "CPU"


@st.cache_data(show_spinner=False)
def load_kamus_alay() -> dict:
    """Load kamus alay sekali, cache di memori."""
    url = (
        "https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/"
        "kamus/nasalsabila_kamus-alay/_json_combined.json"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────
# PREPROCESSING FUNCTIONS
# ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)           # non-ASCII / emoji
    text = re.sub(r"https?://\S+", " ", text)             # URL http/https
    text = re.sub(r"pic\.twitter\.com\S*", " ", text)     # pic.twitter.com
    text = re.sub(r"@[\w]+", " ", text)                   # mention
    text = re.sub(r"#[\w]+", " ", text)                   # hashtag
    text = re.sub(r"[!$%^&*@#()_+|~=`{}\[\]%\-:;\"'<>?,./]", " ", text)
    text = re.sub(r"\d+", "", text)                       # angka
    text = re.sub(r"([a-zA-Z])\1{2,}", r"\1", text)      # karakter berulang ≥3
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def case_fold(text: str) -> str:
    return text.lower() if isinstance(text, str) else ""


def normalize_text(text: str, kamus: dict) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    return " ".join(kamus.get(w, w) for w in text.split())


def predict_indobertweet(text: str, model) -> tuple[str, float]:
    if isinstance(text, list):
        text = " ".join(text)
    if not isinstance(text, str) or not text.strip():
        return "Positif", 0.0
    try:
        results = model(text[:512], truncation=True, top_k=None)
        if isinstance(results[0], list):
            results = results[0]
        pos, neg = 0.0, 0.0
        for item in results:
            lbl = item["label"].lower()
            if "pos" in lbl or lbl == "label_2":
                pos = item["score"]
            elif "neg" in lbl or lbl == "label_0":
                neg = item["score"]
        total = pos + neg
        if total == 0:
            return "Positif", 0.0
        pn, nn = pos / total, neg / total
        return ("Positif", pn) if pn >= nn else ("Negatif", nn)
    except Exception:
        return "Positif", 0.0


# ─────────────────────────────────────────────────────────────
# SESSION STATE HELPERS
# ─────────────────────────────────────────────────────────────
_ALL_KEYS = [
    "uploaded_file_name", "selected_col",
    "df_raw", "df_cleaned", "df_casefolded", "df_filtered",
    "df_normalized", "df_labeled",
    "X_train", "X_test", "y_train", "y_test", "labels_sorted",
    "tfidf", "X_train_tfidf", "X_test_tfidf",
    "nb_model", "svm_model", "y_pred_nb", "y_pred_svm",
    "df_doc_freq", "valid_keywords",
]


def init_state() -> None:
    for k in _ALL_KEYS:
        if k not in st.session_state:
            st.session_state[k] = None


def _clear_keys(*keys: str) -> None:
    for k in keys:
        st.session_state[k] = None
    # clear pagination
    for pk in [k for k in st.session_state if k.startswith("_pag_")]:
        del st.session_state[pk]


def clear_from(step: str) -> None:
    order = [
        "df_cleaned", "df_casefolded", "df_filtered", "df_normalized",
        "df_labeled", "X_train", "X_test", "y_train", "y_test", "labels_sorted",
        "tfidf", "X_train_tfidf", "X_test_tfidf",
        "nb_model", "svm_model", "y_pred_nb", "y_pred_svm",
        "df_doc_freq", "valid_keywords",
    ]
    if step in order:
        _clear_keys(*order[order.index(step):])
    else:
        _clear_keys(*order)


def clear_all() -> None:
    _clear_keys(*_ALL_KEYS)


# ─────────────────────────────────────────────────────────────
# PAGINATION
# ─────────────────────────────────────────────────────────────
def show_paginated(df: pd.DataFrame, uid: str, rows: int = 5) -> None:
    n = len(df)
    if n == 0:
        st.info("Tidak ada data.")
        return
    n_pages = max(1, (n + rows - 1) // rows)
    pk = f"_pag_{uid}"
    if pk not in st.session_state:
        st.session_state[pk] = 1
    st.session_state[pk] = max(1, min(st.session_state[pk], n_pages))
    cur = st.session_state[pk]

    c1, c2, c3, c4, c5 = st.columns([1, 1, 4, 1, 1])
    if c1.button("⏮", key=f"{uid}_first"):
        st.session_state[pk] = 1
    if c2.button("◀", key=f"{uid}_prev"):
        st.session_state[pk] = max(1, cur - 1)
    c3.markdown(
        f"<div style='text-align:center;padding-top:7px'>"
        f"Halaman <b>{st.session_state[pk]}</b> / <b>{n_pages}</b>"
        f" &nbsp;|&nbsp; Total <b>{n}</b> baris</div>",
        unsafe_allow_html=True,
    )
    if c4.button("▶", key=f"{uid}_next"):
        st.session_state[pk] = min(n_pages, cur + 1)
    if c5.button("⏭", key=f"{uid}_last"):
        st.session_state[pk] = n_pages

    p = st.session_state[pk]
    s = (p - 1) * rows
    st.dataframe(
        df.iloc[s : s + rows].reset_index(drop=True),
        use_container_width=True,
    )


# ─────────────────────────────────────────────────────────────
# VISUALIZATION HELPERS
# ─────────────────────────────────────────────────────────────
def fig_dist(df: pd.DataFrame, col: str = "label") -> plt.Figure:
    counts = df[col].value_counts()
    pal = {"Positif": "#27ae60", "Negatif": "#e74c3c"}
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[pal.get(l, "#3498db") for l in counts.index],
        width=0.5,
    )
    for b in bars:
        ax.annotate(
            f"{int(b.get_height())}",
            (b.get_x() + b.get_width() / 2, b.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )
    ax.set_title("Distribusi Kelas Sentimen", fontweight="bold", pad=12)
    ax.set_xlabel("Kelas")
    ax.set_ylabel("Jumlah")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def fig_cm(y_true, y_pred, title: str, labels: list) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_title(f"Confusion Matrix – {title}", fontweight="bold", pad=12)
    ax.set_xlabel("Prediksi", fontsize=11)
    ax.set_ylabel("Aktual", fontsize=11)
    plt.tight_layout()
    return fig


def fig_bar_freq(
    df_freq: pd.DataFrame,
    term_col: str,
    freq_col: str,
    title: str,
    top_n: int = 30,
) -> plt.Figure:
    df_top = df_freq.head(top_n).copy()
    h = max(6, len(df_top) * 0.38)
    fig, ax = plt.subplots(figsize=(10, h))
    bars = ax.barh(range(len(df_top)), df_top[freq_col], color="steelblue")
    ax.set_yticks(range(len(df_top)))
    ax.set_yticklabels(df_top[term_col], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Document Frequency", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=12)
    mx = df_top[freq_col].max() if len(df_top) else 1
    for i, v in enumerate(df_top[freq_col]):
        ax.text(v + mx * 0.01, i, str(int(v)), va="center", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def fig_wc(text: str, title: str, cmap: str = "Greens") -> plt.Figure:
    wc = WordCloud(
        width=800, height=400, background_color="white",
        colormap=cmap, max_words=100, collocations=False,
    )
    wc.generate(text if text.strip() else "tidak ada data")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontweight="bold", fontsize=14, pad=12)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# SIDEBAR PIPELINE STATUS
# ─────────────────────────────────────────────────────────────
def sidebar_status() -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Status Pipeline**")
    items = [
        ("Data Dimuat", "df_raw"),
        ("Cleaning", "df_cleaned"),
        ("Case Folding", "df_casefolded"),
        ("Filter Keyword & Tahun", "df_filtered"),
        ("Normalisasi", "df_normalized"),
        ("Pelabelan IndoBERTweet", "df_labeled"),
        ("Data Split", "X_train"),
        ("TF-IDF", "tfidf"),
        ("Model NB & SVM", "nb_model"),
    ]
    for label, key in items:
        done = st.session_state.get(key) is not None
        icon = "✅" if done else "⬜"
        st.sidebar.markdown(f"{icon} {label}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main() -> None:
    init_state()

    st.sidebar.title("🔍 Analisis Sentimen Polri")
    st.sidebar.markdown("---")
    menu = st.sidebar.radio(
        "Menu Utama",
        ["📊 Analisis Dataset", "✍️ Analisis Teks Tunggal"],
    )
    sidebar_status()

    if menu == "📊 Analisis Dataset":
        page_dataset()
    else:
        page_single()


# ─────────────────────────────────────────────────────────────
# PAGE 1 – DATASET ANALYSIS
# ─────────────────────────────────────────────────────────────
def page_dataset() -> None:
    st.title("📊 Analisis Sentimen Dataset")

    # ── File Upload ──────────────────────────────────────────
    st.header("① Upload Dataset")
    st.info(
        "⚠️ **Catatan:** Hanya memuat dan memproses kolom teks yang Anda pilih. "
        "Kolom numerik, tanggal, dan lainnya tidak ikut diproses ke model."
    )

    uploaded = st.file_uploader(
        "Pilih file CSV atau Excel (.xlsx / .xls)",
        type=["csv", "xlsx", "xls"],
        key="file_uploader",
    )

    if uploaded is None:
        st.warning("⏳ Silakan upload file dataset untuk memulai.")
        return

    # Detect new file → clear all
    if st.session_state.get("uploaded_file_name") != uploaded.name:
        clear_all()
        st.session_state["uploaded_file_name"] = uploaded.name

    # Load raw data
    if st.session_state["df_raw"] is None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df_raw = pd.read_csv(
                    uploaded, sep=",", skipinitialspace=True, na_values="?"
                )
            else:
                df_raw = pd.read_excel(uploaded)
            df_raw = df_raw.dropna(how="all").reset_index(drop=True)
            st.session_state["df_raw"] = df_raw
        except Exception as exc:
            st.error(f"❌ Gagal memuat file: {exc}")
            return

    df_raw: pd.DataFrame = st.session_state["df_raw"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Baris", f"{len(df_raw):,}")
    c2.metric("Total Kolom", len(df_raw.columns))
    c3.metric("File", uploaded.name)

    # ── Column Selection ─────────────────────────────────────
    st.header("② Pilih Kolom Teks")

    text_cols = df_raw.select_dtypes(include=["object"]).columns.tolist()
    options = text_cols if text_cols else df_raw.columns.tolist()
    prev_col = st.session_state.get("selected_col")
    default_idx = options.index(prev_col) if prev_col in options else 0

    sel_col: str = st.selectbox(
        "Kolom yang berisi teks untuk dianalisis:",
        options=options,
        index=default_idx,
        help="Pilih kolom yang berisi konten teks (misal: full_text, tweet, content, dll)",
    )

    if prev_col != sel_col:
        clear_from("df_cleaned")
        st.session_state["selected_col"] = sel_col
    elif st.session_state["selected_col"] is None:
        st.session_state["selected_col"] = sel_col

    with st.expander("👀 Preview 5 baris pertama kolom terpilih"):
        st.dataframe(df_raw[[sel_col]].head(5), use_container_width=True)

    st.markdown("---")
    st.header("③ Preprocessing Pipeline")

    # ── STEP 1: CLEANING ─────────────────────────────────────
    with st.expander("📌 Langkah 1 – Text Cleaning", expanded=True):
        st.markdown(
            '<p class="step-note">Menghapus HTML entities, URL, mention (@), '
            "hashtag (#), simbol, angka, karakter berulang ≥ 3, dan spasi berlebih.</p>",
            unsafe_allow_html=True,
        )

        if st.button("▶ Jalankan Text Cleaning", key="btn_clean"):
            df = df_raw.copy()
            df = df.dropna(subset=[sel_col])
            n0 = len(df)
            df["original_text"] = df[sel_col].astype(str)
            df["cleaned_text"] = df[sel_col].astype(str).apply(clean_text)
            df = df[df["cleaned_text"].str.strip().astype(bool)]
            df = df.drop_duplicates(subset=["cleaned_text"]).reset_index(drop=True)
            nf = len(df)
            clear_from("df_cleaned")
            st.session_state["df_cleaned"] = df
            c1, c2, c3 = st.columns(3)
            c1.metric("Data Awal", f"{n0:,}")
            c2.metric("Tersisa", f"{nf:,}")
            c3.metric("Dihapus", f"{n0 - nf:,}")
            st.success("✅ Text cleaning selesai!")

        if st.session_state["df_cleaned"] is not None:
            st.markdown("**Hasil Text Cleaning (5 baris/hal):**")
            show_paginated(
                st.session_state["df_cleaned"][["original_text", "cleaned_text"]].rename(
                    columns={"original_text": "Teks Asli", "cleaned_text": "Teks Bersih"}
                ),
                "clean",
            )

    # ── STEP 2: CASE FOLDING ─────────────────────────────────
    with st.expander(
        "📌 Langkah 2 – Case Folding",
        expanded=st.session_state["df_cleaned"] is not None,
    ):
        st.markdown(
            '<p class="step-note">Mengubah semua teks menjadi huruf kecil (lowercase).</p>',
            unsafe_allow_html=True,
        )
        if st.session_state["df_cleaned"] is None:
            st.warning("⚠️ Selesaikan Text Cleaning terlebih dahulu.")
        else:
            if st.button("▶ Jalankan Case Folding", key="btn_cf"):
                df = st.session_state["df_cleaned"].copy()
                df["case_folded_text"] = df["cleaned_text"].apply(case_fold)
                clear_from("df_casefolded")
                st.session_state["df_casefolded"] = df
                st.success("✅ Case folding selesai!")

            if st.session_state["df_casefolded"] is not None:
                st.markdown("**Hasil Case Folding (5 baris/hal):**")
                show_paginated(
                    st.session_state["df_casefolded"][
                        ["cleaned_text", "case_folded_text"]
                    ].rename(
                        columns={
                            "cleaned_text": "Teks Bersih",
                            "case_folded_text": "Teks Case Folded",
                        }
                    ),
                    "cf",
                )

    # ── STEP 3: FILTER ───────────────────────────────────────
    with st.expander(
        "📌 Langkah 3 – Filter Keyword & Tahun",
        expanded=st.session_state["df_casefolded"] is not None,
    ):
        st.markdown(
            '<p class="step-note">Data yang tidak mengandung keyword pilihan '
            "dan/atau di luar rentang tahun akan dihapus.</p>",
            unsafe_allow_html=True,
        )
        if st.session_state["df_casefolded"] is None:
            st.warning("⚠️ Selesaikan Case Folding terlebih dahulu.")
        else:
            df_cf = st.session_state["df_casefolded"]
            has_date = "created_at" in df_cf.columns

            # Keyword config
            use_custom = st.checkbox("✏️ Kustomisasi keyword", key="chk_custom_kw")
            if use_custom:
                kw_raw = st.text_area(
                    "Keyword (pisahkan dengan koma):",
                    value=", ".join(DEFAULT_KEYWORDS),
                    height=130,
                    key="ta_keywords",
                )
                keywords = [k.strip() for k in kw_raw.split(",") if k.strip()]
                st.caption(f"{len(keywords)} keyword dikonfigurasi.")
            else:
                keywords = DEFAULT_KEYWORDS.copy()
                st.info(f"ℹ️ Menggunakan **{len(keywords)} keyword Polri** bawaan.")

            # Year filter
            st.markdown("**Filter Tahun:**")
            enable_year = st.checkbox(
                "Aktifkan filter tahun",
                value=has_date,
                disabled=not has_date,
                key="chk_year",
            )
            if not has_date:
                st.warning(
                    "⚠️ Kolom `created_at` tidak ditemukan — filter tahun tidak tersedia."
                )

            y_start, y_end = 2022, 2025
            if enable_year and has_date:
                c1, c2 = st.columns(2)
                y_start = c1.number_input(
                    "Tahun Mulai:", 2000, 2030, 2022, key="ni_ystart"
                )
                y_end = c2.number_input(
                    "Tahun Akhir:", 2000, 2030, 2025, key="ni_yend"
                )

            if st.button("▶ Jalankan Filter", key="btn_filter"):
                df = df_cf.copy()
                n0 = len(df)

                if enable_year and has_date:
                    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
                    target = list(range(int(y_start), int(y_end) + 1))
                    df["_yr"] = df["created_at"].dt.year
                    df = df[df["_yr"].isin(target)].drop(columns=["_yr"])
                    st.info(f"📅 Setelah filter tahun {y_start}–{y_end}: {len(df):,} baris")

                if keywords:
                    pat = r"\b(?:" + "|".join(map(re.escape, keywords)) + r")\b"
                    mask = df["case_folded_text"].str.contains(
                        pat, flags=re.IGNORECASE, na=False
                    )
                    df = df[mask].reset_index(drop=True)

                if df.empty:
                    st.error(
                        "❌ Tidak ada data tersisa. Ubah keyword atau rentang tahun."
                    )
                else:
                    clear_from("df_filtered")
                    st.session_state["df_filtered"] = df
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Awal", f"{n0:,}")
                    c2.metric("Tersisa", f"{len(df):,}")
                    c3.metric("Dihapus", f"{n0 - len(df):,}")
                    st.success(f"✅ Filter selesai! {len(df):,} baris tersisa.")

                    # year distribution
                    if has_date and "created_at" in df.columns:
                        yr = (
                            pd.to_datetime(df["created_at"], errors="coerce")
                            .dt.year.value_counts()
                            .sort_index()
                        )
                        fig, ax = plt.subplots(figsize=(7, 4))
                        ax.bar(yr.index.astype(str), yr.values, color="steelblue", width=0.5)
                        for i, v in enumerate(yr.values):
                            ax.text(i, v, str(v), ha="center", va="bottom", fontweight="bold")
                        ax.set_title("Distribusi Tweet per Tahun", fontweight="bold")
                        ax.set_xlabel("Tahun")
                        ax.set_ylabel("Jumlah")
                        ax.spines["top"].set_visible(False)
                        ax.spines["right"].set_visible(False)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

            if st.session_state["df_filtered"] is not None:
                st.markdown("**Hasil Filter (5 baris/hal):**")
                show_paginated(
                    st.session_state["df_filtered"][["case_folded_text"]].rename(
                        columns={"case_folded_text": "Teks Setelah Filter"}
                    ),
                    "filter",
                )

    # ── STEP 4: NORMALIZATION ────────────────────────────────
    with st.expander(
        "📌 Langkah 4 – Normalisasi (Kamus Alay)",
        expanded=st.session_state["df_filtered"] is not None,
    ):
        st.markdown(
            '<p class="step-note">Mengganti kata tidak baku / slang menggunakan kamus alay. '
            "Membutuhkan koneksi internet untuk mengunduh kamus.</p>",
            unsafe_allow_html=True,
        )
        if st.session_state["df_filtered"] is None:
            st.warning("⚠️ Selesaikan Filter terlebih dahulu.")
        else:
            if st.button("▶ Jalankan Normalisasi", key="btn_norm"):
                with st.spinner("Mengunduh kamus alay…"):
                    kamus = load_kamus_alay()

                if kamus:
                    st.info(f"📚 {len(kamus):,} kata dalam kamus normalisasi.")
                else:
                    st.warning(
                        "⚠️ Kamus tidak dapat diunduh. Teks asli digunakan tanpa normalisasi."
                    )

                df = st.session_state["df_filtered"].copy()
                df["normalized_text"] = df["case_folded_text"].apply(
                    lambda t: normalize_text(t, kamus)
                )
                clear_from("df_normalized")
                st.session_state["df_normalized"] = df
                st.success("✅ Normalisasi selesai!")

            if st.session_state["df_normalized"] is not None:
                st.markdown("**Hasil Normalisasi (5 baris/hal):**")
                show_paginated(
                    st.session_state["df_normalized"][
                        ["case_folded_text", "normalized_text"]
                    ].rename(
                        columns={
                            "case_folded_text": "Sebelum Normalisasi",
                            "normalized_text": "Sesudah Normalisasi",
                        }
                    ),
                    "norm",
                )

    # ── STEP 5: INDOBERTWEET LABELING ────────────────────────
    with st.expander(
        "📌 Langkah 5 – Pelabelan Sentimen (IndoBERTweet)",
        expanded=st.session_state["df_normalized"] is not None,
    ):
        st.markdown(
            '<p class="step-note">Melabeli sentimen menggunakan model '
            "<code>Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis</code>. "
            "Proses akan memakan waktu untuk dataset besar.</p>",
            unsafe_allow_html=True,
        )
        if st.session_state["df_normalized"] is None:
            st.warning("⚠️ Selesaikan Normalisasi terlebih dahulu.")
        else:
            df_norm = st.session_state["df_normalized"]
            n_total = len(df_norm)
            st.info(
                f"📊 Total data: **{n_total:,}** baris  \n"
                "⏱️ Estimasi waktu sangat tergantung hardware (GPU jauh lebih cepat)."
            )

            if st.button("▶ Mulai Pelabelan IndoBERTweet", key="btn_label"):
                with st.spinner("Memuat model IndoBERTweet (pertama kali butuh waktu)…"):
                    model, dev = load_indobertweet_model()
                st.info(f"✅ Model dimuat – perangkat: **{dev}**")

                prog = st.progress(0.0, text="Memulai pelabelan…")
                labels, scores = [], []

                for i, txt in enumerate(df_norm["normalized_text"].tolist()):
                    lbl, sc = predict_indobertweet(txt, model)
                    labels.append(lbl)
                    scores.append(sc)
                    if (i + 1) % 100 == 0 or (i + 1) == n_total:
                        pct = (i + 1) / n_total
                        prog.progress(pct, text=f"{i+1:,}/{n_total:,} diproses…")

                prog.empty()

                df = df_norm.copy()
                df["label"] = labels
                df["indobertweet_score"] = scores
                clear_from("df_labeled")
                st.session_state["df_labeled"] = df
                st.success("✅ Pelabelan selesai!")

                lc = df["label"].value_counts()
                c1, c2, c3 = st.columns(3)
                c1.metric("Total", f"{len(df):,}")
                c2.metric("Positif", f"{lc.get('Positif', 0):,}")
                c3.metric("Negatif", f"{lc.get('Negatif', 0):,}")
                st.pyplot(fig_dist(df))
                plt.close()

            if st.session_state["df_labeled"] is not None:
                st.markdown("**Hasil Pelabelan (5 baris/hal):**")
                show_paginated(
                    st.session_state["df_labeled"][
                        ["normalized_text", "label", "indobertweet_score"]
                    ].rename(
                        columns={
                            "normalized_text": "Teks",
                            "label": "Label",
                            "indobertweet_score": "Score",
                        }
                    ),
                    "label",
                )

    # ── STEP 6: DATA SPLIT ───────────────────────────────────
    with st.expander(
        "📌 Langkah 6 – Data Split",
        expanded=st.session_state["df_labeled"] is not None,
    ):
        st.markdown(
            '<p class="step-note">Membagi dataset menjadi data latih dan data uji '
            "dengan rasio yang dapat dikonfigurasi.</p>",
            unsafe_allow_html=True,
        )
        if st.session_state["df_labeled"] is None:
            st.warning("⚠️ Selesaikan Pelabelan terlebih dahulu.")
        else:
            df_lbl = st.session_state["df_labeled"]
            total = len(df_lbl)

            c1, c2 = st.columns(2)
            train_pct = c1.number_input(
                "Data Latih (%):", 50, 90, 70, 5, key="ni_train"
            )
            test_pct = int(100 - train_pct)
            c2.metric("Data Uji (%)", test_pct)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Data", f"{total:,}")
            c2.metric("Estimasi Train", f"{int(total * train_pct / 100):,}")
            c3.metric("Estimasi Test", f"{total - int(total * train_pct / 100):,}")

            if st.button("▶ Lakukan Data Split", key="btn_split"):
                X = df_lbl["normalized_text"].astype(str)
                y = df_lbl["label"]
                lbls_sorted = sorted(y.unique())

                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=test_pct / 100, random_state=42, stratify=y
                )
                clear_from("tfidf")
                st.session_state.update(
                    {
                        "X_train": X_tr,
                        "X_test": X_te,
                        "y_train": y_tr,
                        "y_test": y_te,
                        "labels_sorted": lbls_sorted,
                    }
                )
                st.success("✅ Data split selesai!")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Distribusi Data Latih:**")
                    for lbl, cnt in y_tr.value_counts().items():
                        st.write(f"- {lbl}: {cnt:,}")
                    st.write(
                        f"P(Positif) = {(y_tr == 'Positif').sum() / len(y_tr):.4f}"
                    )
                    st.write(
                        f"P(Negatif) = {(y_tr == 'Negatif').sum() / len(y_tr):.4f}"
                    )
                with c2:
                    st.markdown("**Distribusi Data Uji:**")
                    for lbl, cnt in y_te.value_counts().items():
                        st.write(f"- {lbl}: {cnt:,}")

    # ── STEP 7: TF-IDF ───────────────────────────────────────
    with st.expander(
        "📌 Langkah 7 – TF-IDF Vectorization",
        expanded=st.session_state.get("X_train") is not None,
    ):
        st.markdown(
            '<p class="step-note">Mengubah teks menjadi representasi vektor numerik '
            "menggunakan TF-IDF dengan dukungan n-gram.</p>",
            unsafe_allow_html=True,
        )
        if st.session_state.get("X_train") is None:
            st.warning("⚠️ Selesaikan Data Split terlebih dahulu.")
        else:
            c1, c2, c3 = st.columns(3)
            max_feat = c1.number_input(
                "Max Features:", 500, 30000, 5000, 500, key="ni_mf"
            )
            ng_min = c2.number_input("N-gram Min:", 1, 3, 1, key="ni_ngmin")
            ng_max = c3.number_input("N-gram Max:", 1, 3, 2, key="ni_ngmax")
            sublinear = st.checkbox("Sublinear TF", value=True, key="chk_sub")

            if st.button("▶ Jalankan TF-IDF", key="btn_tfidf"):
                with st.spinner("Membuat TF-IDF matrix…"):
                    tfidf = TfidfVectorizer(
                        ngram_range=(int(ng_min), int(ng_max)),
                        max_features=int(max_feat),
                        sublinear_tf=sublinear,
                    )
                    X_tr_t = tfidf.fit_transform(st.session_state["X_train"])
                    X_te_t = tfidf.transform(st.session_state["X_test"])

                    doc_freq = np.asarray((X_tr_t > 0).sum(axis=0)).flatten()
                    features = tfidf.get_feature_names_out()

                    df_doc = (
                        pd.DataFrame({"Term": features, "Document_Frequency": doc_freq})
                        .sort_values("Document_Frequency", ascending=False)
                        .reset_index(drop=True)
                    )
                    valid_kw = [k for k in DEFAULT_KEYWORDS if k in set(features)]

                    clear_from("nb_model")
                    st.session_state.update(
                        {
                            "tfidf": tfidf,
                            "X_train_tfidf": X_tr_t,
                            "X_test_tfidf": X_te_t,
                            "df_doc_freq": df_doc,
                            "valid_keywords": valid_kw,
                        }
                    )

                st.success(f"✅ TF-IDF selesai! {len(features):,} fitur unik.")
                c1, c2 = st.columns(2)
                c1.metric("Train Matrix", f"{X_tr_t.shape[0]:,} × {X_tr_t.shape[1]:,}")
                c2.metric("Test Matrix", f"{X_te_t.shape[0]:,} × {X_te_t.shape[1]:,}")

            if st.session_state.get("df_doc_freq") is not None:
                df_doc = st.session_state["df_doc_freq"]
                valid_kw = st.session_state.get("valid_keywords", [])

                tab1, tab2 = st.tabs(["📊 Top 30 Terms (DF)", "🔑 Keyword Polri (DF)"])

                with tab1:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Term Unik", f"{len(df_doc):,}")
                    c2.metric("Rata-rata DF", f"{df_doc['Document_Frequency'].mean():.2f}")
                    c3.metric("Max DF", f"{int(df_doc['Document_Frequency'].max()):,}")
                    c4.metric("Min DF", int(df_doc["Document_Frequency"].min()))
                    st.pyplot(
                        fig_bar_freq(
                            df_doc,
                            "Term",
                            "Document_Frequency",
                            "Top 30 Terms – Document Frequency",
                            30,
                        )
                    )
                    plt.close()
                    with st.expander("Lihat tabel Top 30"):
                        st.dataframe(df_doc.head(30), use_container_width=True)

                with tab2:
                    if valid_kw:
                        df_kw = (
                            pd.DataFrame(
                                {
                                    "Keyword": valid_kw,
                                    "Document_Frequency": [
                                        df_doc.loc[
                                            df_doc["Term"] == k, "Document_Frequency"
                                        ].values[0]
                                        if k in df_doc["Term"].values
                                        else 0
                                        for k in valid_kw
                                    ],
                                }
                            )
                            .sort_values("Document_Frequency", ascending=False)
                            .reset_index(drop=True)
                        )
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Keyword Ditemukan", len(df_kw))
                        c2.metric(
                            "Max DF",
                            f"{df_kw['Document_Frequency'].max():,} "
                            f"({df_kw.iloc[0]['Keyword']})",
                        )
                        c3.metric(
                            "Rata-rata DF",
                            f"{df_kw['Document_Frequency'].mean():.2f}",
                        )
                        st.pyplot(
                            fig_bar_freq(
                                df_kw,
                                "Keyword",
                                "Document_Frequency",
                                "Keyword Polri – Document Frequency",
                                len(df_kw),
                            )
                        )
                        plt.close()
                        with st.expander("Lihat tabel keyword"):
                            st.dataframe(df_kw, use_container_width=True)
                    else:
                        st.warning("Tidak ada keyword Polri yang ditemukan dalam fitur TF-IDF.")

    # ── STEP 8: MODEL ────────────────────────────────────────
    with st.expander(
        "📌 Langkah 8 – Model Machine Learning",
        expanded=st.session_state.get("X_train_tfidf") is not None,
    ):
        st.markdown(
            '<p class="step-note">Melatih Naive Bayes (MultinomialNB) dan '
            "SVM (LinearSVC) menggunakan fitur TF-IDF.</p>",
            unsafe_allow_html=True,
        )
        if st.session_state.get("X_train_tfidf") is None:
            st.warning("⚠️ Selesaikan TF-IDF terlebih dahulu.")
        else:
            if st.button("▶ Latih Semua Model", key="btn_train"):
                X_tr_t = st.session_state["X_train_tfidf"]
                X_te_t = st.session_state["X_test_tfidf"]
                y_tr = st.session_state["y_train"]
                y_te = st.session_state["y_test"]
                lbls = st.session_state["labels_sorted"]

                with st.spinner("Melatih Naive Bayes…"):
                    nb = MultinomialNB()
                    nb.fit(X_tr_t, y_tr)
                    y_pred_nb = nb.predict(X_te_t)

                with st.spinner("Melatih SVM…"):
                    svm = LinearSVC(random_state=42, max_iter=3000)
                    svm.fit(X_tr_t, y_tr)
                    y_pred_svm = svm.predict(X_te_t)

                st.session_state.update(
                    {
                        "nb_model": nb,
                        "svm_model": svm,
                        "y_pred_nb": y_pred_nb,
                        "y_pred_svm": y_pred_svm,
                    }
                )
                st.success("✅ Semua model berhasil dilatih!")

            if st.session_state.get("nb_model") is not None:
                y_te = st.session_state["y_test"]
                lbls = st.session_state["labels_sorted"]
                y_pred_nb = st.session_state["y_pred_nb"]
                y_pred_svm = st.session_state["y_pred_svm"]

                tab1, tab2 = st.tabs(["🔵 Naive Bayes (MultinomialNB)", "🟠 SVM (LinearSVC)"])

                with tab1:
                    acc = accuracy_score(y_te, y_pred_nb)
                    st.metric("Akurasi", f"{acc:.4f}  ({acc * 100:.2f}%)")
                    rep = classification_report(
                        y_te, y_pred_nb, labels=lbls, zero_division=0, output_dict=True
                    )
                    st.dataframe(
                        pd.DataFrame(rep)
                        .transpose()
                        .style.format("{:.4f}", na_rep="-"),
                        use_container_width=True,
                    )

                with tab2:
                    acc = accuracy_score(y_te, y_pred_svm)
                    st.metric("Akurasi", f"{acc:.4f}  ({acc * 100:.2f}%)")
                    rep = classification_report(
                        y_te, y_pred_svm, labels=lbls, zero_division=0, output_dict=True
                    )
                    st.dataframe(
                        pd.DataFrame(rep)
                        .transpose()
                        .style.format("{:.4f}", na_rep="-"),
                        use_container_width=True,
                    )

    # ── STEP 9: CONFUSION MATRIX ─────────────────────────────
    with st.expander(
        "📌 Langkah 9 – Confusion Matrix",
        expanded=st.session_state.get("nb_model") is not None,
    ):
        if st.session_state.get("nb_model") is None:
            st.warning("⚠️ Selesaikan training model terlebih dahulu.")
        else:
            y_te = st.session_state["y_test"]
            lbls = st.session_state["labels_sorted"]
            y_pred_nb = st.session_state["y_pred_nb"]
            y_pred_svm = st.session_state["y_pred_svm"]

            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(fig_cm(y_te, y_pred_nb, "Naive Bayes", lbls))
                plt.close()
            with c2:
                st.pyplot(fig_cm(y_te, y_pred_svm, "SVM (LinearSVC)", lbls))
                plt.close()

    # ── STEP 10: WORD CLOUD ──────────────────────────────────
    with st.expander(
        "📌 Langkah 10 – Word Cloud",
        expanded=st.session_state.get("df_labeled") is not None,
    ):
        if st.session_state.get("df_labeled") is None:
            st.warning("⚠️ Selesaikan Pelabelan terlebih dahulu.")
        else:
            if st.button("▶ Generate Word Cloud", key="btn_wc"):
                df_lbl = st.session_state["df_labeled"]
                pos_txt = " ".join(
                    df_lbl[df_lbl["label"] == "Positif"]["normalized_text"]
                    .dropna()
                    .astype(str)
                )
                neg_txt = " ".join(
                    df_lbl[df_lbl["label"] == "Negatif"]["normalized_text"]
                    .dropna()
                    .astype(str)
                )
                c1, c2 = st.columns(2)
                with c1:
                    st.pyplot(
                        fig_wc(pos_txt, "Word Cloud – Sentimen Positif", "Greens")
                    )
                    plt.close()
                with c2:
                    st.pyplot(
                        fig_wc(neg_txt, "Word Cloud – Sentimen Negatif", "Reds")
                    )
                    plt.close()


# ─────────────────────────────────────────────────────────────
# PAGE 2 – SINGLE TEXT ANALYSIS
# ─────────────────────────────────────────────────────────────
def page_single() -> None:
    st.title("✍️ Analisis Teks Tunggal")

    has_ml = (
        st.session_state.get("nb_model") is not None
        and st.session_state.get("svm_model") is not None
        and st.session_state.get("tfidf") is not None
    )

    if has_ml:
        st.success(
            "✅ Model NB & SVM tersedia dari analisis dataset. "
            "Prediksi ketiga model (IndoBERTweet + NB + SVM) akan ditampilkan."
        )
    else:
        st.info(
            "ℹ️ Model NB & SVM belum tersedia. Hanya IndoBERTweet yang akan digunakan.  \n"
            "Untuk prediksi NB & SVM beserta metrik lengkap, selesaikan pipeline "
            "**Analisis Dataset** terlebih dahulu."
        )

    input_text = st.text_area(
        "Masukkan teks yang ingin dianalisis:",
        height=140,
        placeholder="Contoh: Polisi berhasil mengamankan pelaku dengan cepat dan profesional...",
        key="ta_single",
    )

    if st.button("🔍 Analisis Sentimen", type="primary", key="btn_analyze"):
        if not input_text.strip():
            st.error("❌ Teks tidak boleh kosong!")
            return

        with st.spinner("Memproses teks…"):
            cleaned = clean_text(input_text)
            folded = case_fold(cleaned)
            kamus = load_kamus_alay()
            normed = normalize_text(folded, kamus)

            model, dev = load_indobertweet_model()
            lbl_bert, sc_bert = predict_indobertweet(normed, model)

        st.markdown("---")
        st.subheader("📊 Hasil Analisis")

        # ── IndoBERTweet Result ──────────────────────────────
        st.markdown("### 🤖 IndoBERTweet")
        icon_b = "🟢" if lbl_bert == "Positif" else "🔴"
        c1, c2, c3 = st.columns(3)
        c1.metric("Sentimen", f"{icon_b} {lbl_bert}")
        c2.metric("Confidence Score", f"{sc_bert:.4f}")
        c3.metric("Keyakinan (%)", f"{sc_bert * 100:.2f}%")

        # Confidence bar
        fig, ax = plt.subplots(figsize=(7, 1.4))
        clr = "#27ae60" if lbl_bert == "Positif" else "#e74c3c"
        ax.barh([""], [sc_bert], color=clr, height=0.5)
        ax.barh([""], [1 - sc_bert], left=[sc_bert], color="#ecf0f1", height=0.5)
        ax.set_xlim(0, 1)
        ax.text(
            sc_bert / 2, 0, f"{sc_bert * 100:.1f}%",
            ha="center", va="center", color="white", fontweight="bold", fontsize=11,
        )
        ax.set_title(f"Confidence: {lbl_bert} ({sc_bert * 100:.1f}%)", fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Preprocessing Detail ──────────────────────────────
        with st.expander("🔍 Detail Preprocessing"):
            rows = [
                ("Teks Asli", input_text),
                ("Setelah Cleaning", cleaned),
                ("Setelah Case Folding", folded),
                ("Setelah Normalisasi", normed),
            ]
            for lbl, val in rows:
                st.markdown(f"**{lbl}:**")
                st.write(val or "_(kosong setelah diproses)_")

        # ── ML Model predictions ──────────────────────────────
        if has_ml:
            st.markdown("### 🤖 Prediksi Model ML")

            tfidf = st.session_state["tfidf"]
            nb_mdl = st.session_state["nb_model"]
            svm_mdl = st.session_state["svm_model"]
            y_te = st.session_state["y_test"]
            y_pred_nb = st.session_state["y_pred_nb"]
            y_pred_svm = st.session_state["y_pred_svm"]
            lbls = st.session_state["labels_sorted"]

            vec = tfidf.transform([normed])
            pred_nb = nb_mdl.predict(vec)[0]
            pred_svm = svm_mdl.predict(vec)[0]

            try:
                proba_nb = dict(zip(nb_mdl.classes_, nb_mdl.predict_proba(vec)[0]))
            except Exception:
                proba_nb = {}

            c1, c2 = st.columns(2)
            with c1:
                icon_nb = "🟢" if pred_nb == "Positif" else "🔴"
                st.metric("Naive Bayes (MultinomialNB)", f"{icon_nb} {pred_nb}")
                if proba_nb:
                    for cls, p in proba_nb.items():
                        st.write(f"- P({cls}): {p:.4f}")
            with c2:
                icon_svm = "🟢" if pred_svm == "Positif" else "🔴"
                st.metric("SVM (LinearSVC)", f"{icon_svm} {pred_svm}")

            # ── Performance metrics ───────────────────────────
            st.markdown("### 📈 Performa Model dari Dataset Uji")

            acc_nb = accuracy_score(y_te, y_pred_nb)
            acc_svm = accuracy_score(y_te, y_pred_svm)
            rep_nb = classification_report(
                y_te, y_pred_nb, labels=lbls, zero_division=0, output_dict=True
            )
            rep_svm = classification_report(
                y_te, y_pred_svm, labels=lbls, zero_division=0, output_dict=True
            )

            def wa(rep: dict, key: str) -> float:
                return rep.get("weighted avg", {}).get(key, 0.0)

            tab1, tab2 = st.tabs(["🔵 Naive Bayes", "🟠 SVM"])

            with tab1:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Akurasi", f"{acc_nb:.4f}")
                c2.metric("Presisi (WA)", f"{wa(rep_nb, 'precision'):.4f}")
                c3.metric("Recall (WA)", f"{wa(rep_nb, 'recall'):.4f}")
                c4.metric("F1-Score (WA)", f"{wa(rep_nb, 'f1-score'):.4f}")
                st.dataframe(
                    pd.DataFrame(rep_nb).transpose().style.format("{:.4f}", na_rep="-"),
                    use_container_width=True,
                )

            with tab2:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Akurasi", f"{acc_svm:.4f}")
                c2.metric("Presisi (WA)", f"{wa(rep_svm, 'precision'):.4f}")
                c3.metric("Recall (WA)", f"{wa(rep_svm, 'recall'):.4f}")
                c4.metric("F1-Score (WA)", f"{wa(rep_svm, 'f1-score'):.4f}")
                st.dataframe(
                    pd.DataFrame(rep_svm).transpose().style.format("{:.4f}", na_rep="-"),
                    use_container_width=True,
                )


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
