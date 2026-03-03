import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from linearmodels.panel import PooledOLS, PanelOLS, RandomEffects
from scipy import stats
from scipy.stats import chi2, norm
import io
import base64

st.set_page_config(page_title="Data Science Econometrics", layout="wide")

# =========================
# SESSION STATE
# =========================
if 'df' not in st.session_state:
    st.session_state.df = None
if 'daerah_col' not in st.session_state:
    st.session_state.daerah_col = None
if 'tahun_col' not in st.session_state:
    st.session_state.tahun_col = None
if 'dep_var' not in st.session_state:
    st.session_state.dep_var = None
if 'indep_vars' not in st.session_state:
    st.session_state.indep_vars = []
if 'daerah_order' not in st.session_state:
    st.session_state.daerah_order = None
if 'original_indep_vars' not in st.session_state:
    st.session_state.original_indep_vars = []

# =========================
# SIDEBAR
# =========================

# Identitas
st.sidebar.image("Logo_UPN.png", use_container_width=True)
st.sidebar.markdown("## Universitas Pembangunan Nasional 'Veteran' Jawa Timur")
st.sidebar.markdown("### Mohamad Ibnu Fajar Maulana")
st.sidebar.markdown("21083010106")

st.sidebar.markdown("---")

# Navigasi
menu = st.sidebar.radio("Pilih Menu:", [
    "Home",
    "Data dan Analisis",
    "Uji Spesifikasi Model",
    "Uji Signifikansi Parameter",
    "Estimasi Model Regresi Panel",
    "Uji Asumsi Residual",
    "Evaluasi & Rekomendasi Model"
])

# =========================
# HELPER FUNCTIONS
# =========================
def check_data_ready():
    """Check if data is uploaded and configured"""
    if st.session_state.df is None:
        st.warning("⚠️ Silakan upload data terlebih dahulu di menu **Data dan Analisis**")
        return False
    if st.session_state.daerah_col is None or st.session_state.tahun_col is None:
        st.warning("⚠️ Silakan konfigurasi kolom cross-section dan time di menu **Data dan Analisis**")
        return False
    if st.session_state.dep_var is None or len(st.session_state.indep_vars) == 0:
        st.warning("⚠️ Silakan pilih variabel Respon dan Prediktor di menu **Data dan Analisis**")
        return False
    return True

def get_ordered_df():
    """Get dataframe with proper daerah ordering"""
    df = st.session_state.df.copy()
    daerah_col = st.session_state.daerah_col
    tahun_col = st.session_state.tahun_col
    
    # Jawa Barat standard order (from notebook)
    jabar_order = [
        "Kabupaten Bogor", "Kabupaten Sukabumi", "Kabupaten Cianjur", "Kabupaten Bandung",
        "Kabupaten Garut", "Kabupaten Tasikmalaya", "Kabupaten Ciamis", "Kabupaten Kuningan",
        "Kabupaten Cirebon", "Kabupaten Majalengka", "Kabupaten Sumedang", "Kabupaten Indramayu",
        "Kabupaten Subang", "Kabupaten Purwakarta", "Kabupaten Karawang", "Kabupaten Bekasi",
        "Kabupaten Bandung Barat", "Kabupaten Pangandaran", "Kota Bogor", "Kota Sukabumi",
        "Kota Bandung", "Kota Cirebon", "Kota Bekasi", "Kota Depok", "Kota Cimahi",
        "Kota Tasikmalaya", "Kota Banjar"
    ]
    
    # Apply daerah ordering
    if st.session_state.daerah_order is not None and len(st.session_state.daerah_order) > 0:
        # Use user-specified order
        df[daerah_col] = pd.Categorical(
            df[daerah_col],
            categories=st.session_state.daerah_order,
            ordered=True
        )
    else:
        # Auto-detect Jawa Barat data and apply standard order
        current_daerah = df[daerah_col].unique().tolist()
        is_jabar = all(d in jabar_order for d in current_daerah) and len(current_daerah) > 0
        if is_jabar:
            final_order = [d for d in jabar_order if d in current_daerah]
            df[daerah_col] = pd.Categorical(
                df[daerah_col],
                categories=final_order,
                ordered=True
            )
            st.session_state.daerah_order = final_order  # Save for consistency
    
    df = df.sort_values([daerah_col, tahun_col]).reset_index(drop=True)
    return df

def get_full_models():
    """Build full OWFE and TWFE models with ALL original variables (for Uji F & Residual tests)"""
    df = get_ordered_df()
    
    dummies_daerah = pd.get_dummies(df[st.session_state.daerah_col], drop_first=True, dtype=float)
    dummies_tahun = pd.get_dummies(df[st.session_state.tahun_col], drop_first=True, dtype=float)
    
    y = df[st.session_state.dep_var].astype(float)
    
    # Use original_indep_vars (all variables before backward elimination)
    # Fallback to indep_vars if original_indep_vars is empty
    indep_vars = st.session_state.original_indep_vars if st.session_state.original_indep_vars else st.session_state.indep_vars
    
    # If still empty, try to infer from data (for Jawa Barat TPT analysis)
    if not indep_vars:
        possible_vars = ["TPAK", "IPM", "UMK", "APS"]
        indep_vars = [v for v in possible_vars if v in df.columns]
    
    # Save to session state for consistency
    if not st.session_state.original_indep_vars and indep_vars:
        st.session_state.original_indep_vars = indep_vars.copy()
    
    # For Jawa Barat data (TPT analysis), use specific variable order from notebook: TPAK, IPM, UMK, APS
    jabar_indep_order = ["TPAK", "IPM", "UMK", "APS"]
    if all(v in indep_vars for v in jabar_indep_order) and len(indep_vars) == 4:
        indep_vars = jabar_indep_order
    
    # OWFE (full model)
    X_owfe = pd.concat([df[indep_vars].astype(float), dummies_daerah], axis=1)
    X_owfe = sm.add_constant(X_owfe, has_constant="add").astype(float)
    models_owfe = sm.OLS(y, X_owfe).fit()
    
    # TWFE (full model)
    X_twfe = pd.concat([df[indep_vars].astype(float), dummies_daerah, dummies_tahun], axis=1)
    X_twfe = sm.add_constant(X_twfe, has_constant="add").astype(float)
    models_twfe = sm.OLS(y, X_twfe).fit()
    
    return models_owfe, models_twfe, df, y, dummies_daerah, dummies_tahun, indep_vars

def get_models():
    """Build OWFE and TWFE models with current indep_vars (for backward elimination)"""
    df = get_ordered_df()
    
    dummies_daerah = pd.get_dummies(df[st.session_state.daerah_col], drop_first=True, dtype=float)
    dummies_tahun = pd.get_dummies(df[st.session_state.tahun_col], drop_first=True, dtype=float)
    
    y = df[st.session_state.dep_var].astype(float)
    
    # OWFE
    X_owfe = pd.concat([df[st.session_state.indep_vars].astype(float), dummies_daerah], axis=1)
    X_owfe = sm.add_constant(X_owfe, has_constant="add").astype(float)
    model_owfe = sm.OLS(y, X_owfe).fit()
    
    # TWFE
    X_twfe = pd.concat([df[st.session_state.indep_vars].astype(float), dummies_daerah, dummies_tahun], axis=1)
    X_twfe = sm.add_constant(X_twfe, has_constant="add").astype(float)
    model_twfe = sm.OLS(y, X_twfe).fit()
    
    return model_owfe, model_twfe, df, y, dummies_daerah, dummies_tahun

def get_backward_models():
    """
    Build OWFE and TWFE models with SIGNIFICANT variables only (after backward elimination).
    This is model_owfe/model_twfe - used for Estimasi Model.
    """
    df = get_ordered_df()
    
    dummies_daerah = pd.get_dummies(df[st.session_state.daerah_col], drop_first=True, dtype=float)
    dummies_tahun = pd.get_dummies(df[st.session_state.tahun_col], drop_first=True, dtype=float)
    
    y = df[st.session_state.dep_var].astype(float)
    
    # Start with all original variables
    indep_vars_orig = st.session_state.original_indep_vars if st.session_state.original_indep_vars else st.session_state.indep_vars
    
    # For Jawa Barat data, use specific order
    jabar_indep_order = ["TPAK", "IPM", "UMK", "APS"]
    if all(v in indep_vars_orig for v in jabar_indep_order) and len(indep_vars_orig) == 4:
        indep_vars_orig = jabar_indep_order
    
    alpha = 0.05
    
    # ==================
    # BACKWARD ELIMINATION FOR OWFE
    # ==================
    indep_vars_owfe = indep_vars_orig.copy()
    while True:
        X_main = df[indep_vars_owfe].astype(float)
        X = pd.concat([X_main, dummies_daerah], axis=1)
        X = sm.add_constant(X, has_constant="add").astype(float)
        model = sm.OLS(y, X).fit()
        
        pvals = model.pvalues[indep_vars_owfe]
        max_p = pvals.max()
        worst_var = pvals.idxmax()
        
        if max_p > alpha and len(indep_vars_owfe) > 1:
            indep_vars_owfe.remove(worst_var)
        else:
            model_owfe = model
            break
    
    # ==================
    # BACKWARD ELIMINATION FOR TWFE
    # ==================
    indep_vars_twfe = indep_vars_orig.copy()
    while True:
        X_main = df[indep_vars_twfe].astype(float)
        X = pd.concat([X_main, dummies_daerah, dummies_tahun], axis=1)
        X = sm.add_constant(X, has_constant="add").astype(float)
        model = sm.OLS(y, X).fit()
        
        pvals = model.pvalues[indep_vars_twfe]
        max_p = pvals.max()
        worst_var = pvals.idxmax()
        
        if max_p > alpha and len(indep_vars_twfe) > 1:
            indep_vars_twfe.remove(worst_var)
        else:
            model_twfe = model
            break
    
    return model_owfe, model_twfe, df, y, dummies_daerah, dummies_tahun, indep_vars_owfe, indep_vars_twfe

# =========================
# 1. HOME
# =========================
if menu == "Home":
    st.title("Pemodelan Tingkat Pengangguran Terbuka dengan Pendekatan Regresi Data Panel")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Selamat Datang! 👋
        
        Aplikasi ini dirancang untuk melakukan **analisis regresi data panel** secara komprehensif.
        Data panel adalah gabungan data cross-section dan time series yang memungkinkan analisis
        lebih mendalam tentang variasi antar individu dan waktu.
        
        ### Fitur Utama:
        
        1. **Data dan Analisis**
           - Upload data (CSV/Excel)
           - Eksplorasi Data (EDA)
           - Statistik Deskriptif
           - Uji Multikolinearitas (VIF)
        
        2. **Uji Spesifikasi Model**
           - Uji Chow (CEM vs FEM)
           - Uji Hausman (FEM vs REM)
           - Uji Lagrange Multiplier (CEM vs REM)
        
        3. **Uji Signifikansi Parameter**
           - Uji F (Simultan)
           - Uji t dengan Backward Elimination
        
        4. **Estimasi Model Regresi Panel**
           - One-Way Fixed Effect Model (OWFE)
           - Two-Way Fixed Effect Model (TWFE)
        
        5. **Uji Asumsi Residual**
           - Uji Autokorelasi (Breusch-Godfrey)
           - Uji Heteroskedastisitas (Glejser)
           - Uji Normalitas (Jarque-Bera)
        
        6. **Evaluasi & Rekomendasi Model**
           - R-Squared, Adjusted R-Squared
           - AIC,MAPE
        
        ### 🚀 Cara Menggunakan:
        1. Pilih menu **Data dan Analisis** di sidebar
        2. Upload file data panel Anda (CSV atau Excel)
        3. Konfigurasi kolom cross-section, time, dan variabel
        4. Lanjutkan ke analisis sesuai kebutuhan
        
        ---
        *Aplikasi ini mendukung data panel dari seluruh provinsi di Indonesia*
        """)
    
    with col2:
        # Display a relevant image
        st.markdown("### 📈 Data Panel Analysis")
        
        # Create a sample visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Sample panel data visualization
        np.random.seed(42)
        years = [2019, 2020, 2021, 2022, 2023, 2024]
        regions = ['Region A', 'Region B', 'Region C', 'Region D']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, region in enumerate(regions):
            values = np.random.randn(len(years)).cumsum() + 10 + i*2
            ax.plot(years, values, marker='o', linewidth=2, label=region, color=colors[i])
        
        ax.set_xlabel('Tahun', fontsize=12)
        ax.set_ylabel('Nilai', fontsize=12)
        ax.set_title('Ilustrasi Data Panel', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.info("""
        **💡 Tips:**
        - Pastikan data sudah dalam format panel (cross-section × time)
        - Periksa missing value sebelum analisis
        - Gunakan VIF untuk cek multikolinearitas
        """)

# =========================
# 2. DATA DAN ANALISIS
# =========================
elif menu == "Data dan Analisis":
    st.title("Data dan Analisis")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["📤 Upload Data", "📊 Eksplorasi Data (EDA)"])
    
    # =====================
    # TAB 1: UPLOAD DATA
    # =====================
    with tab1:
        st.header("Upload Data Panel")
        
        uploaded_file = st.file_uploader(
            "Pilih file CSV atau Excel",
            type=['csv', 'xlsx', 'xls'],
            help="Upload file data panel dalam format CSV atau Excel"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.df = df
                st.success(f"✅ Data berhasil diupload! ({len(df)} baris, {len(df.columns)} kolom)")
                
            except Exception as e:
                st.error(f"❌ Error membaca file: {str(e)}")
        
        if st.session_state.df is not None:
            df = st.session_state.df
            
            st.subheader("Preview Data")
            st.dataframe(df.head(20), width='stretch')
            
            st.subheader("Konfigurasi Variabel")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Cross-section column
                daerah_col = st.selectbox(
                    "Kolom Cross-Section (Daerah/Provinsi/Kabupaten):",
                    options=df.columns.tolist(),
                    index=0 if st.session_state.daerah_col is None else df.columns.tolist().index(st.session_state.daerah_col) if st.session_state.daerah_col in df.columns else 0,
                    help="Pilih kolom yang berisi identitas cross-section (misal: daerah, provinsi)"
                )
                st.session_state.daerah_col = daerah_col
                
                # Time column
                tahun_col = st.selectbox(
                    "Kolom Time (Tahun/Periode):",
                    options=df.columns.tolist(),
                    index=1 if st.session_state.tahun_col is None else df.columns.tolist().index(st.session_state.tahun_col) if st.session_state.tahun_col in df.columns else 1,
                    help="Pilih kolom yang berisi identitas waktu (misal: tahun)"
                )
                st.session_state.tahun_col = tahun_col
            
            with col2:
                # Response variable - filter out index columns
                all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [c for c in all_numeric_cols if c != daerah_col and c != tahun_col]
                
                dep_var = st.selectbox(
                    "Variabel Respon (Y):",
                    options=numeric_cols,
                    index=0 if st.session_state.dep_var is None or st.session_state.dep_var not in numeric_cols else numeric_cols.index(st.session_state.dep_var),
                    help="Pilih variabel Respon yang akan dianalisis"
                )
                st.session_state.dep_var = dep_var
                
                # Prediktor variables - exclude dep_var
                available_indep = [c for c in numeric_cols if c != dep_var]
                default_indep = [v for v in st.session_state.indep_vars if v in available_indep]
                if not default_indep:
                    default_indep = available_indep[:4] if len(available_indep) >= 4 else available_indep
                
                indep_vars = st.multiselect(
                    "Variabel Prediktor (X):",
                    options=available_indep,
                    default=default_indep,
                    help="Pilih variabel Prediktor untuk model"
                )
                st.session_state.indep_vars = indep_vars
                # Save original_indep_vars for full model (Uji F & Uji Asumsi Residual)
                st.session_state.original_indep_vars = indep_vars.copy()
            
            st.markdown("---")
            
            # Daerah Ordering Configuration
            st.subheader("Konfigurasi Urutan Cross-Section")
            
            # Default order for Jawa Barat (from notebook)
            jabar_order = [
                "Kabupaten Bogor", "Kabupaten Sukabumi", "Kabupaten Cianjur", "Kabupaten Bandung",
                "Kabupaten Garut", "Kabupaten Tasikmalaya", "Kabupaten Ciamis", "Kabupaten Kuningan",
                "Kabupaten Cirebon", "Kabupaten Majalengka", "Kabupaten Sumedang", "Kabupaten Indramayu",
                "Kabupaten Subang", "Kabupaten Purwakarta", "Kabupaten Karawang", "Kabupaten Bekasi",
                "Kabupaten Bandung Barat", "Kabupaten Pangandaran", "Kota Bogor", "Kota Sukabumi",
                "Kota Bandung", "Kota Cirebon", "Kota Bekasi", "Kota Depok", "Kota Cimahi",
                "Kota Tasikmalaya", "Kota Banjar"
            ]
            
            current_daerah_list = df[daerah_col].unique().tolist()
            
            # Auto-detect Jawa Barat data
            is_jabar = all(d in jabar_order for d in current_daerah_list) and len(current_daerah_list) > 0
            
            if is_jabar:
                # Automatically apply Jawa Barat ordering
                final_order = [d for d in jabar_order if d in current_daerah_list]
                st.session_state.daerah_order = final_order
                st.success("✅ Data terdeteksi sebagai data Jawa Barat. Urutan otomatis: Kab. Bogor → Kota Banjar")
                
                with st.expander("Lihat Urutan Daerah"):
                    for i, d in enumerate(final_order, 1):
                        st.write(f"{i}. {d}")
            else:
                st.info("⚠️ **Penting:** Urutan daerah mempengaruhi koefisien fixed effect.")
                
                use_custom_order = st.checkbox(
                    "Gunakan urutan kustom untuk cross-section",
                    value=st.session_state.daerah_order is not None,
                    help="Centang untuk mengatur urutan daerah secara manual"
                )
                
                if use_custom_order:
                    order_text = st.text_area(
                        "Masukkan urutan daerah (satu per baris):",
                        value="\n".join(current_daerah_list),
                        height=300,
                        help="Urutkan daerah sesuai kebutuhan, satu nama per baris"
                    )
                    
                    if order_text.strip():
                        custom_order = [d.strip() for d in order_text.strip().split("\n") if d.strip()]
                        missing = [d for d in current_daerah_list if d not in custom_order]
                        extra = [d for d in custom_order if d not in current_daerah_list]
                        
                        if missing:
                            st.error(f"Daerah berikut tidak ada dalam urutan: {missing}")
                        elif extra:
                            st.error(f"Daerah berikut tidak ada dalam data: {extra}")
                        else:
                            st.session_state.daerah_order = custom_order
                            st.success("✅ Urutan daerah tersimpan")
                else:
                    st.session_state.daerah_order = None
            
            st.markdown("---")
            
            # Data Info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Jumlah Cross-Section", df[daerah_col].nunique())
            with col2:
                st.metric("Jumlah Periode", df[tahun_col].nunique())
            with col3:
                st.metric("Total Observasi", len(df))
            
            # Missing Value Check
            st.subheader("Pengecekan Missing Value")
            missing_counts = df.isnull().sum()
            missing_df = pd.DataFrame({
                'Kolom': missing_counts.index,
                'Missing': missing_counts.values,
                'Persentase (%)': (missing_counts.values / len(df) * 100).round(2)
            })
            
            if missing_counts.sum() == 0:
                st.success("✅ Tidak ditemukan missing value pada data.")
            else:
                st.warning("⚠️ Ditemukan missing value pada data:")
                st.dataframe(missing_df[missing_df['Missing'] > 0], width='stretch')
            
            # Data Types
            st.subheader("Tipe Data")
            dtype_df = pd.DataFrame({
                'Kolom': df.dtypes.index,
                'Tipe Data': df.dtypes.values.astype(str)
            })
            st.dataframe(dtype_df, width='stretch')
    
    # =====================
    # TAB 2: EDA
    # =====================
    with tab2:
        st.header("Eksplorasi Data (EDA)")
        
        if st.session_state.df is None:
            st.warning("⚠️ Silakan upload data terlebih dahulu di tab **Upload Data**")
        else:
            df = st.session_state.df
            daerah_col = st.session_state.daerah_col
            tahun_col = st.session_state.tahun_col
            dep_var = st.session_state.dep_var
            indep_vars = st.session_state.indep_vars
            
            all_vars = [dep_var] + indep_vars if dep_var else indep_vars
            
            # Dropdown for variable selection
            selected_var = st.selectbox(
                "Pilih Variabel untuk Eksplorasi:",
                options=all_vars,
                help="Pilih variabel yang ingin dieksplorasi"
            )
            
            if selected_var:
                st.markdown("---")
                
                # ========================
                # GRAFIK PERKEMBANGAN
                # ========================
                st.subheader(f"📈 Perkembangan {selected_var} per Wilayah")
                
                fig, ax = plt.subplots(figsize=(14, 8))
                for d in df[daerah_col].unique():
                    subset = df[df[daerah_col] == d]
                    ax.plot(subset[tahun_col], subset[selected_var], marker='o', linestyle='--', label=d)
                ax.set_xlabel("Tahun")
                ax.set_ylabel(selected_var)
                ax.set_title(f"Perkembangan {selected_var} per Wilayah")
                ax.grid(True)
                ax.legend(title="Wilayah", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                
                # ========================
                # RATA-RATA PER WILAYAH
                # ========================
                st.subheader(f"📊 Rata-rata {selected_var} per Wilayah")
                
                var_wilayah = df.groupby(daerah_col, observed=True)[selected_var].mean()
                x = np.arange(len(var_wilayah))
                mean_var = var_wilayah.mean()
                colors = np.where(var_wilayah.values >= mean_var, "tab:green", "tab:orange")
                
                fig2, ax2 = plt.subplots(figsize=(16, 7))
                ax2.scatter(x, var_wilayah.values, c=colors, s=60)
                ax2.plot(x, var_wilayah.values, linestyle="--", alpha=0.5)
                ax2.axhline(mean_var, linestyle=":", linewidth=2, color='red')
                ax2.text(x[-1] + 0.3, mean_var, f"Rata-rata = {mean_var:.2f}", va="center")
                ax2.set_xticks(x)
                ax2.set_xticklabels(var_wilayah.index, rotation=75, fontsize=10)
                ax2.set_xlabel("Wilayah")
                ax2.set_ylabel(f"Rata-rata {selected_var}")
                ax2.set_title(f"Rata-rata {selected_var} per Wilayah")
                ax2.grid(axis="y", linestyle="--", alpha=0.6)
                plt.tight_layout()
                st.pyplot(fig2)
            
            st.markdown("---")
            
            # ========================
            # STATISTIK DESKRIPTIF
            # ========================
            st.subheader("📋 Statistik Deskriptif")
            
            if len(all_vars) > 0:
                desc_total = (
                    df[all_vars]
                    .agg(["mean", "median", "std", "min", "max"])
                    .T
                )
                desc_total = desc_total.rename(columns={
                    "mean": "Mean", "median": "Median", "std": "Std. Dev", "min": "Min", "max": "Max"
                })
                st.dataframe(desc_total.style.format("{:.3f}"), width='stretch')
            
            st.markdown("---")
            
            # ========================
            # HEATMAP KORELASI
            # ========================
            st.subheader("🔥 Heatmap Korelasi")
            
            if len(all_vars) > 1:
                corr_matrix = df[all_vars].corr()
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f",
                            linewidths=0.5, square=True, cbar_kws={"shrink": 0.8}, ax=ax)
                ax.set_title("Heatmap Korelasi Antarvariabel")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Minimal 2 variabel diperlukan untuk heatmap korelasi")
            
            st.markdown("---")
            
            # ========================
            # UJI MULTIKOLINEARITAS
            # ========================
            st.subheader("🔍 Uji Multikolinearitas (VIF)")
            
            if len(indep_vars) > 0:
                X = df[indep_vars].dropna()
                X_const = sm.add_constant(X)
                
                vif_data = pd.DataFrame()
                vif_data["Variabel"] = X_const.columns
                vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
                
                def interpret_vif(vif):
                    if vif < 5:
                        return "✅ Tidak ada multikolinearitas"
                    elif vif < 10:
                        return "⚠️ Indikasi multikolinearitas sedang"
                    else:
                        return "❌ Multikolinearitas tinggi"
                
                vif_data["VIF"] = vif_data["VIF"].round(3)
                vif_data["Keterangan"] = vif_data["VIF"].apply(interpret_vif)
                
                st.dataframe(vif_data, width='stretch')
                
                st.info("📌 **Interpretasi:** VIF < 5 = Tidak ada multikolinearitas, VIF 5-10 = Sedang, VIF > 10 = Tinggi")
            else:
                st.info("Pilih variabel Prediktor untuk uji multikolinearitas")

# =========================
# 3. UJI SPESIFIKASI MODEL
# =========================
elif menu == "Uji Spesifikasi Model":
    st.title("Uji Spesifikasi Model")
    st.markdown("---")
    
    if not check_data_ready():
        st.stop()
    
    df = get_ordered_df()  # Use ordered dataframe
    daerah_col = st.session_state.daerah_col
    tahun_col = st.session_state.tahun_col
    dep_var = st.session_state.dep_var
    # Use original_indep_vars for specification tests
    indep_vars = st.session_state.original_indep_vars if st.session_state.original_indep_vars else st.session_state.indep_vars
    
    # Prepare panel data
    df_panel = df.set_index([daerah_col, tahun_col]).sort_index()
    y = df_panel[dep_var].astype(float)
    X = sm.add_constant(df_panel[indep_vars].astype(float))
    
    tab1, tab2, tab3 = st.tabs(["Uji Chow", "Uji Hausman", "Uji Lagrange Multiplier"])
    
    # =====================
    # UJI CHOW
    # =====================
    with tab1:
        st.header("Uji Chow (Redundant Fixed Effects Test)")
        st.markdown("""
        **Hipotesis:**
        - H0: Common Effect Model (CEM) lebih sesuai
        - H1: Fixed Effect Model (FEM) lebih sesuai
        """)
        
        model_cem = PooledOLS(y, X)
        res_cem = model_cem.fit()
        
        model_fem = PanelOLS(y, X, entity_effects=True)
        res_fem = model_fem.fit()
        
        RSS_pooled = np.sum(res_cem.resids ** 2)
        RSS_fixed = np.sum(res_fem.resids ** 2)
        
        n = df_panel.shape[0]
        k = X.shape[1]
        N = df_panel.index.get_level_values(0).nunique()
        
        F = ((RSS_pooled - RSS_fixed) / (N - 1)) / (RSS_fixed / (n - N - k))
        p_value_F = 1 - stats.f.cdf(F, N - 1, n - N - k)
        
        LL_cem = res_cem.loglik
        LL_fem = res_fem.loglik
        chi2_LR = 2 * (LL_fem - LL_cem)
        p_value_chi2_LR = 1 - stats.chi2.cdf(chi2_LR, N - 1)
        
        # =========================
        # TEST RESULTS TABLE
        # =========================
        st.subheader("Hasil Uji Chow")
        chow_df = pd.DataFrame({
            "Effects Test": ["Cross-section F", "Cross-section Chi-square"],
            "Statistic": [f"{F:.6f}", f"{chi2_LR:.6f}"],
            "d.f.": [f"({N-1},{n-N-k})", f"{N-1}"],
            "Prob.": [f"{p_value_F:.4f}", f"{p_value_chi2_LR:.4f}"]
        })
        st.dataframe(chow_df, width='stretch', hide_index=True)
        
        # =========================
        # FEM STATISTICS (seperti EViews)
        # =========================
        st.subheader("Statistik Model Fixed Effect")
        
        try:
            r2_fem = res_fem.rsquared.overall if hasattr(res_fem.rsquared, 'overall') else res_fem.rsquared
        except:
            r2_fem = res_fem.rsquared
        
        k_no_const = k - 1
        adj_r2_fem = 1 - ((1 - r2_fem) * (n - 1) / (n - k_no_const - 1))
        
        residuals_fem = res_fem.resids
        SSE_fem = np.sum(residuals_fem ** 2)
        sigma_fem = np.sqrt(SSE_fem / (n - k_no_const - 1))
        
        # Durbin-Watson
        dw_fem = sm.stats.durbin_watson(residuals_fem)
        
        # Information criteria
        loglik_fem = -0.5 * n * (np.log(2 * np.pi) + 1 + np.log(SSE_fem / n))
        aic_fem = -2 * loglik_fem + 2 * (k_no_const + 1)
        sc_fem = -2 * loglik_fem + (k_no_const + 1) * np.log(n)
        hq_fem = -2 * loglik_fem + 2 * (k_no_const + 1) * np.log(np.log(n))
        
        col1, col2 = st.columns(2)
        
        with col1:
            fem_stats1 = pd.DataFrame({
                "Metrik": ["R-squared", "Adjusted R-squared", "S.E. of regression", "Sum squared resid"],
                "Nilai": [f"{r2_fem:.6f}", f"{adj_r2_fem:.6f}", f"{sigma_fem:.6f}", f"{SSE_fem:.6f}"]
            })
            st.dataframe(fem_stats1, width='stretch', hide_index=True)
        
        with col2:
            fem_stats2 = pd.DataFrame({
                "Metrik": ["Log likelihood", "Durbin-Watson stat", "Akaike info criterion", "Schwarz criterion", "Hannan-Quinn criter."],
                "Nilai": [f"{loglik_fem:.6f}", f"{dw_fem:.6f}", f"{aic_fem:.6f}", f"{sc_fem:.6f}", f"{hq_fem:.6f}"]
            })
            st.dataframe(fem_stats2, width='stretch', hide_index=True)
        
        st.markdown("---")
        if p_value_F < 0.05:
            st.success("✅ **Keputusan:** Tolak H0 → Model Fixed Effect lebih sesuai dibandingkan Common Effect")
        else:
            st.warning("⚠️ **Keputusan:** Gagal Tolak H0 → Model Common Effect lebih sesuai dibandingkan Fixed Effect")
    
    # =====================
    # UJI HAUSMAN
    # =====================
    with tab2:
        st.header("Uji Hausman (Correlated Random Effects Test)")
        st.markdown("""
        **Hipotesis:**
        - H0: Random Effect Model (REM) lebih sesuai
        - H1: Fixed Effect Model (FEM) lebih sesuai
        """)
        
        res_fe = PanelOLS(y, X, entity_effects=True).fit(cov_type='unadjusted')
        res_re = RandomEffects(y, X).fit(cov_type='unadjusted')
        
        b_fe = res_fe.params.drop('const')
        b_re = res_re.params.drop('const')
        
        common_idx = b_fe.index.intersection(b_re.index)
        b_diff = b_fe[common_idx] - b_re[common_idx]
        
        cov_fe = res_fe.cov.loc[common_idx, common_idx]
        cov_re = res_re.cov.loc[common_idx, common_idx]
        cov_diff = cov_fe - cov_re
        
        inv_cov = np.linalg.pinv(cov_diff)
        chi2_stat = float(b_diff.T @ inv_cov @ b_diff)
        df_hausman = len(b_diff)
        p_value = 1 - stats.chi2.cdf(chi2_stat, df_hausman)
        
        hausman_df = pd.DataFrame({
            "Test Summary": ["Cross-section random"],
            "Chi-Sq. Statistic": [f"{chi2_stat:.6f}"],
            "Chi-Sq. d.f.": [df_hausman],
            "Prob.": [f"{p_value:.4f}"]
        })
        st.dataframe(hausman_df, width='stretch')
        
        if p_value < 0.05:
            st.success("✅ **Keputusan:** Tolak H0 → Model Fixed Effect lebih sesuai")
        else:
            st.warning("⚠️ **Keputusan:** Gagal Tolak H0 → Model Random Effect lebih sesuai")
    
    # =====================
    # UJI LAGRANGE MULTIPLIER
    # =====================
    with tab3:
        st.header("Uji Lagrange Multiplier (Random Effects)")
        st.markdown("""
        **Hipotesis:**
        - H0: Tidak ada efek random (Common Effect Model lebih sesuai)
        - H1: Terdapat efek random (Random Effect Model lebih sesuai)
        """)
        
        res_pooled = PooledOLS(y, X).fit()
        resid = res_pooled.resids
        
        entities = df_panel.index.get_level_values(0).unique()
        times = df_panel.index.get_level_values(1).unique()
        N_bp, T = len(entities), len(times)
        
        resid_matrix = (
            df_panel.assign(resid=resid)
            .reset_index()
            .pivot(index=daerah_col, columns=tahun_col, values="resid")
        )
        
        sigma_e2 = np.nanvar(resid_matrix.stack(), ddof=1)
        sigma_mu2 = np.nanvar(resid_matrix.mean(axis=1), ddof=1)
        sigma_lambda2 = np.nanvar(resid_matrix.mean(axis=0), ddof=1)
        resid_mean = resid_matrix.stack().mean()
        
        # =========================
        # Breusch-Pagan LM
        # =========================
        LM_cross = (T**2 / (2*(T-1))) * np.sum((resid_matrix.mean(axis=1) - resid_mean)**2) / sigma_e2
        LM_time = (N_bp**2 / (2*(N_bp-1))) * np.sum((resid_matrix.mean(axis=0) - resid_mean)**2) / sigma_e2
        LM_both = LM_cross + LM_time
        
        p_cross = 1 - chi2.cdf(LM_cross, 1)
        p_time = 1 - chi2.cdf(LM_time, 1)
        p_both = 1 - chi2.cdf(LM_both, 2)
        
        # =========================
        # Honda / King-Wu
        # =========================
        h_cross = np.sqrt(T/2) * np.sqrt(sigma_mu2 / sigma_e2)
        h_time = np.sqrt(N_bp/2) * np.sqrt(sigma_lambda2 / sigma_e2)
        h_both = np.sqrt(h_cross**2 + h_time**2)
        
        p_h_cross = 1 - norm.cdf(h_cross)
        p_h_time = 1 - norm.cdf(h_time)
        p_h_both = 1 - norm.cdf(h_both)
        
        # =========================
        # Standardized Honda
        # =========================
        std_h_cross = h_cross / np.sqrt(2)
        std_h_time = h_time / np.sqrt(2)
        std_h_both = np.sqrt(std_h_cross**2 + std_h_time**2)
        
        p_std_h_cross = 1 - norm.cdf(std_h_cross)
        p_std_h_time = 1 - norm.cdf(std_h_time)
        p_std_h_both = 1 - norm.cdf(std_h_both)
        
        # =========================
        # Display Results (format like EViews/notebook)
        # =========================
        st.subheader("Lagrange Multiplier Tests for Random Effects")
        
        # Create table with all tests
        lm_data = []
        
        # Breusch-Pagan
        lm_data.append({
            "Test": "Breusch-Pagan",
            "Cross-section": f"{LM_cross:.6f}",
            "Cross-section (Prob)": f"({p_cross:.4f})",
            "Time": f"{LM_time:.6f}",
            "Time (Prob)": f"({p_time:.4f})",
            "Both": f"{LM_both:.6f}",
            "Both (Prob)": f"({p_both:.4f})"
        })
        
        # Honda
        lm_data.append({
            "Test": "Honda",
            "Cross-section": f"{h_cross:.6f}",
            "Cross-section (Prob)": f"({p_h_cross:.4f})",
            "Time": f"{h_time:.6f}",
            "Time (Prob)": f"({p_h_time:.4f})",
            "Both": f"{h_both:.6f}",
            "Both (Prob)": f"({p_h_both:.4f})"
        })
        
        # Standardized Honda
        lm_data.append({
            "Test": "Standardized Honda",
            "Cross-section": f"{std_h_cross:.6f}",
            "Cross-section (Prob)": f"({p_std_h_cross:.4f})",
            "Time": f"{std_h_time:.6f}",
            "Time (Prob)": f"({p_std_h_time:.4f})",
            "Both": f"{std_h_both:.6f}",
            "Both (Prob)": f"({p_std_h_both:.4f})"
        })
        
        # Gourieroux et al.
        lm_data.append({
            "Test": "Gourieroux et al.",
            "Cross-section": "--",
            "Cross-section (Prob)": "",
            "Time": "--",
            "Time (Prob)": "",
            "Both": f"{LM_both:.6f}",
            "Both (Prob)": f"({p_both:.4f})"
        })
        
        lm_df = pd.DataFrame(lm_data)
        st.dataframe(lm_df, width='stretch', hide_index=True)
        
        # Detailed interpretation
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Cross-section:**")
            if p_cross < 0.05:
                st.success(f"Prob = {p_cross:.4f} < 0.05 → Ada efek cross-section")
            else:
                st.info(f"Prob = {p_cross:.4f} ≥ 0.05 → Tidak ada efek cross-section")
                
        with col2:
            st.markdown("**Time:**")
            if p_time < 0.05:
                st.success(f"Prob = {p_time:.4f} < 0.05 → Ada efek time")
            else:
                st.info(f"Prob = {p_time:.4f} ≥ 0.05 → Tidak ada efek time")
                
        with col3:
            st.markdown("**Both:**")
            if p_both < 0.05:
                st.success(f"Prob = {p_both:.4f} < 0.05 → Random Effect lebih sesuai")
            else:
                st.info(f"Prob = {p_both:.4f} ≥ 0.05 → Common Effect lebih sesuai")
        
        st.markdown("---")
        if p_both < 0.05:
            st.success("✅ **Keputusan:** Tolak H0 → Terdapat efek random, Model Random Effect lebih sesuai dibandingkan Common Effect")
        else:
            st.warning("⚠️ **Keputusan:** Gagal Tolak H0 → Tidak ada efek random, Model Common Effect lebih sesuai")

# =========================
# 4. UJI SIGNIFIKANSI PARAMETER
# =========================
elif menu == "Uji Signifikansi Parameter":
    st.title("Uji Signifikansi Parameter")
    st.markdown("---")
    
    if not check_data_ready():
        st.stop()
    
    tab1, tab2 = st.tabs(["Uji F (Simultan)", "Uji t (Backward Elimination)"])
    
    # Get both full models (for Uji F) and current models (for backward)
    models_owfe, models_twfe, df_work, y, dummies_daerah, dummies_tahun, full_indep_vars = get_full_models()
    
    # =====================
    # UJI F (using full models - models_owfe/models_twfe)
    # =====================
    with tab1:
        st.header("Uji F (Uji Simultan)")
        st.markdown("""
        **Hipotesis:**
        - H0: Semua koefisien variabel Prediktor = 0
        - H1: Minimal satu koefisien ≠ 0
        
        **Note:** Menggunakan model lengkap dengan semua variabel Prediktor.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("One-Way Fixed Effect (OWFE)")
            
            # Koefisien Variabel Utama
            st.markdown("**Koefisien Variabel Utama:**")
            coef_vars = ["const"] + full_indep_vars
            coef_owfe = pd.DataFrame({
                "Variabel": coef_vars,
                "Koefisien": [f"{models_owfe.params[v]:.6f}" for v in coef_vars]
            })
            st.dataframe(coef_owfe, width='stretch', hide_index=True)
            
            st.markdown("**Statistik Model:**")
            f_df_owfe = pd.DataFrame({
                "Metrik": ["F-statistic", "Prob (F-statistic)", "Adj. R-squared"],
                "Nilai": [
                    f"{models_owfe.fvalue:.6f}",
                    f"{models_owfe.f_pvalue:.6f}",
                    f"{models_owfe.rsquared_adj:.6f}"
                ]
            })
            st.dataframe(f_df_owfe, width='stretch', hide_index=True)
            
            if models_owfe.f_pvalue < 0.05:
                st.success("✅ Tolak H0 → Model signifikan secara simultan")
            else:
                st.warning("⚠️ Gagal Tolak H0 → Model tidak signifikan")
        
        with col2:
            st.subheader("Two-Way Fixed Effect (TWFE)")
            
            # Koefisien Variabel Utama
            st.markdown("**Koefisien Variabel Utama:**")
            coef_twfe = pd.DataFrame({
                "Variabel": coef_vars,
                "Koefisien": [f"{models_twfe.params[v]:.6f}" for v in coef_vars]
            })
            st.dataframe(coef_twfe, width='stretch', hide_index=True)
            
            st.markdown("**Statistik Model:**")
            f_df_twfe = pd.DataFrame({
                "Metrik": ["F-statistic", "Prob (F-statistic)", "Adj. R-squared"],
                "Nilai": [
                    f"{models_twfe.fvalue:.6f}",
                    f"{models_twfe.f_pvalue:.6f}",
                    f"{models_twfe.rsquared_adj:.6f}"
                ]
            })
            st.dataframe(f_df_twfe, width='stretch', hide_index=True)
            
            if models_twfe.f_pvalue < 0.05:
                st.success("✅ Tolak H0 → Model signifikan secara simultan")
            else:
                st.warning("⚠️ Gagal Tolak H0 → Model tidak signifikan")
    
    # =====================
    # UJI t (BACKWARD ELIMINATION)
    # =====================
    with tab2:
        st.header("Uji t dengan Backward Elimination")
        st.markdown("""
        **Metode:** Eliminasi variabel dengan p-value > 0.05 secara bertahap
        
        **Note:** Dimulai dengan semua variabel Prediktor, kemudian eliminasi satu per satu.
        """)
        
        model_type = st.radio("Pilih Model:", ["One-Way Fixed Effect (OWFE)", "Two-Way Fixed Effect (TWFE)"])
        
        # Start with all original Prediktor variables
        indep_vars = st.session_state.original_indep_vars.copy() if st.session_state.original_indep_vars else st.session_state.indep_vars.copy()
        alpha = 0.05
        step = 1
        elimination_log = []
        
        if model_type == "One-Way Fixed Effect (OWFE)":
            dummies = dummies_daerah
        else:
            dummies = pd.concat([dummies_daerah, dummies_tahun], axis=1)
        
        while True:
            X_main = df_work[indep_vars].astype(float)
            X = pd.concat([X_main, dummies], axis=1)
            X = sm.add_constant(X, has_constant="add").astype(float)
            model = sm.OLS(y, X).fit()
            
            y_hat = model.fittedvalues
            rmse = np.sqrt(np.mean((y - y_hat) ** 2))
            mape = np.mean(np.abs((y - y_hat) / y)) * 100
            
            pvals = model.pvalues[indep_vars]
            max_p = pvals.max()
            worst_var = pvals.idxmax()
            
            elimination_log.append({
                "Step": step,
                "Variabel": ", ".join(indep_vars),
                "Adj R²": f"{model.rsquared_adj:.6f}",
                "AIC": f"{model.aic:.4f}",
                "Var. Dihapus": worst_var if max_p > alpha else "-",
                "P-value Tertinggi": f"{max_p:.4f}"
            })
            
            if max_p > alpha and len(indep_vars) > 1:
                indep_vars.remove(worst_var)
                step += 1
            else:
                break
        
        st.subheader("Proses Backward Elimination")
        st.dataframe(pd.DataFrame(elimination_log), width='stretch')
        
        # Final Model Results
        st.subheader("Hasil Uji t Final (Model Terpilih)")
        
        vars_final = ["const"] + indep_vars
        result_t = pd.DataFrame({
            "Variable": vars_final,
            "Coefficient": [model.params[v] for v in vars_final],
            "Std. Error": [model.bse[v] for v in vars_final],
            "t-Statistic": [model.tvalues[v] for v in vars_final],
            "Prob.": [model.pvalues[v] for v in vars_final]
        })
        
        st.dataframe(result_t.style.format({
            "Coefficient": "{:.6f}", "Std. Error": "{:.6f}",
            "t-Statistic": "{:.6f}", "Prob.": "{:.4f}"
        }), width='stretch')
        
        st.info(f"📌 **Variabel signifikan dalam model final:** {', '.join(indep_vars)}")

# =========================
# 5. ESTIMASI MODEL REGRESI PANEL
# =========================
elif menu == "Estimasi Model Regresi Panel":
    st.title("Estimasi Model Regresi Panel (FEM)")
    st.markdown("---")
    
    if not check_data_ready():
        st.stop()
    
    model_owfe, model_twfe, df_work, y, dummies_daerah, dummies_tahun, sig_vars_owfe, sig_vars_twfe = get_backward_models()
    
    st.info("**Note:** Model estimasi menggunakan variabel signifikan saja (hasil backward elimination)")
    
    tab1, tab2 = st.tabs(["One-Way Fixed Effect (OWFE)", "Two-Way Fixed Effect (TWFE)"])
    
    daerah_col = st.session_state.daerah_col
    tahun_col = st.session_state.tahun_col
    
    if st.session_state.daerah_order is not None:
        unique_daerah = st.session_state.daerah_order
    else:
        unique_daerah = df_work[daerah_col].unique().tolist()
    
    # =====================
    # OWFE
    # =====================
    with tab1:
        st.header("One-Way Fixed Effect Model (OWFE)")
        st.markdown(f"**Model:** {st.session_state.dep_var} ~ {' + '.join(sig_vars_owfe)} + Fixed Effects (Cross-section)")
        st.caption(f"📌 Variabel signifikan: {', '.join(sig_vars_owfe)}")
        
        # ===== TABEL KOEFISIEN =====
        st.subheader("Hasil Estimasi Koefisien")
        
        coef_df = pd.DataFrame({
            "Variabel": model_owfe.params.index,
            "Koefisien": model_owfe.params.values,
            "Std Error": model_owfe.bse.values,
            "t-value": model_owfe.tvalues.values,
            "p-value": model_owfe.pvalues.values
        })
        
        # tampilkan hanya konstanta + variabel utama
        coef_df = coef_df[coef_df["Variabel"].isin(["const"] + sig_vars_owfe)]
        coef_df = coef_df.round(6)
        
        st.dataframe(coef_df, width="stretch")
        
        # ===== INTERCEPT INDIVIDU =====
        st.subheader("Intercept Individu (μ_i)")
        
        mu_i_list = []
        for d in unique_daerah:
            if d in model_owfe.params.index:
                mu_i_list.append({"Wilayah": d, "μ_i": model_owfe.params[d]})
            else:
                mu_i_list.append({"Wilayah": d, "μ_i": 0.0})
        
        tabel_mu = pd.DataFrame(mu_i_list)
        st.dataframe(tabel_mu.style.format({"μ_i": "{:.6f}"}), width='stretch')
    
    
    # =====================
    # TWFE
    # =====================
    with tab2:
        st.header("Two-Way Fixed Effect Model (TWFE)")
        st.markdown(f"**Model:** {st.session_state.dep_var} ~ {' + '.join(sig_vars_twfe)} + Fixed Effects (Cross-section + Time)")
        st.caption(f"📌 Variabel signifikan: {', '.join(sig_vars_twfe)}")
        
        # ===== TABEL KOEFISIEN =====
        st.subheader("Hasil Estimasi Koefisien")
        
        coef_df = pd.DataFrame({
            "Variabel": model_twfe.params.index,
            "Koefisien": model_twfe.params.values,
            "Std Error": model_twfe.bse.values,
            "t-value": model_twfe.tvalues.values,
            "p-value": model_twfe.pvalues.values
        })
        
        coef_df = coef_df[coef_df["Variabel"].isin(["const"] + sig_vars_twfe)]
        coef_df = coef_df.round(6)
        
        st.dataframe(coef_df, width="stretch")
        
        col1, col2 = st.columns(2)
        
        # ===== INTERCEPT INDIVIDU =====
        with col1:
            st.subheader("Intercept Individu (μ_i)")
            
            mu_i_list = []
            for d in unique_daerah:
                if d in model_twfe.params.index:
                    mu_i_list.append({"Wilayah": d, "μ_i": model_twfe.params[d]})
                else:
                    mu_i_list.append({"Wilayah": d, "μ_i": 0.0})
            
            tabel_mu = pd.DataFrame(mu_i_list)
            st.dataframe(tabel_mu.style.format({"μ_i": "{:.6f}"}), width='stretch')
        
        # ===== INTERCEPT WAKTU =====
        with col2:
            st.subheader("Intercept Waktu (λ_t)")
            
            unique_tahun = sorted(df_work[tahun_col].unique())
            lambda_t_list = []
            for t in unique_tahun:
                if t in model_twfe.params.index:
                    lambda_t_list.append({"Tahun": t, "λ_t": model_twfe.params[t]})
                else:
                    lambda_t_list.append({"Tahun": t, "λ_t": 0.0})
            
            tabel_lambda = pd.DataFrame(lambda_t_list)
            st.dataframe(tabel_lambda.style.format({"λ_t": "{:.6f}"}), width='stretch')

# =========================
# 6. UJI ASUMSI RESIDUAL
# =========================
elif menu == "Uji Asumsi Residual":
    st.title("Uji Asumsi Residual")
    st.markdown("---")
    
    if not check_data_ready():
        st.stop()
    
    # Get both models as per notebook:
    # - models_owfe/models_twfe (full model) for Uji Autokorelasi
    # - model_owfe/model_twfe (backward) for Uji Hetero & Normalitas
    models_owfe, models_twfe, df_work, y, dummies_daerah, dummies_tahun, full_indep_vars = get_full_models()
    model_owfe, model_twfe, _, _, _, _, sig_vars_owfe, sig_vars_twfe = get_backward_models()
    
    tab1, tab2, tab3 = st.tabs(["Uji Autokorelasi", "Uji Heteroskedastisitas", "Uji Normalitas"])
    
    # =====================
    # UJI AUTOKORELASI - uses models_owfe/models_twfe (full model)
    # =====================
    with tab1:
        st.header("Uji Autokorelasi (Breusch-Godfrey)")
        st.markdown("""
        **Hipotesis:**
        - H0: Tidak ada autokorelasi
        - H1: Terdapat autokorelasi
        """)
        
        bg_owfe = acorr_breusch_godfrey(models_owfe, nlags=1)
        bg_twfe = acorr_breusch_godfrey(models_twfe, nlags=1)
        
        bg_df = pd.DataFrame({
            "Statistic": ["LM Statistic", "Prob. Chi-Square(1)", "F-Statistic", "Prob. F"],
            "OWFE": [f"{bg_owfe[0]:.6f}", f"{bg_owfe[1]:.6f}", f"{bg_owfe[2]:.6f}", f"{bg_owfe[3]:.6f}"],
            "TWFE": [f"{bg_twfe[0]:.6f}", f"{bg_twfe[1]:.6f}", f"{bg_twfe[2]:.6f}", f"{bg_twfe[3]:.6f}"]
        })
        st.dataframe(bg_df, width='stretch')
        
        col1, col2 = st.columns(2)
        with col1:
            if bg_owfe[1] > 0.05:
                st.success("✅ OWFE: Tidak terdapat autokorelasi")
            else:
                st.warning("⚠️ OWFE: Terdapat autokorelasi")
        with col2:
            if bg_twfe[1] > 0.05:
                st.success("✅ TWFE: Tidak terdapat autokorelasi")
            else:
                st.warning("⚠️ TWFE: Terdapat autokorelasi")
    
    # =====================
    # UJI HETEROSKEDASTISITAS - uses model_owfe/model_twfe (backward) but X_glejser with all vars
    # =====================
    with tab2:
        st.header("Uji Heteroskedastisitas (Glejser)")
        st.markdown("""
        **Hipotesis:**
        - H0: Tidak ada heteroskedastisitas (homoskedastis)
        - H1: Terdapat heteroskedastisitas
        """)
        
        def glejser_lm_test(model, df_data, indep_vars):
            abs_resid = np.abs(model.resid)
            X_glejser = sm.add_constant(df_data[indep_vars])
            model_glejser = sm.OLS(abs_resid, X_glejser).fit()
            n_obs = int(model_glejser.nobs)
            r2 = model_glejser.rsquared
            LM_stat = n_obs * r2
            df_chi = X_glejser.shape[1] - 1
            p_value = 1 - chi2.cdf(LM_stat, df_chi)
            return LM_stat, p_value
        
        # Use model_owfe/model_twfe (backward) but X_glejser with ALL 4 variables
        lm_owfe, p_owfe = glejser_lm_test(model_owfe, df_work, full_indep_vars)
        lm_twfe, p_twfe = glejser_lm_test(model_twfe, df_work, full_indep_vars)
        
        glejser_df = pd.DataFrame({
            "Keterangan": ["Obs*R-squared", "Prob(Chi-square)"],
            "OWFE": [f"{lm_owfe:.6f}", f"{p_owfe:.6f}"],
            "TWFE": [f"{lm_twfe:.6f}", f"{p_twfe:.6f}"]
        })
        st.dataframe(glejser_df, width='stretch')
        
        col1, col2 = st.columns(2)
        with col1:
            if p_owfe > 0.05:
                st.success("✅ OWFE: Tidak terdapat heteroskedastisitas")
            else:
                st.warning("⚠️ OWFE: Terdapat heteroskedastisitas")
        with col2:
            if p_twfe > 0.05:
                st.success("✅ TWFE: Tidak terdapat heteroskedastisitas")
            else:
                st.warning("⚠️ TWFE: Terdapat heteroskedastisitas")
    
    # =====================
    # UJI NORMALITAS - uses model_owfe/model_twfe (backward)
    # =====================
    with tab3:
        st.header("Uji Normalitas (Jarque-Bera)")
        st.markdown("""
        **Hipotesis:**
        - H0: Residual berdistribusi normal
        - H1: Residual tidak berdistribusi normal
        """)
        
        jb_owfe = stats.jarque_bera(model_owfe.resid)
        jb_twfe = stats.jarque_bera(model_twfe.resid)
        
        jb_df = pd.DataFrame({
            "Model": ["OWFE", "TWFE"],
            "JB Statistic": [f"{jb_owfe.statistic:.6f}", f"{jb_twfe.statistic:.6f}"],
            "Prob": [f"{jb_owfe.pvalue:.6f}", f"{jb_twfe.pvalue:.6f}"]
        })
        st.dataframe(jb_df, width='stretch')
        
        col1, col2 = st.columns(2)
        with col1:
            if jb_owfe.pvalue > 0.05:
                st.success("✅ OWFE: Residual berdistribusi normal")
            else:
                st.warning("⚠️ OWFE: Residual tidak normal")
        with col2:
            if jb_twfe.pvalue > 0.05:
                st.success("✅ TWFE: Residual berdistribusi normal")
            else:
                st.warning("⚠️ TWFE: Residual tidak normal")
        
        # Q-Q Plot
        st.subheader("Q-Q Plot Residual")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sm.qqplot(model_owfe.resid, line='45', ax=axes[0])
        axes[0].set_title("Q-Q Plot Residual OWFE")
        sm.qqplot(model_twfe.resid, line='45', ax=axes[1])
        axes[1].set_title("Q-Q Plot Residual TWFE")
        plt.tight_layout()
        st.pyplot(fig)

# =========================
# 7. EVALUASI & REKOMENDASI MODEL
# =========================
elif menu == "Evaluasi & Rekomendasi Model":
    st.title("Evaluasi & Rekomendasi Model")
    st.markdown("---")
    
    if not check_data_ready():
        st.stop()
    
    # Ambil model
    model_owfe, model_twfe, df_work, y, dummies_daerah, dummies_tahun, sig_vars_owfe, sig_vars_twfe = get_backward_models()
    dep_var = st.session_state.dep_var
    daerah_col = st.session_state.daerah_col
    tahun_col = st.session_state.tahun_col

    st.info("Evaluasi model menggunakan variabel signifikan (hasil backward elimination)")
    st.caption(f"OWFE: {', '.join(sig_vars_owfe)} | TWFE: {', '.join(sig_vars_twfe)}")

    # =====================
    # HITUNG PREDIKSI
    # =====================
    df_work["Pred_OWFE"] = model_owfe.fittedvalues
    df_work["Pred_TWFE"] = model_twfe.fittedvalues

    # =====================================================
    # 1️⃣ EVALUASI MODEL
    # =====================================================
    st.header("1️⃣ Evaluasi Model (Perbandingan Metrik)")

    def calc_metrics(model, y_true, y_pred):
        adj_r2 = model.rsquared_adj
        aic = model.aic
        return adj_r2, aic

    adj_r2_owfe, aic_owfe = calc_metrics(model_owfe, y, df_work["Pred_OWFE"])
    adj_r2_twfe, aic_twfe = calc_metrics(model_twfe, y, df_work["Pred_TWFE"])

    comparison_df = pd.DataFrame({
        "Metrik": ["Adj. R-squared", "AIC"],
        "OWFE": [f"{adj_r2_owfe:.6f}", f"{aic_owfe:.2f}"],
        "TWFE": [f"{adj_r2_twfe:.6f}", f"{aic_twfe:.2f}"]
    })

    st.dataframe(comparison_df, width="stretch")

    # =====================
    # SKORING MODEL
    # =====================
    score_owfe = 0
    score_twfe = 0

    if adj_r2_owfe > adj_r2_twfe:
        score_owfe += 1
    else:
        score_twfe += 1

    if aic_owfe < aic_twfe:
        score_owfe += 1
    else:
        score_twfe += 1


    st.write(f"Skor OWFE: {score_owfe} | Skor TWFE: {score_twfe}")

    # =====================================================
    # 2️⃣ REKOMENDASI MODEL TERBAIK
    # =====================================================
    st.markdown("---")
    st.header("2️⃣ Rekomendasi Model Terbaik")

    if score_twfe > score_owfe:
        st.success("Model terbaik berdasarkan mayoritas indikator adalah Two-Way Fixed Effect (TWFE).")
    elif score_owfe > score_twfe:
        st.success("Model terbaik berdasarkan mayoritas indikator adalah One-Way Fixed Effect (OWFE).")
    else:
        st.info("Kedua model memiliki performa yang relatif seimbang.")

    # =====================
    # GRAFIK TWFE
    # =====================
    st.subheader("Grafik Aktual vs Prediksi (TWFE)")
    
    x_axis = np.arange(len(df_work))
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(x_axis, y, label="Nilai Aktual", linewidth=1.8)
    ax.plot(x_axis, df_work["Pred_TWFE"], label="Nilai Prediksi (TWFE)", linewidth=1.8)
    ax.set_title(f"Perbandingan Nilai Aktual dan Prediksi {dep_var} (TWFE)")
    ax.set_xlabel("Observasi (Wilayah × Tahun)")
    ax.set_ylabel(dep_var)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    # =====================
    # TABEL AKTUAL & PREDIKSI TWFE
    # =====================
    st.subheader("Tabel Nilai Aktual & Prediksi (TWFE)")

    df_display = df_work[[daerah_col, tahun_col]].copy()
    df_display["Aktual"] = df_work[dep_var]
    df_display["Prediksi"] = df_work["Pred_TWFE"]
    df_display["Residual"] = df_display["Aktual"] - df_display["Prediksi"]

    st.dataframe(df_display, width="stretch", height=500)

    # =====================
    # TABEL KOEFISIEN TWFE
    # =====================
    st.subheader("Tabel Koefisien Model TWFE")

    coef_twfe = pd.DataFrame({
        "Variabel": model_twfe.params.index,
        "Koefisien": model_twfe.params.values,
        "Std Error": model_twfe.bse.values,
        "t-value": model_twfe.tvalues.values,
        "p-value": model_twfe.pvalues.values
    })

    coef_twfe = coef_twfe[coef_twfe["Variabel"].isin(["const"] + sig_vars_twfe)]
    coef_twfe = coef_twfe.round(6)

    st.dataframe(coef_twfe, width="stretch")
    
    # =====================
    # INTERPRETASI MODEL TWFE
    # =====================
    st.markdown("---")

    st.success("""
    ### 💡 Insight

    Model TWFE menunjukkan bahwa:

    - **TPAK berpengaruh negatif terhadap TPT (β = -0,0775)**
    - **IPM berpengaruh negatif terhadap TPT (β = -1,1722)**

    berpengaruh negatif bermakna bahwa peningkatan variabel tersebut akan **menurunkan tingkat pengangguran**, 
    begitupun sebaliknuya jika Pengangguran naik maka Partisipasi Angkatan Kerja dan Indeks Pembangunan Manusia
    mengalami penurunan pada kabupaten/kota di Jawa Barat.

    ---

    ### 🔎 Simulasi Dampak

    - Jika **TPAK naik 1%**, maka **TPT turun sekitar 0,0775%**.  
    Contoh: TPT 8% → menjadi sekitar **7,92%**.

    - Jika **IPM naik 1 poin**, maka **TPT turun sekitar 1,17%**.  
    Contoh: TPT 8% → menjadi sekitar **6,83%**.

    ---

    ### 📈 Makna Ekonomi

    Hasil ini sejalan dengan konsep pertumbuhan ekonomi yang inklusif sebagaimana dijelaskan oleh 
    **Todaro & Smith (2015)**, yang menekankan bahwa pembangunan tidak hanya berfokus pada 
    peningkatan output ekonomi, tetapi juga pada pemerataan kesempatan kerja dan peningkatan kualitas sumber daya manusia.

    Dalam konteks penelitian ini, peningkatan TPAK menunjukkan semakin luasnya akses masyarakat 
    terhadap aktivitas ekonomi, sehingga tekanan pengangguran menurun.

    Sementara itu, peningkatan IPM yang mencerminkan kualitas pendidikan, kapasitas ekonomi, 
    dan kesiapan tenaga kerja mendorong peningkatan produktivitas serta daya saing di pasar kerja. 
    Melalui pendidikan dan pengembangan keterampilan yang relevan dengan kebutuhan industri, 
    tenaga kerja menjadi lebih adaptif dan memiliki peluang kerja yang lebih besar. 

    Dengan demikian, pembangunan manusia menjadi fondasi penting dalam menekan pengangguran 
    secara berkelanjutan dan menciptakan pertumbuhan ekonomi yang lebih inklusif.

    ---

    ### 📝 Implikasi Kebijakan

    Sesuai dengan prinsip pembangunan inklusif (Todaro & Smith, 2015), 
    Penurunan pengangguran di Jawa Barat perlu difokuskan pada:

    - Peningkatan kualitas pendidikan yang relevan dengan kebutuhan industri,
    - Penguatan pelatihan kerja, sertifikasi, dan program upskilling–reskilling,
    - Penciptaan lapangan kerja produktif yang mampu menyerap angkatan kerja secara optimal.

    Dengan demikian, pengembangan sumber daya manusia tidak hanya meningkatkan kualitas hidup, 
    tetapi juga memperkuat daya saing tenaga kerja di pasar kerja.
    """)



# =========================
# FOOTER
# =========================
st.sidebar.markdown("---")
st.sidebar.markdown("📊 **Sains Data Ekonometrika**")
st.sidebar.markdown("*Mendukung data panel dari seluruh provinsi*")







