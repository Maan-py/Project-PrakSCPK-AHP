import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Proyek Akhir Praktikum SCPK - AHP : Mencari Bibit Kedelai Terbaik",
    layout="wide",
)
st.title("SPK AHP: Mencari Bibit Kedelai Terbaik 🌱")

df = pd.read_csv("Advanced Soybean Agricultural Dataset.csv")

# 5 Kriteria
criteria = [
    "Seed Yield per Unit Area (SYUA)",  # Hasil biji kedelai yang dihasilkan dalam satu satuan luas lahan (misalnya kg/ha).
    "Number of Pods (NP)",  #  Jumlah polong per tanaman.
    "Protein Content (PCO)",  #  Kandungan total protein dalam biji kedelai.
    "Relative Water Content in Leaves (RWCL)",  # Persentase kandungan air relatif dalam daun tanaman.
    "Weight of 300 Seeds (W3S)",  # Total berat dari 300 butir biji kedelai.
]

st.subheader("Data Kriteria yang Digunakan dalam AHP")

# Alternatif
alternatives = df["Parameters"].unique()
bibit_df = pd.DataFrame(alternatives, columns=["Bibit"])

# Alternatif + Kriteria
st.subheader("Dataset Advanced Soybean Agricultural Dataset")
st.dataframe(df[["Parameters"] + criteria])

st.subheader("Alternatif Bibit")
st.dataframe(bibit_df)

grouped = df.groupby("Parameters")[criteria].mean()

# Tampilkan hasil
st.subheader("Rata-rata Kriteria Tiap Varietas Unik")
st.dataframe(grouped)

# Dropdown untuk memilih salah satu bibit
selected_bibit = st.selectbox("Pilih Bibit untuk Analisis:", bibit_df["Bibit"].tolist())

# Tampilkan data yang sesuai dengan pilihan
filtered_df = df[df["Parameters"] == selected_bibit][["Parameters"] + criteria]
st.subheader(f"Data untuk Bibit: {selected_bibit}")
st.dataframe(filtered_df)

n = len(criteria)
pairwise_matrix = np.ones((n, n))


# Fungsi untuk mendapatkan index perbandingan unik di atas diagonal
def get_comparisons(criteria):
    comparisons = []
    for i in range(len(criteria)):
        for j in range(i + 1, len(criteria)):
            comparisons.append((i, j))
    return comparisons


# Skala pilihan
scale = {
    "1 - Sama penting": 1,
    "3 - Sedikit lebih penting": 3,
    "5 - Cukup lebih penting": 5,
    "7 - Sangat lebih penting": 7,
    "9 - Mutlak lebih penting": 9,
}

comparisons = get_comparisons(criteria)

with st.sidebar:
    st.subheader("🧮 Masukkan Nilai Perbandingan Berpasangan")
    for i, j in comparisons:
        label = f"{criteria[i]} vs {criteria[j]}"
        value = st.slider(
            label, min_value=1, max_value=9, value=1, step=1, key=f"{i}-{j}"
        )
        pairwise_matrix[i][j] = value
        pairwise_matrix[j][i] = round(1 / value, 3)

st.subheader("📋 Matriks Perbandingan Berpasangan")
df_matrix = pd.DataFrame(pairwise_matrix, index=criteria, columns=criteria)
st.dataframe(df_matrix.style.format(precision=3))


# Langkah 1: Normalisasi matriks
def calc_norm(M):
    if M.ndim == 1:
        sM = np.sum(M)
        return M / sM
    else:
        sM = np.sum(M, axis=0)
        return M / sM


norm_matrix = calc_norm(pairwise_matrix)

st.subheader("📊 Matriks Normalisasi")
df_norm = pd.DataFrame(norm_matrix, index=criteria, columns=criteria)
st.dataframe(df_norm.style.format(precision=3))

# Langkah 2: Hitung bobot prioritas (eigen vector)
m, n = norm_matrix.shape
V = np.zeros(m)
for i in range(m):
    V[i] = np.sum(norm_matrix[i, :])
weights = V / m

st.subheader("⭐ Bobot Prioritas (Eigen Vector)")
for i in range(n):
    st.write(f"- {criteria[i]}: **{weights[i]:.4f}**")

# # Langkah 3: Konsistensi (λ_max, CI, CR)
# # Hitung λ_max
# Aw = np.dot(pairwise_matrix, weights)
# lambda_max = np.sum(Aw / weights) / n

# # Consistency Index (CI)
# CI = (lambda_max - n) / (n - 1)

# # Random Index (RI) tabel (hanya sampai n = 10)
# RI_dict = {
#     1: 0.00,
#     2: 0.00,
#     3: 0.58,
#     4: 0.90,
#     5: 1.12,
#     6: 1.24,
#     7: 1.32,
#     8: 1.41,
#     9: 1.45,
#     10: 1.49,
# }
# RI = RI_dict.get(n, 1.49)  # default pakai 1.49 jika n > 10
# CR = CI / RI if RI != 0 else 0

# st.subheader("✅ Konsistensi Matriks")
# st.write(f"λ maks: **{lambda_max:.4f}**")
# st.write(f"CI (Consistency Index): **{CI:.4f}**")
# st.write(f"CR (Consistency Ratio): **{CR:.4f}**")

# if CR < 0.1:
#     st.success("Matriks konsisten (CR < 0.1).")
# else:
#     st.warning(
#         "Matriks tidak konsisten (CR ≥ 0.1), silakan sesuaikan penilaian pairwise."
#     )

# Buat dataframe nilai alternatif per kriteria
alt_data = df[df["Parameters"].isin(alternatives)][["Parameters"] + criteria]

# Hitung nilai rata-rata tiap alternatif (jika ada multiple data per alternatif)
alt_means = alt_data.groupby("Parameters").mean()

st.subheader("📈 Nilai Rata-rata Alternatif per Kriteria")
st.dataframe(alt_means.style.format(precision=3))

# Normalisasi (anggap semua kriteria benefit, sesuaikan kalau ada cost)
normalized_alt = alt_means / alt_means.max()

st.subheader("🔄 Nilai Alternatif setelah Normalisasi")
st.dataframe(normalized_alt.style.format(precision=3))

# Hitung skor akhir per alternatif
scores = normalized_alt.dot(weights)

st.subheader("🏆 Skor Akhir dan Ranking Alternatif Bibit")
result_df = pd.DataFrame({"Skor": scores})
result_df = result_df.sort_values(by="Skor", ascending=False)
result_df["Ranking"] = range(1, len(result_df) + 1)

st.dataframe(result_df.style.format({"Skor": "{:.4f}"}))

# Ambil alternatif terbaik (skor tertinggi)
best_bibit = result_df.index[0]
best_score = result_df.iloc[0]["Skor"]

st.subheader("🎉 Bibit Terbaik")
# st.success(f"**Bibit:** {best_bibit}")
# st.success(f"**Skor:** {best_score:.4f}")
st.success(f"Alternatif Terbaik: **{best_bibit}** dengan skor **{best_score:.4f}**")
