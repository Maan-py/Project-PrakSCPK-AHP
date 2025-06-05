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
kriteria = [
    "Seed Yield per Unit Area (SYUA)",  # Hasil biji kedelai yang dihasilkan dalam satu satuan luas lahan (misalnya kg/ha).
    "Number of Pods (NP)",  #  Jumlah polong per tanaman.
    "Protein Content (PCO)",  #  Kandungan total protein dalam biji kedelai.
    "Relative Water Content in Leaves (RWCL)",  # Persentase kandungan air relatif dalam daun tanaman.
    "Weight of 300 Seeds (W3S)",  # Total berat dari 300 butir biji kedelai.
]


# Alternatif
alternatif = df["Parameters"].unique()
bibit_df = pd.DataFrame(alternatif, columns=["Bibit"])


# Normalisasi matriks
def calc_norm(M):
    if M.ndim == 1:
        sM = np.sum(M)
        return M / sM
    else:
        sM = np.sum(M, axis=0)
        return M / sM


# Fungsi untuk mendapatkan index perbandingan unik di atas diagonal
def get_comparisons(kriteria):
    comparisons = []
    for i in range(len(kriteria)):
        for j in range(i + 1, len(kriteria)):
            comparisons.append((i, j))
    return comparisons


def validity_check(M, W):
    """Fungsi untuk mengecek validitas bobot"""
    n = len(M)
    RI = {
        2: 0.00,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.51,
        11: 1.53,
        12: 1.54,
        13: 1.56,
        14: 1.57,
    }
    # menghitung CV
    CV = M @ W / W  # operator @ mengalikan matriks sesuai dengan prinsipnya
    st.write(f"CV: {CV}\n")
    # menghitung nilai eigen
    eigen = np.mean(CV)
    st.write(f"Eigen: {eigen}\n")
    # mencari nilai RI
    st.write(f"RI: {RI[n]}\n")

    # menghitung CI
    CI = (eigen - n) / (n - 1)
    st.write(f"CI: {CI}\n")
    # menghitung CR
    CR = CI / RI[n]
    st.write(f"CR: {CR}\n")
    if CR <= 0.1:
        st.success("✅ Konsistensi matriks dapat diterima (CR ≤ 0.1)")
    else:
        st.warning(
            "⚠️ Konsistensi matriks **tidak dapat diterima** (CR > 0.1). Silakan ubah nilai perbandingan."
        )


# Alternatif + Kriteria
st.write("#### Dataset")
with st.expander("Dataset : "):
    st.write("#### 📃 Dataset Advanced Soybean Agricultural Dataset")
    st.dataframe(df[["Parameters"] + kriteria])

st.write("#### Alternatif")
with st.expander("Alternatif Bibit : "):
    st.write("#### 🌱 Alternatif Bibit")
    st.dataframe(bibit_df)

grouped = df.groupby("Parameters")[kriteria].mean()

st.write("#### Group by Alternatif Dataset")
with st.expander("Data Setelah Di-groupby : "):
    # st.write(pd.DataFrame(grouped, index=alternatif).style.format("{:.2f}"))
    st.dataframe(grouped.style.format("{:.2f}"))

with st.sidebar:
    scale = {
        "1 - Sama penting": 1,
        "3 - Sedikit lebih penting": 3,
        "5 - Cukup lebih penting": 5,
        "7 - Sangat lebih penting": 7,
        "9 - Mutlak lebih penting": 9,
    }
    st.write("## 🧮 Masukkan Nilai Perbandingan Berpasangan")
    st.table(scale)

    n = len(kriteria)
    MPBk = np.ones((n, n))

    comparisons = get_comparisons(kriteria)

    for i, j in comparisons:
        label = f"{kriteria[i]} vs {kriteria[j]}"
        value = st.slider(
            label, min_value=0.1, max_value=9.0, value=1.0, step=0.1, key=f"{i}-{j}"
        )
        MPBk[i][j] = value
        MPBk[j][i] = round(1 / value, 3)

st.write("#### Kriteria")
with st.expander("Detail : "):
    st.write("#### 📋 Matriks Perbandingan Berpasangan")
    st.write(pd.DataFrame(MPBk, columns=kriteria, index=kriteria))
    # df_matrix = pd.DataFrame(MPBk, index=kriteria, columns=kriteria)
    # st.dataframe(df_matrix.style.format(precision=3))

    # Langkah 2: Hitung bobot prioritas (eigen vector)
    w_MPB = calc_norm(MPBk)
    st.write("#### 📊 Normalisasi Matriks")
    st.write(pd.DataFrame(w_MPB, columns=kriteria, index=kriteria))

    m, n = w_MPB.shape
    V = np.zeros(m)
    for i in range(m):
        V[i] = np.sum(w_MPB[i, :])
    w_MPB = V / m

    st.write("#### ⭐ Bobot Prioritas (Eigen Vector)")
    st.write(pd.DataFrame(w_MPB, columns=["Eigenvektor"], index=kriteria))

    # validity_check(np.array(MPBk), np.array(w_MPB))
    # for i in range(n):
    #     st.write(f"- {kriteria[i]}: **{w_MPB[i]:.4f}**")

# Tampilkan hasil
# st.write("## Rata-rata Kriteria Tiap Varietas Unik")
# st.dataframe(grouped)
st.write("#### Alternatif 1 - Seed Yield per Unit Area (SYUA)")
with st.expander("Detail : "):
    st.write(
        "#### 📋 Perbandingan Seed Yield per Unit Area (SYUA): Alternatif Kuantitatif"
    )
    st.dataframe(grouped[["Seed Yield per Unit Area (SYUA)"]].style.format("{:.2f}"))
    # st.write(
    #     pd.DataFrame(
    #         grouped[["Seed Yield per Unit Area (SYUA)"]], index=alternatif
    #     ).style.format("{:.2f}")
    # )

    # Normalisasi kolom SYUA
    norm_SYUA = calc_norm(grouped["Seed Yield per Unit Area (SYUA)"].values)

    # Simpan sebagai DataFrame
    w_SYUA = pd.DataFrame(norm_SYUA, columns=["Eigenvector"], index=grouped.index)

    st.write(
        "#### 📊 Eigenvector (Bobot Alternatif untuk Seed Yield per Unit Area (SYUA)"
    )
    st.dataframe(w_SYUA.style.format("{:.4f}"))

st.write("#### Alternatif 2 - Number of Pods (NP)")
with st.expander("Detail : "):
    st.write("#### 📋 Perbandingan Number of Pods (NP): Alternatif Kuantitatif")
    st.dataframe(grouped[["Number of Pods (NP)"]].style.format("{:.2f}"))

    # Normalisasi kolom NP
    norm_NP = calc_norm(grouped["Number of Pods (NP)"].values)

    # Simpan sebagai DataFrame
    w_NP = pd.DataFrame(norm_NP, columns=["Eigenvector"], index=grouped.index)

    st.write("#### 📊 Eigenvector (Bobot Alternatif untuk Number of Pods (NP))")
    st.dataframe(w_NP.style.format("{:.4f}"))

st.write("#### Alternatif 3 - Protein Content (PCO)")
with st.expander("Detail : "):
    st.write("#### 📋 Perbandingan Protein Content (PCO): Alternatif Kuantitatif")
    st.dataframe(grouped[["Protein Content (PCO)"]].style.format("{:.2f}"))

    # Normalisasi kolom PCO
    norm_PCO = calc_norm(grouped["Protein Content (PCO)"].values)

    # Simpan sebagai DataFrame
    w_PCO = pd.DataFrame(norm_PCO, columns=["Eigenvector"], index=grouped.index)

    st.write("#### 📊 Eigenvector (Bobot Alternatif untuk Protein Content (PCO)")
    st.dataframe(w_PCO.style.format("{:.4f}"))

st.write("#### Alternatif 4 - Relative Water Content in Leaves (RWCL)")
with st.expander("Detail : "):
    st.write(
        "#### 📋 Perbandingan Relative Water Content in Leaves (RWCL): Alternatif Kuantitatif"
    )
    st.dataframe(
        grouped[["Relative Water Content in Leaves (RWCL)"]].style.format("{:.2f}")
    )

    # Normalisasi kolom RWCL
    norm_RWCL = calc_norm(grouped["Relative Water Content in Leaves (RWCL)"].values)

    # Simpan sebagai DataFrame
    w_RWCL = pd.DataFrame(norm_RWCL, columns=["Eigenvector"], index=grouped.index)

    st.write(
        "#### 📊 Eigenvector (Bobot Alternatif untuk Relative Water Content in Leaves (RWCL)"
    )
    st.dataframe(w_RWCL.style.format("{:.4f}"))

st.write("#### Alternatif 5 - Weight of 300 Seeds (W3S)")
with st.expander("Detail : "):
    st.write("#### 📋 Perbandingan Weight of 300 Seeds (W3S): Alternatif Kuantitatif")
    st.dataframe(grouped[["Weight of 300 Seeds (W3S)"]].style.format("{:.2f}"))

    # Normalisasi kolom W3S
    norm_W3S = calc_norm(grouped["Weight of 300 Seeds (W3S)"].values)

    # Simpan sebagai DataFrame
    w_W3S = pd.DataFrame(norm_W3S, columns=["Eigenvector"], index=grouped.index)

    st.write("#### 📊 Eigenvector (Bobot Alternatif untuk Weight of 300 Seeds (W3S)")
    st.dataframe(w_W3S.style.format("{:.4f}"))

# Menyusun matrix alternatif (wM)
wM = np.column_stack((w_SYUA, w_NP, w_PCO, w_RWCL, w_W3S))
st.write("## Jawaban Akhir dan Vector Keputusan")

st.write("#### 📋 Matriks Bobot Alternatif")
st.dataframe(
    pd.DataFrame(wM, columns=kriteria, index=grouped.index).style.format("{:.4f}")
)

st.write("#### 📊 Nilai Akhir (Skor Masing-masing Bibit)")
w_MPB = np.array(w_MPB).flatten()
MC_Scores = np.dot(wM, w_MPB)

scores_df = pd.DataFrame({"Nilai Akhir": MC_Scores}, index=grouped.index)
scores_df["Ranking"] = scores_df["Nilai Akhir"].rank(ascending=False).astype(int)
result_df = scores_df.sort_values(by="Nilai Akhir", ascending=False)
st.dataframe(result_df.style.format({"Nilai Akhir": "{:.4f}", "Ranking": "{:.0f}"}))

max_bibit_score = np.max(MC_Scores)
max_score_index = np.argmax(MC_Scores)
best_alternatif = alternatif[max_score_index]

st.subheader("🎉 Bibit Terbaik")
st.success(
    f"Bibit terbaik terpilih berdasarkan kriteria adalah **{best_alternatif}** dengan nilai akhir **{max_bibit_score:.4f}**"
)

# # # Langkah 3: Konsistensi (λ_max, CI, CR)
# # # Hitung λ_max
# Aw = np.dot(MPBk, w_MPB)
# lambda_max = np.sum(Aw / w_MPB) / n

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
# st.write(f"λ Maksimum (λ_max): {lambda_max:.4f}")
# st.write(f"Consistency Index (CI): {CI:.4f}")
# st.write(f"Random Index (RI): {RI:.2f}")
# st.write(f"Consistency Ratio (CR): {CR:.4f}")

# if CR < 0.1:
#     st.success("Matriks konsisten (CR < 0.1).")
# else:
#     st.warning(
#         "Matriks tidak konsisten (CR ≥ 0.1), silakan sesuaikan penilaian pairwise."
#     )


# grouped = df.groupby("Parameters")[kriteria].mean()
# # Tampilkan hasil
# st.subheader("Rata-rata Kriteria Tiap Varietas Unik")
# st.dataframe(grouped)

# # Dropdown untuk memilih salah satu bibit
# selected_bibit = st.selectbox("Pilih Bibit untuk Analisis:", bibit_df["Bibit"].tolist())

# # Tampilkan data yang sesuai dengan pilihan
# filtered_df = df[df["Parameters"] == selected_bibit][["Parameters"] + kriteria]
# st.subheader(f"Data untuk Bibit: {selected_bibit}")
# st.dataframe(filtered_df)


# # Skala pilihan


# # st.subheader("📊 Matriks Normalisasi")
# # df_norm = pd.DataFrame(w_MPB, index=kriteria, columns=kriteria)
# # st.dataframe(df_norm.style.format(precision=3))
