import streamlit as st
import matplotlib.pyplot as plt
from functions import create_plotly_table, local_css, percentage_format
from bayesian import Bayesian
from frequentist import Frequentist

st.set_page_config(
    page_title="AB Test Calculator",
    page_icon="https://rfoxdata.co.uk/assets/favicon/favicon-32x32.png",
)

roboto = {"fontname": "Roboto", "size": "12"}
roboto_title = {"fontname": "Roboto", "size": "14", "weight": "bold"}
roboto_bold = {"fontname": "Roboto", "size": "12", "weight": "bold"}
roboto_small = {"fontname": "Roboto", "size": "10"}

local_css("style.css")

font = {"family": "sans-serif", "sans-serif": "roboto", "size": 11}

plt.rc("font", **font)

"""
# AB test calculator

_Masukkan data pengujian Anda ke sidebar dan pilih pendekatan pengujian Bayesian atau Frequentist. Dibahwa ini adalah Vayesian secara default._

---
"""

# Sidebar
st.sidebar.markdown(
    """
## Approach
"""
)

method = st.sidebar.radio("Bayesian vs. Frequentist", ["Bayesian", "Frequentist"])


st.sidebar.markdown(
    """
## Test data
"""
)

visitors_A = st.sidebar.number_input("Pengunjung A", value=50000, step=100)
conversions_A = st.sidebar.number_input("Konversi A", value=1500, step=10)
visitors_B = st.sidebar.number_input("Pengunjung B", value=50000, step=100)
conversions_B = st.sidebar.number_input("Konversi B", value=1560, step=10)

st.sidebar.markdown(
    """
## Frequentist settings
"""
)

alpha_input = 1 - st.sidebar.slider(
    "Significance level", value=0.95, min_value=0.5, max_value=0.99
)
tails_input = st.sidebar.selectbox(
    "One vs. two tail", ["One-tail", "Two-tail"], index=1
)

if tails_input == "One-tail":
    two_tails_bool = False
else:
    two_tails_bool = True

b = Bayesian(visitors_A, conversions_A, visitors_B, conversions_B)

# Bayesian Method
if method == "Bayesian":

    try:
        b.generate_posterior_samples()
        b.calculate_probabilities()
        b.plot_bayesian_probabilities()

        st.text("")

        bayesian_data = {
            "<b>Varians</b>": ["A", "B"],
            "<b>Pengunjung</b>": [f"{b.visitors_A:,}", f"{b.visitors_B:,}"],
            "<b>Konversi</b>": [b.conversions_A, b.conversions_B],
            "<b>Rasio KOnversi</b>": [f"{b.control_cr:.2%}", f"{b.variant_cr:.2%}"],
            "<b>Peningkatan</b>": ["", f"{b.relative_difference:.2%}"],
            "<b>Likelihood of being better</b>": [f"{b.prob_A:.2%}", f"{b.prob_B:.2%}"],
        }

        create_plotly_table(bayesian_data)

        """
        Grafik di bawah memplot perbedaan yang disimulasikan antara dua
        distribusi posterior untuk varians. Ini men-highlight 
        rentang perbedaan antara kedua varians. Lebih banyak data akan mengurangi
        jangkauan.
        """

        st.text("")

        b.plot_simulation_of_difference()

    

       

    except ValueError:

        t = """
        <img class='error'
            src='https://www.flaticon.com/svg/static/icons/svg/595/595067.svg'>
        """
        st.markdown(t, unsafe_allow_html=True)

        """
        Terjadi kesalahan, harap periksa input data pengujian dan coba lagi.

        Untuk perhitungan bayesian, tingkat konversi harus antara 0 dan 1.
        """


else:  # Frequentist

    f = Frequentist(
        visitors_A,
        conversions_A,
        visitors_B,
        conversions_B,
        alpha=alpha_input,
        two_tails=two_tails_bool,
    )

    z_score, p_value = f.z_test()

    power = f.get_power()

    if p_value < alpha_input:
        t = """
        <h3 class='frequentist_title'>Significant</h3>
        <img class='frequentist_icon'
            src='https://raw.githubusercontent.com/rjjfox/ab-test-calculator/master/img/positive-vote.png'>
        """
        st.markdown(t, unsafe_allow_html=True)

        if f.relative_difference < 0:
            t = (
                """
            <p>B's conversion rate is <span class='lower'>"""
                + "{:.2%}".format(abs(f.relative_difference))
                + """ lower</span> than A's CR."""
            )
            st.markdown(t, unsafe_allow_html=True)
        else:
            t = (
                """
            <p>B's conversion rate is <span class='higher'>"""
                + "{:.2%}".format(abs(f.relative_difference))
                + """ higher</span> than A's CR."""
            )
            st.markdown(t, unsafe_allow_html=True)

        f"""
        Anda dapat {1-alpha_input:.0%} yakin bahwa hasilnya benar dan karena
        perubahan dilakukan. Ada kemungkinan {alpha_input:.0%} bahwa hasilnya
        adalah false positive atau error tipe I  yang berarti hasilnya adalah karena
        kebetulan acak.
        """

    else:
        t = """
        <h3 class='frequentist_title'>Not significant</h3>
        <img class='frequentist_icon'
            src='https://raw.githubusercontent.com/rjjfox/ab-test-calculator/master/img/negative-vote.png'>
        """
        st.markdown(t, unsafe_allow_html=True)

        f"""
        Tidak ada cukup bukti untuk membuktikan adanya 
        {f.relative_difference:.2%} perbedaan rasio konversi antara 
        varians A and B.
        """

        """
        Kumpulkan lebih banyak data untuk mencapai presisi yang lebih tinggi dalam pengujian Anda,
        atau simpulkan pengujian sebagai tidak menyakinkan.
        """

    frequentist_data = {
        "<b>Varians</b>": ["A", "B"],
        "<b>Pengunjung</b>": [f"{f.visitors_A:,}", f"{f.visitors_B:,}"],
        "<b>Konversi</b>": [f.conversions_A, f.conversions_B],
        "<b>Rasio Konversi</b>": [f"{f.control_cr:.2%}", f"{f.variant_cr:.2%}"],
        "<b>Peningkatan</b>": ["", f"{f.relative_difference:.2%}"],
        "<b>Power</b>": ["", f"{power:.4f}"],
        "<b>Z-score</b>": ["", f"{z_score:.4f}"],
        "<b>P-value</b>": ["", f"{p_value:.4f}"],
    }

    create_plotly_table(frequentist_data)

    z = f.get_z_value()

    """
    Menurut hipotesis nol, tidak ada perbedaan antara rata-rata.
    Plot di bawah ini menunjukkan distribusi perbedaan rata-rata 
    kita harap di bawah hypothesis nol.
    """

    f.plot_test_visualisation()

    if p_value < alpha_input:
        f"""
        Area yang diarsir mencakup {alpha_input:.0%} dari distribusi. Itu karena
        rata-rata yang diamati dari varian termasuk dalam area ini sehingga
        kita dapat menolak hypothesis nol dengan {1-alpha_input:.0%} confidence.
        """
    else:
        f"""
        Area yang diarsir mencakup {alpha_input:.0%} dari distribusi. Itu karena
        rata-rata yang diamati dari varian tidak termasuk dalam area ini sehingga
        kita tidak dapat menolak hypothesis nol dan mendapatkan hasil yang signifikan.
        Perbedaan lebih dari
        {f.se_difference*z/f.control_cr:.2%} is needed.
        """

    """
    #### Statistical Power
    """

    f"""
    Kekuatan adalah ukuran seberapa besar kemungkinan kita mendeteksi perbedaan ketika ada
    isatu dengan 80% menjadi ambang batas yang diterima secara umum untuk validitas statistik.
    **Kekuatan untuk pengujian Anda adalah {power:.2%}**
    """

    f"""
    Cara alternatif untuk mendefinisikan kekuatan adalah kemungkinan kita
    menghindari kesalahan Tipe II atau kesalahan negatif. Oleh karena itu, kebalikan dari
    kekuatan adalah 1 - {power:.2%} = {1-power:.2%} merupakan kemungkinan kita untuk
    kesalahan tipe II.
    """

    f.plot_power()

 
