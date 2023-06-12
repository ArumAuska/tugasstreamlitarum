import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs
import matplotlib.ticker as mtick
import seaborn as sns
import streamlit as st

roboto = {"fontname": "Roboto", "size": "12"}
roboto_title = {"fontname": "Roboto", "size": "14", "weight": "bold"}
roboto_bold = {"fontname": "Roboto", "size": "12", "weight": "bold"}
roboto_small = {"fontname": "Roboto", "size": "10"}


class Frequentist(object):
    """
    Kelas untuk mewakili data uji yag digunakan untuk analisis Frequentist

    ...

    Attributes
    ---------
    visitors_A, visitors_B : int
        Jumlah pengunjung di kedua variasi
    conversions_A, conversions_B : int
        Jumlah konversi di salah satu variase
    alpha: float (optional)
        Type I error probability (default = 0.05)
    two_tails : bool (optional)
        Boolean mendefinisikan a two-tail or one-tail test
        (default = True)
    control_cr, variant_cr : float
        Rasio konversi untuk A dan B, diberi label A sebagai kontrol dan
        B sebagai varian
    relative_difference : float
        Persentase perbedaan antara A dan B
    control_se, variant_se : float
        The standard error of the means
    se_difference : float
        Kesalahan standar dari perbedaan rata-rata

    Methods
    -------


    """

    def __init__(
        self,
        visitors_A,
        conversions_A,
        visitors_B,
        conversions_B,
        alpha=0.05,
        two_tails=True,
    ):
        self.visitors_A = visitors_A
        self.conversions_A = conversions_A
        self.visitors_B = visitors_B
        self.conversions_B = conversions_B
        self.alpha = alpha
        self.two_tails = two_tails
        self.control_cr = conversions_A / visitors_A
        self.variant_cr = conversions_B / visitors_B
        self.relative_difference = self.variant_cr / self.control_cr - 1
        self.control_se = (self.control_cr * (1 - self.control_cr) / visitors_A) ** 0.5
        self.variant_se = (self.variant_cr * (1 - self.variant_cr) / visitors_B) ** 0.5
        self.se_difference = (self.control_se ** 2 + self.variant_se ** 2) ** 0.5
        if two_tails is False:
            if self.relative_difference < 0:
                self.tail_direction = "right"
            else:
                self.tail_direction = "left"
        else:
            self.tail_direction = "two"

    def z_test(self):
        """Jalankan Z-test dengan data anda, kembalikan Z-score and p-value.

        Returns
        -------
        z_score : float
            Jumlah simpangan baku antara rata-rata distribusi
            tingkat konversi kontrol dan tingkat konversi varian
        p_value : float
            Probabilitas memperoleh hasil tes setidaknya sama ekstrimnya
            dengan hasil yang diamati, di bawah kondisi hipotesis nol
        """

        combined_cr = (self.conversions_A + self.conversions_B) / (
            self.visitors_A + self.visitors_B
        )
        self.combined_se = (
            combined_cr
            * (1 - combined_cr)
            * (1 / self.visitors_A + 1 / self.visitors_B)
        ) ** 0.5

        # z-score
        self.z_score = (self.variant_cr - self.control_cr) / self.combined_se

        # Calculate the p-value dependent on one or two tails
        if self.tail_direction == "left":
            self.p_value = scs.norm.cdf(-self.z_score)
        elif self.tail_direction == "right":
            self.p_value = scs.norm.cdf(self.z_score)
        else:
            self.p_value = 2 * scs.norm.cdf(-abs(self.z_score))

        return self.z_score, self.p_value

    def get_power(self):
        """Returns observed power dari hasil pengujian."""

        n = self.visitors_A + self.visitors_B

        if self.two_tails:
            qu = scs.norm.ppf(1 - self.alpha / 2)
        else:
            qu = scs.norm.ppf(1 - self.alpha)

        diff = abs(self.variant_cr - self.control_cr)
        avg_cr = (self.control_cr + self.variant_cr) / 2

        control_var = self.control_cr * (1 - self.control_cr)
        variant_var = self.variant_cr * (1 - self.variant_cr)
        avg_var = avg_cr * (1 - avg_cr)

        power_lower = scs.norm.cdf(
            (n ** 0.5 * diff - qu * (2 * avg_var) ** 0.5)
            / (control_var + variant_var) ** 0.5
        )
        power_upper = 1 - scs.norm.cdf(
            (n ** 0.5 * diff + qu * (2 * avg_var) ** 0.5)
            / (control_var + variant_var) ** 0.5
        )

        self.power = power_lower + power_upper

        return self.power

    def get_z_value(self):
        z_dist = scs.norm()
        if self.two_tails:
            self.alpha = self.alpha / 2
            area = 1 - self.alpha
        else:
            area = 1 - self.alpha

        self.z = z_dist.ppf(area)
        return self.z

    def plot_test_visualisation(self):
        """Memplot visualisasi uji Z dan hasilnya."""

        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        xA = np.linspace(0 - 4 * self.se_difference, 0 + 4 * self.se_difference, 1000)
        yA = scs.norm(0, self.se_difference).pdf(xA)
        ax.plot(xA, yA, c="#181716")

        diff = self.variant_cr - self.control_cr

        ax.axvline(
            x=diff, ymax=ax.get_ylim()[1], c="tab:orange", alpha=0.5, linestyle="--"
        )
        ax.text(
            ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.8,
            ax.get_ylim()[1] * 0.8,
            "Observed\ndifference: {:.2%}".format(self.relative_difference),
            color="tab:orange",
            **roboto,
        )

        if self.tail_direction == "left":
            ax.fill_between(
                xA,
                0,
                yA,
                where=(xA > 0 + self.se_difference * self.z),
                color="green",
                alpha=0.2,
            )
        elif self.tail_direction == "right":
            ax.fill_between(
                xA,
                0,
                yA,
                where=(xA < 0 - self.se_difference * self.z),
                color="green",
                alpha=0.2,
            )
        else:
            ax.fill_between(
                xA,
                0,
                yA,
                where=(xA > 0 + self.se_difference * self.z)
                | (xA < 0 - self.se_difference * self.z),
                color="green",
                alpha=0.2,
            )

        ax.get_xaxis().set_major_formatter(
            mtick.FuncFormatter(lambda x, p: format(x / self.control_cr, ".0%"))
        )

        plt.xlabel("Perbedaan relative rata-rata")

        ax.text(
            ax.get_xlim()[0],
            ax.get_ylim()[1] * 1.25,
            "Z-test visualisation",
            **roboto_title,
        )

        ax.text(
            ax.get_xlim()[0],
            ax.get_ylim()[1] * 1.18,
            "Menampilkan distribusi yang diharapkan dari perbedaan antara"
            "rata-rata di bawah hypothesis nol.",
            **roboto,
        )

        sns.despine(left=True)
        ax.get_yaxis().set_visible(False)
        fig.tight_layout()

        st.write(fig)

    def plot_power(self):
        """Mengembalikan figur plot streamlit yang memvisualisasikan kekuatan berdasarkan
        hasil tes AB."""

        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

        # Plot the distribution of A
        xA = np.linspace(
            self.control_cr - 4 * self.control_se,
            self.control_cr + 4 * self.control_se,
            1000,
        )
        yA = scs.norm(self.control_cr, self.control_se).pdf(xA)
        ax.plot(xA, yA, label="A")

        # Plot the distribution of B
        xB = np.linspace(
            self.variant_cr - 4 * self.variant_se,
            self.variant_cr + 4 * self.variant_se,
            1000,
        )
        yB = scs.norm(self.variant_cr, self.variant_se).pdf(xB)
        ax.plot(xB, yB, label="B")

        # Label A at its apex
        ax.text(
            self.control_cr,
            max(yA) * 1.03,
            "A",
            color="tab:blue",
            horizontalalignment="center",
            **roboto_bold,
        )

        # Label B at its apex
        ax.text(
            self.variant_cr,
            max(yB) * 1.03,
            "B",
            color="tab:orange",
            horizontalalignment="center",
            **roboto_bold,
        )

        # Add critical value lines depending on two vs. one tail and left vs. right
        if self.tail_direction == "left":
            ax.axvline(
                x=self.control_cr + self.control_se * self.z,
                c="tab:blue",
                alpha=0.5,
                linestyle="--",
            )
            ax.text(
                self.control_cr + self.control_se * self.z,
                max(yA) * 0.4,
                "Critical value",
                color="tab:blue",
                rotation=270,
                **roboto_small,
            )
        elif self.tail_direction == "right":
            ax.axvline(
                x=self.control_cr - self.control_se * self.z,
                c="tab:blue",
                alpha=0.5,
                linestyle="--",
            )
            ax.text(
                self.control_cr - self.control_se * self.z,
                max(yA) * 0.4,
                "Critical value",
                color="tab:blue",
                rotation=270,
                **roboto_small,
            )
        else:
            ax.axvline(
                x=self.control_cr - self.control_se * self.z,
                c="tab:blue",
                alpha=0.5,
                linestyle="--",
            )
            ax.text(
                self.control_cr - self.control_se * self.z,
                max(yA) * 0.4,
                "Critical value",
                color="tab:blue",
                rotation=270,
                **roboto_small,
            )

            ax.axvline(
                x=self.control_cr + self.control_se * self.z,
                c="tab:blue",
                alpha=0.5,
                linestyle="--",
            )
            ax.text(
                self.control_cr + self.control_se * self.z,
                max(yA) * 0.4,
                "Critical value",
                color="tab:blue",
                rotation=270,
                **roboto_small,
            )

        # Fill in the power and annotate
        if self.variant_cr > self.control_cr:
            ax.fill_between(
                xB,
                0,
                yB,
                where=(xB > self.control_cr + self.control_se * self.z),
                color="green",
                alpha=0.2,
            )
        else:
            ax.fill_between(
                xB,
                0,
                yB,
                where=(xB < self.control_cr - self.control_se * self.z),
                color="green",
                alpha=0.2,
            )

        # Display power value on graph
        ax.text(
            ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.8,
            ax.get_ylim()[1] * 0.8,
            f"Power: {self.power:.2%}",
            horizontalalignment="left",
            **roboto,
        )

        # Title
        ax.text(
            ax.get_xlim()[0],
            ax.get_ylim()[1] * 1.25,
            "Statistical power",
            **roboto_title,
        )

        # Subtitle
        ax.text(
            ax.get_xlim()[0],
            ax.get_ylim()[1] * 1.17,
            "Menggambarkan kemungkinan menghindari kesalahan negatif/tipe II" " error",
            **roboto,
        )

        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))
        plt.xlabel("Converted Proportion")

        sns.despine(left=True)
        ax.get_yaxis().set_visible(False)
        fig.tight_layout()

        st.write(fig)