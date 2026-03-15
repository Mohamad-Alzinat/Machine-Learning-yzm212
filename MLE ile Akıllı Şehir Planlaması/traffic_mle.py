"""
YZM212 Makine Öğrenmesi 2 - Laboratuvar Ödevi
MLE ile Akıllı Şehir Planlaması

Bu dosya, ödevde istenen sayısal MLE çözümünü ve görselleştirmeleri üretir.
GitHub üzerinde doğrudan çalışması için .py biçiminde hazırlanmıştır.
"""

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.special import gammaln

# =========================================================
# Bölüm 2 - Veriyi Yükleme
# =========================================================
# Gözlemlenen trafik verisi: 1 dakikada geçen araç sayısı
traffic_data = np.array([12, 15, 10, 8, 14, 11, 13, 16, 9, 12, 11, 14, 10, 15])


# =========================================================
# Bölüm 2 - Negatif Log-Likelihood Fonksiyonu
# =========================================================
def negative_log_likelihood(lam, data):
    """
    Poisson dağılımı için negatif log-likelihood hesaplar.

    Parametreler:
        lam  : Poisson parametresi (lambda > 0)
        data : Gözlenen veri dizisi

    Not:
    log(k!) terimi türev alırken sabit kalsa da, burada fonksiyonu
    tam ve sayısal olarak kararlı biçimde yazmak için scipy.special.gammaln
    kullanılmıştır. Çünkü log(k!) = gammaln(k + 1) eşitliği vardır.
    """
    lam = float(np.atleast_1d(lam)[0])

    # Geçersiz lambda değerlerini engelle
    if lam <= 0:
        return np.inf

    # Toplam log-likelihood:
    # sum(k_i * log(lam) - lam - log(k_i!))
    log_likelihood = np.sum(data * np.log(lam) - lam - gammaln(data + 1))

    # Optimizasyon minimizasyon yaptığı için negatifini döndürüyoruz
    nll = -log_likelihood
    return nll


# =========================================================
# Bölüm 2 - Sayısal MLE Hesabı
# =========================================================
initial_guess = [1.0]

result = opt.minimize(
    negative_log_likelihood,
    initial_guess,
    args=(traffic_data,),
    bounds=[(0.001, None)]
)

lambda_mle_numerical = result.x[0]
lambda_mle_analytical = np.mean(traffic_data)

print("=== Bölüm 2: Sayısal MLE Sonucu ===")
print(f"Sayısal Tahmin (MLE lambda): {lambda_mle_numerical:.6f}")
print(f"Analitik Tahmin (Ortalama):  {lambda_mle_analytical:.6f}")


# =========================================================
# Bölüm 3 - Model Karşılaştırma ve Görselleştirme
# =========================================================
k_values = np.arange(int(traffic_data.min()) - 2, int(traffic_data.max()) + 3)
pmf_values = poisson.pmf(k_values, mu=lambda_mle_numerical)

plt.figure(figsize=(9, 5.5))
bins = np.arange(traffic_data.min() - 0.5, traffic_data.max() + 1.5, 1)

# Histogram yoğunluk biçiminde çiziliyor ki PMF ile karşılaştırılabilsin
plt.hist(
    traffic_data,
    bins=bins,
    density=True,
    alpha=0.6,
    edgecolor="black",
    label="Gerçek Veri Histogramı"
)

plt.plot(
    k_values,
    pmf_values,
    marker="o",
    linewidth=2,
    label=f"Poisson PMF (λ = {lambda_mle_numerical:.4f})"
)

plt.xlabel("1 Dakikadaki Araç Sayısı")
plt.ylabel("Olasılık / Yoğunluk")
plt.title("Trafik Verisi Histogramı ve Poisson MLE Uyum Grafiği")
plt.legend()
plt.tight_layout()
plt.savefig("trafik_pmf_histogram.png", dpi=200)
print("\nGrafik kaydedildi: trafik_pmf_histogram.png")


# =========================================================
# Bölüm 4 - Outlier Analizi
# =========================================================
traffic_data_with_outlier = np.append(traffic_data, 200)

lambda_with_outlier = np.mean(traffic_data_with_outlier)

print("\n=== Bölüm 4: Outlier Analizi ===")
print(f"Outlier öncesi λ (ortalama):  {lambda_mle_analytical:.6f}")
print(f"Outlier sonrası λ (ortalama): {lambda_with_outlier:.6f}")

# İsteğe bağlı ikinci görselleştirme
k_values_outlier = np.arange(0, 60)

plt.figure(figsize=(9, 5.5))
plt.hist(
    traffic_data_with_outlier,
    bins=20,
    density=True,
    alpha=0.5,
    edgecolor="black",
    label="Outlier Eklenmiş Veri"
)
plt.plot(
    k_values_outlier,
    poisson.pmf(k_values_outlier, mu=lambda_mle_analytical),
    marker="o",
    linewidth=1.8,
    label=f"Orijinal λ = {lambda_mle_analytical:.2f}"
)
plt.plot(
    k_values_outlier,
    poisson.pmf(k_values_outlier, mu=lambda_with_outlier),
    marker="s",
    linewidth=1.8,
    label=f"Outlier Sonrası λ = {lambda_with_outlier:.2f}"
)
plt.xlim(0, 60)
plt.xlabel("1 Dakikadaki Araç Sayısı")
plt.ylabel("Olasılık / Yoğunluk")
plt.title("Outlier Etkisinin λ Üzerindeki Değişimi")
plt.legend()
plt.tight_layout()
plt.savefig("outlier_etkisi.png", dpi=200)
print("Grafik kaydedildi: outlier_etkisi.png")
