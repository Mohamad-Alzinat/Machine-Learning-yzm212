# YZM212 Makine Öğrenmesi 2 - Laboratuvar Ödevi

## Ödev Başlığı
**MLE ile Akıllı Şehir Planlaması**

## Problem Tanımı
Bu ödevde, bir ana caddeden **1 dakikada geçen araç sayısı** verisi Poisson dağılımı ile modellenmiştir. Amaç, Poisson dağılımının parametresi olan **λ (lambda)** değerini **Maximum Likelihood Estimation (MLE)** yöntemiyle bulmaktır.

## Veri
Kullanılan trafik verisi:
`[12, 15, 10, 8, 14, 11, 13, 16, 9, 12, 11, 14, 10, 15]`

Gözlem sayısı: **14**

## Yöntem
Çalışma iki şekilde yapılmıştır:

1. **Analitik çözüm:**  
   Poisson olabilirlik fonksiyonu yazılmış, log-likelihood alınmış ve türev sıfıra eşitlenerek
   `λ_MLE = örnek ortalaması` sonucu gösterilmiştir.

2. **Sayısal çözüm (Python):**  
   `negative_log_likelihood` fonksiyonu tanımlanmış ve `scipy.optimize.minimize` ile minimum nokta bulunmuştur.

## Sonuçlar
- **Analitik λ tahmini:** `12.142857`
- **Sayısal λ tahmini:** `12.142855`

Bu iki sonuç neredeyse aynıdır. Bu da sayısal optimizasyonun doğru çalıştığını gösterir.

## Görselleştirme
Dosya çalıştırıldığında iki grafik üretilir:

- `trafik_pmf_histogram.png`  
  Orijinal veri histogramı ile Poisson PMF grafiğini birlikte gösterir.

- `outlier_etkisi.png`  
  Veri setine `200` değerli aykırı gözlem eklendiğinde λ değerindeki değişimi gösterir.

## Yorum / Tartışma
Orijinal veri için bulunan λ değeri yaklaşık `12.1429` olup, histogram ile Poisson eğrisi genel olarak uyumludur. Veri değerleri 8 ile 16 arasında toplandığı için model, tipik trafik akışını makul biçimde temsil eder.

Ancak veri setine **200** gibi çok büyük bir aykırı değer eklendiğinde ortalama `24.6667` seviyesine çıkar. Bu durum MLE'nin ortalamaya dayalı olduğu için **uç değerlere hassas** olduğunu gösterir. Gerçek hayatta böyle bir hata, belediyenin trafiği olduğundan çok daha yoğun sanmasına ve gereksiz kapasite artışı, yanlış bütçe planlaması veya hatalı yol genişletme kararı almasına neden olabilir.

## Dosyalar
- `traffic_mle.py` → Python çözümü
- `report.pdf` → İstenen teknik rapor
- `trafik_pmf_histogram.png` → Ana görselleştirme
- `outlier_etkisi.png` → Aykırı değer etkisi grafiği

## Çalıştırma
```bash
pip install numpy scipy matplotlib
python traffic_mle.py
```
