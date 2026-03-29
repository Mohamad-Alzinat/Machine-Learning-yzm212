# Özdeğerler ve Özvektörler - Ödev Raporu

## Ders Bilgisi
- Ders: YZM212 Makine Öğrenmesi
- Ödev: III. Laboratuvar Değerlendirmesi
- Konu: Matris manipülasyonu, özdeğerler ve özvektörler

---

## 1) Makine öğrenmesi ile matris manipülasyonu, özdeğerler ve özvektörlerin ilişkisi

### 1.1 Kısa tanımlar

**Matris manipülasyonu**, verilerin ve doğrusal dönüşümlerin matrisler üzerinde toplama, çarpma, transpoz alma, ters alma, parçalama ve benzeri işlemlerle ele alınmasıdır. Makine öğrenmesinde veri kümeleri çoğunlukla bir tasarım matrisi X olarak tutulur; satırlar gözlemleri, sütunlar ise özellikleri temsil eder.

**Özdeğer** ve **özvektör** kavramları, karesel bir A matrisi için
A v = λ v
eşitliğini sağlayan v != 0 vektörü ve λ skalerini ifade eder. Burada v özvektör, λ ise buna karşılık gelen özdeğerdir.

### 1.2 Makine öğrenmesindeki rolü

Makine öğrenmesinde matrisler yalnızca veri saklamak için değil, modelin matematiksel yapısını kurmak için de kullanılır:

1. **Veri gösterimi**
   - Özellik matrisi X, hedef vektörü y, ağırlıklar w gibi kavramlar doğrusal cebir ile ifade edilir.
   - Birçok algoritma, tahmini Xw biçiminde üretir.

2. **Dönüşüm ve projeksiyon**
   - Özellikle boyut indirgeme yöntemlerinde veri, yeni eksenlere projekte edilir.
   - Bu yeni eksenler çoğu zaman kovaryans matrisinin özvektörleri ile bulunur.

3. **Sayısal kararlılık ve optimizasyon**
   - Hessian, kovaryans ve Gram matrislerinin spektral özellikleri, optimizasyon davranışı hakkında bilgi verir.
   - En büyük ya da en küçük özdeğerler yakınsama, koşul sayısı ve kararlılık açısından önemlidir.

### 1.3 Hangi yöntemlerde kullanılır?

#### PCA (Principal Component Analysis)
PCA, verideki en yüksek varyans yönlerini bulur. Bunun için genellikle kovaryans matrisinin özdeğerleri ve özvektörleri hesaplanır. En büyük özdeğere sahip özvektörler, veriyi en iyi açıklayan ana bileşenlerdir.

#### Spektral kümeleme
Graf temelli kümeleme yaklaşımlarında benzerlik matrisi veya Laplasyen matrisinin özvektörleri kullanılır. Böylece veri, kümelenmeye daha uygun bir uzaya taşınır.

#### Lineer cebir tabanlı boyut indirgeme
PCA dışında bazı matris ayrışımı yaklaşımlarında da eigendecomposition önemli rol oynar. Özellikle simetrik matrislerin analizi için yararlıdır.

#### Markov süreçleri ve geçiş matrisleri
Olasılıksal modellerde geçiş matrislerinin özdeğer yapısı, sistemin uzun dönem davranışı hakkında fikir verir.

#### Optimizasyon ve ikinci mertebe yöntemler
Kayıp fonksiyonunun Hessian matrisinin özdeğerleri, eğriliği gösterir. Bu bilgi; öğrenme oranı, kararlılık ve yakınsama analizlerinde önemlidir.

### 1.4 Kısa değerlendirme

Özetle, matris manipülasyonu makine öğrenmesinin temel hesaplama dilidir. Özdeğerler ve özvektörler ise bu dilin veri dönüşümü, yapı analizi ve boyut indirgeme tarafında çok önemli araçlarıdır. Özellikle PCA ve spektral yöntemlerde doğrudan kullanılırlar.

---

## 2) numpy.linalg.eig fonksiyonunun incelenmesi

### 2.1 Fonksiyonun amacı

`numpy.linalg.eig(a)` fonksiyonu, karesel bir matrisin özdeğerlerini ve sağ özvektörlerini hesaplar. Dönüş değeri iki parçadır:

- `eigenvalues`: özdeğerler dizisi
- `eigenvectors`: karşılık gelen özvektörlerin sütunlarda tutulduğu matris

Yani `eigenvectors[:, i]`, `eigenvalues[i]` özdeğerine karşılık gelen özvektördür.

### 2.2 Dokümantasyondan çıkan önemli noktalar

NumPy dokümantasyonuna göre:

- Girdi matrisinin karesel olması gerekir.
- Sonuçtaki özdeğerler sıralı olmak zorunda değildir.
- Gerçel matrislerde sonuçlar gerçel olabilir ya da karmaşık eşlenik çiftler halinde gelebilir.
- Özvektörler normalize edilmiş olarak döndürülür.
- Hesaplama yakınsamazsa `LinAlgError` oluşabilir.
- Genel karesel matrisler için altta LAPACK `_geev` rutinleri kullanılır.

### 2.3 Kaynak kod açısından işleyiş

NumPy kaynak kodunda `eig` fonksiyonunun akışı özetle şöyledir:

1. Girdi diziye çevrilir.
2. Matrisin en az 2 boyutlu ve karesel olduğu kontrol edilir.
3. Uygun veri tipi belirlenir.
4. Gerçel ve karmaşık durumlara göre uygun imza (`signature`) seçilir.
5. Alt seviyede `_umath_linalg.eig` çağrılır.
6. Dönen sonuçlar uygun veri tipine çevrilip kullanıcıya verilir.

Bu yapı, kullanıcıya basit bir arayüz sunarken ağır hesabı LAPACK seviyesinde yaptırır.

### 2.4 Neden önemlidir?

Bu fonksiyon pratikte çok kullanışlıdır çünkü:

- Hazır ve güvenilir bir özdeğer/özvektör çözümü verir.
- Küçük ve orta boyutlu matrislerde hızlıdır.
- Sonuçların doğrulanmasında referans çözüm olarak kullanılabilir.

---

## 3) Hazır eig kullanmadan özdeğer hesabı ve karşılaştırma

### 3.1 Referans çalışmanın özeti

LucasBN tarafından paylaşılan GitHub deposunda özdeğerler, doğrudan `numpy.linalg.eig` çağrılmadan hesaplanmaktadır. Yaklaşımın ana fikri şudur:

1. A - λI biçiminde karakteristik matris kurulur.
2. det(A - λI) polinomu elde edilir.
3. Bu polinomun kökleri bulunarak özdeğerler hesaplanır.

Depodaki örnek matris:

[[6, 1, -1],
 [0, 7, 0],
 [3, -1, 2]]

### 3.2 Bu çalışmada izlenen yöntem

Bu ödevde aynı fikir yeniden uygulanmıştır:

- Karakteristik denklem el ile kuran yardımcı fonksiyonlar yazılmıştır.
- Determinant polinom biçiminde çıkarılmıştır.
- `numpy.roots` ile özdeğerler elde edilmiştir.
- Sonuçlar daha sonra `numpy.linalg.eig` ile karşılaştırılmıştır.

Not: Referans depoda doğrudan özvektör hesabı yoktur. Bu nedenle bu teslimde özdeğerler referans yaklaşımla, özvektörler ise bulunan her özdeğer için (A-λI)v=0 sisteminin boşluk (null space) çözümü üzerinden elde edilmiştir. Böylece `eig` kullanılmadan da karşılaştırma yapılabilmiştir.

### 3.3 Elde edilen sonuçlar

Seçilen matris için bulunan özdeğerler:
λ = 7, 5, 3

`numpy.linalg.eig` ile de aynı özdeğerler elde edilmiştir (sıra farkı olabilir).

Karşılaştırma sonucunda:
- Özdeğerler aynıdır.
- Özvektörler doğrultu olarak aynıdır.
- İşaret veya ölçek farkı olabilir; bu durum normaldir çünkü özvektörler skaler katsayı ile çarpılabilir.

### 3.4 Sonuç yorumu

Bu karşılaştırma, `eig` fonksiyonunun verdiği sonucun altında yatan temel matematiksel mantığın karakteristik polinom yaklaşımı olduğunu göstermektedir. Ancak büyük matrislerde doğrudan determinant tabanlı yöntemler hem maliyetli hem de sayısal olarak daha hassas olabilir. Bu nedenle pratik uygulamalarda NumPy gibi optimize edilmiş kütüphane çözümleri tercih edilir.

---

## Sonuç

Bu ödev kapsamında:

- Matris manipülasyonu, özdeğerler ve özvektörlerin makine öğrenmesindeki yeri açıklanmıştır.
- `numpy.linalg.eig` fonksiyonunun dokümantasyon ve kaynak kod mantığı incelenmiştir.
- Hazır `eig` fonksiyonu kullanılmadan özdeğer hesabı yeniden uygulanmış ve sonuçlar NumPy ile karşılaştırılmıştır.

Genel sonuç olarak, özdeğer-özvektör yaklaşımı özellikle boyut indirgeme, spektral analiz ve doğrusal dönüşümlerin anlaşılması açısından makine öğrenmesinin temel araçlarından biridir.

---

## Kaynakça

1. NumPy Developers. numpy.linalg.eig - NumPy v2.1 Manual
   https://numpy.org/doc/2.1/reference/generated/numpy.linalg.eig.html
   Erişim tarihi: 29.03.2026

2. NumPy Developers. Linear algebra (numpy.linalg)
   https://numpy.org/doc/2.1/reference/routines.linalg.html
   Erişim tarihi: 29.03.2026

3. NumPy Developers. NumPy source code (`numpy/linalg/_linalg.py`, v2.1.0)
   https://github.com/numpy/numpy/blob/v2.1.0/numpy/linalg/_linalg.py
   Erişim tarihi: 29.03.2026

4. Lucas BN. Eigenvalues-and-Eigenvectors
   https://github.com/LucasBN/Eigenvalues-and-Eigenvectors
   Erişim tarihi: 29.03.2026

5. Jason Brownlee. Introduction to Matrices and Matrix Arithmetic for Machine Learning
   https://www.machinelearningmastery.com/introduction-matrices-machine-learning/
   Erişim tarihi: 29.03.2026

6. Jason Brownlee. Gentle Introduction to Eigenvalues and Eigenvectors for Machine Learning
   https://www.machinelearningmastery.com/introduction-to-eigendecomposition-eigenvalues-and-eigenvectors/
   Erişim tarihi: 29.03.2026
