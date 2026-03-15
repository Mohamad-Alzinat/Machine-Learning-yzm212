
# HMM ile Konuşma Tanıma – YZM212 Makine Öğrenmesi

## Problem Tanımı
Bu projede Hidden Markov Model (HMM) kullanarak basit bir konuşma tanıma sistemi tasarlanmıştır.
Amaç, gözlem dizilerine bakarak "EV" ve "OKUL" gibi kelimeleri sınıflandırabilen bir model oluşturmaktır.

## Yöntem
Sistemde fonemler gizli durumlar (hidden states) olarak modellenmiştir.
Mikrofondan gelen ses özellikleri ise gözlemler (observations) olarak kabul edilir.

Verilen gözlem dizisine göre en olası fonem dizisini bulmak için Viterbi algoritması kullanılmıştır.

Python tarafında model oluşturmak için `hmmlearn` kütüphanesi kullanılmıştır.

## Veri
Bu çalışmada örnek olması amacıyla sentetik gözlem verileri kullanılmıştır.

Kodlama:
- High frekans → 0
- Low frekans → 1

Örnek gözlem dizisi:

[High, Low] → [0,1]

## Sonuçlar
Viterbi algoritması uygulanarak gözlem dizisi [High, Low] için en olası fonem dizisi:

e → v

olarak bulunmuştur. Bu da sistemin **EV kelimesini tanıdığını** göstermektedir.

## Tartışma

### Gürültünün Etkisi
Ses verisindeki gürültü, konuşma sinyalinin spektral özelliklerini değiştirir.
Bu durum emisyon olasılıklarının doğru tahmin edilmesini zorlaştırır ve modelin hata yapma ihtimalini artırır.

### Günümüzde Neden Derin Öğrenme Kullanılıyor?
Modern konuşma tanıma sistemleri genellikle derin öğrenme modelleri kullanmaktadır çünkü:

- Daha karmaşık ses örüntülerini öğrenebilirler
- Büyük veri setleri ile daha iyi çalışırlar
- Özellik çıkarımını otomatik olarak öğrenebilirler
