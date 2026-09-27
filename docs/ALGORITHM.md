# Nihai Algoritma — tek kaynak (single source of truth)

> Bu dosya **her zaman güncel "asıl algoritmayı"** tanımlar. Araştırmada yeni bir aday mevcut algoritmayı
> geçtiğinde bu dosya güncellenir ve en alttaki değişiklik geçmişine satır eklenir. Deneylerin tamamı
> `EXPERIMENTS.md`'de, damıtılmış bulgular `docs/knowhow.md`'de, kurallar `docs/rules.md`'de.

| | |
|---|---|
| **Sürüm** | v1 — "best13" (EXP-069) + funded çekim modu (EXP-070/071) |
| **Son güncelleme** | 2026-09-27 |
| **Enstrüman / hesap** | XAUUSD, FundingPips 2-Step Standard $100k |
| **Kod (araştırma)** | `research/best_report.py` → `book_signals()`; bacaklar `research/combo8.py::build()` |
| **Kod (MT5 EA)** | `mql5/Experts/XauScalper/XauPortfolio.mq5` (9 bacaklı sürüm; v1'in 4 eki henüz EA'da yok) |
| **Durum** | Araştırma adayı. Demo forward test yapılmadı. Haftalık 2R hedefi **karşılanmıyor** (bkz. §6) |

---

## 1. Değişmez kısıtlar (her sürüm uymak zorunda)

| Kısıt | Değer |
|---|---|
| İşlem başı risk | ≤ %0.5 (funded modda başlangıç bakiyesinin %0.5'i = $500), her işlemde sunucu tarafı SL |
| Günlük zarar | İç limit %3 (hard), %2'de yeni giriş durur. Referans = günün başında max(bakiye, equity) — **her gün yeniden hesaplanır** |
| Toplam zarar | İç limit %8 (statik, $92k), %6.5'te risk yarıya iner |
| Sharpe | ≥ 1.5 (günlük getiriler, √252) |
| Hedge | Yok — aynı anda yalnız tek yön, en fazla 6 pozisyon, açık risk toplamı ≤ %3 |
| Haber | USD yüksek etkili haberde −30/+30 dk yeni giriş yok; açık pozisyon haberden 10 dk önce kapatılır |
| Hafta sonu | Cuma son bardan önce tüm pozisyonlar kapanır (overnight hafta içi serbest, swap maliyeti hesapta) |
| Yasak teknikler | Grid, martingale, averaging, HFT/tick scalping, latency/arb yok |
| Funded çekim | Döngü kârı ≥ %3 → çekim, bakiye $100k'ya döner |
| **Tutarlılık (%35)** | Döngünün en iyi günü döngü kârının ≤ %35'i olmalı, yoksa çekim bekler. **Kalıcı kısıt** (On-Demand payout) — algoritma bunu karşılayacak şekilde geliştirilir |
| Haftalık hedef | ≥ 2R/hafta (≥ %1) — henüz karşılanmıyor |

## 2. Veri ve icra modeli

- Karar zaman dilimleri: M15 / H4 / D1 (M1'den oluşturulur); icra M1 barlarında.
- Sunucu saati UTC+2/+3 (NY-close). Gün 01:00'de başlar; ilk giriş 01:05, son giriş 23:10, gün içi bacaklar 23:45'te kapanır.
- Maliyetler: gerçek spread (min 15 pt), 5 pt slipaj/taraf, $7/lot komisyon, swap long −$79.48 / short +$34.41 lot/gece (Çarşamba ×3).
- SL ve TP aynı barda ise SL sayılır (kötümser). Lot aşağı yuvarlanır (0.01).
- ATR_D = D1 ATR(14), dünün değeri (look-ahead yok). ATR_H1 / ATR_H4 aynı şekilde son kapanmış bar.

## 3. Bacaklar (13 sinyal kaynağı)

Hepsi aynı risk birimiyle (%0.5) girer; "season" yarım risk (%0.25). "Gün içi" bacaklarda iz süren stop =
bacağın kendi stop mesafesinin 1.5 katı (EXP-068).

| # | Bacak | Yön | Giriş | Stop | Çıkış | Tür |
|---|---|---|---|---|---|---|
| 1 | **trendH4** | Long | H4 kapanış > önceki 180 H4 barın en yükseği (Donchian-180) | 2 × ATR_H4(20) | İz stop 6 × ATR_H4, TP 50 ATR, max 240 H4 bar, hafta sonu flat | Swing |
| 2 | **tday** (18:00) | D1 trend yönü | D1 durumu (close>EMA50 & EMA20>EMA50 = up; tersi = down). 18:00'de gün açılışından ≥ 0.3 × ATR_D trend yönünde ve fiyat günün aralığının üst (alt) %25'inde | 1 × ATR_D | 23:45 / 330 dk | Gün içi |
| 3 | **tday900** (15:00) | D1 trend yönü | Aynı kural, karar 15:00 | 1 × ATR_D | 510 dk | Gün içi |
| 4 | **drift** | Long | Salı–Cuma 01:15 (Asya açılışı); yalnız **düşük volatilite günleri**: ATR_D(bps) ≤ geçmiş medyan | 4 × ATR_H1 | 475 dk sonra (~09:10) | Gün içi |
| 5 | **friday** | Long | Cuma 01:05 | 1.5 × ATR_D | 23:00 | Gün içi |
| 6 | **fri_close** | Long | Cuma 23:05 | 4 × ATR_M15 | 50 dk (~23:55) | Gün içi |
| 7 | **tom** | Long | Ayın ilk işlem günü 01:10 | 2 × ATR_D | 3 gün | Swing |
| 8 | **lw** (Larry Williams) | İki yön | 10:00 sonrası M15 kapanış > gün açılışı + 0.4 × dünkü aralık (short: altı), günün ilk tetiği | 1 × ATR_D | Gün sonu | Gün içi |
| 9 | **inside** | D1 trend yönü | Dün inside day ise dünün yükseği/düşüğü kırılımı (trend yönünde) | Dünün karşı ucu (≤ 1.5 ATR_D) | Gün sonu | Gün içi |
| 10 | **nr7** | D1 trend yönü | Dün NR7 ise aynı kırılım kuralı | Aynı | Gün sonu | Gün içi |
| 11 | **strong_close** | Long | Dünkü kapanış günün aralığının üst bölgesinde (CLV > 0.6) → ertesi gün 01:05 long | 1 × ATR_D | 23:30 | Gün içi |
| 12 | **season** | Long | Ocak, Temmuz, Ağustos her gün 01:07, **yarım risk** | 1 × ATR_D | 23:47 | Gün içi |
| — | (trail) | | Gün içi bacaklarda (2–6, 8–12) iz süren stop = 1.5 × kendi stop mesafesi | | | |

Portföy kuralları: aynı anda en fazla 6 pozisyon, hepsi aynı yönde (ters sinyal açık pozisyon varken yok sayılır),
açık risk ≤ %3, günde en fazla 8 giriş, FP korumaları (§1).

## 4. Çalışma modları

| Faz | Mod | Ayar |
|---|---|---|
| Challenge P1 (%8) / P2 (%5) | Bileşik risk (bakiyenin %0.5'i), günlük kâr tavanı yok | `payout_sim.guards` yerine `final_candidate.g(True, 6)` |
| Funded | Risk sabit $500, %3'te çekim, %35 tutarlılık; **günlük kâr tavanı %1.25** (gün kârı ≥ %1.25 → hepsini kapat, o gün işlem yok) | `payout_sim.guards(3.0, 35.0, 1.25)` |

Tavan seçimi (EXP-071): %1.25 toplam çekimi en az bozan ve medyan çekim aralığını en çok kısaltan ayar.

## 5. Performans (Python motoru, M1 icra, 2019-01 → 2026-09)

| Ölçüt | Challenge modu (bileşik) | Funded modu (sabit $500, tavan %1.25) |
|---|---|---|
| Sharpe | 1.53 (2021–26: 1.72; 2025–26: 2.57) | 1.54 |
| Net / yıl | +$138.7k (CAGR %11.9) | 23.7 R/yıl ≈ $11.9k/yıl (çekilen $78.2k / 7.7 yıl) |
| Tepe DD | %8.6 (tepe-dip); statik %8 ihlali yok | %7.6 |
| En kötü gün | −%2.1 | −%1.3 |
| Haftalık R | 0.69 ort. (bileşik), %20 hafta ≥ 2R | 0.41 ort. |
| Challenge (P1+P2 ≤ 250 gün) | %54 geçer, %0 ihlal, medyan 185 gün | — |
| Çekim | — | 16 çekim / 7.7 yıl, medyan 56 gün, ortalama 174 gün, $78.2k toplam |
| Yıllar | Her yıl pozitif | — |

MT5 tester uyumu (9 bacaklı EA, EXP-050): 3,111 vs 3,101 işlem, equity DD %7.7 vs %7.9.

## 6. Bilinen zayıflıklar / açık işler

1. **Haftalık 2R hedefi karşılanmıyor** (funded 0.41R, bileşik 0.69R). Asıl araştırma yönü bu.
   Hesap (EXP-073 sonrası): %8 DD ile haftada 2R için Sharpe ≈ 4 gerekir; mevcut 1.5. Risk artırmak (haberde tutmak,
   daha çok pozisyon) R'yi artırır ama DD'yi aynı oranda artırır → hedefe ancak **yeni, bağımsız getiri kaynaklarıyla** gidilir.
2. İşlemlerin ~%50'si haber kapatmasıyla bitiyor. Test edildi (EXP-072/073): yeniden giriş zararlı, haberde tutmak
   Sharpe'ı değiştirmiyor (yalnızca riski artırıyor) → mevcut haber kuralı kalıyor.
3. Tutarlılık kuralı çekimleri geciktiriyor; çekim sıklığının üst sınırı haftalık R ile belirleniyor (+%3 = 6R).
   Sabit %1.25 tavan, denenen alternatiflerden (dinamik tavan, swing bacaklarını çıkarmak) daha iyi (EXP-077).
4. Kazanç yoğun olarak long tarafta (altın boğa piyasası 2019–26); 2013–18 tipi ayı piyasasında long bacaklar
   yatay kalıyor (EXP-051), zarar değil.
5. v1'in 4 eki (tday900, strong_close, drift düşük-vol filtresi, season, trail 1.5×) EA'ya taşınmadı.
6. Demo forward test yapılmadı (yalnızca kullanıcının ayrı demo hesabında yapılacak).

## 7. Değişiklik geçmişi

| Tarih | Sürüm | Değişiklik | Kanıt |
|---|---|---|---|
| 2026-09-27 | v0 | 9 bacaklı portföy (trendH4, tday, drift, friday, tom, lw, inside, nr7, fri_close) | EXP-047, EXP-050 |
| 2026-09-27 | v1 | + tday900, strong_close, drift yalnız düşük-vol, season ×0.5, gün içi trail 1.5×; funded modu %3 çekim + %35 tutarlılık + %1.25 gün tavanı | EXP-052..071 |
