# Nihai Algoritma — tek kaynak (single source of truth)

> Bu dosya **her zaman güncel "asıl algoritmayı"** tanımlar. Araştırmada yeni bir aday mevcut algoritmayı
> geçtiğinde bu dosya güncellenir ve en alttaki değişiklik geçmişine satır eklenir. Deneylerin tamamı
> `EXPERIMENTS.md`'de, damıtılmış bulgular `docs/knowhow.md`'de, kurallar `docs/rules.md`'de.

| | |
|---|---|
| **Sürüm** | **v2.2** — v2 (XAUUSD best13 + NAS100/DJ30 endeks bacakları) + tampon-oranlı risk kuralı (QM-008/009, kilitli kutuda doğrulandı) |
| **Son güncelleme** | 2026-09-27 |
| **Enstrüman / hesap** | XAUUSD + NAS100 + DJ30 (US30), FundingPips 2-Step Standard $100k |
| **Kod (araştırma)** | `research/best_v2.py` (rapor) · XAU bacakları `research/best_report.py::book_signals()` · endeks bacakları `research/index_legs.py::index_legs()` · motor `research/engine_multi.py` |
| **Kod (MT5 EA)** | `mql5/Experts/XauScalper/XauIdxPortfolio.mq5` v2.20 = v2.2 (tester 2019–26: 6,630 vs Python 6,609 işlem, bacak bazında 0–7 fark; tampon kuralı tester günlüğüyle doğrulandı, EXP-100) |
| **Durum** | Araştırma adayı. Haftalık 2R dışındaki **tüm kapılar geçiyor**. Demo forward test yapılmadı (bkz. §6) |

---

## 1. Değişmez kısıtlar (her sürüm uymak zorunda)

| Kısıt | Değer |
|---|---|
| İşlem başı risk | ≤ %0.5 (funded modda başlangıç bakiyesinin %0.5'i = $500), her işlemde sunucu tarafı SL |
| Günlük zarar | İç limit %3 (hard), %2'de yeni giriş durur. Referans = günün başında max(bakiye, equity) — **her gün yeniden hesaplanır** |
| Toplam zarar | İç limit %8 (statik, $92k). **Tampon kuralı:** statik DD %2'yi aşınca risk doğrusal azalır: risk × (8 − DD)/(8 − 2), en az ×0.1 (eski kural: %6.5'te yarıya) |
| Sharpe | ≥ 1.5 (günlük getiriler, √252) |
| Hedge | Yok — her sembolde aynı anda yalnız tek yön; toplam en fazla 6 pozisyon, açık risk toplamı ≤ %3 |
| Haber (v2) | Sembolün para birimlerindeki yüksek etkili haberde **−10…+10 dk yeni pozisyon yok**; açık pozisyonlar haberden 10 dk önce kapatılır (FP funded ±5 dk kuralından sıkı). Ayrıntı: `docs/rules.md` |
| Hafta sonu | Cuma son bardan önce tüm pozisyonlar kapanır (overnight hafta içi serbest, swap maliyeti hesapta) |
| Yasak teknikler | Grid, martingale, averaging, HFT/tick scalping, latency/arb yok |
| Funded çekim | Döngü kârı ≥ %3 → çekim, bakiye $100k'ya döner |
| **Tutarlılık (%35)** | Döngünün en iyi günü döngü kârının ≤ %35'i olmalı, yoksa çekim bekler. **Kalıcı kısıt** (On-Demand payout) — algoritma bunu karşılayacak şekilde geliştirilir |
| Haftalık hedef | ≥ 2R/hafta (≥ %1) — henüz karşılanmıyor |

## 2. Veri ve icra modeli

- Karar zaman dilimleri: M15 / H4 / D1 (M1'den oluşturulur); icra M1 barlarında.
- Sunucu saati UTC+2/+3 (NY-close). Gün 01:00'de başlar; ilk giriş 01:05, son giriş 23:10, gün içi bacaklar 23:45'te kapanır.
- Maliyetler XAU: gerçek spread (min 15 pt), 5 pt slipaj/taraf, $7/lot komisyon, swap long −$79.48 / short +$34.41 lot/gece (Çarşamba ×3).
- Maliyetler endeks: gerçek spread (NAS100 min 0.5, DJ30 min 1.0 endeks puanı), slipaj 0.5 / 1.0 puan/taraf, komisyon yok (FP), swap NAS100 −$6.08 / DJ30 −$10.61 lot/gece; lot adımı 0.1.
- Tüm semboller tek M1 zaman çizgisinde, tek hesapta (`engine_multi`); her sinyal kendi sembolünün ilk açık barında girer.
- SL ve TP aynı barda ise SL sayılır (kötümser). Lot aşağı yuvarlanır (0.01).
- ATR_D = D1 ATR(14), dünün değeri (look-ahead yok). ATR_H1 / ATR_H4 aynı şekilde son kapanmış bar.

## 3. Bacaklar

### 3a. XAUUSD (12 bacak — v1'den aynen)

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

### 3b. NAS100 ve DJ30 (her ikisinde aynı 6 kural — literatür öncüllü, endeks başına seçim yapılmadı)

Giriş 01:05, çıkış 23:30 (sunucu); gün içi olanlarda iz süren stop = 1.5 × kendi stop mesafesi. Hepsi long.

| # | Bacak | Giriş koşulu | Stop | Çıkış |
|---|---|---|---|---|
| 13 | **mon** | Pazartesi | 1.5 × ATR_D | 23:30 |
| 14 | **dip_low20** | Dünkü kapanış 20 günlük aralığın en alt %10'unda | 1 × ATR_D | 23:30 |
| 15 | **dip_clv** | Dünkü kapanış günün aralığının alt bölgesinde (CLV < −0.6) | 1 × ATR_D | 23:30 |
| 16 | **hi20** | Dünkü kapanış 20 günlük aralığın en üst %10'unda (momentum) | 1 × ATR_D | 23:30 |
| 17 | **prefomc** | FOMC kararından 24 saat önce | 1.5 × ATR_D | Haber kuralıyla FOMC'den 10 dk önce |
| 18 | **tom** | Ayın son işlem günü | 2 × ATR_D | 4 gün (Cuma kapanır), trail yok |

GER40 test edildi, aynı bacaklar ~0 katkı verip DD'yi artırdığı için **çıkarıldı** (EXP-086).

Portföy kuralları: toplam en fazla 6 pozisyon, her sembolde tek yön (o sembolde ters sinyal yok sayılır),
açık risk ≤ %3, günde en fazla 8 giriş, FP korumaları (§1).

## 4. Çalışma modları

| Faz | Mod | Ayar |
|---|---|---|
| Challenge P1 (%8) / P2 (%5) | Bileşik risk (bakiyenin %0.5'i), günlük kâr tavanı yok | `portfolio_v2.guards(6, 3.0)` |
| Funded | Risk sabit $500, %3'te çekim, %35 tutarlılık; **günlük kâr tavanı %1.25** (gün kârı ≥ %1.25 → hepsini kapat, o gün işlem yok) | `portfolio_v2.guards(6, 3.0, risk_on_initial=True, payout_pct=3, consistency_pct=35, day_profit_cap_pct=1.25)` |

Tavan seçimi (EXP-071): %1.25 toplam çekimi en az bozan ve medyan çekim aralığını en çok kısaltan ayar.

## 5. Performans (Python motoru, M1 icra, 2019-01 → 2026-09, EXP-087)

| Ölçüt | Challenge modu (bileşik) | Funded modu (sabit $500, tavan %1.25) |
|---|---|---|
| Sharpe | **1.91** (v1: 1.53) · Sortino 3.35 | 1.85 |
| Net | +$234k / 7.7 yıl (CAGR %16.9) | 25 çekim, toplam **$116k** brüt (kâr payı öncesi) |
| Tepe DD | **%7.57** (v1: %8.6) · MC95 %7.5 | %6.7, ihlal yok |
| En kötü gün | −%2.4 | −%2.0 |
| Haftalık R | 1.16 ort. (bileşik), %30 hafta ≥ 2R | 0.60 ort. |
| Challenge (P1+P2 ≤ 250 gün) | **%66 geçer, %0 ihlal**, medyan 154 gün | — |
| Çekim sıklığı | — | yılda 3.2 · medyan **71 gün** · %88'i tutarlılık için bekledi (ort. 34 gün) |
| Yıllar | Her yıl pozitif (2021 en zayıf: +$2.6k) | 2021'de çekim yok |
| Katkı | XAU 177R · NAS100 53R · DJ30 32R; endekslerin XAU ile günlük korelasyonu ≈ 0 | |

Kapılar: Sharpe ✓ · günlük DD ✓ · toplam DD ✓ · PF ✓ · işlem sayısı ✓ · MC95 ✓ · **haftalık 2R ✗**.
MT5 tester uyumu yalnızca 9 bacaklı XAU EA için yapıldı (EXP-050).

## 6. Bilinen zayıflıklar / açık işler

1. **Haftalık 2R hedefi karşılanmıyor** (funded 0.60R, bileşik 1.16R). %8 DD ile haftada 2R için Sharpe ≈ 4 gerekir;
   v2 1.9. İlerleme yolu yeni, **bağımsız** getiri kaynakları (v1→v2 böyle geldi).
2. Funded modda her çekimden sonra bakiye $100k'ya döndüğü için tampon sıfırlanır: çekimden hemen sonra %8 düşüş
   hesabı durdurur. v2'de 7.7 yılda olmadı, ama K10 + GER40 varyantında oldu (EXP-086) → izlenmeli.
3. Endeks bacakları 2019–26 boğa piyasasında test edildi; 2013–18 D1 dönemleri de pozitif (EXP-083), ama uzun bir
   ayı piyasası (2022 gibi) sınırlı sayıda.
4. Seçim yanlılığı: GER40'ın çıkarılması ve endeks bacak listesi 2013–26 verisine bakılarak yapıldı (literatür öncülü
   olsa da). Endeks başına seçilmiş set (SR 2.03) kullanılmadı.
5. EA v2.20: bacak bazında uyumlu. MT5 kârı Python'dan yüksek (+$279k vs +$234k) çünkü tester komisyonu 0 ve slipaj yok; Python daha kötümser.
6. FX (EURUSD/USDJPY) ve gümüş: bu broker maliyetleriyle katkı yok (EXP-078..081). FundingPips spread/komisyonları
   farklıysa yeniden değerlendirilecek.
7. Demo forward test 2026-09-27'de hazırlandı (EA v2.21, MetaQuotes demo, challenge modu $100k, `deploy/forward/KURULUM.md`);
   sonuçlar `reports/forward/` altında haftalık izlenecek (`tools/forward_report.py --shadow`).
   **2026-09-28 başladı:** MetaQuotes-Demo, EA v2.22, yalnızca altın bacakları (bu sunucuda USTEC/US30 işleme kapalı).
8. **Reddedilen aday v2.1 (örüntü bacağı, EXP-089..098):** walk-forward örüntü seçimi portföy Sharpe'ını 1.91 → 1.96'ya
   çıkardı, ama placebo testinde (getiriler karıştırılınca) aynı veya daha iyi sonuç %11–31 olasılıkla şans eseri
   çıkıyor → istatistiksel olarak kanıtlanmadı; seçilen örüntüler büyük ölçüde mevcut bacakları (zayıf/güçlü kapanış,
   Cuma) yeniden buluyor, katkısı çoğunlukla ek long pozisyon. **Algoritmaya alınmadı.**

9. **Endeks genişlemesi reddedildi (EXP-102):** FundingPips'in diğer endekslerine (SP500, UK100, JP225, GER40) aynı
   literatür bacakları eklendi. SP500 tek başına t 2.9 ama NAS/DJ ile aynı günlerde aynı yönde bahis → SR 1.91 → 1.86,
   DD %7.6 → %8.7 (sınır aşımı). UK100 (t 0.4), JP225 (t −0.8), GER40 (t 0.3) bacakları kenarsız; hepsi birlikte SR 1.50.
   Bu bacaklar ABD hisse piyasasına özgü (hafta sonu etkisi, dip alımı). FX majörleri de kenarsız (FXR-001..003).

## 7. Değişiklik geçmişi

| Tarih | Sürüm | Değişiklik | Kanıt |
|---|---|---|---|
| 2026-09-27 | v0 | 9 bacaklı portföy (trendH4, tday, drift, friday, tom, lw, inside, nr7, fri_close) | EXP-047, EXP-050 |
| 2026-09-27 | v1 | + tday900, strong_close, drift yalnız düşük-vol, season ×0.5, gün içi trail 1.5×; funded modu %3 çekim + %35 tutarlılık + %1.25 gün tavanı | EXP-052..071 |
| 2026-09-27 | v2 | + NAS100/DJ30 bacakları (mon, dip_low20, dip_clv, hi20, prefomc, tom), çok sembollü motor, haber kuralı v2 (±10 dk giriş yasağı, 10 dk önce kapat); GER40 test edilip çıkarıldı | EXP-082..087 |
| 2026-09-27 | v2 (değişmedi) | v2.1 adayı (örüntü bacağı) placebo testinde anlamsız çıktı → reddedildi; v2 EA yazıldı | EXP-088..098 |
| 2026-09-27 | v2.2 | Tampon-oranlı risk kuralı (Grossman–Zhou/CPPI tipi) — MC: challenge başarısızlık %2.2 → %0.03, funded 2 yılda hesap kaybı %5.6 → %0.03; kilitli kutu 2025–26 geçti; EA v2.20'ye eklendi | QM-008/009, QM-LOCKBOX, EXP-100 |
| 2026-09-27 | v2.2 (değişmedi) | FX majör taraması (FXR-001..003) ve endeks genişlemesi SP500/UK100/JP225/GER40 (EXP-102) → kenar yok veya SR düşürüyor, reddedildi | FXR-001..003, EXP-102 |
| 2026-09-27 | v2.2 (EA v2.21) | Canlı çalışma güvenlikleri: sunucu saat dilimi kontrolü, durum dosyası (yeniden başlatmaya dayanıklı), anlık işlem kaydı, haber/FOMC dosya kapsam kontrolü; tester sonucu v2.20 ile birebir aynı; demo forward test paketi | EXP-103 |
| 2026-09-28 | v2.2 (EA v2.22) | Canlı: yalnız demo koruması, kapalı sembol kontrolü, broker seans sonuna uyum (son 15 dk giriş yok, gün içi pozisyonlar ve Cuma kitabı seans bitiminden 5 dk önce kapanır); demo forward test başladı (yalnız altın) | EXP-104 |
