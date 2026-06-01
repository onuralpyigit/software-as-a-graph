# Yapısal Metrik Formülasyon Referansı

## Kapsam ve Bağlam

Bu belge, aşağıdaki makale kapsamında geliştirilen matematiksel formülasyonun tamamını içermektedir:

> **"Yayınla-Abone Ol Tabanlı Dağıtık Sistemlerde Yapısal Etkileşim Örüntülerinin Çizge Tabanlı Statik Analiz ile İncelenmesi"**
>
> Mustafa Can Çalışkan, İbrahim Onuralp Yiğit, Feza Buzluca
>
> **UYMS'26** — 17. Ulusal Yazılım Mühendisliği Sempozyumu, 14–16 Mayıs 2026, Muğla, Türkiye kapsamında sunulmuştur.

### Makalenin Ele Aldığı Problem

Yayınla-abone ol mimarileri çalışma zamanında gevşek bağlanma sağlarken, uygulamalar arası etkileşimlerin **örtük** kalmasına yol açmaktadır — yalnızca kaynak koda bakarak hangi uygulamanın hangisiyle iletişim kurduğunu görmek kolay değildir. Zamanla bu durum, fark edilmesi güç yapısal yoğunlaşmalara neden olabilir: aşırı merkezî uygulamalar, yoğun kullanılan iletişim kanalları veya belirli çalışma düğümlerinde kümelenen etkileşimler bunlara örnek gösterilebilir.

### Ne Yapıldı

Makale, bu örtük örüntüleri ortaya çıkarmak için **statik analiz** tabanlı (çalışma zamanı verisine ihtiyaç duymayan) bir yaklaşım önermektedir. İş hattı üç aşamadan oluşmaktadır:

1. **İlişkilerin çıkarılması:** Kaynak koddan statik analiz yoluyla (ana yürütme noktasından erişilebilen yayınla/abone ol çağrılarını izleyen CodeQL sorguları ile) etkileşimler tespit edilir.
2. **Çizge tabanlı temsil oluşturulması:** Sistem; uygulamalar, konular, çalışma düğümleri ve ortak kütüphaneler şeklinde düğümlerle, aralarındaki yayınlama/abonelik/dağıtım/bağımlılık kenarlarıyla modellenir.
3. **Yapısal metriklerin hesaplanması ve göreli değerlendirilmesi:** Çizge üzerinden hesaplanan metrikler, mutlak eşikler yerine sistemin kendi dağılımındaki çeyreklik değerlerine göre (göreli olarak) yorumlanır ve **kural tabanlı örüntü tespiti** ile birleştirilir.

### Belgenin Organizasyonu

Bu belgenin geri kalanı, makalede yer alan tüm formülleri düzeylere göre (uygulama, konu, çalışma düğümü, kütüphane) gruplayarak sunmakta; ardından göreli yorumlama şeması, yapısal aykırılık örüntüleri ve birleşik aykırılık skorlama mekanizmasını açıklamaktadır. Her formül, anlaşılır bir düz metin açıklamasıyla desteklenmiştir.

---

## 1. Notasyonlar

Formüllere geçmeden önce, kullanılan semboller aşağıda özetlenmiştir. Sistemi, dört tür düğüm ve aralarındaki kenarlardan oluşan yönlü bir çizge olarak düşünebilirsiniz.

| Sembol | Anlamı |
|--------|--------|
| $a \in \mathcal{A}$ | Sistem içerisindeki bir **uygulama** (bağımsız bir yazılım birimi). |
| $t \in \mathcal{T}$ | Bir **konu** (uygulamaların yayın yaptığı veya abone olduğu adlandırılmış iletişim kanalı). |
| $n \in \mathcal{N}$ | Bir **çalışma düğümü** (uygulamaların üzerinde çalıştığı fiziksel veya sanal makine). |
| $l \in \mathcal{L}$ | Bir **ortak kütüphane** (birden fazla uygulama tarafından kullanılan ortak yazılım bileşeni). |

### İlişki Kümeleri

Bu kümeler çizgedeki kenarları tanımlar. Her biri doğrudan statik analizden veya dağıtım yapılandırmasından elde edilir.

| Sembol | Tür | Anlamı |
|--------|-----|--------|
| $PUB(a) \subseteq \mathcal{T}$ | Uygulama → Konular | Uygulama $a$ tarafından **yayın yapılan** konular kümesi. |
| $SUB(a) \subseteq \mathcal{T}$ | Uygulama → Konular | Uygulama $a$ tarafından **abone olunan** konular kümesi. |
| $PUB(t) \subseteq \mathcal{A}$ | Konu → Uygulamalar | Konu $t$'ye **yayın yapan** uygulamalar kümesi. |
| $SUB(t) \subseteq \mathcal{A}$ | Konu → Uygulamalar | Konu $t$'ye **abone olan** uygulamalar kümesi. |
| $RUNS(n) \subseteq \mathcal{A}$ | Düğüm → Uygulamalar | Çalışma düğümü $n$ üzerinde **konumlanan** uygulamalar kümesi. |
| $USES(a) \subseteq \mathcal{L}$ | Uygulama → Kütüphaneler | Uygulama $a$ tarafından **kullanılan** ortak kütüphaneler kümesi. |
| $USES(l) \subseteq \mathcal{A}$ | Kütüphane → Uygulamalar | Kütüphane $l$'yi **kullanan** uygulamalar kümesi. |

> **Okuma notu:** $PUB$ ve $SUB$ aşırı yüklenmiş (overloaded) kullanılmıştır — argüman bir uygulama olduğunda konuları, argüman bir konu olduğunda uygulamaları döndürür. Anlam, bağlamdan her zaman açıktır.

---

## 2. Uygulama Düzeyinde Yapısal Metrikler

Bu metrikler, bireysel uygulamaları mimari yapı içerisinde karakterize eder — ne kadar geniş bir alanda etkileşime girdiklerini, hangi rolleri üstlendiklerini ve ne kadar bağımlılık taşıdıklarını ölçer.

### 2.1 Etki Alanı (R — Reach)

$$R(a) = \big|\{ a' \in \mathcal{A} \setminus \{a\} \mid (\exists\, t \in PUB(a): a' \in SUB(t)) \lor (\exists\, t \in SUB(a): a' \in PUB(t)) \}\big|$$

**Ne ölçer:** Uygulama $a$'nın, konular aracılığıyla her iki yönde iletişim kurduğu **benzersiz diğer uygulama sayısı**.

**Nasıl okunur:** $a$'nın yayın yaptığı her konuya bakın — o konulara abone olan herkes bir iletişim ortağıdır. Sonra $a$'nın abone olduğu her konuya bakın — o konulara yayın yapan herkes de bir iletişim ortağıdır. Tüm benzersiz ortakları sayın ($a$'nın kendisi hariç). Yüksek bir Reach değeri, uygulamanın bir iletişim merkezi olduğunu gösterir — sistem genelinde birçok uygulamaya dokunmaktadır.

**Örnek:** $a$ uygulaması $t_1$ konusuna yayın yapıyorsa ($b$ ve $c$ uygulamaları bu konuya abone) ve $t_2$ konusuna abone ise ($d$ uygulaması bu konuya yayın yapıyor), o zaman $R(a) = 3$ (ortaklar: $b$, $c$, $d$).

---

### 2.2 Yoğunlaştırma (AMP — Amplification)

$$AMP(a) = \frac{R(a)}{|PUB(a)| + 1}$$

**Ne ölçer:** Bir uygulamanın **yayın kanalı başına ne kadar erişim** sağladığını ölçer. Sınırlı sayıda konu üzerinden geniş bir etki alanı oluşturup oluşturmadığını yakalar.

**Nasıl okunur:** Toplam etki alanını, uygulamanın yayın yaptığı konu sayısına bölün (sıfıra bölmeyi önlemek ve yayın yapmayan uygulamaları ele almak için +1 eklenir). Yüksek bir AMP, uygulamanın az sayıda konu üzerinden birçok ortağa ulaştığı anlamına gelir — mesajları geniş bir yelpazede yayılmaktadır. Yoğunlaştırma çıkış etkisiyle ilgili olduğundan, yalnızca yayıncı rolü dikkate alınmıştır.

**Örnek:** $R(a) = 12$ ve $a$ 2 konuya yayın yapıyorsa, $AMP(a) = 12 / 3 = 4.0$. Her yayın kanalı ortalama 4 uygulamaya ulaşmaktadır.

---

### 2.3 Rol Asimetrisi (RA — Role Asymmetry)

$$RA(a) = \frac{|PUB(a)| - |SUB(a)|}{|PUB(a)| + |SUB(a)| + 1}$$

**Ne ölçer:** **Üretici (yayıncı) ve tüketici (abone) rolleri arasındaki denge**. Sonuç $(-1, +1)$ aralığında bir değerdir.

**Nasıl okunur:**
- $RA(a) > 0$ → uygulama abone olduğundan daha fazla konuya yayın yapar (üretici ağırlıklı).
- $RA(a) < 0$ → uygulama yayın yaptığından daha fazla konuya abone olur (tüketici ağırlıklı).
- $RA(a) \approx 0$ → yaklaşık dengeli.

Paydadaki $+1$, yayın/abonelik aktivitesi olmayan uygulamalarda sıfıra bölmeyi engeller.

**Örnek:** 8 konuya yayın yapan ve 2 konuya abone olan bir uygulamanın değeri: $RA = (8-2)/(8+2+1) = 6/11 \approx 0.55$ — belirgin biçimde üretici yanlı.

---

### 2.4 Bağlam Çeşitliliği (TC — Topic Context Diversity)

$$TC(a) = \big|\{ \text{category}(t) \mid t \in PUB(a) \cup SUB(a) \}\big|$$

**Ne ölçer:** Uygulamanın tüm konuları üzerinden etkileşimde bulunduğu **farklı işlevsel kategori sayısı**.

**Nasıl okunur:** Her konu bir kategoriye aittir (konu adlarındaki hiyerarşik öneklerden türetilir — ör. `navigasyon/konum` ve `navigasyon/yön` ikisi de `navigasyon` kategorisindedir). Uygulamanın dokunduğu tüm konularda kaç farklı kategori göründüğünü sayın. Yüksek bir TC, uygulamanın birçok işlevsel alana yayıldığını gösterir — uzmanlaşmış değil, çapraz-kesişen (cross-cutting) bir yapıdadır.

**Örnek:** $a$ uygulaması {navigasyon, silah, sensör, görüntüleme} kategorilerindeki konuları kullanıyorsa, $TC(a) = 4$.

---

### 2.5 Kütüphane Maruziyeti (LE — Library Exposure)

$$LE(a) = |USES(a)|$$

**Ne ölçer:** Uygulamanın bağımlı olduğu **ortak kütüphane sayısı**.

**Nasıl okunur:** Uygulama tarafından kullanılan ortak kütüphaneleri saymanız yeterlidir. Yüksek bir değer, uygulamanın çok sayıda dışsal bağımlılığa sahip olduğunu gösterir; bu durum ortak kod aracılığıyla diğer bileşenlerle bağlantısını artırabilir.

---

## 3. Konu Düzeyinde Yapısal Metrikler

Bu metrikler, bireysel konuları karakterize eder — kaç uygulama tarafından kullanıldıklarını, kullanımlarının dengeli olup olmadığını ve katılımcılarının fiziksel olarak ne kadar yayıldığını ölçer.

### 3.1 Kapsayıcılık (C — Coverage)

$$C(t) = |SUB(t)| + |PUB(t)|$$

**Ne ölçer:** Konu $t$ ile etkileşimde bulunan **toplam uygulama sayısı** (hem yayıncılar hem de aboneler birlikte).

**Nasıl okunur:** Yüksek bir Kapsayıcılık, konunun merkezî bir iletişim kanalı olduğu anlamına gelir — birçok uygulama ona bağımlıdır. Böyle bir konuda sorun yaşanırsa (şema değişiklikleri, teslimat hataları), etki alanı geniş olur.

---

### 3.2 Dengesizlik (I — Imbalance)

$$I(t) = \frac{\big||SUB(t)| - |PUB(t)|\big|}{|SUB(t)| + |PUB(t)| + 1}$$

**Ne ölçer:** Bir konunun yayıncı ve abone dağılımındaki **asimetri**. Sonuç $[0, 1)$ aralığındadır.

**Nasıl okunur:**
- $I(t) \approx 0$ → konu, yaklaşık eşit sayıda yayıncı ve aboneye sahiptir (dengeli, "omurga" niteliğinde kanal).
- $I(t) \to 1$ → konu belirgin biçimde tek taraflıdır (ör. çok sayıda abone ama tek bir yayıncı, veya tersi).

Not: Paydaki mutlak değer, bu metriği yön bağımsız kılar — yalnızca dengesizliğin büyüklüğünü ölçer, hangi tarafın ağır bastığını değil.

---

### 3.3 Fiziksel Yayılım (PS — Physical Spread)

$$PS(t) = \big|\{ n \in \mathcal{N} \mid \exists\, a \in SUB(t) \cup PUB(t),\ a \in RUNS(n) \}\big|$$

**Ne ölçer:** Konu $t$ etrafındaki iletişime dahil olan **farklı çalışma düğümü sayısı**.

**Nasıl okunur:** $t$'ye yayın yapan veya abone olan tüm uygulamaları toplayın, ardından bu uygulamaların hangi düğümlerde çalıştığına bakın. Benzersiz düğümleri sayın. Yüksek bir PS, konunun iletişiminin birçok fiziksel/sanal makineye yayıldığı anlamına gelir — düğüm sınırlarını aşar ve ağ ek yükü anlamına gelir.

---

### 3.4 Düşük Bağlantılı Uygulama Oranı (LCR — Low Connectivity Ratio)

$$LCR(t) = \frac{|\{a \in PUB(t) \cup SUB(t) : |PUB(a) \cup SUB(a)| \leq k\}|}{|PUB(t) \cup SUB(t)| + 1}$$

**Ne ölçer:** Bir konunun katılımcıları arasındaki **zayıf bağlantılı uygulama oranı**.

**Nasıl okunur:** Konu $t$ ile etkileşimde bulunan tüm uygulamalar arasında, sistem genelinde (sadece $t$ değil, tüm konular dahil) toplam konu bağlantısı en fazla $k$ olan kaçının bulunduğunu sayın. Toplam katılımcıya bölün (+1). Yüksek bir LCR, bu konunun iletişim ağına yeterince entegre olmayan uygulamaları bir araya topladığı anlamına gelir — bu uygulamalar, bu konuyu az sayıdaki bağlantılarından biri olarak kullanmaktadır. $k$ eşik parametresidir (vaka çalışmasında 2 olarak belirlenmiştir).

**Örnek:** Konu $t$'nin 10 katılımcı uygulaması varsa ve bunların 7'si sistem genelinde $\leq 2$ konuyla etkileşimde bulunuyorsa, $LCR(t) = 7/11 \approx 0.64$.

---

## 4. Çalışma Düğümü Düzeyinde Yapısal Metrikler

Bu metrikler, çalışma düğümlerini karakterize eder — ne kadar yüklü olduklarını ve üzerlerindeki uygulamaların ne yoğunlukta etkileştiğini ölçer.

### 4.1 Düğüm Yoğunluğu (ND — Node Density)

$$ND(n) = |RUNS(n)|$$

**Ne ölçer:** Düğüm $n$ üzerinde **konumlanan uygulama sayısı**.

**Nasıl okunur:** Bu düğümde kaç uygulama çalıştığını saymanız yeterlidir. Yüksek bir değer, bir dağıtım yoğunlaşma noktasına işaret edebilir.

---

### 4.2 Düğüm İçi Etkileşim Yoğunluğu (NID — Node Interaction Density)

Önce iki uygulamanın ne zaman **etkileşimde** olduğunu tanımlayalım: $a_i$ ve $a_j$ uygulamaları, ancak ve ancak en az bir konuda birinin yayın yapıp diğerinin abone olması durumunda etkileşimde kabul edilir:

$$a_i \leftrightarrow a_j \iff \exists\, t \in \mathcal{T} : (a_i \in PUB(t) \land a_j \in SUB(t)) \lor (a_j \in PUB(t) \land a_i \in SUB(t))$$

Ardından, Düğüm İçi Etkileşim Yoğunluğu, **aynı düğümdeki tüm etkileşen çiftleri** sayar:

$$NID(n) = \big|\{ (a_i, a_j) \subseteq RUNS(n) \mid a_i \leftrightarrow a_j \}\big|$$

**Ne ölçer:** Düğüm $n$ üzerinde konumlanan ve **en az bir ortak konu aracılığıyla iletişim kuran uygulama çifti sayısı**.

**Nasıl okunur:** Aynı düğümde çalışan tüm uygulamalar arasında, kaç çift gerçekten birbirleriyle konuşuyor? Yüksek bir NID, düğümün yalnızca çok sayıda uygulama barındırmakla kalmayıp, bu uygulamaların aynı zamanda yoğun biçimde birbirine bağlı olduğu anlamına gelir. Bu, tek başına yoğunluktan daha anlamlıdır — çünkü birbirleriyle iletişim kurmayan çok sayıda izole uygulamayı barındıran bir düğüm, her uygulamanın diğerleriyle konuştuğu bir düğümden yapısal olarak farklıdır.

---

## 5. Kütüphane Düzeyinde Yapısal Metrikler

Bu metrikler, ortak kütüphaneleri kullanım genişliği ve fiziksel yoğunlaşma açısından karakterize eder.

### 5.1 Kütüphane Yaygınlığı (LC — Library Coverage)

$$LC(l) = |USES(l)|$$

**Ne ölçer:** Kütüphane $l$'yi **kullanan uygulama sayısı**.

**Nasıl okunur:** Yüksek bir değer, kütüphanenin yaygın biçimde bağımlılık oluşturduğu anlamına gelir. Kütüphanedeki değişiklikler birçok uygulamaya yayılabilir.

---

### 5.2 Kütüphane Yoğunlaşması (LCon — Library Concentration)

$$LCon(l) = \max_{n \in \mathcal{N}} \big|RUNS(n) \cap USES(l)\big|$$

**Ne ölçer:** Herhangi bir tek düğümdeki **kütüphane $l$ kullanıcılarının maksimum sayısı**.

**Nasıl okunur:** Her çalışma düğümü için, o düğümdeki uygulamalardan kaçının $l$ kütüphanesini kullandığını sayın. Tüm düğümler arasındaki maksimumu alın. Yüksek bir LCon, kütüphanenin kullanımının tek bir düğümde yoğunlaştığı anlamına gelir — bu kütüphanede bir hata varsa, aynı makinedeki birçok uygulamayı eşzamanlı olarak etkileyebilir.

---

## 6. Göreli Yorumlama Şeması

Makale bilinçli olarak **mutlak eşiklerden** (ör. "Reach > 10 kötüdür") kaçınmaktadır. Bunun yerine her metrik, çeyreklik değerler kullanılarak **sistemin kendi dağılımına göre göreli olarak** yorumlanmaktadır.

Herhangi bir $M$ metriği için, ilgili bileşen türündeki tüm varlıklar üzerinden $Q_1(M)$ (25. yüzdelik) ve $Q_3(M)$ (75. yüzdelik) hesaplanır. Ardından:

$$M(x)\!\uparrow \iff M(x) \geq Q_3(M)$$

$$M(x)\!\downarrow \iff M(x) \leq Q_1(M)$$

**Bunun anlamı:**
- $M(x)\!\uparrow$ — varlığın metrik değeri **görece yüksek** (sistemin 75. yüzdeliğinde veya üzerinde).
- $M(x)\!\downarrow$ — varlığın metrik değeri **görece düşük** (sistemin 25. yüzdeliğinde veya altında).

**Sınır durumu:** $Q_1 = Q_3$ olduğunda (çok düşük varyans — çoğu varlık aynı değere sahip), yorumlama yalnızca mutlak uç değerlere (minimum ve maksimum) sahip bileşenler üzerinden gerçekleştirilir.

> **Neden göreli?** Neyin "yüksek" bir etki alanı oluşturduğu tamamen sisteme bağlıdır. 10 uygulamalı bir sistemde Reach=5 anlamlı olabilir. 500 uygulamalı bir sistemde ise dikkat çekmeyebilir. Çeyreklik tabanlı yorumlama, sistemin ölçeğine otomatik olarak uyum sağlar.

---

## 7. Yapısal Aykırılık Örüntüleri

Tekil metrikler yalnızca tek bir boyutu yakalar. Aşağıdaki örüntüler, **yapısal olarak dikkat çekici** varlıkları belirlemek üzere birden fazla metriği birleştirir. Bir örüntü, bileşen metrikleri eşzamanlı olarak ilgili eşiklere ulaştığında tetiklenir.

### 7.1 Uygulama Düzeyi Örüntüler

**Geniş Etki Alanı (WR — Wide Reach):** Uygulama hem yüksek etki alanına hem de yüksek yoğunlaştırmaya sahiptir — görece az sayıda kanal üzerinden birçok uygulamayı etkiler.

$$R(a)\!\uparrow \;\land\; AMP(a)\!\uparrow \;\Rightarrow\; WR(a)$$

**Rol Dengesizliği (RS — Role Skew):** Uygulama, üretici veya tüketici rollerinden birinde belirgin biçimde yoğunlaşmıştır.

$$RA(a)\!\uparrow \;\lor\; RA(a)\!\downarrow \;\Rightarrow\; RS(a)$$

> Not: Bu örüntü, RA **her iki** uçta olduğunda tetiklenir — çok pozitif (üretici baskın) veya çok negatif (tüketici baskın).

**Bağlam Yayılımı (CS — Context Spread):** Uygulama, birçok farklı işlevsel alanda etkileşimde bulunur.

$$TC(a)\!\uparrow \;\Rightarrow\; CS(a)$$

**Ortak Bağımlılık Maruziyeti (SD — Shared Dependency Exposure):** Uygulama çok sayıda ortak kütüphaneye bağımlıdır.

$$LE(a)\!\uparrow \;\Rightarrow\; SD(a)$$

---

### 7.2 Konu Düzeyi Örüntüler

**İletişim Omurgası (CB — Communication Backbone):** Konu, dengeli yayıncı/abone kullanımıyla merkezî bir iletişim kanalı olarak hizmet eder.

$$C(t)\!\uparrow \;\land\; I(t)\!\downarrow \;\Rightarrow\; CB(t)$$

> Yüksek kapsayıcılık VE düşük dengesizliğe sahip bir konu "omurga"dır — birçok uygulama bu kanala hem yayın yapar hem de abone olur, yaklaşık olarak eşit biçimde.

**Yönlü Yoğunlaşma (DC — Directional Concentration):** Konu, yayıncı/abone dağılımında belirgin biçimde tek taraflıdır.

$$I(t)\!\uparrow \;\Rightarrow\; DC(t)$$

**Çevresel Toplayıcı (PA — Peripheral Aggregator):** Konu, sistemin geri kalanına zayıf bağlı olan uygulamaları bir araya toplar.

$$LCR(t)\!\uparrow \;\Rightarrow\; PA(t)$$

> Yüksek bir LCR, konunun katılımcılarının çoğunun çok az sayıda başka konu bağlantısına sahip olduğu anlamına gelir. Konu, merkeze yeterince entegre olmayan çevresel uygulamalar için bir toplanma noktası işlevi görür.

---

### 7.3 Çalışma Düğümü Düzeyi Örüntüler

**Yoğunlaşmış Etkileşim Kümesi (IH — Interaction Hotspot):** Düğüm hem çok sayıda uygulama barındırır hem de bu uygulamalar birbirleriyle aktif biçimde iletişim kurar.

$$ND(n)\!\uparrow \;\land\; NID(n)\!\uparrow \;\Rightarrow\; IH(n)$$

> Her iki koşul da sağlanmalıdır — birbirleriyle konuşmayan çok sayıda uygulamayı barındıran bir düğüm, etkileşim yoğunlaşma noktası değildir.

---

### 7.4 Kütüphane Düzeyi Örüntüler

**Yaygın Ortak Kütüphane (WUL — Widely Used Library):** Kütüphane, çok sayıda uygulama tarafından kullanılmaktadır.

$$LC(l)\!\uparrow \;\Rightarrow\; WUL(l)$$

**Yoğunlaşmış Ortak Kütüphane (CL — Concentrated Library):** Kütüphanenin kullanımı belirli çalışma düğümlerinde yoğunlaşmıştır.

$$LCon(l)\!\uparrow \;\Rightarrow\; CL(l)$$

---

## 8. Birleşik Aykırılık Skoru

Birleşik skor, varlıkları yapısal olarak ne kadar aykırı olduklarına göre **sıralar**; "iyi" veya "kötü" olarak sınıflandırmaz. İki bileşenin birleşiminden oluşur.

### 8.1 Örüntü Tabanlı Aykırılık Skoru

Her varlık için, ilgili bileşen türü için tanımlanan tüm örüntüler üzerinden toplam alınır. Her aktif örüntü, o örüntüyü kaç varlığın tetiklediğiyle ters orantılı bir ağırlıkla katkıda bulunur:

$$OS^{P}_{\mathcal{A}}(a) = \sum_{p \in \mathcal{P}_{\mathcal{A}}} \frac{1}{|\{a' \in \mathcal{A} \mid p(a')\}|} \cdot \mathbb{I}[p(a)]$$

Burada:
- $\mathcal{P}_{\mathcal{A}} = \{WR, RS, CS, SD\}$, uygulamalar için tanımlanan örüntüler kümesidir.
- $\mathbb{I}[p(a)]$, $p$ örüntüsü $a$ uygulaması için tetikleniyorsa 1, aksi hâlde 0'dır (Iverson bracket).
- Paydadaki $|\{a' \in \mathcal{A} \mid p(a')\}|$, $p$ örüntüsünü kaç uygulamanın tetiklediğini sayar.

**Neden ters frekans ağırlığı?** Bir örüntü uygulamaların %80'inde tetikleniyorsa, çok ayırt edici değildir — dolayısıyla skora katkısı küçüktür ($1/0.8N$). Bir örüntü yalnızca 2 uygulamada tetikleniyorsa, nadirdir ve yüksek ayırt edicidir — katkısı büyüktür ($1/2$). Bu, yaygın örüntülerin skoru salt yaygınlıkları nedeniyle domine etmesini, önsel önem sıralaması gerektirmeksizin engeller.

Aynı formül konular ($\mathcal{P}_{\mathcal{T}}$), düğümler ($\mathcal{P}_{\mathcal{N}}$) ve kütüphaneler ($\mathcal{P}_{\mathcal{L}}$) için ilgili örüntü kümeleriyle uygulanır.

---

### 8.2 Tek-Boyutlu Aykırılık Katkısı

Örüntü tabanlı skor, yalnızca **tek bir** metrikte aşırı uç olan ancak çok-metrik örüntü tetiklemeyen varlıkları kaçırabilir. Bu durumların tamamen göz ardı edilmemesi için sınırlı bir tek-metrik katkısı eklenir.

Her $M$ metriği için, varlığın dağılımın üst kuyruğunda ne kadar uzakta olduğunu yansıtan $u_M(x) \in [0, 1]$ üst-kuyruk uçluk değeri hesaplanır. Ardından üst sınırla kısıtlanır:

$$c_M(x) = \min(u_M(x),\; \tau)$$

Varlığın türüne ait tüm metrikler üzerinden toplam alınır:

$$UNI(x) = \sum_{M \in \mathcal{M}_x} c_M(x)$$

Burada:
- $\mathcal{M}_x$, $x$ varlığının türü için tanımlı metrikler kümesidir.
- $\tau$, üst sınırdır (vaka çalışmasında **0.30** olarak belirlenmiştir) — hiçbir tek metrik skora $\tau$'dan fazla katkıda bulunamaz, bu da tek bir aşırı değerin sıralamayı domine etmesini engeller.

---

### 8.3 Nihai Birleşik Skor

$$Score(x) = OS^{P}(x) + \lambda \cdot UNI(x)$$

Burada:
- $OS^{P}(x)$, örüntü tabanlı skordur (sıralamanın birincil belirleyicisi).
- $UNI(x)$, tek-boyutlu katkıdır (ikincil eşitlik kırıcı/güvenlik ağı).
- $\lambda$, küçük bir ağırlık katsayısıdır (vaka çalışmasında **0.30** olarak belirlenmiştir); örüntü tabanlı değerlendirmenin baskın kalmasını sağlar.

**Yorumlama:** Bu skor, bir sınıflandırıcı değil **göreli bir sıralama aracıdır**. "Bu uygulama kötüdür" demez. "Tüm uygulamalar arasında, bu uygulama en olağandışı yapısal özellikler kombinasyonunu sergilemektedir" der. Mühendisler, mimari değerlendirmelerde hangi bileşenlerin öncelikli incelenmesi gerektiğini belirlemek için bu sıralamayı kullanabilir.

---

## 9. Hiperparametreler

Vaka çalışmasında üç hiperparametre kullanılmıştır. Bunlar sezgisel olarak belirlenmiştir:

| Parametre | Değer | Amacı |
|-----------|-------|-------|
| $k$ | 2 | LCR için düşük bağlantılılık eşiği — toplam konu bağlantısı $\leq 2$ olan bir uygulama "zayıf bağlantılı" kabul edilir. |
| $\tau$ | 0.30 | Tek-metrik aykırılık katkısının üst sınırı — herhangi bir tek metriğin birleşik skoru domine etmesini engeller. |
| $\lambda$ | 0.30 | Nihai skordaki tek-boyutlu katkı ağırlığı — örüntü tabanlı değerlendirmenin birincil sıralama faktörü olarak kalmasını sağlar. |

---