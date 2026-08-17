# Job Application Assistant — Multi-Agent LLM Framework

Open-source, LangGraph tabanlı bir multi-agent orchestration sistemi. Bir CV ve bir iş ilanı verildiğinde; agent'lar ilanı analiz eder, CV ile eşleştirir, mülakat hazırlığı üretir ve sonuçları tek bir raporda birleştirir.

## Motivasyon

- Portfolyo/CV projesi olarak: RAG (InfoGuide Pilot-1) tecrübesinin ötesinde **agent orkestrasyonu, karar verme ve çoklu-agent koordinasyonu** becerisini gösterir.
- Pratik motivasyon: kendi aktif iş arama sürecinde (CV-ilan eşleştirme, mülakat hazırlığı) doğrudan kullanılabilir bir araç.

## Kullanım Senaryosu

Kullanıcı bir CV metni ve bir iş ilanı metni girer. Sistem:
1. İlanı analiz eder (gerekli beceriler, seviye, şirket bağlamı)
2. CV'yi ilanla karşılaştırır (eşleşen/eksik yetkinlikler, öne çıkarılacak projeler)
3. Muhtemel mülakat sorularını ve konuşma noktalarını üretir
4. Hepsini okunabilir tek bir raporda birleştirir

## Mimari

**Pattern:** Supervisor pattern (LangGraph StateGraph)

```
User Input (CV + İlan)
        ↓
   Supervisor ──→ Job Analyzer ──┐
        ↑                         │
        └──────── CV Matcher ←────┘
        ↓
   Interview Prep
        ↓
   Report Writer
        ↓
   Final Output
```

Supervisor, her agent tamamlandığında state'e bakarak bir sonraki agent'a yönlendirme kararını verir (LLM tabanlı yönlendirme — deterministik sıralamadan daha güçlü bir "agentic" hikaye sunar).

### Agent'lar

| Agent | Görev | Çıktı |
|---|---|---|
| **Supervisor** | Akışı yönetir, sıradaki agent'ı belirler | routing kararı |
| **Job Analyzer** | İlan metnini parse eder | `job_requirements`: skills, seniority, keywords, company context (JSON) |
| **CV Matcher** | CV'yi ilan gereksinimleriyle karşılaştırır | `match_analysis`: eşleşme skoru, güçlü/zayıf noktalar, öne çıkarılacak projeler |
| **Interview Prep** | Zayıf noktalara özel mülakat soruları üretir | `interview_prep`: olası sorular, konuşma noktaları |
| **Report Writer** | Tüm çıktıları tek raporda birleştirir | `final_report` (Markdown/HTML) |

### State (LangGraph StateGraph)

- `cv_text`, `job_posting_text` — ham girdi
- `job_requirements` — Job Analyzer çıktısı
- `match_analysis` — CV Matcher çıktısı
- `interview_prep` — Interview Prep çıktısı
- `final_report` — Report Writer çıktısı
- `messages`, `next_agent` — supervisor yönlendirme mantığı

## Teknik Yığın

- **Orchestration:** LangGraph (StateGraph, checkpointing, Store API)
- **LLM:** Açık kaynak model, Ollama üzerinden (Qwen / Mistral)
- **Arayüz:** Streamlit
- **Observability:** LangSmith trace'leri (opsiyonel, README'de demo materyali olarak kullanılacak)

## Veri

Dataset zorunlu değil — ana akış kullanıcının kendi CV ve iş ilanı metnini girdi olarak alır. Test/örnekler için `examples/` klasöründe örnek CV + ilan çiftleri tutulacak.

## Repo Yapısı

```
job-app-assistant/
├── agents/
│   ├── supervisor.py
│   ├── job_analyzer.py
│   ├── cv_matcher.py
│   ├── interview_prep.py
│   └── report_writer.py
├── graph.py          # StateGraph tanımı
├── app.py            # Streamlit arayüzü
├── tests/
├── examples/          # örnek CV + ilan çiftleri
└── README.md
```

## Durum

Mimari planlama tamamlandı, kodlama aşamasına geçiliyor.
