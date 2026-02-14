# NLP/Voice Feature Extraction Strategy

> This document describes how we would extract structured health features from unstructured text or voice input in a production deployment.

## Overview

In real-world deployment, Hea users provide health data through conversational interfaces (chat, voice). This document outlines our strategy for converting unstructured input into the structured features our model expects.

## Pipeline Architecture

```
Voice/Text Input → Transcription → Entity Extraction → Feature Mapping → Model Input
```

## 1. Speech-to-Text (Voice Input)

**Framework:** OpenAI Whisper (open-source)

**Why Whisper:**
- Open-source, runs locally (cost efficient)
- Multilingual support
- Robust to background noise
- Word-level timestamps for pause analysis

**Implementation:**
```python
import whisper
model = whisper.load_model("base")
result = model.transcribe("user_audio.mp3")
text = result["text"]
```

## 2. Entity Extraction

**Framework:** spaCy + custom NER or LLM-based extraction

**Target Entities:**
| Entity Type | Examples | Maps to Feature |
|-------------|----------|-----------------|
| SYMPTOM | "headache", "tired", "chest pain" | symptom_count, symptom_severity |
| MOOD | "stressed", "happy", "anxious" | mood_score (PHQ-2 proxy) |
| SLEEP | "slept 5 hours", "insomnia" | sleep_hours, sleep_quality |
| ACTIVITY | "walked 2 miles", "sedentary" | activity_level |
| DIET | "skipped breakfast", "ate fast food" | diet_quality_score |
| PAIN | "back pain 7/10" | pain_level |

**LLM-based Extraction (Alternative):**
```python
prompt = """
Extract health indicators from this text:
"{user_input}"

Return JSON with: mood, energy_level, sleep_hours, symptoms, pain_level
"""
```

## 3. Temporal Aggregation

Since our model uses longitudinal data, we aggregate daily inputs into weekly/monthly features:

- `avg_mood_7d` — 7-day rolling average mood score
- `sleep_variance_30d` — Sleep consistency over 30 days
- `symptom_trend` — Increasing/decreasing symptom frequency

## 4. Feature Mapping to Model Input

Map extracted entities to RAND HRS-equivalent features:

| Extracted | Model Feature | Transformation |
|-----------|---------------|----------------|
| mood_score | RwCESD (depression) | Scale 1-10 → 0-8 CESD |
| self_rated_health | RwSHLT | "good"→2, "fair"→3, "poor"→4 |
| sleep_hours | sleep_quality | <6h=poor, 6-8=good, >8=check |
| activity_level | RwACTIV | Categorical mapping |

## 5. Handling Noisy Input

**Strategies:**
1. **Missing data:** Use historical average or flag as missing
2. **Ambiguous input:** Ask follow-up clarification question
3. **Out-of-range values:** Clip to valid range with warning
4. **Inconsistent reports:** Use median of recent values

## 6. Privacy Considerations

- All processing can run on-device (Whisper, local LLM)
- No raw audio stored after transcription
- Text anonymized before cloud processing (if needed)

## Cost Efficiency

| Component | Cost | Alternative |
|-----------|------|-------------|
| Whisper | Free (local) | — |
| spaCy NER | Free | — |
| LLM extraction | ~$0.01/query | Use local Llama |

**Total per user/day:** <$0.01

---

*This strategy enables seamless integration with Hea's conversational interface while maintaining the structured input our prediction model requires.*
