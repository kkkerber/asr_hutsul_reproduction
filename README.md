# ASR for the Hutsul Dialect

## Supported model families

| Model                              | Family       | Strategy                |      WER     |     CER      |
|------------------------------------|--------------|-------------------------|--------------|--------------|
| `openai/whisper-small`             | Whisper      | LoRA (q_proj, v_proj)   |              |              |
| `openai/whisper-medium`            | Whisper      | LoRA                    |              |              |
| `openai/whisper-large-v3`          | Whisper      | LoRA                    |              |              |
| `arampacha/whisper-large-uk-2`     | Whisper      | LoRA                    |              |              |
| `facebook/wav2vec2-large-xlsr-53`  | Wav2Vec2     | full CTC fine-tuning    |              |              |
| `Yehor/w2v-bert-uk-v2.1`           | Wav2Vec2-BERT| adapter tuning          |              |              |
| OmniASR-300M                       | OmniASR      | tri-stage CTC           |              |              |
| OmniASR-1B                         | OmniASR      | tri-stage CTC           |              |              |

---

## Repository layout

```
asr_hutsul_reproduction/
├── requirements.txt
├── README.md
├── config.py                 
├── preprocess.py             
├── metrics.py                
├── evaluate.py              
├── train.py                  
├── configs/                  
│   ├── whisper.yaml
│   ├── wav2vec2.yaml
│   ├── wav2vec2_bert.yaml
│   └── omniasr.yaml
├── models/                  
│   ├── whisper_trainer.py
│   ├── wav2vec2_trainer.py
│   ├── wav2vec2_bert_trainer.py
│   └── omniasr_trainer.py
├── utils/                   
│   ├── augmentation.py       
│   ├── text_normalization.py 
│   ├── collators.py          
│   └── callbacks.py          
├── outputs/                  
└── notebooks/
    ├── colab_training.ipynb  ← main Colab entry point
    └── inference_demo.ipynb  ← single-file transcription demo
```
---


| Package        | Pinned range          |
|----------------|-----------------------|
| torch          | `>=2.2,<2.6`          |
| transformers   | `>=4.45,<4.50`        |
| datasets       | `>=2.20,<3.3`         |
| evaluate       | `>=0.4.2,<0.5`        |
| peft           | `>=0.12,<0.15`        |
| accelerate     | `>=0.34,<1.5`         |
| jiwer          | `>=3.0.4`             |
| audiomentations| `>=0.36.0`            |

---

## Hyper-parameters

### Whisper (LoRA)

| Setting                   | small | medium | large-v3 | large-uk-2 |
|---------------------------|-------|--------|----------|------------|
| `max_steps`               | 8 000 | 8 000  | 8 000    | 4 000      |
| `per_device_train_batch`  | 4     | 8      | 4        | 4          |
| `gradient_accumulation`   | 4     | 4      | 4        | 4          |
| effective batch           | 16    | 32     | 16       | 16         |
| LoRA r / α / dropout      | 16 / 32 / 0.05      ||||
| LoRA target               | `q_proj`, `v_proj`  ||||
| `learning_rate`           | 1e-4              ||||
| `warmup_steps`            | 500               ||||
| `generation_max_length`   | 225               ||||
| metric for best model     | CER               ||||

### Wav2Vec2-XLSR (CTC)

| Setting                   | xlsr-300m-uk |
|---------------------------|--------------|
| `max_steps`               | 5 000        |
| `per_device_train_batch`  | 8            |
| `gradient_accumulation`   | 8            |
| effective batch           | 64           |
| `learning_rate`           | 1e-4         |
| `warmup_steps`            | 1 000        |
| feature encoder           | frozen       |
| metric for best model     | CER          |

### Wav2Vec2-BERT-UK-v2.1

| Setting                              | value          |
|--------------------------------------|----------------|
| `max_steps`                          | 5 000          |
| `per_device_train_batch`             | 8              |
| `gradient_accumulation`              | 4 (eff. 32)    |
| `learning_rate`                      | 5e-5           |
| `warmup_steps`                       | 800            |
| feature encoder                      | frozen         |
| lower transformer layers frozen      | 6              |
| adapters trainable                   | yes            |
| augmentation                         | strong preset  |
| metric for best model                | CER            |

### OmniASR

| Setting                   | 300M    | 1B    |
|---------------------------|---------|-------|
| `max_steps`               | 48 000  | 36 000|
| `per_device_train_batch`  | 8       | 4     |
| `gradient_accumulation`   | 4       | 8     |
| effective batch           | 32      | 32    |
| `learning_rate`           | 5e-5    | 5e-5  |
| scheduler                 | tri-stage (warmup 0.10, hold 0.40, decay 0.50) ||
| metric for best model     | WER     | WER   |

---

## Augmentation pipeline

`utils/augmentation.py` implements two stages:

1. **Waveform** (`audiomentations.Compose`):
   * `AddGaussianNoise`,
   * `PitchShift` (±2 semitones, ±3 in *strong* preset),
   * `TimeStretch` (0.8 × – 1.2 ×),
   * `Gain` (±6 dB / ±8 dB *strong*).

2. **Feature-domain** SpecAugment with independent time and
   frequency masking.



