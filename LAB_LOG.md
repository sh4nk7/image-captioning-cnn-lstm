# Laboratory Log

This file records the executed experiments and their real outputs.

---

## Environment Record

- Date range: 2026-04-17 to 2026-04-24
- Working directory: `C:\Users\Giuseppe\Desktop\cnn-lstm-image-captioning\cnn-lstm-image-captioning`
- Environment: `captioning`
- Python executable used for final runs: `C:\Users\Giuseppe\anaconda3\envs\captioning\python.exe`
- PyTorch version: `2.9.1+cpu`
- Device used for experiments: `cpu`

---

## Data Preparation

- Command: `python src/prepare_data.py`
- Output files created:
  - `data/vocab.json`
  - `data/Flickr8k_text/train.csv`
  - `data/Flickr8k_text/val.csv`
  - `data/Flickr8k_text/test_captions.json`
- Vocabulary size: `2985`
- Dataset sizes:
  - train rows: `30000`
  - validation rows: `5000`
  - test images: `1000`
- Notes:
  - Flickr8k captions were cleaned and split into reproducible train/validation/test artifacts.

---

## Executed Experiments

### baseline_cpu

- Purpose: establish the reference experiment
- Config: `configs/baseline_cpu.yml`
- Epochs: `5`
- Trainable parameters: `4,709,177`
- Best validation loss: `4.1927`
- Final training loss: `4.8016`
- Final validation loss: `4.2332`
- BLEU-1: `0.2553`
- BLEU-2: `0.1556`
- BLEU-3: `0.0928`
- BLEU-4: `0.0472`
- Logs:
  - `logs/baseline_cpu_train.log`
  - `logs/baseline_cpu_eval.log`
- Outputs:
  - `checkpoints/baseline_cpu/model_best.pt`
  - `results/baseline_cpu_predictions.json`
  - `figures/baseline_cpu/`

### ablation_no_ss_cpu

- Purpose: measure the effect of removing scheduled sampling
- Config: `configs/ablation_no_scheduled_sampling_cpu.yml`
- Epochs: `5`
- Trainable parameters: `4,709,177`
- Best validation loss: `4.0106`
- Final training loss: `3.5373`
- Final validation loss: `4.0234`
- BLEU-1: `0.2232`
- BLEU-2: `0.1242`
- BLEU-3: `0.0717`
- BLEU-4: `0.0396`
- Logs:
  - `logs/ablation_no_ss_cpu_train.log`
  - `logs/ablation_no_ss_cpu_eval.log`
- Outputs:
  - `checkpoints/ablation_no_ss_cpu/model_best.pt`
  - `results/ablation_no_ss_cpu_predictions.json`
  - `figures/ablation_no_ss_cpu/`
- Notes:
  - Training loss improved relative to baseline, but BLEU degraded.
  - This indicates weaker generalization without scheduled sampling.

### ablation_no_ls_cpu

- Purpose: measure the effect of removing label smoothing
- Config: `configs/ablation_no_label_smoothing_cpu.yml`
- Epochs: `5`
- Trainable parameters: `4,709,177`
- Best validation loss: `3.5029`
- Final training loss: `4.0860`
- Final validation loss: `3.5738`
- BLEU-1: `0.2404`
- BLEU-2: `0.1435`
- BLEU-3: `0.0809`
- BLEU-4: `0.0394`
- Logs:
  - `logs/ablation_no_ls_cpu_train.log`
  - `logs/ablation_no_ls_cpu_train_resume.log`
  - `logs/ablation_no_ls_cpu_eval.log`
- Outputs:
  - `checkpoints/ablation_no_ls_cpu/model_best.pt`
  - `results/ablation_no_ls_cpu_predictions.json`
  - `figures/ablation_no_ls_cpu/`
- Notes:
  - Label smoothing removal reduced BLEU relative to baseline.

### ablation_finetune_cpu

- Purpose: measure the effect of CNN fine-tuning
- Config: `configs/ablation_finetune_cpu.yml`
- Epochs: `5`
- Trainable parameters: `4,709,177`
- Best validation loss: `3.6614`
- Final training loss: `4.7512`
- Final validation loss: `4.2861`
- BLEU-1: `0.3301`
- BLEU-2: `0.2022`
- BLEU-3: `0.1189`
- BLEU-4: `0.0649`
- Logs:
  - `logs/ablation_finetune_cpu_train.log`
  - `logs/ablation_finetune_cpu_train_resume.log`
  - `logs/ablation_finetune_cpu_eval.log`
- Outputs:
  - `checkpoints/ablation_finetune_cpu/model_best.pt`
  - `results/ablation_finetune_cpu_predictions.json`
  - `figures/ablation_finetune_cpu/`
- Notes:
  - This was the best-performing setup by BLEU score.
  - The experiment was first run for 2 epochs, then resumed and extended to 5 epochs.

---

## Comparative Conclusions

- Scheduled sampling improved BLEU compared with the `no_ss` ablation.
- Label smoothing improved BLEU compared with the `no_ls` ablation.
- CNN fine-tuning produced the best final captioning quality in this CPU-only study.

---

## Final Result Files For Documentation

- `results/metrics_summary.csv`
- `results/experiment_summary.csv`
- `results/*_predictions.json`
- `logs/*`
- `figures/*`
