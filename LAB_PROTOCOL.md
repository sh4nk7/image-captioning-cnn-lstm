# Laboratory Protocol

This document defines the rigid experimental protocol for the CNN-LSTM
image captioning project. The goal is to produce reproducible, comparable,
and documentable results.

## Principles

- Run one controlled experiment at a time
- Change one main variable per ablation
- Save command, config, runtime, metrics, and observations
- Do not claim any metric that was not actually measured
- Keep raw logs, checkpoints, and output files for every experiment

## Fixed Inputs

- Dataset: Flickr8k
- Vocabulary source: `data/Flickr8k_text/Flickr8k.token.txt`
- Image directory: `data/Flickr8k_Dataset/`
- Prepared files:
  - `data/vocab.json`
  - `data/Flickr8k_text/train.csv`
  - `data/Flickr8k_text/val.csv`
  - `data/Flickr8k_text/test_captions.json`

## Controlled Variables

- Encoder architecture: InceptionV3
- Decoder architecture: single-layer LSTM
- Embedding size: 300
- Hidden size: 512
- Max caption length during evaluation: 20
- Test references: 5 captions per image

## Planned Experiments

1. `debug_cpu`
- Purpose: verify pipeline correctness and checkpoint creation
- Config: `configs/debug_cpu.yml`

2. `baseline_cpu`
- Purpose: establish the main CPU baseline
- Config: `configs/baseline_cpu.yml`

3. `ablation_no_ss_cpu`
- Purpose: measure the impact of scheduled sampling
- Config: `configs/ablation_no_scheduled_sampling_cpu.yml`

4. `ablation_no_ls_cpu`
- Purpose: measure the impact of label smoothing
- Config: `configs/ablation_no_label_smoothing_cpu.yml`

5. `ablation_finetune_cpu`
- Purpose: measure the runtime/quality trade-off of CNN fine-tuning
- Config: `configs/ablation_finetune_cpu.yml`

## Data To Record For Every Training Run

- Experiment name
- Config file used
- Command executed
- Date and time of start
- Date and time of end
- Device used
- CPU model and RAM if available
- Number of trainable parameters
- Epoch count
- Batch size
- Best validation loss
- Runtime per epoch
- Total runtime
- Checkpoints produced
- Observed warnings or runtime errors

## Data To Record For Every Evaluation Run

- Experiment name
- Checkpoint path
- Evaluation command
- BLEU-1
- BLEU-2
- BLEU-3
- BLEU-4
- Qualitative observations
- Output JSON path

## Storage Policy

- Keep training outputs in experiment-specific checkpoint folders
- Keep TensorBoard logs in experiment-specific run folders
- Keep evaluation outputs in experiment-specific result files
- Keep the execution trace in `LAB_LOG.md`
