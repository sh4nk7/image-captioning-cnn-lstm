# CNN-LSTM Image Captioning on Flickr8k

Project for the University of Parma course `Deep Learning and Generative Models`.

This repository implements an image captioning pipeline based on:
- `InceptionV3` as CNN encoder
- `LSTM` as sequence decoder
- text cleaning and vocabulary building from Flickr8k captions
- scheduled sampling, label smoothing, gradient clipping
- BLEU-1 to BLEU-4 evaluation on 5 captions per test image
- optional Streamlit demo for interactive inference

## Repository Structure

```text
src/
  dataset.py
  eval.py
  model.py
  prepare_data.py
  train.py
  ui_streamlit.py
  visualize.py
  vocab_builder.py
configs/
  config.yml
  baseline_cpu.yml
  ablation_no_scheduled_sampling_cpu.yml
  ablation_no_label_smoothing_cpu.yml
  ablation_finetune_cpu.yml
data/
  Flickr8k_Dataset/
  Flickr8k_text/
checkpoints/
figures/
logs/
results/
```

## Environment Setup

```bash
conda create -n captioning python=3.10 -y
conda activate captioning
pip install -r requirements.txt
```

Dataset source: <https://github.com/goodwillyoga/Flickr8k_dataset>

## 1. Prepare Vocabulary and Splits

Quick command:

```bash
python src/prepare_data.py
```

Equivalent explicit command:

This command:
- cleans the Flickr8k captions
- builds `data/vocab.json`
- creates `train.csv` and `val.csv`
- creates `test_captions.json` with 5 reference captions per image

```bash
python src/vocab_builder.py \
  --captions_file data/Flickr8k_text/Flickr8k.token.txt \
  --freq_threshold 5 \
  --output data/vocab.json \
  --train_split data/Flickr8k_text/Flickr_8k.trainImages.txt \
  --val_split data/Flickr8k_text/Flickr_8k.devImages.txt \
  --test_split data/Flickr8k_text/Flickr_8k.testImages.txt \
  --train_out data/Flickr8k_text/train.csv \
  --val_out data/Flickr8k_text/val.csv \
  --test_json_out data/Flickr8k_text/test_captions.json
```

## 2. Train

General training command:

```bash
python src/train.py --config configs/config.yml
```

Recommended reproducible experiment commands used in the project:

```bash
python src/train.py --config configs/baseline_cpu.yml
python src/train.py --config configs/ablation_no_scheduled_sampling_cpu.yml
python src/train.py --config configs/ablation_no_label_smoothing_cpu.yml
python src/train.py --config configs/ablation_finetune_cpu.yml
```

Typical outputs:
- experiment checkpoints inside `checkpoints/<experiment_name>/`
- text logs inside `logs/`
- prediction files and metric summaries inside `results/`

## 3. Evaluate

```bash
python src/eval.py \
  --images_dir data/Flickr8k_Dataset \
  --captions_json data/Flickr8k_text/test_captions.json \
  --vocab_path data/vocab.json \
  --weights checkpoints/baseline_cpu/model_best.pt
```

For the full experiment set, evaluation results are saved as:
- `results/baseline_cpu_predictions.json`
- `results/ablation_no_ss_cpu_predictions.json`
- `results/ablation_no_ls_cpu_predictions.json`
- `results/ablation_finetune_cpu_predictions.json`

Summary tables:
- `results/metrics_summary.csv`
- `results/experiment_summary.csv`

## 4. Visualize Predictions

```bash
python src/visualize.py \
  --weights checkpoints/baseline_cpu/model_best.pt \
  --vocab_path data/vocab.json \
  --images_dir data/Flickr8k_Dataset \
  --captions_json data/Flickr8k_text/test_captions.json \
  --output_dir figures/baseline_cpu
```

## 5. Streamlit Demo

```bash
streamlit run src/ui_streamlit.py
```

The UI defaults to:
- `checkpoints/model_mini_demo.pt`
- `data/vocab.json`
- `data/Flickr8k_text/test_captions.json`

## Notes

- The vocabulary is now built from the real Flickr8k tab-separated caption format.
- The training loss is aligned on next-token prediction instead of scoring the `<bos>` token.
- CNN fine-tuning is now actually enabled when `train_cnn: true` in `configs/config.yml`.
- The final repository includes a small ablation study on scheduled sampling, label smoothing, and CNN fine-tuning.
- The main experimental artifacts are stored in `checkpoints/`, `logs/`, `results/`, and `figures/`.
