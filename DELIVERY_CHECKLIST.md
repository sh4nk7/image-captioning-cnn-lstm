# Delivery Checklist

Use this checklist before the exam or before packaging the project.

## 1. Environment

- Activate the correct Conda environment:
  - `conda activate captioning`
- Install dependencies if needed:
  - `pip install -r requirements.txt`

## 2. Prepare Data

- Run:
  - `python src/prepare_data.py`
- Check that these files exist:
  - `data/vocab.json`
  - `data/Flickr8k_text/train.csv`
  - `data/Flickr8k_text/val.csv`
  - `data/Flickr8k_text/test_captions.json`

## 3. Training

- Launch training:
  - `python src/train.py --config configs/config.yml`
- Check outputs:
  - `checkpoints/last.pt`
  - `checkpoints/model_best.pt`
  - `runs/tensorboard/`

## 4. Evaluation

- Run evaluation:
  - `python src/eval.py --images_dir data/Flickr8k_Dataset --captions_json data/Flickr8k_text/test_captions.json --vocab_path data/vocab.json --weights checkpoints/model_best.pt`
- Check output:
  - `results/predictions.json`

## 5. Qualitative Results

- Generate visualizations:
  - `python src/visualize.py --weights checkpoints/model_best.pt --vocab_path data/vocab.json --images_dir data/Flickr8k_Dataset --captions_json data/Flickr8k_text/test_captions.json --output_dir figures/`

## 6. Demo

- Run the interface:
  - `streamlit run src/ui_streamlit.py`
- Default demo checkpoint:
  - `checkpoints/model_mini_demo.pt`

## 7. Oral Exam Talking Points

- Dataset: Flickr8k, 5 captions per image
- Text cleaning: lowercase, punctuation removal, digit removal, single-letter removal
- Vocabulary with special tokens: `<pad>`, `<bos>`, `<eos>`, `<unk>`
- Encoder: pretrained InceptionV3
- Decoder: LSTM with linear projection to vocabulary
- Training details: scheduled sampling, label smoothing, gradient clipping
- Evaluation: BLEU-1 to BLEU-4 with 5 references per test image

## 8. Safe Claims

You can confidently say that the repository now includes:
- reproducible data preparation
- training/validation/test split generation
- aligned next-token loss
- real CNN fine-tuning support
- BLEU evaluation on 5 captions per image

Avoid claiming results you have not actually measured yet in your runtime environment.
