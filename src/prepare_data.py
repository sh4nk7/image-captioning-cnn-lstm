"""
prepare_data.py
---------------
Convenience script to build the Flickr8k vocabulary and export the
processed train/validation/test files expected by the project.
"""

import argparse

from vocab_builder import build_vocabulary, save_vocabulary, export_dataset_files


def main():
    parser = argparse.ArgumentParser(description="Prepare Flickr8k data for CNN-LSTM captioning.")
    parser.add_argument("--captions_file", type=str, default="data/Flickr8k_text/Flickr8k.token.txt")
    parser.add_argument("--train_split", type=str, default="data/Flickr8k_text/Flickr_8k.trainImages.txt")
    parser.add_argument("--val_split", type=str, default="data/Flickr8k_text/Flickr_8k.devImages.txt")
    parser.add_argument("--test_split", type=str, default="data/Flickr8k_text/Flickr_8k.testImages.txt")
    parser.add_argument("--freq_threshold", type=int, default=5)
    parser.add_argument("--vocab_out", type=str, default="data/vocab.json")
    parser.add_argument("--train_out", type=str, default="data/Flickr8k_text/train.csv")
    parser.add_argument("--val_out", type=str, default="data/Flickr8k_text/val.csv")
    parser.add_argument("--test_json_out", type=str, default="data/Flickr8k_text/test_captions.json")
    args = parser.parse_args()

    vocab = build_vocabulary(args.captions_file, args.freq_threshold)
    save_vocabulary(vocab, args.vocab_out)
    export_dataset_files(
        captions_file=args.captions_file,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        train_out=args.train_out,
        val_out=args.val_out,
        test_json_out=args.test_json_out,
    )

    print("[INFO] Data preparation completed successfully.")


if __name__ == "__main__":
    main()
