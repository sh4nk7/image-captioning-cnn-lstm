"""
model.py
--------
Definition of the CNN-LSTM architecture for Image Captioning on Flickr8k.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """CNN encoder based on InceptionV3."""

    def __init__(self, embed_size, train_cnn=False):
        super().__init__()

        self.inception = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1
        )

        for param in self.inception.parameters():
            param.requires_grad = train_cnn

        self.inception.fc = nn.Identity()

        self.fc = nn.Linear(2048, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        if self.training:
            self.inception.train()
        else:
            self.inception.eval()

        if any(param.requires_grad for param in self.inception.parameters()):
            outputs = self.inception(images)
        else:
            with torch.no_grad():
                outputs = self.inception(images)

        if hasattr(outputs, "logits"):
            features = outputs.logits
        else:
            features = outputs

        features = self.bn(self.fc(features))
        return features


class DecoderLSTM(nn.Module):
    """LSTM decoder that generates captions word by word."""

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.3):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)

        lstm_out, _ = self.lstm(self.dropout(embeddings))
        outputs = self.linear(lstm_out)
        return outputs

    def sample(self, features, max_len=20):
        sampled_ids = []
        states = None
        inputs = features.unsqueeze(1)

        for _ in range(max_len):
            lstm_out, states = self.lstm(inputs, states)
            outputs = self.linear(lstm_out.squeeze(1))
            _, predicted = outputs.max(1)

            sampled_ids.append(predicted)
            inputs = self.embed(predicted).unsqueeze(1)

        sampled_ids = torch.stack(sampled_ids, dim=1)
        return sampled_ids


class CNNtoLSTM(nn.Module):
    """Wrapper model combining the CNN encoder and the LSTM decoder."""

    def __init__(self, embed_size, hidden_size, vocab_size,
                 num_layers=1, dropout=0.3, train_cnn=False):
        super().__init__()

        self.encoder = EncoderCNN(embed_size, train_cnn)
        self.decoder = DecoderLSTM(embed_size, hidden_size, vocab_size,
                                   num_layers, dropout)

    def forward(self, images, captions):
        features = self.encoder(images)
        return self.decoder(features, captions)

    def generate(self, images, max_len=20):
        features = self.encoder(images)
        return self.decoder.sample(features, max_len)


if __name__ == "__main__":
    vocab_size = 5000
    model = CNNtoLSTM(embed_size=300, hidden_size=512, vocab_size=vocab_size)

    images = torch.randn(4, 3, 299, 299)
    captions = torch.randint(0, vocab_size, (4, 15))

    outputs = model(images, captions)
    print("Output shape:", outputs.shape)
