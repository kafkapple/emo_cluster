import pytorch_lightning as pl
import torch
import torch.nn.functional as F

class EmotionRecognitionModule(pl.LightningModule):
    def __init__(self, audio_model, text_model, fusion_type="early", learning_rate=0.001):
        super(EmotionRecognitionModule, self).__init__()
        self.audio_model = audio_model
        self.text_model = text_model
        self.fusion_type = fusion_type
        self.learning_rate = learning_rate
        self.classifier = torch.nn.Linear(768 + 768, 8)  # Example for 8 classes

    def forward(self, audio, text):
        audio_embed = self.audio_model(audio)
        text_embed = self.text_model(text)
        if self.fusion_type == "early":
            fused = torch.cat((audio_embed, text_embed), dim=1)
        else:
            fused = (audio_embed + text_embed) / 2
        return self.classifier(fused)

    def training_step(self, batch, batch_idx):
        audio, text, labels = batch
        logits = self(audio, text)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
