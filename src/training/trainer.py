
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

def train_model(audio_model, dataloader, config):
    wandb_logger = WandbLogger(name=config.experiment.name, project="emotion_recognition")
    model = EmotionRecognitionModule(audio_model=audio_model, text_model=None)
    trainer = pl.Trainer(max_epochs=config.training.max_epochs, logger=wandb_logger)
    trainer.fit(model, dataloader)
