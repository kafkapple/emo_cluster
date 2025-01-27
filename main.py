import hydra
from omegaconf import DictConfig
from src.data.download import DatasetDownloader
from src.models.wav2vec_model import Wav2VecModel
from src.models.text_model import TextModel
from src.training.trainer import train_model

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Dataset Download
    DatasetDownloader.download_and_extract(cfg.dataset.download_url, cfg.dataset.download_path)

    # Model Initialization
    audio_model = Wav2VecModel(cfg.model.audio_model)
    text_model = TextModel(cfg.model.text_model)

    # Dataloader (Mockup for simplicity)
    dataloader = None  # Replace with actual DataLoader

    # Train
    train_model(cfg, audio_model, text_model, dataloader)

if __name__ == "__main__":
    main()
