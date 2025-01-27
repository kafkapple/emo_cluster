import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from data.preprocess import AudioPreprocessor

class EmotionDataset(Dataset):
    def __init__(self, data_dir, labels, audio_preprocessor):
        self.audio_files = glob.glob(os.path.join(data_dir, "*.wav"))
        self.labels = labels  # Dictionary {filename: label}
        self.audio_preprocessor = audio_preprocessor

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        label = self.labels[os.path.basename(audio_file)]
        audio_features = self.audio_preprocessor.preprocess_audio(audio_file)
        return audio_features, torch.tensor(label)

def create_dataloader(data_dir, labels, batch_size=32):
    audio_preprocessor = AudioPreprocessor()
    dataset = EmotionDataset(data_dir, labels, audio_preprocessor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


import os
import glob
import torchaudio
from transformers import Wav2Vec2Processor, AutoTokenizer, AutoModel
import torch

class AudioPreprocessor:
    def __init__(self, model_name="facebook/wav2vec2-base"):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = torch.nn.Identity()  # Optional: Replace with frozen Wav2Vec2Model

    def preprocess_audio(self, audio_file):
        waveform, sample_rate = torchaudio.load(audio_file)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            embedding = self.model(inputs["input_values"]).mean(dim=1)  # Mean pooling
        return embedding

class TextPreprocessor:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def preprocess_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embedding = self.model(**inputs).last_hidden_state.mean(dim=1)  # Mean pooling
        return embedding
