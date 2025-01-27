import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

class Wav2VecModel(torch.nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base"):
        super(Wav2VecModel, self).__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)

    def forward(self, audio_waveform):
        inputs = self.processor(audio_waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            embedding = self.model(**inputs).last_hidden_state.mean(dim=1)
        return embedding
