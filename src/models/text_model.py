from transformers import AutoModel, AutoTokenizer
import torch
class TextModel(torch.nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(TextModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embedding = self.model(**inputs).last_hidden_state.mean(dim=1)
        return embedding
