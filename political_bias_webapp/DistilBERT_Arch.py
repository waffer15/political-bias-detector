import torch.nn as nn
import transformers
from transformers import DistilBertModel

class DistilBERT_Arch(nn.Module):
    def __init__(self, bert):
      super(DistilBERT_Arch, self).__init__()
      self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
      self.pre_classifier = nn.Linear(768, 768)
      self.dropout = nn.Dropout(0.3)
      self.classifier = nn.Linear(768, 2)

    #define the forward pass
    def forward(self, input_ids, attention_mask):
      output_l = self.l1(input_ids=input_ids, attention_mask=attention_mask)
      hidden_state = output_l[0]
      pooler = hidden_state[:, 0]
      pooler = self.pre_classifier(pooler)
      pooler = nn.ReLU()(pooler)
      pooler = self.dropout(pooler)
      output = self.classifier(pooler)
      return output