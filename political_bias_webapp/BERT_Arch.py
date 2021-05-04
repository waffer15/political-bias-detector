import torch.nn as nn
class BERT_Arch(nn.Module):

    def __init__(self, bert):
      
      super(BERT_Arch, self).__init__()

      self.bert = bert
      
      # relu activation function
      self.relu =  nn.ReLU()

      # dense layer 1
      self.fc1 = nn.Linear(768,64)
      
      # dense layer 2 (Output layer)
      self.fc2 = nn.Linear(64,64)
      self.fc3 = nn.Linear(64,64)
      self.fc4 = nn.Linear(64,2)

      #softmax activation function
      self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):

      #pass the inputs to the model  
      _, cls_hs = self.bert(sent_id, attention_mask=mask)
      
      x = self.fc1(cls_hs)

      x = self.relu(x)

      # output layer
      x = self.fc2(x)
      x = self.relu(x)
      x = self.fc3(x)
      x = self.relu(x)
      x = self.fc4(x)
      # apply softmax activation
      x = self.softmax(x)

      return x