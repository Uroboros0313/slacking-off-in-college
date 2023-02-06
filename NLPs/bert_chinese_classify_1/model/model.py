import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class BertChineseClassifier(nn.Module):
    def __init__(
        self,
        model_path,
        max_length=512,
        freeze=True,
        device='cpu'):
        super().__init__()
        
        self.model_path = model_path
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)
        
        if freeze:
            for par in self.model.parameters():
                par.requires_grad = False
                
        self.fc = nn.Linear(768, 1)
        self.device = device
        self.to(device)
        
    def forward(self, batch_sentences):
        sentence_tokenized = self.tokenizer(batch_sentences,
                                                truncation=True, # 截断超出最大长度
                                                padding=True, # padding长度不足句子
                                                max_length=self.max_length, # 最大长度
                                                add_special_tokens=True) # 添加默认token
        
        input_ids = torch.tensor(sentence_tokenized['input_ids']).to(self.device)
        attention_mask = torch.tensor(sentence_tokenized['attention_mask']).to(self.device)
        
        bert_output = self.model(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:, 0, :]# 提取[CLS]
        out = self.fc(bert_cls_hidden_state)
        
        return out
    
    def predict(self, x):
        return torch.sigmoid(self.forward(x))