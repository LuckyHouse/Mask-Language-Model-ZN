import torch.nn as nn
from model.bert import BERT
import config.hparams as hp


class BERTLM(nn.Module):
    """
    BERT Language Model
    Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.mask_lm = MaskedLanguageModel(hp.hidden, vocab_size)
        self.init_model()

    def forward(self, x, pos):
        x, attn_list = self.bert(x, pos)
        return self.mask_lm(x), attn_list

    def init_model(self):
        un_init = ['bert.embed.weight', 'bert.pos_emb.weight']
        for n, p in self.named_parameters():
            if n not in un_init and p.dim() > 1:
                nn.init.xavier_uniform_(p)


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))




class BertForSA(nn.Module):
    """
    BERT Language Model
    Masked Language Model
    """

    def __init__(self, bert: BERT, num_labels=2):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.fc = nn.Linear(hp.hidden, hp.hidden)
        self.activ = nn.Tanh()
        self.drop = nn.Dropout(0.1)
        self.classifier = nn.Linear(hp.hidden, num_labels)

        self.init_layer(self.classifier)

    def forward(self, x, pos):
        x, attn_list = self.bert(x, pos)
        pooled_h = self.activ(self.fc(x[:, 0]))
        logits = self.classifier(self.drop(pooled_h))

        return logits


    def init_layer(self, layers):
        for p in layers.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)