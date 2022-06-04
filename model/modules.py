import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from model.layers import RNNEncoder, BertEncoder


class ClassificationLoss(nn.Module):
    def __init__(self):
        super(ClassificationLoss, self).__init__()
        self.loss = CrossEntropyLoss()

    def forward(self, inputs, target):
        return self.loss(inputs, target)

class BilstmEncoder(nn.Module):
    def __init__(self, bert_hidden_size, rnn_hidden_size, rnn_dropout, sequence_size, num_label, batch_size):
        super(BilstmEncoder, self).__init__()
        self.bert_hidden_size = bert_hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_dropout = rnn_dropout
        self.sequence_size = sequence_size
        self.num_label = num_label
        self.batch_size = batch_size
        self.bert = BertEncoder('bert-base-uncased')
        self.rnn = RNNEncoder(input_size=self.bert_hidden_size, hidden_size=self.rnn_hidden_size)
        self.fc = nn.Linear(self.rnn_hidden_size, self.num_label)
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        # hidden_states 是层数为13，每层的feature size 是 torch.Size([64, 128, 768] batch * bert_max_len * input_size
        utterance = data["utterance"]
        input_seq = utterance.input_ids
        token_type_ids = utterance.token_type_ids
        attn_mask = utterance.attention_mask
        bert_out = self.bert(input_seq, token_type_ids=token_type_ids, attn_mask=attn_mask)
        # last_hidden_state (batch_size, sequence_length, hidden_size)
        # Sequence of hidden-states at the output of the last layer of the model.
        bert_hidden_state = bert_out["last_hidden_state"]
        # Calculate lengths by attn_mask([64, 128])
        lengths = attn_mask.sum(1)
        feature, last_hidden_state = self.rnn(inputs=(bert_hidden_state, lengths))
        # feature, (h, c) = self.bilstm(hidden_states[-1]) # bilstm采用comcat torch.Size([32, 64, 60])
        # last_hidden_state = feature[:, -1, :]
        linear_state = self.fc(last_hidden_state[-1])
        # logits = self.softmax(linear_state)
        return linear_state

# class KUDC(nn.Module):
#     def __init__(self, bert_model, kwargs):
#         super(KUDC, self).__init__()
#
#         self.hidden_size = kwargs['hidden_size']
#         self.rnn_dropout = kwargs['rnn_dropout']
#         self.sequence_size = kwargs['sequence_size']
#         self.num_class = kwargs['num_class']
#         self.att_state_size = kwargs['att_state_size']
#         self.bert = bert_model
#         self.T2M = nn.Sequential(
#             nn.Linear(2 * self.sequence_size, self.att_state_size), # u*2*d  2*d*1 -> u*1
#             nn.Tanh(),
#             nn.Linear(self.att_state_size, 1, bias=False), # 1*u  u*1 -> 1
#             nn.Softmax(dim=1)
#         )
#         self.gate1 = nn.Sequential(
#             nn.Linear(2 * self.sequence_size, 1),
#             nn.Sigmoid()
#         )
#         self.bilstm = nn.LSTM(input_size=(self.sequence_size), hidden_size=self.hidden_size, bidirectional=True,
#                               batch_first=True)
#         self.activation = nn.Tanh()
#         self.fc2 = nn.Linear(2 * self.hidden_size, self.num_class)
#         self.softmax = nn.Softmax(dim=1)
#         self.loss = nn.CrossEntropyLoss()
#
#     def forward(self, input_seq, attn_mask, labels, type_seq, stage='train'):
#         # hidden_states 是层数为13，每层的feature size 是 torch.Size([32, 64, 768] batch * bert_max_len * input_size
#         hidden_states = self.bert(input_seq, attention_mask=attn_mask, output_hidden_states=True, return_dict=True)[
#             "hidden_states"][-1]
#
#         # word-level knowledge
#         entity_emb = hidden_states[1,3] # entity embedding 求和
#         type_emb = self.bert(type_seq, attention_mask=attn_mask, output_hidden_states=True, return_dict=True)[
#             "hidden_states"][-1]
#
#         alpha = self.T2M(torch.cat((entity_emb, type_emb), 1))
#         type_feature = alpha * type_emb
#         lamb = self.gate1(torch.cat((entity_emb, type_feature), 1))
#
#
#
#
#
#
#
#         feature, (h, c) = self.bilstm(hidden_states)  # bilstm采用comcat torch.Size([32, 64, 60])
#         last_hidden_state = feature[:, -1, :]
#
#         logits = self.softmax(self.fc(last_hidden_state))
#         prediction = torch.argmax(logits, 1)
#         loss = self.loss(logits, labels)
#         return logits, prediction, loss
