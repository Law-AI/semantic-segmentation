from model.submodels import *

'''
    Top-level module which uses a Hierarchical-LSTM-CRF to classify.
    
    If pretrained = False, each example is represented as a sequence of sentences, which themselves are sequences of word tokens. Individual sentences are passed to LSTM_Sentence_Encoder to generate sentence embeddings. 
    If pretrained = True, each example is represented as a sequence of fixed-length pre-trained sentence embeddings.
    
    Sentence embeddings are then passed to LSTM_Emitter to generate emission scores, and finally CRF is used to obtain optimal tag sequence. 
    Emission scores are fed to the CRF to generate optimal tag sequence.
'''
class Hier_LSTM_CRF_Classifier(nn.Module):
    def __init__(self, n_tags, sent_emb_dim, sos_tag_idx, eos_tag_idx, pad_tag_idx, vocab_size = 0, word_emb_dim = 0, pad_word_idx = 0, pretrained = False, device = 'cuda'):
        super().__init__()
        
        self.emb_dim = sent_emb_dim
        self.pretrained = pretrained
        self.device = device
        self.pad_tag_idx = pad_tag_idx
        self.pad_word_idx = pad_word_idx
        
        # sentence encoder is not required for pretrained embeddings
        self.sent_encoder = LSTM_Sentence_Encoder(vocab_size, word_emb_dim, sent_emb_dim).to(self.device) if not self.pretrained else None
            
        self.emitter = LSTM_Emitter(n_tags, sent_emb_dim, sent_emb_dim).to(self.device)
        self.crf = CRF(n_tags, sos_tag_idx, eos_tag_idx, pad_tag_idx).to(self.device)
        
    
    def forward(self, x):
        batch_size = len(x)
        seq_lengths = [len(doc) for doc in x]
        max_seq_len = max(seq_lengths)
        
        if not self.pretrained: ## x: list[batch_size, sents_per_doc, words_per_sent]
            tensor_x = []
            for doc in x:
                sents = [torch.tensor(s, dtype = torch.long) for s in doc]
                sent_lengths = [len(s) for s in doc]
                
                ## list[sents_per_doc, words_per_sent] --> tensor[sents_per_doc, max_sent_len]
                sents = nn.utils.rnn.pad_sequence(sents, batch_first = True, padding_value = self.pad_word_idx).to(self.device)
                
                ## tensor[sents_per_doc, max_sent_len] --> tensor[sents_per_doc, sent_emb_dim]
                sents = self.sent_encoder(sents, sent_lengths)
        
                tensor_x.append(sents)
            
        else: ## x: list[batch_size, sents_per_doc, sent_emb_dim]
            tensor_x = [torch.tensor(doc, dtype = torch.float, requires_grad = True) for doc in x]
        
        ## list[batch_size, sents_per_doc, sent_emb_dim] --> tensor[batch_size, max_seq_len, sent_emb_dim]
        tensor_x = nn.utils.rnn.pad_sequence(tensor_x, batch_first = True).to(self.device)        
        
        self.mask = torch.zeros(batch_size, max_seq_len).to(self.device)
        for i, sl in enumerate(seq_lengths):
            self.mask[i, :sl] = 1	
        
        self.emissions = self.emitter(tensor_x)
        _, path = self.crf.decode(self.emissions, mask = self.mask)
        return path
    
    def _loss(self, y):
        ##  list[batch_size, sents_per_doc] --> tensor[batch_size, max_seq_len]
        tensor_y = [torch.tensor(doc, dtype = torch.long) for doc in y]
        tensor_y = nn.utils.rnn.pad_sequence(tensor_y, batch_first = True, padding_value = self.pad_tag_idx).to(self.device)
        
        nll = self.crf(self.emissions, tensor_y, mask = self.mask)
        return nll    
