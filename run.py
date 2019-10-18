from model import *
from prepare_data import *
from train import *

folds = prepare_folds()

x, y, tag2idx = prepare_dataset_emb(folds)

model = Hier_LSTM_CRF_Classifier(len(tag2idx), 100, tag2idx['<start>'], tag2idx['<end>'], tag2idx['<pad>'], pretrained = True).cuda()

opt = torch.optim.Adam(model.parameters(), lr = 0.01)

learn(model, x, y, tag2idx, 3)