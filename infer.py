import argparse
import sent2vec
import os

from model.Hier_BiLSTM_CRF import *
from prepare_data import *
from train import *

def main():
    parser = argparse.ArgumentParser(description = 'Infer tags for unannotated files')

    parser.add_argument('--pretrained', default = False, type = bool, help = 'Whether the model uses pretrained sentence embeddings or not')
    parser.add_argument('--data_path', default = 'infer/data/', type = str, help = 'Folder to store the unannotated text files')
    parser.add_argument('--model_path', default = 'infer/model.tar', type = str, help = 'Path to trained Hierarchical BiLSTM CRF Model')
    parser.add_argument('--sent2vec_model_path', default = 'infer/sent2vec.bin', type = str, help = 'Path to trained sent2vec model, applicable only if pretrained = True')
    parser.add_argument('--save_path', default = 'infer/predictions.txt', type = str, help = 'Path to file where predictions will be saved')
    parser.add_argument('--word2idx_path', default = 'infer/word2idx.json', type = str, help = 'Path to word2idx dict created during training model')
    parser.add_argument('--tag2idx_path', default = 'infer/tag2idx.json', type = str, help = 'Path to tag2idx dict created during training model')
    parser.add_argument('--emb_dim', default = 200, type = int, help = 'Sentence embedding dimension')
    parser.add_argument('--word_emb_dim', default = 100, type = int, help = 'Word embedding dimension, applicable only if pretrained = False')
    parser.add_argument('--device', default = 'cuda', type = str, help = 'cuda / cpu')
    
    args = parser.parse_args()

    with open(args.word2idx_path) as fp:
        args.word2idx = json.load(fp)

    with open(args.tag2idx_path) as fp:
        args.tag2idx = json.load(fp)

    if args.pretrained:
        print('Loading pretrained sent2vec model ...', end = ' ', flush = True)
        sent2vec_model = sent2vec.Sent2vecModel()
        sent2vec_model.load_model(args.sent2vec_model_path)
        print('Done')

    else:
        sent2vec_model = None

    print('\nPreparing data ...', end = ' ', flush = True)
    idx_order = list(map(lambda x: os.fsdecode(x)[:-4], os.listdir(os.fsencode(args.data_path))))
    x = prepare_data_inference(idx_order, args, sent2vec_model)
    print('Done')

    print('\nLoading model ...', end = ' ', flush = True)

    ckpt = torch.load(args.model_path)

    model = Hier_LSTM_CRF_Classifier(len(args.tag2idx), args.emb_dim, args.tag2idx['<start>'], args.tag2idx['<end>'], args.tag2idx['<pad>'], vocab_size = len(args.word2idx), word_emb_dim = args.word_emb_dim, pretrained = args.pretrained, device = args.device).to(args.device)

    model.load_state_dict(ckpt['state_dict'])    

    print('Done')

    pred = infer_step(model, x)

    idx2tag = {v: k for (k, v) in args.tag2idx.items()}

    print('Saving predictions ...', end = ' ', flush = True)    
    with open(args.save_path, 'w') as fp:
        for idx, doc in enumerate(idx_order):
            print(doc, end = '\t', file = fp)
            p = list(map(lambda x: idx2tag[x], pred[idx]))
            print(*p, sep = ',', file = fp)
    print('Done')

if __name__ == '__main__':
	main()
