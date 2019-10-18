import argparse

from model.Hier_BiLSTM_CRF import *
from prepare_data import *
from train import *

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--pretrained', default = False, type = bool)
	parser.add_argument('--data_path', default = 'data/IndianSupremeCourt50/text/', type = str)
	parser.add_argument('--save_path', default = 'saved/', type = str) 	
	parser.add_argument('--dataset_size', default = 50, type = int)
	parser.add_argument('--num_folds', default = 5, type = int)
	parser.add_argument('--device', default = 'cuda', type = str)
	
	parser.add_argument('--batch_size', default = 32, type = int)
	parser.add_argument('--print_every', default = 10, type = int)
	parser.add_argument('--lr', default = 0.01, type = float)
	parser.add_argument('--reg', default = 0, type = float)
	parser.add_argument('--emb_dim', default = 100, type = int)	 
	parser.add_argument('--word_emb_dim', default = 100, type = int)
	parser.add_argument('--epochs', default = 300, type = int)

	parser.add_argument('--val_fold', default = 3, type = int)
	args = parser.parse_args()

	print('\nPreparing data ...', end = ' ')
	idx_order = prepare_folds()
	x, y, word2idx, tag2idx = prepare_data(idx_order, args)
	print('Done')


	print('\nInitializing model ...', end = ' ')
	
	model = Hier_LSTM_CRF_Classifier(len(tag2idx), args.emb_dim, tag2idx['<start>'], tag2idx['<end>'], tag2idx['<pad>'], vocab_size = len(word2idx), word_emb_dim = args.word_emb_dim, pretrained = args.pretrained, device = args.device).to(args.device)
	
	print('Done')

	if args.val_fold < args.num_folds:	
		print('\nEvaluating on fold', args.val_fold, '...')		
		learn(model, x, y, tag2idx, args.val_fold, args)

	else:
		print('\nCross-validation\n')
		for f in range(args.num_folds):
			print('\nEvaluating on fold', f, '...')		
			learn(model, x, y, tag2idx, f, args)

if __name__ == '__main__':
	main()
