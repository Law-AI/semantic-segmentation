import string
from collections import defaultdict

def prepare_folds():
    with open('categories.txt') as fp:
        categories = []
        for i, line in enumerate(fp):
            _, docs = line.strip().split('-->')
            docs = docs.strip().split(',')
            categories.append(docs)

        categories.sort(key = lambda x: len(x))
        n_docs = len(sum(categories, []))
        assert n_docs == 50, "invalid category list"
       
    folds = [[] for f in range(5)]
    f = 0
    for cat in categories:
        for doc in cat:
            folds[f].append(doc)
            f = (f + 1) % 5

    folds = sum(folds, [])
    return folds


def prepare_data(idx_order, args):
    x, y = [], []

    word2idx = defaultdict(lambda: len(word2idx))
    tag2idx = defaultdict(lambda: len(tag2idx))

    # map the special symbols first
    word2idx['<pad>'] = 0
    tag2idx['<pad>'], tag2idx['<start>'], tag2idx['<end>'] = 0, 1, 2

    # iterate over documents
    for doc in idx_order:
        doc_x, doc_y = [], [] 

        with open(args.data_path + doc + '.txt') as fp:
            
            # iterate over sentences
            for sent in fp:
                try:
                	sent_x, sent_y = sent.strip().split('\t')
                except ValueError:
                	continue

                # cleanse text, map words and tags
                if not args.pretrained:
                    sent_x = sent_x.strip().lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
                    sent_x = list(map(lambda x: word2idx[x], sent_x.split()))
                else:
                    sent_x = list(map(float, sent_x.strip().split()[:args.emb_dim]))
                sent_y = tag2idx[sent_y.strip()]

                if sent_x != []:
                    doc_x.append(sent_x)
                    doc_y.append(sent_y)
        
        x.append(doc_x)
        y.append(doc_y)

    return x, y, word2idx, tag2idx
