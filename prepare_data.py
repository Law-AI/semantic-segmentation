import string
from collections import defaultdict

def prepare_folds():
    with open('data/categories.txt') as fp:
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


def prepare_dataset_text():
    x, y = [], []

    word2idx = defaultdict(lambda: len(word2idx))
    tag2idx = defaultdict(lambda: len(tag2idx))

    # map the special symbols first
    word2idx['<pad>'] = 0
    tag2idx['<pad>'], tag2idx['<start>'], tag2idx['<end>'] = 0, 1, 2

    # iterate over documents
    for doc in folds:
        doc_x, doc_y = [], []

        with open('data/features/' + doc + '.txt') as fp:
            
            # iterate over sentences
            for sent in fp:
                try:
                    sent_x, sent_y, _ = sent.strip().split('$$$', 2)
                except ValueError:
                    continue

                # cleanse text, map words and tags
                sent_x = sent_x.strip().lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
                sent_x = list(map(lambda x: word2idx[x], sent_x.split()))

                sent_y = tag2idx[sent_y.strip()]

                if sent_x != []:
                    doc_x.append(sent_x)
                    doc_y.append(sent_y)
        
        x.append(doc_x)
        y.append(doc_y)

    return x, y, word2idx, tag2idx


def prepare_dataset_feat():
    x, y = [], []

    tag2idx = defaultdict(lambda: len(tag2idx))

    # map the special symbols first
    tag2idx['<pad>'], tag2idx['<start>'], tag2idx['<end>'] = 0, 1, 2

    # iterate over documents
    for doc in folds:
        doc_x, doc_y = [], []

        with open('data/features_2008/' + doc + '.txt') as fp:
            
            # iterate over sentences
            for sent in fp:
                try:
                    _, sent_y, sent_x = sent.strip().split('$$$', 2)
                except ValueError:
                    continue

                # map to feature vector of size NUM_FEATS
                sent_x = list(map(float, sent_x.strip().split('$$$')[: NUM_FEATS]))
                sent_y = sent_y.strip()
                
                # merge Fact_Procedural and Fact_Events
                if sent_y in ['Fact_Procedural', 'Fact_Events']:
                    sent_y = 'Facts' 
                # merge Issue into Ratio of the decision
                if sent_y == 'Issue':
                    sent_y = 'Ratio of the decision'
                sent_y = tag2idx[sent_y]

                if sent_x != []:
                    doc_x.append(sent_x)
                    doc_y.append(sent_y)
        
        x.append(doc_x)
        y.append(doc_y)

    return x, y, tag2idx

def prepare_dataset_emb(folds):
    x, y = [], []

    tag2idx = defaultdict(lambda: len(tag2idx))

    # map the special symbols first
    tag2idx['<pad>'], tag2idx['<start>'], tag2idx['<end>'] = 0, 1, 2

    # iterate over documents
    for doc in folds:
        doc_x, doc_y = [], []

        with open('data/embeddings/' + doc + '.txt') as fp:
            
            # iterate over sentences
            for sent in fp:
                try:
                    sent_x, sent_y = sent.strip().split('$$$')
                except ValueError:
                    continue

                # map to sentence embedding of dimension DIM_EMBS
                sent_x = list(map(float, sent_x.strip().split()[: 100]))
                sent_y = sent_y.strip()
                
                # merge Fact_Procedural and Fact_Events
                if sent_y in ['Fact_Procedural', 'Fact_Events']:
                    sent_y = 'Facts' 
                # merge Issue into Ratio of the decision
                if sent_y == 'Issue':
                    sent_y = 'Ratio of the decision'
                sent_y = tag2idx[sent_y]

                if sent_x != []:
                    doc_x.append(sent_x)
                    doc_y.append(sent_y)
        
        x.append(doc_x)
        y.append(doc_y)

    return x, y, tag2idx

