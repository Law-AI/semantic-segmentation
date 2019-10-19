import string
from collections import defaultdict

'''
    This function constructs folds that have a balanced category distribution.
    Folds are stacked up together to give the order of docs in the main data.
    
    idx_order defines the order of documents in the data. Each sequence of (docs_per_fold) documents in idx_order can be treated as a single fold, containing documents balanced across each category.
'''
def prepare_folds(args):
    with open(args.cat_path) as fp:
        
        categories = []
        for line in fp:
            _, docs = line.strip().split('\t')
            docs = docs.strip().split(' ')
            categories.append(docs)

    # categories: list[category, docs_per_category]

    categories.sort(key = lambda x: len(x))
    n_docs = len(sum(categories, []))
    assert n_docs == args.dataset_size, "invalid category list"
           
    docs_per_fold = args.dataset_size // args.num_folds   
    folds = [[] for f in range(docs_per_fold)]
    
    # folds: list[num_folds, docs_per_fold]
    
    f = 0
    for cat in categories:
        for doc in cat:
            folds[f].append(doc)
            f = (f + 1) % 5

    # list[num_folds, docs_per_fold] --> list[num_folds * docs_per_fold]
    idx_order = sum(folds, [])
    return idx_order

'''
    This file prepares the numericalized data in the form of lists, to be used in training mode.
    idx_order is the order of documents in the dataset.

        x:  list[num_docs, sentences_per_doc, words_per_sentence]       if pretrained = False
            list[num_docs, sentences_per_doc, sentence_embedding_dim]   if pretrained = True
        y:  list[num_docs, sentences_per_doc]
'''
def prepare_data(idx_order, args):
    x, y = [], []

    word2idx = defaultdict(lambda: len(word2idx))
    tag2idx = defaultdict(lambda: len(tag2idx))

    # map the special symbols first
    word2idx['<pad>'], word2idx['<unk>'] = 0, 1
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

'''
    This file prepares the numericalized data in the form of lists, to be used in inference mode.
    idx_order is the order of documents in the dataset.

        x:  list[num_docs, sentences_per_doc, words_per_sentence]       if pretrained = False
            list[num_docs, sentences_per_doc, sentence_embedding_dim]   if pretrained = True
'''
def prepare_data_inference(idx_order, args, sent2vec_model):
    x = []

    # iterate over documents
    for doc in idx_order:
        doc_x = []

        with open(args.data_path + doc + '.txt') as fp:
            
            # iterate over sentences
            for sent in fp:
                sent_x = sent.strip()

                # cleanse text, map words and tags
                if not args.pretrained:
                    sent_x = sent_x.lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
                    sent_x = list(map(lambda x: args.word2idx[x] if x in args.word2idx else args.word2idx['<unk>'], sent_x.split()))
                else:
                    sent_x = sent2vec_model.embed_sentence(sent_x).flatten().tolist()[:args.emb_dim]
                
                if sent_x != []:
                    doc_x.append(sent_x)
                    
        x.append(doc_x)

    return x
    
