import torch
import time
import json
import random
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

'''
    Randomly shuffle the data and divide into batches
'''
def batchify(x, y, batch_size):
    idx = list(range(len(x)))
    random.shuffle(idx)
    
    # convert to numpy array for ease of indexing
    x = np.array(x)[idx]
    y = np.array(y)[idx]
    
    i = 0
    while i < len(x):
        j = min(i + batch_size, len(x))
        
        batch_idx = idx[i : j]
        batch_x = x[i : j]
        batch_y = y[i : j]
        
        yield batch_idx, batch_x, batch_y
        
        i = j


'''
    Perform a single training step by iterating over the entire training data once. Data is divided into batches.
'''
def train_step(model, opt, x, y, batch_size):
    ## x: list[num_examples, sents_per_example, features_per_sentence]
    ## y: list[num_examples, sents_per_example]
    
    model.train()
    
    total_loss = 0
    y_pred = [] # predictions
    y_gold = [] # gold standard
    idx = [] # example index
    
    for i, (batch_idx, batch_x, batch_y) in enumerate(batchify(x, y, batch_size)):
        pred = model(batch_x)
        loss = model._loss(batch_y)        

        opt.zero_grad()
        loss.backward()
        opt.step()
        
        total_loss += loss.item()
     
        y_pred.extend(pred)
        y_gold.extend(batch_y)
        idx.extend(batch_idx)
        
    assert len(sum(y, [])) == len(sum(y_pred, [])), "Mismatch in predicted"
    
    return total_loss / (i + 1), idx, y_gold, y_pred

'''
    Perform a single evaluation step by iterating over the entire training data once. Data is divided into batches.
'''
def val_step(model, x, y, batch_size):
    ## x: list[num_examples, sents_per_example, features_per_sentence]
    ## y: list[num_examples, sents_per_example]
    
    model.train()
    
    total_loss = 0
    y_pred = [] # predictions
    y_gold = [] # gold standard
    idx = [] # example index
    
    for i, (batch_idx, batch_x, batch_y) in enumerate(batchify(x, y, batch_size)):
        pred = model(batch_x)
        loss = model._loss(batch_y)
               
        total_loss += loss.item()
     
        y_pred.extend(pred)
        y_gold.extend(batch_y)
        idx.extend(batch_idx)
        
    assert len(sum(y, [])) == len(sum(y_pred, [])), "Mismatch in predicted"
    
    return total_loss / (i + 1), idx, y_gold, y_pred

'''
    Infer predictions for un-annotated data
'''
def infer_step(model, x):
    ## x: list[num_examples, sents_per_example, features_per_sentence]
    
    model.eval()
    y_pred = model(x) # predictions
    
    return y_pred    


'''
    Report all metrics in format using sklearn.metrics.classification_report
'''
def statistics(data_state, tag2idx):
    idx, gold, pred = data_state['idx'], data_state['gold'], data_state['pred']
    
    rev_tag2idx = {v: k for k, v in tag2idx.items()}
    tags = [rev_tag2idx[i] for i in range(len(tag2idx)) if rev_tag2idx[i] not in ['<start>', '<end>', '<pad>']]
    
    # flatten out
    gold = sum(gold, [])
    pred = sum(pred, [])
    
    print(classification_report(gold, pred, target_names = tags, digits = 3))

'''
    Train the model on entire dataset and report loss and macro-F1 after each epoch.
'''
def learn(model, x, y, tag2idx, val_fold, args):
    samples_per_fold = args.dataset_size // args.num_folds

    val_idx = list(range(val_fold * samples_per_fold, val_fold * samples_per_fold + samples_per_fold))
    train_idx = list(range(val_fold * samples_per_fold)) + list(range(val_fold * samples_per_fold + samples_per_fold, args.dataset_size))
    
    train_x = [x[i] for i in train_idx]
    train_y = [y[i] for i in train_idx]
    
    val_x = [x[i] for i in val_idx]
    val_y = [y[i] for i in val_idx]
    
    opt = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.reg)
    
    print("{0:>7}  {1:>10}  {2:>6}  {3:>10}  {4:>6}".format('EPOCH', 'Tr_LOSS', 'Tr_F1', 'Val_LOSS', 'Val_F1'))
    print("-----------------------------------------------------------")
    
    best_val_f1 = 0.0
    
    model_state = {}
    data_state = {}
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):

        train_loss, train_idx, train_gold, train_pred = train_step(model, opt, train_x, train_y, args.batch_size)
        val_loss, val_idx, val_gold, val_pred = val_step(model, val_x, val_y, args.batch_size)

        train_f1 = f1_score(sum(train_gold, []), sum(train_pred, []), average = 'macro')
        val_f1 = f1_score(sum(val_gold, []), sum(val_pred, []), average = 'macro')

        if epoch % args.print_every == 0:
            print("{0:7d}  {1:10.3f}  {2:6.3f}  {3:10.3f}  {4:6.3f}".format(epoch, train_loss, train_f1, val_loss, val_f1))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model_state = {'epoch': epoch, 'arch': model, 'name': model.__class__.__name__, 'state_dict': model.state_dict(), 'best_f1': val_f1, 'optimizer' : opt.state_dict()}
            data_state = {'idx': val_idx, 'loss': val_loss, 'gold': val_gold, 'pred': val_pred}
            
    end_time = time.time()
    
    print("Dumping model and data ...", end = ' ')
    
    torch.save(model_state, args.save_path + 'model_state' + str(val_fold) + '.tar')
    
    with open(args.save_path + 'data_state' + str(val_fold) + '.json', 'w') as fp:
        json.dump(data_state, fp)
    
    print("Done")    

    print('Time taken:', int(end_time - start_time), 'secs')

    statistics(data_state, tag2idx)
