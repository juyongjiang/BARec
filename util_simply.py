import sys
import copy
import random
import numpy as np
import multiprocessing
import time
import os
from collections import defaultdict
from metrics import *
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from tqdm import tqdm
import copy
import time

# Ks = [1, 5, 10, 20, 50, 100]
Ks = [5, 10]
cores = multiprocessing.cpu_count() #// 2

# Set color for tqdm
def set_color(log, color, highlight=True):
    color_set = ['black', 'red', 'green', 'yellow', 'blue', 'pink', 'cyan', 'white']
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = '\033['
    if highlight:
        prev_log += '1;3'
    else:
        prev_log += '0;3'
    prev_log += str(index) + 'm'
    return prev_log + log + '\033[0m'

def record_loss(loss_file, loss_left, kl_loss, loss):
    with open(loss_file, 'a') as f:
        line = "{:.4f}\t{:.4f}\t{:.4f}\n".format(loss_left, kl_loss, loss)
        f.write(line) 

def load_file_and_sort(filename, reverse=False, augdata=None, aug_num=0, M=10):
    data = defaultdict(list)
    max_uind = 0
    max_iind = 0
    with open(filename, 'r') as f:
        for line in f:
            one_interaction = line.rstrip().split("\t") # user, item, timestamp
            uind = int(one_interaction[0]) + 1
            iind = int(one_interaction[1]) + 1
            max_uind = max(max_uind, uind)
            max_iind = max(max_iind, iind)
            t = float(one_interaction[2])
            data[uind].append((iind, t)) #{uind:[(iind, t),...],...}
    
    # data info
    print('data users: ', max_uind)
    print('data items: ', max_iind)
    print('data instances: ', sum([len(ilist) for _, ilist in data.items()]))

    # use pseudo-prior items to augment original sequences
    if augdata:
        for u, ilist in augdata.items():
            sorted_interactions = sorted(ilist, key=lambda x:x[1]) # sorted by timestamp
            # M is the threshold of short sequences
            for i in range(min(aug_num, len(sorted_interactions))): # extract the needed length of aug_items
                if len(data[u]) >= M: continue
                data[u].append((sorted_interactions[i]))
        print('After augmentation:')
        print('data users: ', max_uind)
        print('data items: ', max_iind)
        print('data instances: ', sum([len(ilist) for user, ilist in data.items()])) # bigger than original

    sorted_data = {}
    for u, i_list in data.items():
        if not reverse: # fine-tune
            sorted_interactions = sorted(i_list, key=lambda x:x[1])
        else: # pre-train
            sorted_interactions = sorted(i_list, key=lambda x:x[1], reverse=True)
        seq = [interaction[0] for interaction in sorted_interactions] # delete timestamp info
        sorted_data[u] = seq # {uind:[iind,...],...}

    return sorted_data, max_uind, max_iind


def augdata_load(aug_filename):
    augdata = defaultdict(list)
    with open(aug_filename, 'r') as f:
        for line in f:
            one_interaction = line.rstrip().split("\t")
            uind = int(one_interaction[0]) + 1
            iind = int(one_interaction[1]) + 1
            t = float(one_interaction[2])
            augdata[uind].append((iind, t)) #{uind:[(iind,t),...],...}

    return augdata


def data_load(data_name, args):
    reverseornot = args.reversed == 1 # 1: pre-training, 0: fine-tuning

    train_file = f"./data/{data_name}/train.txt"
    valid_file = f"./data/{data_name}/valid.txt"
    test_file = f"./data/{data_name}/test.txt"

    original_train = None
    augdata = None
    original_train, _, _ = load_file_and_sort(train_file) # {uind:[iind,...],...}
    if args.aug_traindata > 0: # fine-tune
        aug_data_signature = './aug_data/{}/lr_{}_maxlen_{}_hsize_{}_nblocks_{}_drate_{}_l2_{}_nheads_{}_gen_num_'.format(
                               args.dataset, 
                               args.lr, 
                               args.maxlen, 
                               args.hidden_units, 
                               args.num_blocks, 
                               args.dropout_rate, 
                               args.l2_emb, 
                               args.num_heads)

        if os.path.exists(aug_data_signature + '20_M_20.txt'):
            augdata = augdata_load(aug_data_signature + '20_M_20.txt')
            print('load generated items: ', aug_data_signature + '20_M_20.txt')
            time.sleep(3)

    if args.aug_traindata > 0: # fine-tune
        user_train, train_usernum, train_itemnum = load_file_and_sort(train_file, reverse=reverseornot, augdata=augdata, aug_num=args.aug_traindata, M=args.M)
    else:
        user_train, train_usernum, train_itemnum = load_file_and_sort(train_file, reverse=reverseornot) # pre-train
    
    # valid and test files processing
    user_valid, valid_usernum, valid_itemnum = load_file_and_sort(valid_file, reverse=reverseornot)
    user_test, test_usernum, test_itemnum = load_file_and_sort(test_file, reverse=reverseornot)

    usernum = max([train_usernum, valid_usernum, test_usernum])
    itemnum = max([train_itemnum, valid_itemnum, test_itemnum])

    print("data users: {} data items: {} train users: {} valid users: {} test users {}".format(
           usernum, itemnum, len(user_train), len(user_valid), len(user_test)))

    return [user_train, user_valid, user_test, original_train, usernum, itemnum] # {uind:[iind,...],...}


# recursive generation
def data_augment(model, dataset, args, sess, gen_num):
    [train, valid, test, original_train, usernum, itemnum] = copy.deepcopy(dataset)
    all_users = list(train.keys()) # users from train
    cumulative_preds = defaultdict(list)

    items_idx_set = set([i for i in range(itemnum)])
    for num_ind in range(gen_num):
        batch_seq = []
        batch_u = []
        batch_item_idx = []
        
        # print('Gen %d/%d:' % (num_ind+1, gen_num)) # 
        for u_ind, u in enumerate(tqdm(all_users, total=len(all_users), ncols=100, desc=set_color(f"Gen {num_ind+1}/{gen_num}", 'green'))):
            # if u is not existed, return []
            u_data = train.get(u, []) + valid.get(u, []) + test.get(u, []) + cumulative_preds.get(u, []) # [i1,i2,...] + [in, ...] + []

            if len(u_data) == 0 or len(u_data) >= args.M: continue

            # align max sequnece length
            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1 # max index
            for i in reversed(u_data):
                if idx == -1: break
                seq[idx] = i
                idx -= 1
    
            # unique items
            rated = set(u_data)
            item_idx = list(items_idx_set - rated) # the items which are not existed in sequences

            batch_seq.append(seq)
            batch_item_idx.append(item_idx)
            batch_u.append(u)

            if (u_ind + 1) % (args.batch_size*16) == 0 or u_ind + 1 == len(all_users):
                predictions = model.predict(sess, batch_u, batch_seq)
                # re-assign generate top-1 items as pseudo-prior items
                for batch_ind in range(len(batch_item_idx)):
                    test_item_idx = batch_item_idx[batch_ind]

                    test_predictions = predictions[batch_ind][test_item_idx] # select items prob
                    ranked_items_ind = list((-1*np.array(test_predictions)).argsort()) # get sorted index
                    rankeditem_oneuserids = int(test_item_idx[ranked_items_ind[0]])

                    u_batch_ind = batch_u[batch_ind]
                    cumulative_preds[u_batch_ind].append(rankeditem_oneuserids) 

                batch_seq = []
                batch_item_idx = []
                batch_u = []

    return cumulative_preds # {user:[i1, i2, i3, ...], ...}


# rankeditems_list, test_indices, scale_pred_list, test_allitems
def eval_one_interaction(x):
    results = {
            "precision": np.zeros(len(Ks)),
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "hit_ratio": np.zeros(len(Ks)),
            "auc": 0.,
            "mrr": 0.,
    }
    rankeditems = np.array(x[0]) # pred items
    # print("########check the ranked items shape:", rankeditems.shape)
    test_ind = x[1]
    scale_pred = x[2]
    test_item = x[3] # gt
    
    r = np.zeros_like(rankeditems)
    r[rankeditems==test_ind] = 1
    if len(r) != len(scale_pred):
        r = rank_corrected(r, len(r)-1, len(scale_pred))
    gd_prob = np.zeros_like(rankeditems)
    gd_prob[test_ind] = 1

    for ind_k in range(len(Ks)):
        results["precision"][ind_k] += precision_at_k(r, Ks[ind_k])
        results["recall"][ind_k] += recall(rankeditems, [test_ind], Ks[ind_k])
        results["ndcg"][ind_k] += ndcg_at_k(r, Ks[ind_k], 1)
        results["hit_ratio"][ind_k] += hit_at_k(r, Ks[ind_k])
    results["auc"] += auc(gd_prob, scale_pred)
    results["mrr"] += mrr(r)

    return results


def rank_corrected(r, m, n):
    pos_ranks = np.argwhere(r==1)[:,0]
    corrected_r = np.zeros_like(r)
    for each_sample_rank in list(pos_ranks):
        corrected_rank = int(np.floor(((n-1)*each_sample_rank)/m))
        if corrected_rank >= len(corrected_r) - 1:
            continue
        corrected_r[corrected_rank] = 1
    assert np.sum(corrected_r) <= 1
    return corrected_r

def init_metrics():
    metrics_dict = {
            "precision": np.zeros(len(Ks)),
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "hit_ratio": np.zeros(len(Ks)),
            "auc": 0.,
            "mrr": 0.,
    }
    return metrics_dict

def evaluate(model, dataset, args, sess, testorvalid):
    [train, valid, test, original_train, usernum, itemnum] = copy.deepcopy(dataset)
    results = init_metrics()

    rs = []

    if testorvalid == "test":
        eval_data = test
    else:
        eval_data = valid
    num_valid_interactions = 0
    pool = multiprocessing.Pool(cores)
    print(f"{cores} cores are used for evaluation!")

    all_predictions_results = []
    all_item_idx = []
    all_u = []

    batch_seq = []
    batch_u = []
    batch_item_idx = []

    u_ind = 0
    items_idx_set = set([i for i in range(1, itemnum+1)]) # [1, 2, 3, ..., itemnum]
    
    for u, i_list in eval_data.items():
        # for debug
        # if u_ind == 100:
        #     print("Only use 100 user data to evaluation!")
        #     break
        
        u_ind += 1
        if len(train[u]) < 1 or len(eval_data[u]) < 1: continue
        
        # get unique items and remove them in the train sequential data
        rated = set(train[u])
        rated.add(0)
        if testorvalid == "test":
            valid_set = set(valid.get(u, []))
            rated = rated | valid_set
        
        # align max sequence length
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        if testorvalid == "test":
            if u in valid:
                for i in reversed(valid[u]): # for test stage, put valid sequences as part of input
                    if idx == -1: break
                    seq[idx] = i
                    idx -= 1
        for i in reversed(train[u]):
            if idx == -1: break
            seq[idx] = i
            idx -= 1
        
        item_idx = [i_list[0]] # valid or gt, single item
        if args.evalnegsample == -1:
            item_idx += list(items_idx_set - rated - set([i_list[0]])) # set operation
        else:
            item_candiates = list(items_idx_set - rated - set([i_list[0]]))
            if args.evalnegsample >= len(item_candiates):
                item_idx += item_candiates
            else:
                item_idx += list(np.random.choice(item_candiates, size=args.evalnegsample, replace=False))

        batch_seq.append(seq)
        batch_item_idx.append(item_idx)
        batch_u.append(u)

        if len(batch_u) % (args.batch_size*16) == 0 or u_ind == len(eval_data):
            predictions = model.predict(sess, batch_u, batch_seq)
            for pred_ind in range(predictions.shape[0]):
                all_predictions_results.append(predictions[pred_ind])
                all_item_idx.append(batch_item_idx[pred_ind])
                all_u.append(batch_u[pred_ind])

            batch_seq = []
            batch_item_idx = []
            batch_u = []

    print("Stage 1: finished all predictions!")
    
    # calculate the evaluation results
    rankeditems_list = []
    test_indices = []
    scale_pred_list = []
    test_allitems = []

    all_predictions_results_output = []
    for ind in range(len(all_predictions_results)):
        test_item_idx = all_item_idx[ind]
        unk_predictions = all_predictions_results[ind][test_item_idx] # predicted items prob for test_item_idx
        
        # for debug
        if ind < 3:
            print("check the prediction prob.", unk_predictions)

        scaler = MinMaxScaler()
        scale_pred = list(np.transpose(scaler.fit_transform(np.transpose(np.array([unk_predictions]))))[0])

        rankeditems_list.append(list((-1*np.array(unk_predictions)).argsort())) # raned items index
        test_indices.append(0) # label index
        test_allitems.append(test_item_idx[0]) # label item id
        scale_pred_list.append(scale_pred) # normalized prob for label
        
        # test or valid stage with original dataset
        # if 'aug' in args.dataset or 'itemco' in args.dataset or args.aug_traindata > 0:
        #     real_train = original_train
        # else:
        #     real_train = train

        rankeditem_oneuserids = [int(test_item_idx[i]) for i in list((-1*np.array(unk_predictions)).argsort())] # ranked item ids
        rankeditem_scores = [unk_predictions[i] for i in list((-1*np.array(unk_predictions)).argsort())] # corresponding scores

        # {user: user_id, label: item_id, predicted: item_ids}
        one_pred_result = {"u_ind": int(all_u[ind]), "u_pos_gd": int(test_item_idx[0])}
        if args.evalnegsample == -1:
            one_pred_result["predicted"] = [int(item_id_pred) for item_id_pred in rankeditem_oneuserids]
        else:
            one_pred_result["predicted"] = [int(item_id_pred) for item_id_pred in rankeditem_oneuserids[:args.evalnegsample]]
        all_predictions_results_output.append(one_pred_result)
    
    print("Stage 2: finsihed results processing!")

    # overall 
    batch_data = zip(rankeditems_list, test_indices, scale_pred_list, test_allitems)
    batch_result = pool.map(eval_one_interaction, batch_data)
    for re in batch_result:
        results["precision"] += re["precision"]
        results["recall"] += re["recall"]
        results["ndcg"] += re["ndcg"]
        results["hit_ratio"] += re["hit_ratio"]
        results["auc"] += re["auc"]
        results["mrr"] += re["mrr"]
    results["precision"] /= len(eval_data)
    results["recall"] /= len(eval_data)
    results["ndcg"] /= len(eval_data)
    results["hit_ratio"] /= len(eval_data)
    results["auc"] /= len(eval_data)
    results["mrr"] /= len(eval_data)
    print(f"testing #of users: {len(eval_data)}")
    
    print("Stage 3: finish metrics calculation!")
    
    return results, None, None, None, None, None, all_predictions_results_output