import os
import time
import argparse
import tensorflow as tf
from tqdm import tqdm
import traceback, sys
import json
import numpy as np

# self achieve
from model import Model
from util import set_color, data_load, data_augment, evaluate, record_loss


# argument
# def get_arguments():
#     parser = argparse.ArgumentParser()
#     # train
#     parser.add_argument('--aug_data_path', default='aug_data/Beauty', type=str)
#     parser.add_argument('--model_path', default='finetuned_models/Beauty', type=str)
#     parser.add_argument('--evalnegsample', default=10, type=int) # -1: full rank; 100: 100 negative samples
#     parser.add_argument('--aug_traindata', default=15, type=int)
#     parser.add_argument('--M', default=18, type=int, help='threshold of augmenation')
#     parser.add_argument('--clip_k', default=12, type=int)
#     parser.add_argument('--batch_size', default=128, type=int)
#     args = parser.parse_args()
#     return args

# argument
def get_arguments():
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--dataset', default='Beauty', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--evalnegsample', default=-1, type=int)
    parser.add_argument('--del_num', default=0, type=int)
    parser.add_argument('--loss_file', default='fine_tuning_loss.txt', type=str)
    # model
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    # hyper
    parser.add_argument('--lambda_coef', default=1.0, type=float)
    parser.add_argument('--alpha_coef', default=1.0, type=float)
    parser.add_argument('--clip_k', default=12, type=int)
    # aug
    parser.add_argument('--reversed', default=0, type=int)
    parser.add_argument('--reversed_gen_number', default=-1, type=int)
    parser.add_argument('--M', default=50, type=int, help='threshold of augmenation')
    parser.add_argument('--reversed_pretrain', default=-1, type=int, help='indicate whether reversed-pretrained model existing, -1=no and 1=yes')
    parser.add_argument('--aug_traindata', default=-1, type=int)
    parser.add_argument('--pre_trained', default=1, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_arguments()
    # process arguments
    # args.reversed = 0
    # args.reversed_pretrain = 1
    # args.alpha_coef = 1.0
    # args.dataset = args.aug_data_path.split('/')[1]
    # args_list = os.listdir(args.aug_data_path)[0].split('_')[1:-5:2]
    # [args.lr, args.maxlen, args.hidden_units, args.num_blocks, args.dropout_rate, args.l2_emb, args.num_heads] = [float(args_list[0]), 
    #                                                                                                               int(args_list[1]), 
    #                                                                                                               int(args_list[2]), 
    #                                                                                                               int(args_list[3]), 
    #                                                                                                               float(args_list[4]), 
    #                                                                                                               float(args_list[5]), 
    #                                                                                                               int(args_list[6])]
    for arg, value in vars(args).items():
        print(f'{arg}: {value}')
        
    # load augmented dataset
    dataset = data_load(args.dataset, args) 
    [user_train, user_valid, user_test, original_train, usernum, itemnum] = dataset # {uind:[iind,...],...}
   
    # data information
    cc = []
    for u in user_train:
        cc.append(len(user_train[u]))  # len statis
    cc = np.array(cc)
    print('average sequence length: %.2f' % np.mean(cc))
    print('min seq length: %.2f' % np.min(cc)) # 3
    print('max seq length: %.2f' % np.max(cc)) 
    print('quantile 25 percent: %.2f' % np.quantile(cc, 0.25))
    print('quantile 50 percent: %.2f' % np.quantile(cc, 0.50))
    print('quantile 75 percent: %.2f' % np.quantile(cc, 0.75))

    # signature
    config_signature = 'lr_{}_maxlen_{}_hsize_{}_nblocks_{}_drate_{}_l2_{}_nheads_{}'.format(
                        args.lr,
                        args.maxlen,
                        args.hidden_units,
                        args.num_blocks,
                        args.dropout_rate,
                        args.l2_emb,
                        args.num_heads)

    model_signature = '{}_gen_num_{}'.format(config_signature, 50)

    ## create a new session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    # load model
    model = Model(usernum, itemnum, args)
    saver = tf.train.Saver()

    # load fine-tuned model
    # model_files = os.listdir(args.model_path)
    # model_files.remove('checkpoint')
    # model_path = model_files[0].split('.ckpt')[0]
    model_path = './finetuned_models/'+args.dataset+'/'+model_signature+'.ckpt'
    saver.restore(sess, model_path) # f'{args.model_path}/{model_path}.ckpt')
    print(f'Fine-tuned model loaded {model_path}')

    try:
        t0 = time.time()
        # test model stage                        
        print("start testing...")
        t_test, t_test_short_seq, t_test_short37_seq, \
        t_test_short720_seq, t_test_medium2050_seq, \
        t_test_long_seq, test_rankitems = evaluate(model, dataset, args, sess, "test")

        RANK_RESULTS_DIR = f"./rank_results/{args.dataset}"
        if not os.path.isdir(RANK_RESULTS_DIR):
            os.makedirs(RANK_RESULTS_DIR)
        file_prefix = model_path.split('/')[-1].split('.ckpt')[0]
        rank_test_file = f'{RANK_RESULTS_DIR}/{file_prefix}_predictions.json'
        with open(rank_test_file, 'w') as f:
            for eachpred in test_rankitems:
                f.write(json.dumps(eachpred) + '\n')

        print('evaluation time: %fs' % (time.time() - t0))
        print(' testall: ' + str(t_test))
        if args.evalnegsample != -1:
            print(' testshort: ' + str(t_test_short_seq))
            print(' testshort37: ' + str(t_test_short37_seq))
            print(' testshort720: ' + str(t_test_short720_seq))
            print(' testmedium2050: ' + str(t_test_medium2050_seq))
            print(' testlong: ' + str(t_test_long_seq))
    except Exception as e:
        print(e)
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)
        exit(1)
