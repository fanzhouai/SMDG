import numpy as np
import torch


from models.model import*
from dataloaders import*
from configure import*

import argparse
from torchvision.transforms.transforms import *

from method import DG_Semantic
import argparse

parser = argparse.ArgumentParser()


parser.add_argument('--dataname',type=str, help='Which dataset to work on',default = 'pacs')
parser.add_argument('--target',type=str, help='The target domain A C P or S for PACS dataset, V L C S for VLCS dataset')
parser.add_argument('--model',type=str, help='alex or res18',default = 'alex')


parser.add_argument("--lr_fea", type = float, help="learning_rate_fea", default=1e-5)
parser.add_argument("--lr_clf", type = float, help="learning_rate_clf", default=1e-5)
parser.add_argument("--weight_cls_loss", type = float, help="weight_cls_loss", default=1)
parser.add_argument("--weight_decay", type = float, help="weight_decay", default=1e-5)




parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--dataset', default='pacs', type=str,
                    help='choose the dataset for training, pacs, office_caltech or office_home')

parser.add_argument('--initial_lr', default=1e-3, type=float)  # initial lr


parser.add_argument('--down_period', default=5, type=int)

parser.add_argument('--train_batchsize', default=64, type=int)
parser.add_argument('--test_batchsize', default=64, type=int)

parser.add_argument('--lr_decy', default=0.95, type=float)


parser.add_argument('--max_epoch', default=180, type=int)


parser.add_argument("--drift_ratio", type=float, help="ratio for label distribution shift", default=0)
parser.add_argument('--re_weighting', default= False, type=bool)
parser.add_argument('--mv_avg_decay', default= 0.3, type=float)
args = parser.parse_args()



print(args)

data_name = args.dataname

config = all_configs

params = {'fea_lr': args.lr_fea,
          'cls_lr':args.lr_clf,
        'weight_decay': args.weight_decay,
        'batch_size':args.train_batchsize,
        'test_batchsize':args.test_batchsize,
          'mv_avg_decay': args.mv_avg_decay
        }

config[data_name].update(params)







print('args for the experiments', args)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
cuda = torch.cuda.is_available()
max_epoch = args.max_epoch



n_class = config[data_name]['num_classes']
batch_size = config[data_name]['batch_size']
args_train = config[data_name]


print('GPU: {}'.format(args.gpu))

args_train['dataset'] = args.dataset
args_train['lr'] = args.initial_lr
args_train['down_period'] = args.down_period
args_train['lr_decay_rate'] = args.lr_decy

args_train['c3'] = 0.5
print( 'args train', args_train)

target_ = args.target
args_train['target_name'] = target_

source_loaders, target_loader = return_dataset(data_name, target_, 
                          data_lists= data_lists, config = config , mode= 'train', 
                          batch_size=args_train['batch_size'], test_batchsize = args_train['test_batchsize'],need_balance = True)


FE = ResNet18Fc()
cls_net =CLS(512,args_train['num_classes'])


model = [FE ,cls_net]

total_loss =[]
print('NUMBER OF SOURCE LOADERS', len(source_loaders))
DG_algo = DG_Semantic(train_loader = source_loaders, test_loader = target_loader, model = model, args =args_train, dataset = data_name)
for epoch in range(max_epoch):

    total_loss.append(DG_algo.model_fit(epoch=epoch))
    
    target_Acc = DG_algo.model_eval(epoch=epoch,mode = 'test')


    print('\t --------- The target acc is ', np.mean(target_Acc))

print('+++++++++++++++ FINISH ++++++++++++++')
