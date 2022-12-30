import torch

from dataloaders import*
from util import*
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm




class DG_Semantic():
    def __init__(self, train_loader, test_loader, model,args , dataset):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args
        self.device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.decay = self.args['mv_avg_decay']        
        self.num_tsk = len(train_loader)
        self.n_class = self.args['num_classes']

        self.num_tasks = self.num_tsk
        self.centroids = torch.zeros(self.num_tasks,self.n_class, 256)

        self.CEloss, self.MSEloss, self.BCEloss = nn.CrossEntropyLoss(reduction='none'), nn.MSELoss(reduction='none'), nn.BCEWithLogitsLoss(reduction='mean')
        self.cudable = True
        if self.cudable:
           self.CEloss, self.MSEloss, self.BCEloss = self.CEloss.cuda(), self.MSEloss.cuda(), self.BCEloss.cuda()
           self.centroids = self.centroids.cuda()

        self.num_tasks = self.num_tsk

        self.alpha = np.ones((self.num_tsk, self.num_tsk)) * (0.1 / (self.num_tsk - 1))
        np.fill_diagonal(self.alpha, 0.9)

        self.lr = args['lr']
        self.c3_value = args['c3']

        self.down_period = self.args['down_period']
        self.lr_decay_rate = self.args['lr_decay_rate']

        # For different dataset we use different network as feature extractor
        self.FE = model[0]
        self.hypothesis = model[1].to(self.device)
       

        self.FE = self.FE.to(self.device)
        print (self.FE)
        print (self.hypothesis)

    
    def model_fit(self,epoch):

        
        best_acc = 0
        lamb =0.1
        if ((epoch+1)%self.down_period) ==0 and (self.lr>5e-5) :
            self.lr = self.lr*self.lr_decay_rate

        self.optimizer = optim.Adam(list(self.FE.parameters()) + list(self.hypothesis.parameters()),
                                       lr=self.lr,  weight_decay = 1e-5)

        print('======Epoch '+str(epoch)+'===== LR is:'+str(self.lr))


        train_loader = self.train_loader  
        test_loader = self.test_loader

        semt_distnc_mtrx = np.zeros((self.num_tsk, self.num_tsk))

        weigh_loss_hypo_vlue, correct_hypo = np.zeros(self.num_tsk), np.zeros(self.num_tsk)
        Total_loss = 0
        n_batch = 0

        self.FE.train()
        self.hypothesis.train()

        for tasks_batch in zip(*train_loader):
            Loss_1, Loss_2 = 0, 0
            semantic_loss = 0
            n_batch += 1


            inputs = torch.cat([batch[0] for batch in tasks_batch])

            btch_sz = len(tasks_batch[0][0])
            labels = torch.cat([batch[1] for batch in tasks_batch])

            # inputs = (x1,...,xT)  targets = (y1,...,yT)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            features = self.FE(inputs)
            features = features.view(features.size(0), -1)

            _, fc1_s, fc2_s,_ = self.hypothesis(features)

            ce = nn.CrossEntropyLoss(reduction='none')(fc2_s, labels)
            Loss_1 = torch.mean(ce, dim=0, keepdim=True)


            for t in range(self.num_tsk):





                for k in range(t + 1, self.num_tsk):


                    sem_fea_t = fc1_s[t * btch_sz:(t + 1) * btch_sz]
                    sem_fea_k = fc1_s[k * btch_sz:(k + 1) * btch_sz]

                        
                    labels_t = labels[t * btch_sz:(t + 1) * btch_sz]
                    labels_k = labels[k * btch_sz:(k + 1) * btch_sz]

                    _,d = sem_fea_t.shape


                    ones = torch.ones_like(labels_t, dtype=torch.float)
                    zeros = torch.zeros(self.n_class)
                    if self.cudable:
                        zeros = zeros.cuda()
                    # smaples per class
                    t_n_classes = zeros.scatter_add(0, labels_t, ones)
                    k_n_classes = zeros.scatter_add(0, labels_k, ones)

                    # image number cannot be 0, when calculating centroids
                    ones = torch.ones_like(t_n_classes)
                    t_n_classes = torch.max(t_n_classes, ones)
                    k_n_classes = torch.max(k_n_classes, ones)

                    # calculating centroids, sum and divide
                    zeros = torch.zeros(self.n_class, d)
                    if self.cudable:
                        zeros = zeros.cuda()
                    t_sum_feature = zeros.scatter_add(0, torch.transpose(labels_t.repeat(d, 1), 1, 0), sem_fea_t)
                    k_sum_feature = zeros.scatter_add(0, torch.transpose(labels_k.repeat(d, 1), 1, 0), sem_fea_k)
                    current_t_centroid = torch.div(t_sum_feature, t_n_classes.view(self.n_class, 1))
                    current_k_centroid = torch.div(k_sum_feature, k_n_classes.view(self.n_class, 1))

                    # Moving Centroid
                    decay = self.decay
                    t_centroid = (1-decay) * self.centroids[t] + decay * current_t_centroid
                    k_centroid = (1-decay) * self.centroids[k] + decay * current_k_centroid
                        
                    s_loss = self.MSEloss(t_centroid, k_centroid)
                    semantic_loss += torch.mean(s_loss)


                    self.centroids[t] = t_centroid.detach()
                    self.centroids[k]= k_centroid.detach()


                    semt_distnc_mtrx[t, k] +=  torch.mean(s_loss).item()    


            Loss_2 = torch.mean(semantic_loss) 
        
            Loss =  Loss_1+ lamb * Loss_2* (1.0 / self.num_tsk) 
            self.optimizer.zero_grad()
            Loss.backward(retain_graph=True)
            self.optimizer.step()
            Total_loss += Loss.item()



        return  Total_loss
        
    def model_eval(self,epoch,mode = 'test'):
        print('=========testing=============')
        num_batches = len(self.test_loader)
        total_acc_t = 0

        iter_t = iter(self.test_loader)

        for x_t, y_t in tqdm(iter_t,leave=False, total=len(self.test_loader)):
            x_t, y_t = x_t.cuda(), y_t.cuda()
                            
            # Then we shall test the test results on the target domain
            self.FE.eval()
            self.hypothesis.eval()

                    
            with torch.no_grad():
                        
                latent = self.FE(x_t)
                _,_,_, out1 = self.hypothesis(latent)

            total_acc_t    += (out1.max(1)[1] == y_t).float().mean().item()

        acc_t = 100.0* total_acc_t/num_batches

        print('\t ======== EPOCH:{}'.format(epoch))

        print('     Mean acc on target domain is ', acc_t)

        return acc_t
             
    




