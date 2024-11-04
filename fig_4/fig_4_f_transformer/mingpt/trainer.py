"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
import numpy as np
from collections import defaultdict
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import CfgNode as CN
import gc


class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        # setup the dataloader
        self.train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def get_one_batch(self):
        data_iter = iter(self.train_loader)
        batch = next(data_iter)
        batch = [t.to(self.device) for t in batch]
        x, y = batch
        return x, y

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(self.train_loader)
        while True:
            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss, prelogits = model(x, y, pos_loss_beta=config.pos_loss_beta)
            # print(logits.shape, y.shape, prelogits.shape)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break

class Trainer_with_test:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset, test_dataset,test_trials):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.test_trials = test_trials
        # setup the dataloader
        self.train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def get_one_batch(self):
        data_iter = iter(self.train_loader)
        batch = next(data_iter)
        batch = [t.to(self.device) for t in batch]
        x, y = batch
        return x, y
    
    def corr_cal(self,x, y):
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)
        xm = x.sub(mean_x)
        ym = y.sub(mean_y)
        r_num = xm.dot(ym)
        r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
        r_val = r_num / r_den
        return r_val
    def plot_mat(self):

        model = self.model
        model.eval()
        look_back = 4
        test_prelog = []

        with torch.no_grad():

            for i in range(len(self.test_dataset)):
                print(f'in plot mat {i}')
                start = i-look_back if i-look_back>0 else 0
                inp = self.test_dataset[start:i+1]
                inp = inp.unsqueeze(0)

                # get pre-logits activations, for details see model.generate_one
                _, _, pre_log_test = model.generate_one(inp, 1, temperature=1, do_sample=False)

                test_prelog.append(pre_log_test.cpu().numpy()[0,-1,:])

        test_prelog = np.array(test_prelog)

        near_pos = []
        far_pos = []
        tr_len = 23
                
        for i,ind in enumerate(self.test_trials):
            if ind == 0:
                near_pos.append(test_prelog[i*tr_len: (i+1)*tr_len])
            else:
                far_pos.append(test_prelog[i*tr_len: (i+1)*tr_len])

        near_pos = np.mean(np.array(near_pos),0)
        far_pos = np.mean(np.array(far_pos),0)
        

        corr_matrix = np.zeros((23, 23))
        for i in range(23):
            for j in range(23):
                corr, _ = pearsonr(near_pos[i, :], far_pos[j, :])
                corr_matrix[i, j] = corr

        # convert seaborn colormap to matplotlib
        cmap = sns.color_palette("icefire", as_cmap=True)

        # plot the correlation matrix as a heatmap
        fig, ax = plt.subplots()
        im = ax.imshow(corr_matrix, cmap=cmap,vmin= -1,vmax = 1)

        # add axis labels and a colorbar
        ax.set_xlabel('Far_pos')
        ax.set_ylabel('Near_pos')
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Pearson correlation', rotation=-90, va="bottom")
        plt.title(self.iter_num)
        # display the plot
        plt.show()

        del model
        torch.cuda.empty_cache()
        # calling python's garbage collector
        gc.collect()

        return corr_matrix
        

    def run(self):
        
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(self.train_loader)
        loss_curve = []
        accuracy_curve = []
        iter_num_plot = []
        corr_matrix_big = []
        while True:
            model, config = self.model, self.config

            # setup the optimizer
            self.optimizer = model.configure_optimizers(config)

            model.train()
            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch
            #print(f'trial {trial}')

            # forward the model
            logits, self.loss, prelogits = model(x, y, pos_loss_beta=config.pos_loss_beta)
            # print('adding CorrPen')
            
            # h_1 = prelogits[0,23*4:23*5,:]
            # h_2 = prelogits[0,23*5:23*6,:]

            # cov_m = mean_centered_covariance(torch.cat([h_1, h_2]).T)
            # corr_m = covariance_to_correlation(cov_m)
            # corr_loss = corr_m.abs().sum() 

            
            # # Scale the correlation loss before adding to the total loss
            # print(f'prediction loss {self.loss.detach()} CorrPen loss {corr_loss.detach()}')

            # self.loss += 0.001 * corr_loss

        

            loss_curve.append(self.loss.detach().numpy())


            
           
        

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow
            self.model  = model

            if self.iter_num % 20 == 0:
                print(f'iteration {self.iter_num} calculate test accuracy')

                # calculate test accuracy
                reward_error = 0
                reward_count = 0
                look_back = 4
                print(f'look back {look_back}')
                test_prelog = []
                

                model.eval()
                with torch.no_grad():

                    for i in range(len(self.test_dataset)-1):

                        
                        start = i-look_back if i-look_back>0 else 0
                        inp = self.test_dataset[start:i+1]
                        inp = inp.unsqueeze(0)

                        # get pre-logits activations, for details see model.generate_one
                        logits_test, _, pre_logits = model.generate_one(inp, 1, temperature=1, do_sample=False)

                        logits_max_index = torch.argmax(logits_test) 
                        target_max_index = self.test_dataset[i+1]
                        # if target_max_index is 6 or 7, then it's a "reward" instance
                        if target_max_index == 6 or target_max_index == 7:
                            reward_count += 1

                            # if the prediction is incorrect (differs by more than 0.5), then it's a "reward error"
                            if abs(target_max_index.item() - logits_max_index.item()) > 0.5:
                                reward_error += 1


                    
                        test_prelog.append(pre_logits.numpy()[0,-1,:])
                    print(f'reward error {reward_error} reward count {reward_count}')
                    test_prelog = np.array(test_prelog)
                    near_pos = []
                    far_pos = []
                    tr_len = 23
                            
                    for p,ind in enumerate(self.test_trials):
                        if p<= len(self.test_trials) - 2 and p > 2:
                            if ind == 0:
                                near_pos.append(test_prelog[p*tr_len: (p+1)*tr_len])
                            else:
                                far_pos.append(test_prelog[p*tr_len: (p+1)*tr_len])
                    
                
                near_pos = np.mean(np.array(near_pos),0)
                far_pos = np.mean(np.array(far_pos),0)
                

                corr_matrix = np.zeros((23, 23))
                for i in range(23):
                    for j in range(23):
                        corr, _ = pearsonr(near_pos[i, :], far_pos[j, :])
                        corr_matrix[i, j] = corr
                corr_matrix_big.append(corr_matrix)

                cmap = sns.color_palette("icefire", as_cmap=True)

                fig, ax = plt.subplots()
                im = ax.imshow(corr_matrix, cmap=cmap,vmin= -1,vmax = 1)

                # add axis labels and a colorbar
                ax.set_xlabel('Far_pos')
                ax.set_ylabel('Near_pos')
                cbar = ax.figure.colorbar(im, ax=ax)
                cbar.ax.set_ylabel('Pearson correlation', rotation=-90, va="bottom")
                plt.title(self.iter_num)
                # display the plot
                plt.show()


                reward_error_rate = reward_error / reward_count
                accuracy_curve.append(reward_error_rate)
                iter_num_plot.append(self.iter_num)
                
               
               
                # deleting model and clearing cache
                del model
                torch.cuda.empty_cache()

                # calling python's garbage collector
                gc.collect()
           

            
            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
        return loss_curve, accuracy_curve,iter_num_plot,corr_matrix_big
    

def mean_centered_covariance(x):
    # Calculate covariance of mean centered data
    x_centered = x - x.mean(dim=0, keepdim=True)
    cov = x_centered.t().mm(x_centered) / (x_centered.size(0) - 1)
    return cov

def covariance_to_correlation(covariance):
    std_dev = torch.sqrt(torch.diag(covariance))
    correlation = covariance / torch.outer(std_dev, std_dev)
    return correlation
