import torch
import torch.nn as nn
from torch.optim import Adam
import math
import random
from random import randint
from tqdm import tqdm
from copy import deepcopy

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader

from models.gvf_model import gvf_model

use_cuda = torch.cuda.is_available()
if use_cuda:
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor

class GVF_learner():
    def __init__(self, parameters, dataset, policy_func):
        # parameters setting
        self.batch_size = parameters["batch size"]
        self.feature_num = parameters["feature num"]
        self.action_space = parameters["action space"]
        self.state_len = parameters["state length"]
        self.discount_factor = FloatTensor(parameters["discount factor"])
        self.soft_tau = parameters["soft_tau"]
        self.model_replace_freq = parameters["model_replace_freq"]
        self.data_length = len(dataset)
        
        #init data loader
        self.train_dl = self.load_data(dataset)
        self.policy_func = policy_func
        
        # init model
        # state input, action * feature number output
#         self.eval_model = gvf_model(self.state_len, self.action_space * self.feature_num)
#         self.target_model = gvf_model(self.state_len, self.action_space * self.feature_num)
        # state action pair input, features output
        self.input_len = self.state_len + self.action_space
        self.eval_model = gvf_model(self.input_len, self.feature_num)
        self.target_model = gvf_model(self.input_len, self.feature_num)
        if use_cuda:
            self.eval_model = self.eval_model.cuda()
            self.target_model = self.target_model.cuda()
        self.loss_func = nn.MSELoss()
        self.optimizer = Adam(self.eval_model.parameters(), lr = parameters["learning rate"])
        
    def load_data(self, dataset):
        states, actions, features, next_states, dones = [], [], [], [], []
        for data in dataset:
            s, a, _, ns, d, f = data
            states.append(s)
            actions.append(a)
            features.append(f)
            # state action needs
            ns = self.full_state_action(list(ns))
            next_states.append(ns)
            dones.append(d)
        states = FloatTensor(states)
        actions = FloatTensor(actions)
        features = FloatTensor(features)
        next_states = FloatTensor(next_states)
        dones = FloatTensor(dones)
        
        tensor_dataset = torch.utils.data.TensorDataset(states, actions, features, next_states, dones)
        train_indices = list(range(0, self.data_length))
        train_sampler = SubsetRandomSampler(train_indices)
        train_dl = DataLoader(tensor_dataset, self.batch_size, sampler=train_sampler)
        return train_dl
        
    def learn_and_eval(self, train_epochs, test_interval):
        for i in tqdm(range(train_epochs)):
            total_loss = self.learn()
            if i % self.model_replace_freq == 0:
                if self.model_replace_freq == 1:
                    self.soft_replace()
                else:
                    self.hard_replace()
            if i % test_interval == 0:
                print("train loss : {}".format(total_loss))
                self.evaluation()
        return self.eval_model
    
    def full_state_action(self, state):
        next_state_actions = []
        for a in range(self.action_space):
            action_one_hot = [0] * self.action_space
            action_one_hot[a] = 1
            next_state_actions.append(state + action_one_hot)
        return next_state_actions
    
    def learn(self):
        total_loss = 0
        count = 0
        for mini_batch in self.train_dl:
            s, a, f, ns, dones = mini_batch
            a = a.type(torch.LongTensor)
            batch_idx = LongTensor(list(range(s.size()[0])))
            
            # GVF(s)
            s_gvfs = self.eval_model(s)
            
            # actions * features needs
#             s_gvfs = s_gvfs.view(-1, self.action_space, self.feature_num)[batch_idx, a, :]
            
            # state action needs
            na = self.policy_func(ns)
            ns = ns.view(-1, self.input_len)
            ns_gvfs = self.target_model(ns).detach()
            ns_gvf = ns_gvfs.view(-1, self.action_space, self.feature_num)[batch_idx, na, :]
            
            ns_gvf[dones == 1] = 0
            target_gvf = f + self.discount_factor.view(1, -1) * ns_gvf

            loss = self.fit(s_gvfs, target_gvf)
            total_loss += loss
            count += 1
        return total_loss / count
    
    def fit(self, predict_gvf, target_gvf):
        loss = self.loss_func(predict_gvf, target_gvf)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def evaluation(self):
        pass
    
    def soft_replace(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, eval_param in zip(self.target_model.parameters(), self.eval_model.parameters()):
            target_param.data.copy_(self.soft_tau*eval_param.data + (1.0-self.soft_tau)*target_param.data)
    
    def hard_replace(self):
        self.target_model.load_state_dict(self.eval_model.state_dict())
    
    
# -------------------------- test ---------------------------
# test_PARAMETERS = {
#     "batch size" : 64,
#     "learning rate" : 0.0001,
#     "feature num" : 2,
#     "state length" : 4,
#     "discount factor" : [1, 2],
#     "action space": 4
# }
# def get_test_data(size = 100):
#     dataset = []
#     for _ in range(size):
#         state = [random.random()  for _ in range(test_PARAMETERS["state length"])]
#         action = randint(0, test_PARAMETERS["action space"] - 1)
#         reward = state[action] * action
#         feature = [state[action] * action * 1/3, state[action] * action * 2/3]
#         next_state = deepcopy(state)
#         next_state[action] += 0.2
#         done = 1 if sum(next_state) > 3 else 0
# #         dataset.append((state, action, reward, next_state, done, feature))
#         action_one_hot = [0] * test_PARAMETERS["action space"]
#         action_one_hot[action] = 1
#         state_action = state + action_one_hot
#         dataset.append((state_action, action, reward, next_state, done, feature))
        
#     return dataset
    
# def test_policy_func(state):
#     q_value = state[:, :, :test_PARAMETERS["state length"]].max(2)[1]
#     return q_value.max(1)[1]

# dataset = get_test_data()
# # print(dataset)
# gvf_learner = GVF_learner(test_PARAMETERS, dataset, test_policy_func)
# gvf_learner.learn_and_eval(1000, 100)