

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from transformer import transformer
from utils import hyperfanin_for_kernel, hyperfanin_for_bias


class APC(nn.Module):

    def __init__(self, cfg):
        super(APC, self).__init__()
        
        self.dataset = cfg.dataset
        self.epsilon = 1e-1
        
        self.T1 = cfg.T1
        self.T2 = cfg.T2
        self.T = self.T1 * self.T2
        
        self.THR2 = 1.0 - 1/cfg.factor2
        self.THR1 = 1.0 - 1/cfg.factor1
        
        self.cfg = cfg
        self.hyper = cfg.hyper
        self.a2_sz = cfg.a2_sz
        self.a1_sz = cfg.a1_sz
        
        self.BATCH_SIZE = cfg.BATCH_SIZE
        
        self.lr = cfg.lr
        
        self.use_baseline = True
        
        self.H = cfg.dims[0]
        self.W = cfg.dims[1]
        self.C = cfg.dims[2]
        
        self.g1_sz = cfg.g1_sz
        self.g2_sz = cfg.g2_sz
        self.g2_sub_sz = cfg.g2_sub_sz
        self.bottleneck = self.g1_sz * self.g1_sz * self.C
        
        self.num_classes = cfg.num_classes
        
        self.r1_sz = cfg.r1_sz
        self.r2_sz = cfg.r2_sz
        self.z_sz = cfg.z_sz
        
        self.e1_sz = cfg.r1_sz
        self.e2_sz = cfg.r2_sz
        
        self.in_fs1_sz = self.e1_sz + self.r1_sz + 2
        self.in_fa1_sz = self.a1_sz
        
        self.fa1_sz = fa_sz = [2]
        self.fs1_sz = fs_sz = [self.r1_sz]
        
        self.action_layer_collection = []
        self.state_layer_collection = []
        
        self._create_action2()
        self._create_action_hypernet()
        self._create_loc_net2()
        self._create_action_feedback()
        
        self._create_rec()
        
        self._create_state2()
        self._create_state_hypernet()
        self._create_state1_init()
        self._create_state_feedback()
        self._create_r2_init()
        self._create_decoder()
        self._create_decoder2()
        
        self._create_baseline()
        
        self.flatten_e1 = nn.Flatten()
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        self.std1 = cfg.std1
        self.std2 = cfg.std2
        
        self.activations = {}
        
    def _MSE(self, y, y_hat):
        N = self.cfg.BATCH_SIZE
        y_flat = y.view(N, -1)
        y_hat_flat = y_hat.view(N, -1)
        return torch.mean(torch.square(y_flat - y_hat_flat), dim=-1)
        
    def pred_error(self):
        error = 0.0
        for t in range(self.T):
            error += torch.mean(self.activations['pred_errors'][t])
        for t2 in range(self.T2):
            error += 1e-1 * torch.mean(self.activations['pred_errors2'][t2])
        for error_token in self.activations['mse']:
            error += torch.mean(error_token)
        return error
        
    def dense_mse_reward(self):
        T1 = self.T1
        T2 = self.T2
        T = self.T
        N = self.cfg.BATCH_SIZE
        I = self.activations['I']
        
        R = torch.zeros((N, 0), device=I.device)
        
        for t2 in range(self.T2):
            for t1 in range(self.T1):
                t = t2 * self.T1 + t1
                hat0 = self.activations['I_hat'][t]
                hat1 = self.activations['I_hat'][t+1]
                mse0 = self._MSE(I, hat0)
                mse1 = self._MSE(I, hat1)
                
                Rt = torch.unsqueeze(mse0 - mse1, dim=-1)
                R = torch.cat([R, Rt], dim=-1)

        cumulative_reward2 = torch.zeros((N, 0), device=I.device)
        cumulative_reward1 = torch.zeros((N, 0), device=I.device)
        
        for t2 in range(self.T2):
            cumulative_reward2_t = R[:, t2*T1:]
            cumulative_reward2_t = torch.sum(cumulative_reward2_t, dim=-1, keepdim=True)
            cumulative_reward2 = torch.cat([cumulative_reward2, cumulative_reward2_t], dim=-1)
            
            for t1 in range(self.T1):
                t = t2 * T1 + t1
                cumulative_reward1_t = R[:, t:t+1]
                cumulative_reward1_t = torch.sum(cumulative_reward1_t, dim=-1, keepdim=True)
                cumulative_reward1 = torch.cat([cumulative_reward1, cumulative_reward1_t], dim=-1)

        return cumulative_reward2, cumulative_reward1

    def penalty_func(self, x, c):
        pos = torch.square(F.leaky_relu(x - c))
        neg = torch.square(F.leaky_relu(-x - c))
        const = -2 * torch.square(0.2 * c)
        
        penalty = pos + neg - const
        
        penalty = torch.sum(penalty, dim=-1)
        return torch.mean(penalty)
    
    def REINFORCE(self):
        where1_log_prob = torch.zeros((self.cfg.BATCH_SIZE, 0), device=next(self.parameters()).device)
        
        for t2 in range(self.T2):
            for t1 in range(self.T1):
                t = t2 * self.T1 + t1
                loc_mu_t = self.activations['loc1_mu'][t]
                loc_t = self.activations['loc1'][t]
                
                locs_dist = Normal(loc=loc_t, scale=self.std1)
                where1_log_prob_t = torch.sum(locs_dist.log_prob(loc_mu_t), dim=-1, keepdim=True)
                where1_log_prob = torch.cat([where1_log_prob, where1_log_prob_t], dim=-1)
        
        where2_log_prob = torch.zeros((self.cfg.BATCH_SIZE, 0), device=next(self.parameters()).device)
        
        for t2 in range(self.T2):
            loc2_mu_t = self.activations['loc2_mu'][t2]
            loc2_t = self.activations['loc2'][t2]
            
            loc2_dist = Normal(loc=loc2_t, scale=self.std2)
            where2_log_prob_t = torch.sum(loc2_dist.log_prob(loc2_mu_t), dim=-1, keepdim=True)
            where2_log_prob = torch.cat([where2_log_prob, where2_log_prob_t], dim=-1)
            
        THR1 = self.THR1
        THR2 = self.THR2
        
        penalty = 0.0
        
        for t2 in range(self.T2):
            loc2 = self.activations['loc2_mu'][t2]
            penalty += self.penalty_func(loc2, THR2)
            
            for t1 in range(self.T1):
                t = t2 * self.T1 + t1
                
                loc1 = self.activations['loc1_mu'][t]
                penalty += self.penalty_func(loc1, THR1)
        
        cumulative_reward2, cumulative_reward1 = self.dense_mse_reward()
        
        baseline2 = self.activations['baseline2']
        baseline1 = self.activations['baseline1']
        
        baseline_mse = self._MSE(cumulative_reward2.detach(), baseline2)
        baseline_mse += self._MSE(cumulative_reward1.detach(), baseline1)
        baseline_mse = torch.mean(baseline_mse)
    
        advantage2 = cumulative_reward2
        advantage1 = cumulative_reward1
    
        if self.use_baseline:
            advantage2 -= baseline2
            advantage1 -= baseline1
            
        REINFORCE_loss = -torch.mean(where2_log_prob * advantage2.detach(), dim=-1)
        REINFORCE_loss -= torch.mean(where1_log_prob * advantage1.detach(), dim=-1)
        REINFORCE_loss += 1e-5 * penalty
        REINFORCE_loss = torch.mean(REINFORCE_loss)
            
        return REINFORCE_loss, baseline_mse
        
    def attempt_task(self, a2, a1):
        a2_a1 = torch.cat([a2, a1], dim=-1)
        
        I_hat = self.reconstruct(a2_a1)
        self.activations['I_hat'].append(I_hat)
        self.activations['mse'].append(self._MSE(self.activations['I'], I_hat))
        
    def forward_pass(self, inputs):
        I, y = inputs
        
        self.activations['pred_errors'] = []
        self.activations['pred_errors2'] = []
        self.activations['loc1'] = []
        self.activations['loc2'] = []
        self.activations['loc1_mu'] = []
        self.activations['loc2_mu'] = []
        self.activations['loc1_sum'] = []
        self.activations['g1'] = []
        self.activations['g2'] = []
        self.activations['g1_hat'] = []
        self.activations['g2_hat'] = []
        self.activations['g2_sub'] = []
        self.activations['g2_sub_hat'] = []
        self.activations['g1_params'] = []
        self.activations['g2_params'] = []
        self.activations['grids1'] = []
        self.activations['grids2'] = []
        
        self.activations['I'] = I
        self.activations['I_hat'] = []
        self.activations['mse'] = []
        self.activations['y_hat'] = []
        self.activations['baseline1'] = []
        self.activations['baseline2'] = []
        
        N = self.cfg.BATCH_SIZE
    
        init_loc = (torch.rand((N, 2), device=I.device) - 0.5) * 2 * 0.5
        init_loc = init_loc.detach()
        self.activations['init_loc'] = init_loc
        
        init_glimpse, init_params, init_grid = self.glimpse_network(I, [self.g1_sz, self.g1_sz], init_loc, layer=0)
        init_glimpse = init_glimpse.detach()
        
        self.activations['init_glimpse'] = init_glimpse
        self.activations['init_grid1'] = init_grid
        init_glimpse = init_glimpse.view(N, -1)
        r2_in = torch.cat([init_glimpse, init_loc], dim=-1)
        
        r2 = self.init_state2(r2_in)
        a2 = torch.zeros((N, self.a2_sz), device=I.device)
        a2 += self.update_action2(a2, r2.detach(), torch.zeros((N, self.a2_sz), device=I.device))
        
        res2 = torch.zeros((N, self.g2_sub_sz * self.g2_sub_sz * self.C), device=I.device)
        
        ###############################
        ###### MACRO CYCLE START ######
        ###############################
        
        for t2 in range(self.T2):
            loc2_mu, z2 = self.loc_net2(a2.detach())
            loc2 = loc2_mu + self.std2 * torch.randn_like(loc2_mu)
            loc2 = loc2.detach()

            self.activations['loc2_mu'].append(loc2_mu)
            self.activations['loc2'].append(loc2)
            
            fs_hyp_in = torch.cat([r2, z2.detach(), loc2.detach()], dim=-1)
            
            if self.hyper:
                fa1_weights, ln1_weights = self.generate_fa1(z2)
                fa1 = lambda x: F.leaky_relu(self.f_theta(x, fa1_weights, [self.a1_sz], self.a1_sz + self.r1_sz))
                ln1 = lambda x: self.f_theta(x, ln1_weights, [2], self.a1_sz)
                
                fs1_weights = self.generate_fs1(fs_hyp_in)
                fs1 = lambda x: F.leaky_relu(self.f_theta(x, fs1_weights, [self.r1_sz], self.r1_sz + self.bottleneck + 2))
            else:
                fa1, ln1 = self.act1(z2)
                fs1 = self.state1(fs_hyp_in)
            
            g2, g2_params, grids2 = self.glimpse_network(I, [self.g2_sz, self.g2_sz], loc2.detach(), layer=2)
            g2 = g2.detach()
            g2_sub = self.sample_g2(g2)

            self.activations['g2'].append(g2)
            self.activations['g2_sub'].append(g2_sub)
            self.activations['g2_params'].append(g2_params)
            self.activations['grids2'].append(grids2)
            
            ###############################
            ###### MICRO CYCLE START ######
            ###############################
            
            for t1 in range(self.T1):
                if t1 == 0:
                    r1 = self.init_state1(fs_hyp_in)
                    
                    a1 = torch.zeros((N, self.a1_sz), device=I.device)
                    fa1_in = torch.cat([r1.detach(), a1], dim=-1)
                    a1 += fa1(fa1_in)
                    
                    if t2 == 0:
                        self.attempt_task(a2, a1)
                        
                loc1_mu = ln1(a1.detach())
                loc1 = loc1_mu + self.std1 * torch.randn_like(loc1_mu)
                loc1 = loc1.detach()
                
                self.activations['loc1_mu'].append(loc1_mu)
                self.activations['loc1'].append(loc1)
                
                g1, g1_params, grids1 = self.glimpse_network(g2, [self.g1_sz, self.g1_sz], loc1, layer=1, extra_loc=loc2)
                g1 = g1.detach()
                
                self.activations['g1'].append(g1)
                self.activations['g1_params'].append(g1_params)
                self.activations['grids1'].append(grids1)
                
                dec_input = torch.cat([r1, loc1.detach(), loc2.detach()], dim=-1)
                g1_hat = self.decode(dec_input)
                self.activations['g1_hat'].append(g1_hat)
                
                res1 = self.flatten_e1(g1 - g1_hat)
                self.activations['pred_errors'].append(self._MSE(g1, g1_hat))
                
                fs1_in = torch.cat([loc1.detach(), r1, res1], dim=-1)
                r1 += fs1(fs1_in)
                
                fa1_in = torch.cat([r1.detach(), a1], dim=-1)
                a1 += fa1(fa1_in)
                
                r2 += self.update_state2(r2, loc2, self.state_feedback(r1))
                a2 += self.update_action2(a2, r2.detach(), self.action_feedback(a1))
            
                self.attempt_task(a2, a1)
            
            dec2_input = torch.cat([r2, loc2.detach()], dim=-1)
            g2_hat = self.decode2(dec2_input)
            
            self.activations['pred_errors2'].append(self._MSE(g2_sub, g2_hat))
            self.activations['g2_hat'].append(g2_hat)
        
        self.activations['baseline1'] = self.baseline1(torch.ones((N, 64), device=I.device))
        self.activations['baseline2'] = self.baseline2(torch.ones((N, 64), device=I.device))
        
    ###########################################################################################
    ############################## SENSOR #####################################################
    ###########################################################################################
    
    def glimpse_network(self, I, dims, loc, layer=1, extra_loc=None):
        N = I.shape[0]
        gH, gW = dims
        
        if layer == 1:
            rf = -1.0 + 1/self.cfg.factor1
        elif layer == 2:
            rf = -1.0 + 1/self.cfg.factor2
        elif layer == 0:
            rf = -1.0 + 1/self.cfg.factor2 * 1/self.cfg.factor1
        
        rot = torch.zeros((N, 1), device=I.device)
        shear = torch.zeros((N, 1), device=I.device)
        scale = rf + torch.zeros((N, 2), device=I.device)
        theta = torch.cat([loc, rot, scale, shear], dim=-1)

        glimpse, grids = transformer(I, theta, gH, gW, grids=True)
        
        return glimpse, theta, grids
            
    ###########################################################################################
    ############################## DEC NET ####################################################
    ###########################################################################################
    
    def _create_rec(self):
        self.rec1 = nn.Linear(256, 256)
        self.rec2 = nn.Linear(256, 256)
        self.rec3 = nn.Linear(256, self.H * self.W * self.C)
        
        self.action_layer_collection.append(self.rec1)
        self.action_layer_collection.append(self.rec2)
        self.action_layer_collection.append(self.rec3)
        
    def reconstruct(self, x):
        h = F.leaky_relu(self.rec1(x))
        h = F.leaky_relu(self.rec2(h))
        y = F.leaky_relu(self.rec3(h))
        return y.view(-1, self.H, self.W, self.C)
        
    def _create_decoder(self):
        self.dec11 = nn.Linear(256, 256)
        self.dec12 = nn.Linear(256, 256)
        self.dec13 = nn.Linear(256, self.g1_sz * self.g1_sz * self.C)
        
        self.state_layer_collection.append(self.dec11)
        self.state_layer_collection.append(self.dec12)
        self.state_layer_collection.append(self.dec13)
        
    def decode(self, r):
        y = self.dec13(F.leaky_relu(self.dec12(F.leaky_relu(self.dec11(r)))))
        return y.view(-1, self.g1_sz, self.g1_sz, self.C)
        
    def _create_decoder2(self):
        self.dec21 = nn.Linear(256, 256)
        self.dec22 = nn.Linear(256, 256)
        self.dec23 = nn.Linear(256, self.g2_sub_sz * self.g2_sub_sz * self.C)
        
        self.state_layer_collection.append(self.dec21)
        self.state_layer_collection.append(self.dec22)
        self.state_layer_collection.append(self.dec23)
        
    def decode2(self, r):
        y = self.dec23(F.leaky_relu(self.dec22(F.leaky_relu(self.dec21(r)))))
        return y.view(-1, self.g2_sub_sz, self.g2_sub_sz, self.C)
    
    ###########################################################################################
    ############################## ACT2 NET ###################################################
    ###########################################################################################
    
    def _create_action2(self):
        self.fa_prev = nn.Linear(256, 256)
        self.fa_feed_r = nn.Linear(256, 256)
        self.fa_feed_a1 = nn.Linear(256, 256)
        self.fa_feed_merge = nn.Linear(512, 256)
        self.fa_merge = nn.Linear(512, self.a2_sz)
        
        self.action_layer_collection.append(self.fa_prev)
        self.action_layer_collection.append(self.fa_feed_r)
        self.action_layer_collection.append(self.fa_feed_a1)
        self.action_layer_collection.append(self.fa_feed_merge)
        self.action_layer_collection.append(self.fa_merge)
        
    def update_action2(self, a2, r2, a1_feed):
        h_prev = F.leaky_relu(self.fa_prev(a2))
        h_feed_r = F.leaky_relu(self.fa_feed_r(r2))
        h_feed_a1 = F.leaky_relu(self.fa_feed_a1(a1_feed))
        h_feed = F.leaky_relu(self.fa_feed_merge(torch.cat([h_feed_r, h_feed_a1], dim=-1)))

        h = torch.cat([h_prev, h_feed], dim=-1)
        return F.leaky_relu(self.fa_merge(h))
        
    def _create_loc_net2(self):
        self.fa_loc1 = nn.Linear(128, 128)
        self.fa_loc2 = nn.Linear(128, 2)
        self.fa_z1 = nn.Linear(256, 256)
        self.fa_z2 = nn.Linear(256, 256)
        self.fa_z3 = nn.Linear(256, self.z_sz)
        
        # Initialize fa_loc2 with zeros
        nn.init.zeros_(self.fa_loc2.weight)
        nn.init.zeros_(self.fa_loc2.bias)
        
        # Initialize fa_z3 with zeros
        nn.init.zeros_(self.fa_z3.weight)
        nn.init.zeros_(self.fa_z3.bias)
        
        self.action_layer_collection.append(self.fa_loc1)
        self.action_layer_collection.append(self.fa_loc2)
        self.action_layer_collection.append(self.fa_z1)
        self.action_layer_collection.append(self.fa_z2)
        self.action_layer_collection.append(self.fa_z3)
        
    def loc_net2(self, a2):
        return F.leaky_relu(self.fa_loc2(F.leaky_relu(self.fa_loc1(a2)))), self.fa_z3(F.leaky_relu(self.fa_z2(F.leaky_relu(self.fa_z1(a2)))))
        
    ###########################################################################################
    ############################## STATE2 NET #################################################
    ###########################################################################################
        
    def _create_state2(self):
        self.fs_prev = nn.Linear(256, 256)
        self.fs_feed_loc = nn.Linear(256, 256)
        self.fs_feed_r1 = nn.Linear(256, 256)
        self.fs_feed_merge = nn.Linear(512, 256)
        self.fs_merge = nn.Linear(512, self.r2_sz)
            
        self.state_layer_collection.append(self.fs_prev)
        self.state_layer_collection.append(self.fs_feed_loc)
        self.state_layer_collection.append(self.fs_feed_r1)
        self.state_layer_collection.append(self.fs_feed_merge)
        self.state_layer_collection.append(self.fs_merge)
        
    def update_state2(self, r2, loc2, r1_feed):
        h_prev = F.leaky_relu(self.fs_prev(r2))
        h_feed_loc2 = F.leaky_relu(self.fs_feed_loc(loc2))
        h_feed_r1 = F.leaky_relu(self.fs_feed_r1(r1_feed))
        h_feed_merge = F.leaky_relu(self.fs_feed_merge(torch.cat([h_feed_loc2, h_feed_r1], dim=-1)))
        h = torch.cat([h_prev, h_feed_merge], dim=-1)
        
        return F.leaky_relu(self.fs_merge(h))
        
    ###########################################################################################
    ############################## HYPERNETS ##################################################
    ###########################################################################################
        
    ##########################################
    ################# ACTION #################
    ##########################################
    
    def _create_action_hypernet(self):
        if self.hyper:
            self.fa_W1 = nn.Linear((self.a1_sz + self.r1_sz) * self.a1_sz, (self.a1_sz + self.r1_sz) * self.a1_sz)
            self.fa_b1 = nn.Linear(self.a1_sz, self.a1_sz)
            
            # Apply custom initialization
            nn.init.xavier_uniform_(self.fa_W1.weight, gain=hyperfanin_for_kernel(fanin=(self.a1_sz + self.r1_sz)))
            nn.init.zeros_(self.fa_b1.weight)
            nn.init.zeros_(self.fa_b1.bias)
            
            self.action_layer_collection.append(self.fa_W1)
            self.action_layer_collection.append(self.fa_b1)
            
            self.ln_W1 = nn.Linear(self.a1_sz * 2, self.a1_sz * 2)
            self.ln_b1 = nn.Linear(2, 2)
            
            # Apply custom initialization
            nn.init.xavier_uniform_(self.ln_W1.weight, gain=hyperfanin_for_kernel(fanin=self.in_fa1_sz))
            nn.init.zeros_(self.ln_b1.weight)
            nn.init.zeros_(self.ln_b1.bias)

            self.action_layer_collection.append(self.ln_W1)
            self.action_layer_collection.append(self.ln_b1)
            
        else:
            self.act_base_theta1 = nn.Linear(512, 512)
            self.act11 = nn.Linear(512, self.a1_sz)
            
            self.action_layer_collection.append(self.act_base_theta1)
            self.action_layer_collection.append(self.act11)
            
            self.act_loc_theta1 = nn.Linear(256, 256)
            self.loc11 = nn.Linear(256, 2)
            
            self.action_layer_collection.append(self.act_loc_theta1)
            self.action_layer_collection.append(self.loc11)
        
    def act1(self, z):
        theta_act = F.leaky_relu(self.act_base_theta1(z))
        theta_loc = F.leaky_relu(self.act_loc_theta1(z))
        return lambda x: F.leaky_relu(self.act11(torch.cat([x, theta_act], dim=-1))), lambda x: self.loc11(torch.cat([x, theta_loc], dim=-1))
        
    def generate_fa1(self, z):
        fa_W1 = self.fa_W1(z)
        fa_b1 = self.fa_b1(z)
        fa_theta = torch.cat([fa_W1, fa_b1], dim=-1)
        
        ln_W1 = self.ln_W1(z)
        ln_b1 = self.ln_b1(z)
        ln_theta = torch.cat([ln_W1, ln_b1], dim=-1)

        return fa_theta, ln_theta
        
    def _create_baseline(self):
        self.base1 = nn.Linear(64, self.T)
        self.base2 = nn.Linear(64, self.T2)
        
        self.action_layer_collection.append(self.base1)
        self.action_layer_collection.append(self.base2)
        
    def baseline1(self, x):
        return self.base1(x)
        
    def baseline2(self, x):
        return self.base2(x)
        
    ##########################################
    ################# STATE ##################
    ##########################################
    
    def _create_state_hypernet(self):
        self.st_hyp1 = nn.Linear(256, 256)
        self.st_hyp2 = nn.Linear(256, 256)
        
        self.state_layer_collection.append(self.st_hyp1)
        self.state_layer_collection.append(self.st_hyp2)
        
        if self.hyper:
            self.st_hyp3 = nn.Linear(256, self.z_sz)
            self.st_W1 = nn.Linear((self.r1_sz + self.bottleneck + 2) * self.r1_sz, (self.r1_sz + self.bottleneck + 2) * self.r1_sz)
            self.st_b1 = nn.Linear(self.r1_sz, self.r1_sz)
            
            # Apply custom initialization
            nn.init.xavier_uniform_(self.st_W1.weight, gain=hyperfanin_for_kernel(fanin=(self.r1_sz + self.bottleneck + 2)))
            nn.init.zeros_(self.st_W1.weight)
            nn.init.zeros_(self.st_W1.bias)

            self.state_layer_collection.append(self.st_W1)
            self.state_layer_collection.append(self.st_b1)
            self.state_layer_collection.append(self.st_hyp3)
        else:
            self.st_base_theta1 = nn.Linear(512, 512)
            self.state11 = nn.Linear(512, self.r1_sz)
            
            self.state_layer_collection.append(self.st_base_theta1)
            self.state_layer_collection.append(self.state11)
        
    def state1(self, z):
        theta_st = F.leaky_relu(self.st_base_theta1(z))
        return lambda x: F.leaky_relu(self.state11(torch.cat([x, theta_st], dim=-1)))
        
    def generate_fs1(self, r2):
        z = self.st_hyp3(F.leaky_relu(self.st_hyp2(F.leaky_relu(self.st_hyp1(r2)))))
    
        W1 = self.st_W1(z)
        b1 = self.st_b1(z)
        theta = torch.cat([W1, b1], dim=-1)
        return theta
        
    ###########################################################################################
    ############################## AUX NETS ###################################################
    ###########################################################################################
    
    # R2 initialization
    def _create_r2_init(self):
        self.r2_init_flatten = nn.Flatten()
        self.r2_init1 = nn.Linear(256, self.r2_sz)
        self.state_layer_collection.append(self.r2_init_flatten)
        self.state_layer_collection.append(self.r2_init1)
        
    def init_state2(self, x):
        return F.leaky_relu(self.r2_init1(self.r2_init_flatten(x)))
    
    # R2 -> R1 initialization
    def _create_state1_init(self):
        self.state_init1 = nn.Linear(256, self.r1_sz)
        self.state_layer_collection.append(self.state_init1)
        
    def init_state1(self, x):
        return F.leaky_relu(self.state_init1(x))
        
    # A2 -> A1 initialization
    def _create_action1_init(self):
        self.action_init1 = nn.Linear(256, self.a1_sz)
        self.action_layer_collection.append(self.action_init1)
        
    def init_action1(self, x):
        return F.leaky_relu(self.action_init1(x))
        
    # R1 -> R2 feedback
    def _create_state_feedback(self):
        self.fs_feed1 = nn.Linear(256, 256)
        self.fs_feed2 = nn.Linear(256, 256)
        self.fs_feed3 = nn.Linear(256, self.r2_sz)
        self.state_layer_collection.append(self.fs_feed1)
        self.state_layer_collection.append(self.fs_feed2)
        self.state_layer_collection.append(self.fs_feed3)
        
    def state_feedback(self, x):
        return F.leaky_relu(self.fs_feed3(F.leaky_relu(self.fs_feed2(F.leaky_relu(self.fs_feed(x))))))
    
    # A1 -> A2 feedback
    def _create_action_feedback(self):
        self.fa_feed1 = nn.Linear(256, 256)
        self.fa_feed2 = nn.Linear(256, 256)
        self.fa_feed3 = nn.Linear(256, self.a2_sz)
        self.action_layer_collection.append(self.fa_feed1)
        self.action_layer_collection.append(self.fa_feed2)
        self.action_layer_collection.append(self.fa_feed3)
        
    def action_feedback(self, x):
        return F.leaky_relu(self.fa_feed3(F.leaky_relu(self.fa_feed2(F.leaky_relu(self.fa_feed1(x))))))
        
    def sample_g2(self, g2):
        return F.interpolate(g2, size=[self.g2_sub_sz, self.g2_sub_sz], mode='bilinear', align_corners=False)
        
    ###########################################################################################
    ############################## VAR ACCOUNTING #############################################
    ###########################################################################################
    
    def get_state_vars(self):
        var_list = []
    
        for layer in self.state_layer_collection:
            var_list += list(layer.parameters())
        return var_list
        
    def get_action_vars(self):
        var_list = []
    
        for layer in self.action_layer_collection:
            var_list += list(layer.parameters())
        return var_list
        
    ###########################################################################################
    ############################## TRAIN OPS ##################################################
    ###########################################################################################
    
    def train_step(self, inputs):
        self.optimizer.zero_grad()
        
        state_vars = self.get_state_vars()
        action_vars = self.get_action_vars()
        all_vars = action_vars + state_vars
        
        self.forward_pass(inputs)
            
        fp_loss = torch.mean(self.pred_error())
        fa_loss, baseline_mse = self.REINFORCE()
        
        ### Sanity check
        total_size = 0.0
        
        for param in all_vars:
            vr_sz = np.prod(param.shape)
            total_size += vr_sz
            
        print(total_size)
        
        loss = fp_loss + baseline_mse + fa_loss
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(all_vars, 5.0)
        
        self.optimizer.step()
        
        pred = self.activations['I_hat'][-1]
        
        return {'loss': loss.item()}
    
    def test_step(self, inputs):
        with torch.no_grad():
            self.forward_pass(inputs)
                
            pred = self.activations['I_hat'][-1]
            
            return {'pred': pred}
    
    def forward(self, inputs):
        return self.forward_pass(inputs)
        
    ###########################################################################################
    ############################## FUNCTIONAL OPS #############################################
    ###########################################################################################
    
    def f_theta(self, x, theta, sz, in_sz):
        num_layers = len(sz)
        
        offset = 0
        
        y = x.view(-1, 1, in_sz)
        
        for i in range(num_layers):
            out_sz = sz[i]
            
            W_sz = in_sz * out_sz
            b_sz = out_sz
            
            W = theta[:, offset:offset + W_sz]
            offset += W_sz
            
            b = theta[:, offset:offset + b_sz]
            offset += b_sz
            
            W = W.view(-1, in_sz, out_sz)
            b = b.view(-1, 1, out_sz)

            y = torch.matmul(y, W) + b
            
            if i < num_layers - 1:
                y = F.leaky_relu(y)

            in_sz = out_sz

        y = y.squeeze(dim=1)
        
        return y

	
