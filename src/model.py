import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import os
import h5py
import numpy as np
import sklearn.cluster
import copy

import pdb


class EncoderBlock(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, kernel_size=3, conv_act='leaky_relu', dropout=0, num_conv=2):
        super(EncoderBlock, self).__init__()
        if conv_act == 'relu':
            conv_act_layer = nn.ReLU(inplace=True)
        elif conv_act == 'leaky_relu':
            conv_act_layer = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError('No implementation of ', conv_act)

        if num_conv == 1:
            self.conv = nn.Sequential(
                            nn.Conv3d(in_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,
                            nn.Dropout3d(dropout),
                            nn.MaxPool3d(2))
        elif num_conv == 2:
            self.conv = nn.Sequential(
                            nn.Conv3d(in_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,
                            nn.Dropout3d(dropout),
                            nn.Conv3d(out_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,
                            nn.Dropout3d(dropout),
                            nn.MaxPool3d(2))
        else:
            raise ValueError('Number of conv can only be 1 or 2')

        self.init_model()

    def init_model(self):
        for layer in self.conv.children():
            if isinstance(layer, nn.Conv3d):
                for name, weight in layer.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(weight)
                    if 'bias' in name:
                        nn.init.constant_(weight, 0.0)

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, kernel_size=3, conv_act='leaky_relu', dropout=0, num_conv=2):
        super(DecoderBlock, self).__init__()
        if conv_act == 'relu':
            conv_act_layer = nn.ReLU(inplace=True)
        elif conv_act == 'leaky_relu':
            conv_act_layer = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError('No implementation of ', conv_act)

        if num_conv == 1:
            self.conv = nn.Sequential(
                            nn.Conv3d(in_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,
                            nn.Dropout3d(dropout),
                            nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=True))
        elif num_conv == 2:
            self.conv = nn.Sequential(
                            nn.Conv3d(in_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,
                            nn.Dropout3d(dropout),
                            nn.Conv3d(out_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,
                            nn.Dropout3d(dropout),
                            nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=True))
        else:
            raise ValueError('Number of conv can only be 1 or 2')

        self.init_model()

    def init_model(self):
        for layer in self.conv.children():
            if isinstance(layer, nn.Conv3d):
                for name, weight in layer.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(weight)
                    if 'bias' in name:
                        nn.init.constant_(weight, 0.0)

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_num_ch=1, img_size=(64,64,64), inter_num_ch=16, kernel_size=3, conv_act='leaky_relu', num_conv=2, dropout=False):
        super(Encoder, self).__init__()

        if dropout:
            self.conv1 = EncoderBlock(in_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
            self.conv2 = EncoderBlock(inter_num_ch, 2*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0.1, num_conv=num_conv)
            self.conv3 = EncoderBlock(2*inter_num_ch, 4*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0.2, num_conv=num_conv)
        else:
            self.conv1 = EncoderBlock(in_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
            self.conv2 = EncoderBlock(inter_num_ch, 2*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
            self.conv3 = EncoderBlock(2*inter_num_ch, 4*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)

        self.conv4 = nn.Sequential(
                        EncoderBlock(4*inter_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv),
                        nn.Conv3d(inter_num_ch, inter_num_ch, kernel_size=kernel_size, padding=1))
        # self.conv4 = EncoderBlock(4*inter_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        # self.fc = nn.Linear(1024, 1024)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        # (16,4,4,4)
        return conv4.view(x.shape[0], -1), [conv3, conv2, conv1]
        # fc = self.fc(conv4.view(x.shape[0], -1))
        # return fc, [conv3, conv2, conv1]


class Decoder(nn.Module):
    def __init__(self, out_num_ch=1, img_size=(64,64,64), inter_num_ch=16, kernel_size=3, conv_act='leaky_relu', num_conv=2, shortcut=False):
        super(Decoder, self).__init__()
        self.shortcut = shortcut
        # self.fc = nn.Linear(1024, 1024)
        if self.shortcut:
            self.conv4 = DecoderBlock(inter_num_ch, 4*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
            self.conv3 = DecoderBlock(8*inter_num_ch, 2*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
            self.conv2 = DecoderBlock(4*inter_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
            self.conv1 = DecoderBlock(2*inter_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
            self.conv0 = nn.Conv3d(inter_num_ch, out_num_ch, kernel_size=3, padding=1)
        else:
            self.conv4 = DecoderBlock(inter_num_ch, 4*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
            self.conv3 = DecoderBlock(4*inter_num_ch, 2*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
            self.conv2 = DecoderBlock(2*inter_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
            self.conv1 = DecoderBlock(inter_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
            self.conv0 = nn.Conv3d(inter_num_ch, out_num_ch, kernel_size=3, padding=1)

    def forward(self, x, feat_list=[]):
        # fc = self.fc(x)
        # conv4 = self.conv4(fc.view(x.shape[0], 16, 4, 4, 4))
        x_reshaped = x.view(x.shape[0], 16, 4, 4, 4)
        conv4 = self.conv4(x_reshaped)
        if self.shortcut:
            conv3 = self.conv3(torch.cat([conv4, feat_list[0]], 1))
            conv2 = self.conv2(torch.cat([conv3, feat_list[1]], 1))
            conv1 = self.conv1(torch.cat([conv2, feat_list[2]], 1))
        else:
            conv3 = self.conv3(conv4)
            conv2 = self.conv2(conv3)
            conv1 = self.conv1(conv2)
        output = self.conv0(conv1)
        return output

class Encoder_ROI(nn.Module):
    def __init__(self, in_dim, latent_dim, hidden_dim):
        super().__init__()
        self.fcs = nn.Sequential(
                        nn.Linear(in_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim//2),
                        nn.ReLU(),
                        nn.Linear(hidden_dim//2, latent_dim))

    def forward(self, x):
        # return self.fcs(x), None
        return F.normalize(self.fcs(x), dim=1), None

    def encode(self, x):
        return self.fcs(x)

class Decoder_ROI(nn.Module):
    def __init__(self, out_dim, latent_dim, hidden_dim, final_act=False):
        super().__init__()
        self.final_act = final_act
        self.fcs = nn.Sequential(
                        nn.Linear(latent_dim, hidden_dim//2),
                        nn.ReLU(),
                        nn.Linear(hidden_dim//2, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, out_dim))

    def forward(self, x, feat_list=[]):
        if self.final_act:
            return F.softplus(self.fcs(x))    # for PET, SUVR values
            # return F.relu(self.fcs(x))    # for PET, SUVR values
        else:
            return self.fcs(x)    # for MRI


class SOM(nn.Module):
    def __init__(self, config):
        super(SOM, self).__init__()

        self.config = config
        self.device = config['device']
        self.latent_size = config['latent_size']
        self.embedding_size = config['embedding_size']
        self.dataset_name = config['dataset_name']
        init_emb = config['init_emb']

        try:
            if config['input_type'] == 'ROI':
                if config['image_type'] == 'MRI':
                    self.encoder = Encoder_ROI(in_dim=config['num_features'], latent_dim=self.latent_size, hidden_dim=256)
                    self.decoder = Decoder_ROI(out_dim=config['num_features'], latent_dim=self.latent_size, hidden_dim=256, final_act=False)
                else:
                    self.encoder = Encoder_ROI(in_dim=config['num_features'], latent_dim=self.latent_size, hidden_dim=128)
                    self.decoder = Decoder_ROI(out_dim=config['num_features'], latent_dim=self.latent_size, hidden_dim=128, final_act=True)
            else:
                self.encoder = Encoder(in_num_ch=1, inter_num_ch=16, num_conv=2)
                self.decoder = Decoder(out_num_ch=1, inter_num_ch=16, num_conv=2, shortcut=False)
        except:
            self.encoder = Encoder(in_num_ch=1, inter_num_ch=16, num_conv=2)
            self.decoder = Decoder(out_num_ch=1, inter_num_ch=16, num_conv=2, shortcut=False)
        self.embeddings = nn.Parameter(F.normalize(torch.randn(self.embedding_size[0],self.embedding_size[1],self.latent_size,requires_grad=True),dim=2), requires_grad=True)
        # self.embeddings = nn.Parameter(torch.fmod(torch.randn(self.embedding_size[0],self.embedding_size[1],self.latent_size,requires_grad=True),2) * 0.05, requires_grad=True)

    def compute_node_edge_weight(self, emb, idx):
        edge_sum = 0
        edge_count = 0
        row = idx // self.embedding_size[1]
        col = idx % self.embedding_size[1]
        if row != 0:
            edge_sum += np.linalg.norm(emb[idx]-emb[idx-self.embedding_size[1]])
            edge_count += 1
        if row != self.embedding_size[0] - 1:
            edge_sum += np.linalg.norm(emb[idx]-emb[idx+self.embedding_size[1]])
            edge_count += 1
        if col != 0:
            edge_sum += np.linalg.norm(emb[idx]-emb[idx-1])
            edge_count += 1
        if col != self.embedding_size[1] - 1:
            edge_sum += np.linalg.norm(emb[idx]-emb[idx+1])
            edge_count += 1
        return edge_sum / edge_count

    def reorganize_som_embeddings(self, emb_init):
        num_run = 10000
        emb_rand = copy.copy(emb_init.reshape(self.embedding_size[0]*self.embedding_size[1], -1))
        for run_idx in range(num_run):
            idx_sel = np.random.choice(self.embedding_size[0]*self.embedding_size[1], (2,))
            emb_tpm = copy.copy(emb_rand)
            emb_tpm[idx_sel[0]] = emb_rand[idx_sel[1]]
            emb_tpm[idx_sel[1]] = emb_rand[idx_sel[0]]
            weight_old = self.compute_node_edge_weight(emb_rand, idx_sel[0]) + self.compute_node_edge_weight(emb_rand, idx_sel[1])
            weight_new = self.compute_node_edge_weight(emb_tpm, idx_sel[0]) + self.compute_node_edge_weight(emb_tpm, idx_sel[1])
            if weight_new < weight_old:
                emb_rand = emb_tpm
            del emb_tpm
        return emb_rand

    def init_embeddings_ep_weight(self, init):
        if self.config['init_emb']  == 'pretrained-kmeans':
            init_reorganized = init
        else:
            init_reorganized = self.reorganize_som_embeddings(init)
        init = torch.tensor(init_reorganized, requires_grad=True).to(self.device)
        print('Finish embedding initialization by k-means!')
        self.embeddings.data = init.view(self.embedding_size[0], self.embedding_size[1], -1)
        recon_emb = self.recon_embeddings()
        res_path = os.path.join(self.config['ckpt_path'], 'recon_emb_warmup.npy')
        np.save(res_path, recon_emb.detach().cpu().numpy())

    def compute_zq_distance(self, z_e):
        z_dist = torch.sum((z_e.unsqueeze(1).unsqueeze(2) - self.embeddings.unsqueeze(0)) ** 2, dim=-1)
        return z_dist

    # for each z_e, find the nearest embedding z_q
    def compute_zq(self, z_e, global_iter=-1, iter_max=-1):
        z_dist = self.compute_zq_distance(z_e)
        k = torch.argmin(z_dist.view(z_e.shape[0], -1), dim=-1)
        k_1 = k // self.embedding_size[1]
        k_2 = k % self.embedding_size[1]
        k_stacked = torch.stack([k_1, k_2], dim=1)
        z_q = self._gather_nd(self.embeddings, k_stacked)
        return z_q, k

    # find the neighbours of z_q
    def compute_zq_neighbours(self, z_q, k):
        k_1 = k // self.embedding_size[1]
        k_2 = k % self.embedding_size[1]
        k1_down = torch.where(k_1 < self.embedding_size[0] - 1, k_1 + 1, k_1)
        k1_up = torch.where(k_1 > 0, k_1 - 1, k_1)
        k2_right = torch.where(k_2 < self.embedding_size[1] - 1, k_2 + 1, k_2)
        k2_left = torch.where(k_2 > 0, k_2 - 1, k_2)
        z_q_up = self._gather_nd(self.embeddings, torch.stack([k1_up, k_2], dim=1))
        z_q_down = self._gather_nd(self.embeddings, torch.stack([k1_down, k_2], dim=1))
        z_q_left = self._gather_nd(self.embeddings, torch.stack([k_1, k2_left], dim=1))
        z_q_right = self._gather_nd(self.embeddings, torch.stack([k_1, k2_right], dim=1))
        z_q_neighbours = torch.stack([z_q, z_q_up, z_q_down, z_q_left, z_q_right], dim=1)  # check whether gradient get back if no z_q
        # z_q_neighbours = torch.stack([z_q_up, z_q_down, z_q_left, z_q_right], dim=1)  # check whether gradient get back if no z_q
        return z_q_neighbours

    # compute manhattan distance for each embedding to given nearest embedding index k
    def compute_manhattan_distance(self, k):
        k_1 = (k // self.embedding_size[1]).unsqueeze(1).unsqueeze(2)
        k_2 = (k % self.embedding_size[1]).unsqueeze(1).unsqueeze(2)
        row = torch.arange(0, self.embedding_size[0]).long().repeat(self.embedding_size[1],1).transpose(1,0).to(self.device)
        col = torch.arange(0, self.embedding_size[1]).long().repeat(self.embedding_size[0],1).to(self.device)
        row_diff = torch.abs(k_1 - row)
        col_diff = torch.abs(k_2 - col)
        return (row_diff + col_diff).float()

    # gather the elements of given index
    def _gather_nd(self, params, idxes, dims=(0,1)):
        if dims == (0,1):
            outputs = params[idxes[:,0], idxes[:,1]]
        else:
            outputs = params[:, idxes[:,0], idxes[:,1]]
        return outputs

    # reconstruction loss
    def compute_recon_loss(self, x, recon):
        return torch.mean((x - recon) ** 2)

    # standard commitment loss, make sure z_q close to z_e (fix z_e)
    def compute_commit_loss(self, z_e, z_q):
        return torch.mean((z_e.detach() - z_q) ** 2) + self.config['commit_ratio'] * torch.mean((z_e - z_q.detach()) ** 2)

    # compute similarity between z_e and embeddings
    def compute_similarity(self, z_e, sim_type='softmax'):
        z_dist_flatten = self.compute_zq_distance(z_e).view(z_e.shape[0], -1)
        if sim_type == 'softmax':
            sim_flatten = F.softmax(- z_dist_flatten / torch.std(z_dist_flatten, dim=1).unsqueeze(1))
        else:
            raise ValueError('Not supporting this similarity type')
        return sim_flatten

    # som loss based on grid distance to zq, from XADLiME paper
    def compute_som_loss(self, z_e, k, iter=-1, iter_max=-1, Tmax=1., Tmin=0.1):
        Tmin = self.config['Tmin']
        Tmax = self.config['Tmax']
        dis_ze_emb = self.compute_zq_distance(z_e.detach())
        dis_zq_manhattan = self.compute_manhattan_distance(k)               # (bs, 32, 32)
        if iter != -1 and iter_max != -1:
            self.T = Tmax * (Tmin / Tmax)**(iter / iter_max)
        else:
            self.T = Tmin
        weight = torch.exp(-0.5 * dis_zq_manhattan**2 / (self.embedding_size[0] * self.embedding_size[1] * self.T**2))
        weight_normalized = weight / weight.view(-1, self.embedding_size[0] * self.embedding_size[1]).sum(1).unsqueeze(1).unsqueeze(2)
        som = torch.mean(weight_normalized * dis_ze_emb)
        return som

    def recon_embeddings(self):
        emb_resize = self.embeddings.reshape(-1, self.latent_size)
        recon_emb_list = []
        i = 0
        while(1):
            if (i+1) * 64 < emb_resize.shape[0]:
                recon_emb = self.decoder(emb_resize[i*64:(i+1)*64, :])
                recon_emb_list.append(recon_emb)
            else:
                recon_emb = self.decoder(emb_resize[i*64:, :])
                recon_emb_list.append(recon_emb)
                break
            i += 1
        recon_emb_list = torch.cat(recon_emb_list, 0)
        return recon_emb_list

    def forward(self, x, global_iter=-1, iter_max=-1):
        # pdb.set_trace()
        z_e, feat_list = self.encoder(x)
        z_q, k = self.compute_zq(z_e, global_iter, iter_max)
        sim = self.compute_similarity(z_e, sim_type='softmax')
        recon_ze = self.decoder(z_e, feat_list)
        recon_zq = self.decoder(z_q, feat_list)
        return [recon_ze, recon_zq], [z_e, z_q, k, sim]


class SOMPairVisitDirection(SOM):
    def __init__(self, config):
        super(SOM, self).__init__()
        self.config = config
        self.device = config['device']
        self.latent_size = config['latent_size']
        self.embedding_size = config['embedding_size']
        self.dataset_name = config['dataset_name']
        init_emb = config['init_emb']

        try:
            if config['input_type'] == 'ROI':
                if config['image_type'] == 'MRI':
                    self.encoder = Encoder_ROI(in_dim=config['num_features'], latent_dim=self.latent_size, hidden_dim=256)
                    self.decoder = Decoder_ROI(out_dim=config['num_features'], latent_dim=self.latent_size, hidden_dim=256, final_act=False)
                else:
                    self.encoder = Encoder_ROI(in_dim=config['num_features'], latent_dim=self.latent_size, hidden_dim=128)
                    self.decoder = Decoder_ROI(out_dim=config['num_features'], latent_dim=self.latent_size, hidden_dim=128, final_act=True)
            else:
                self.encoder = Encoder(in_num_ch=1, inter_num_ch=16, num_conv=2)
                self.decoder = Decoder(out_num_ch=1, inter_num_ch=16, num_conv=2, shortcut=False)
        except:
            self.encoder = Encoder(in_num_ch=1, inter_num_ch=16, num_conv=2)
            self.decoder = Decoder(out_num_ch=1, inter_num_ch=16, num_conv=2, shortcut=False)
        self.embeddings = nn.Parameter(F.normalize(torch.randn(self.embedding_size[0],self.embedding_size[1],self.latent_size,requires_grad=True),dim=2), requires_grad=True)

    def init_embeddings_ep_weight(self, init):
        if self.config['init_emb']  == 'pretrained-kmeans':
            init_reorganized = init
        else:
            init_reorganized = self.reorganize_som_embeddings(init)
        init = torch.tensor(init_reorganized, requires_grad=True).to(self.device)
        print('Finish embedding initialization by k-means!')
        self.embeddings.data = init.view(self.embedding_size[0], self.embedding_size[1], -1)
        recon_emb = self.recon_embeddings()
        res_path = os.path.join(self.config['ckpt_path'], 'recon_emb_warmup.npy')
        np.save(res_path, recon_emb.detach().cpu().numpy())

    def forward_pair_z(self, x1, x2, interval):
        bs = x1.shape[0]
        z_e, feat_list = self.encoder(torch.cat([x1, x2], dim=0))
        z_e1, z_e2 = z_e[:bs], z_e[bs:]

        z_e_diff = (z_e2 - z_e1) / interval.unsqueeze(1)
        recon_ze = self.decoder(z_e, feat_list)
        recon_ze1, recon_ze2 = recon_ze[:bs], recon_ze[bs:]

        z_q, k = self.compute_zq(z_e)
        sim = self.compute_similarity(z_e, sim_type='softmax')
        recon_zq = self.decoder(z_q, feat_list)

        z_q1, z_q2 = z_q[:bs], z_q[bs:]
        recon_zq1, recon_zq2 = recon_zq[:bs], recon_zq[bs:]
        k1, k2 = k[:bs], k[bs:]
        sim1, sim2 = sim[:bs], sim[bs:]

        return [[recon_ze1, recon_ze2], [recon_zq1, recon_zq2]], [[z_e1, z_e2, z_e_diff], [z_q1, z_q2], [k1, k2], [sim1, sim2]]

    # regularize sim2 to be on the right side of sim1 based on the sum of each col
    def compute_direction_loss(self, sim1, sim2, alpha=0.1):
        sim_sum1 = sim1.view(-1, self.embedding_size[0],self.embedding_size[1]).sum(dim=1)
        sim_sum2 = sim2.view(-1, self.embedding_size[0],self.embedding_size[1]).sum(dim=1)

        weight = torch.arange(1, self.embedding_size[1]+1).to(self.device)
        mean_idx1 = torch.sum(sim_sum1 * weight, dim=1)
        mean_idx2 = torch.sum(sim_sum2 * weight, dim=1)
        loss = F.relu(mean_idx1 - mean_idx2 + alpha)
        return loss.mean()
