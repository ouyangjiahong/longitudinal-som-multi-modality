import os
import glob
import time
import torch
import torch.optim as optim
import numpy as np
import yaml
import pdb
import psutil
import sklearn.cluster

from model import *
from util import *

# set seed
seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True

_, config = load_config_yaml('config_pair_roi_multimodal.yaml')
config['device'] = torch.device('cuda:'+ config['gpu'])

if config['ckpt_timelabel'] and (config['phase'] == 'test' or config['continue_train'] == True):
    time_label = config['ckpt_timelabel']
else:
    localtime = time.localtime(time.time())
    time_label = str(localtime.tm_year) + '_' + str(localtime.tm_mon) + '_' + str(localtime.tm_mday) + '_' + str(localtime.tm_hour) + '_' + str(localtime.tm_min)
print(time_label)

# ckpt folder, load yaml config
config['ckpt_path'] = os.path.join('../ckpt/', config['dataset_name'], config['model_name'], time_label)
if not os.path.exists(config['ckpt_path']):     # test, not exists
    os.makedirs(config['ckpt_path'])
    save_config_yaml(config['ckpt_path'], config)
elif config['load_yaml']:       # exist and use yaml config
    print('Load config ', os.path.join(config['ckpt_path'], 'config_pair_roi_multimodal.yaml'))
    flag, config_load = load_config_yaml(os.path.join(config['ckpt_path'], 'config_pair_roi_multimodal.yaml'))
    if flag:    # load yaml success
        print('load yaml config file')
        for key in config_load.keys():  # if yaml has, use yaml's param, else use config
            if key == 'phase' or key == 'gpu' or key == 'continue_train' or key == 'ckpt_name' or key == 'ckpt_path':
                continue
            if key in config.keys():
                config[key] = config_load[key]
            else:
                print('current config do not have yaml param')
    else:
        save_config_yaml(config['ckpt_path'], config)
print(config)


# define dataset
# load data csv
df = pd.read_csv(config['data_csv_path'])
print('No. of images', len(df))
print('No. of subjects', df['RID'].unique().shape[0])

# select cohorts, split train/test
df_sel = df

df_train_idx = pd.read_csv(config['train_idx_csv_path'])
df_val_idx = pd.read_csv(config['val_idx_csv_path'])
df_test_idx = pd.read_csv(config['test_idx_csv_path'])
train_idx_list_1 = np.array(df_train_idx['idx_1'])
train_idx_list_2 = np.array(df_train_idx['idx_2'])
val_idx_list_1 = np.array(df_val_idx['idx_1'])
val_idx_list_2 = np.array(df_val_idx['idx_2'])
test_idx_list_1 = np.array(df_test_idx['idx_1'])
test_idx_list_2 = np.array(df_test_idx['idx_2'])

print('Num. of train pairs', len(train_idx_list_1))
print('Num. of val pairs', len(val_idx_list_1))
print('Num. of test pairs', len(test_idx_list_1))


# select feature columns
def select_feature_columns(mode):
    if mode == 'MRI':
        cols_sel = [col for col in df if 'ST' in col and col != 'STATUS' and '_' not in col and len(col)<8]   # 313
    elif mode == 'Amyloid':
        cols_sel = [col for col in df if '_SUVR' in col]   # 160
    print('Num. of image features', len(cols_sel))
    return cols_sel


cols_sel_m1 = select_feature_columns(config['image_type_1'])
config['num_features_1'] = len(cols_sel_m1)
features_m1 =  np.array(df_sel[cols_sel_m1])

cols_sel_m2 = select_feature_columns(config['image_type_2'])
config['num_features_2'] = len(cols_sel_m2)
features_m2 =  np.array(df_sel[cols_sel_m2])

days_after_baseline = np.array(df_sel['days_after_bl'])

train_feat_m1t1 = features_m1[train_idx_list_1]
train_feat_m1t2 = features_m1[train_idx_list_2]
train_feat_m2t1 = features_m2[train_idx_list_1]
train_feat_m2t2 = features_m2[train_idx_list_2]
train_interval = (days_after_baseline[train_idx_list_2] - days_after_baseline[train_idx_list_1]) / 365.0

val_feat_m1t1 = features_m1[val_idx_list_1]
val_feat_m1t2 = features_m1[val_idx_list_2]
val_feat_m2t1 = features_m2[val_idx_list_1]
val_feat_m2t2 = features_m2[val_idx_list_2]
val_interval = (days_after_baseline[val_idx_list_2] - days_after_baseline[val_idx_list_1]) / 365.0

test_feat_m1t1 = features_m1[test_idx_list_1]
test_feat_m1t2 = features_m1[test_idx_list_2]
test_feat_m2t1 = features_m2[test_idx_list_1]
test_feat_m2t2 = features_m2[test_idx_list_2]
test_interval = (days_after_baseline[test_idx_list_2] - days_after_baseline[test_idx_list_1]) / 365.0

# define loader
# the seven tensors are:
# features of modality 1 timestep 1,
# features of modality 2 timestep 1,
# index of timestep 1 (not used)
# features of modality 1 timestep 2,
# features of modality 2 timestep 2,
# index of timestep 2 (not used)
# time interval between timestep 1 and 2
trainDataLoader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.tensor(train_feat_m1t1), torch.tensor(train_feat_m2t1), torch.tensor(train_idx_list_1), torch.tensor(train_feat_m1t2), torch.tensor(train_feat_m2t2), torch.tensor(train_idx_list_2), torch.tensor(train_interval)),
    batch_size=config['batch_size'],
    shuffle=True,
)
valDataLoader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.tensor(val_feat_m1t1), torch.tensor(val_feat_m2t1), torch.tensor(val_idx_list_1), torch.tensor(val_feat_m1t2), torch.tensor(val_feat_m2t2), torch.tensor(val_idx_list_2), torch.tensor(val_interval)),
    batch_size=config['batch_size'],
    shuffle=False,
)
testDataLoader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.tensor(test_feat_m1t1), torch.tensor(test_feat_m2t1), torch.tensor(test_idx_list_1), torch.tensor(test_feat_m1t2), torch.tensor(test_feat_m2t2), torch.tensor(test_idx_list_2), torch.tensor(test_interval)),
    batch_size=config['batch_size'],
    shuffle=False,
)
iter_max = len(trainDataLoader) * config['epochs']


# define model
config_m1 = config.copy()
config_m1['image_type'] = config['image_type_1']
config_m1['num_features'] = config['num_features_1']
config_m2 = config.copy()
config_m2['image_type'] = config['image_type_2']
config_m2['num_features'] = config['num_features_2']

if config['model_name'] == 'SOMPairVisitDirection':
    model_m1 = SOMPairVisitDirection(config=config_m1).to(config['device'])
    model_m2 = SOMPairVisitDirection(config=config_m2).to(config['device'])
else:
    raise ValueError('Not support other models yet!')

# define optimizer
if config['jointly_train']:
    optimizer = optim.SGD(list(model_m1.parameters()) + list(model_m2.parameters()), lr=config['lr'], momentum=0.9)
else:
    optimizer = optim.SGD(model_m1.parameters(), lr=config['lr'], momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999), weight_decay=1e-5, amsgrad=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-5)

# load pretrained model
if config['continue_train'] or config['phase'] == 'test':
    [optimizer, scheduler, model_m1, model_m2], start_epoch = load_checkpoint_by_key([optimizer, scheduler, model_m1, model_m2], config['ckpt_path'], ['optimizer', 'scheduler', 'model_m1', 'model_m2'], config['device'], config['ckpt_name'])
    # [optimizer, model], start_epoch = load_checkpoint_by_key([optimizer, model], config['ckpt_path'], ['optimizer', 'model'], config['device'], config['ckpt_name'])
    # [model], start_epoch = load_checkpoint_by_key([model], config['ckpt_path'], ['model'], config['device'], config['ckpt_name'])
    print('starting lr:', optimizer.param_groups[0]['lr'])
else:
    [model_m1], start_epoch = load_checkpoint_by_key([model_m1], config['pretrain_model_path_1'], ['model'], config['device'], config['ckpt_name'])
    [model_m2], start_epoch = load_checkpoint_by_key([model_m2], config['pretrain_model_path_2'], ['model'], config['device'], config['ckpt_name'])
    start_epoch = -1

def train():
    global_iter = len(trainDataLoader) * (start_epoch+1)
    monitor_metric_best = 100
    start_time = time.time()

    # print(stat)
    for epoch in range(start_epoch+1, config['epochs']):
        print('Epoch', epoch)

        model_m1.train()
        if config['jointly_train']:
            model_m2.train()
        else:
            model_m2.eval()

        loss_all_dict_m1 = {'all': 0, 'recon': 0., 'recon_zq': 0., 'som': 0., 'commit': 0., 'dir': 0., 'multimodal': 0.}
        loss_all_dict_m2 = {'all': 0, 'recon': 0., 'recon_zq': 0., 'som': 0., 'commit': 0., 'dir': 0., 'multimodal': 0.}
        global_iter0 = global_iter

        for iter, sample in enumerate(trainDataLoader, 0):
            global_iter += 1

            feat_m1t1, feat_m2t1, _, feat_m1t2, feat_m2t2, _, interval = sample
            interval = interval.to(config['device'], dtype=torch.float)

            # run moddality 1
            feat_m1t1 = feat_m1t1.to(config['device'], dtype=torch.float)
            feat_m1t2 = feat_m1t2.to(config['device'], dtype=torch.float)
            recons_m1, zs_m1 = model_m1.forward_pair_z(feat_m1t1, feat_m1t2, interval)

            # run moddality 2
            feat_m2t1 = feat_m2t1.to(config['device'], dtype=torch.float)
            feat_m2t2 = feat_m2t2.to(config['device'], dtype=torch.float)
            recons_m2, zs_m2 = model_m2.forward_pair_z(feat_m2t1, feat_m2t2, interval)

            # print(zs_m2[2][0])
            # pdb.set_trace()

            def compute_losses(model, feat1, feat2, recons, zs, interval, loss_all_dict, modality="m1"):
                recon_ze1, recon_ze2 = recons[0][0], recons[0][1]
                recon_zq1, recon_zq2 = recons[1][0], recons[1][1]
                z_e1, z_e2, z_e_diff = zs[0][0], zs[0][1], zs[0][2]
                z_q1, z_q2 = zs[1][0], zs[1][1]
                k1, k2 = zs[2][0], zs[2][1]
                sim1, sim2 = zs[3][0], zs[3][1]

                if modality == "m1":
                    lambda_commit = config['lambda_commit_m1']
                    lambda_som = config['lambda_som_m1']
                    lambda_dir = config['lambda_dir_m1']
                else:
                    lambda_commit = config['lambda_commit_m2']
                    lambda_som = config['lambda_som_m2']
                    lambda_dir = config['lambda_dir_m2']
                # loss
                loss = 0
                if config['lambda_recon'] > 0:
                    loss_recon = 0.5 * (model.compute_recon_loss(feat1, recon_ze1) + model.compute_recon_loss(feat2, recon_ze2))
                    loss += config['lambda_recon'] * loss_recon
                else:
                    loss_recon = torch.tensor(0.)

                if config['lambda_recon_zq'] > 0 and epoch >= config['warmup_epochs']:
                    loss_recon_zq = 0.5 * (model.compute_recon_loss(feat1, recon_zq1) + model.compute_recon_loss(feat2, recon_zq2))
                    loss += config['lambda_recon_zq'] * loss_recon_zq
                else:
                    loss_recon_zq = torch.tensor(0.)

                if lambda_commit > 0 and epoch >= config['warmup_epochs']:
                    loss_commit = 0.5 * (model.compute_commit_loss(z_e1, z_q1) + model.compute_commit_loss(z_e2, z_q2))
                    loss += lambda_commit * loss_commit
                else:
                    loss_commit = torch.tensor(0.)

                if lambda_som > 0 and epoch >= config['warmup_epochs']:
                    loss_som = 0.5 * (model.compute_som_loss(z_e1, k1, global_iter-config['warmup_epochs']*len(trainDataLoader), iter_max) + \
                                    model.compute_som_loss(z_e2, k2, global_iter-config['warmup_epochs']*len(trainDataLoader), iter_max))
                    loss += lambda_som * loss_som
                else:
                    loss_som = torch.tensor(0.)

                if lambda_dir > 0 and epoch >= config['warmup_epochs']:
                    loss_dir = model.compute_direction_loss(sim1, sim2, alpha=config['dir_thres'])
                    loss += lambda_dir * loss_dir
                else:
                    loss_dir = torch.tensor(0.)


                loss_all_dict['all'] += loss.item()
                loss_all_dict['recon'] += loss_recon.item()
                loss_all_dict['recon_zq'] += loss_recon_zq.item()
                loss_all_dict['commit'] += loss_commit.item()
                loss_all_dict['som'] += loss_som.item()
                loss_all_dict['dir'] += loss_dir.item()
                # print(loss_recon)

                return loss

            loss_m1 = compute_losses(model_m1, feat_m1t1, feat_m1t2, recons_m1, zs_m1, interval, loss_all_dict_m1)
            # cross-modal loss
            if config['lambda_multimodal'] > 0 and epoch >= config['warmup_epochs']:
                loss_multimodal = 0.5 * (model_m1.compute_direction_loss(zs_m1[3][0], zs_m2[3][0], alpha=config['multimodal_thres']) + \
                                    model_m1.compute_direction_loss(zs_m1[3][1], zs_m2[3][1], alpha=config['multimodal_thres']))
                loss_m1 += config['lambda_multimodal'] * loss_multimodal
                loss_all_dict_m1['multimodal'] += loss_multimodal.item()

            if config['jointly_train']:
                loss_m2 = compute_losses(model_m2, feat_m2t1, feat_m2t2, recons_m2, zs_m2, interval, loss_all_dict_m2, modality="m2")
                loss = loss_m1 + config['lambda_m2_ratio'] * loss_m2
            else:
                loss = loss_m1
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            # add this for ROI
            with torch.no_grad():
                model_m1.embeddings.data.div_(torch.norm(model_m1.embeddings.data, dim=2, keepdim=True))
                model_m2.embeddings.data.div_(torch.norm(model_m2.embeddings.data, dim=2, keepdim=True))

        # save train result
        num_iter = global_iter - global_iter0
        for key in loss_all_dict_m1.keys():
            loss_all_dict_m1[key] /= num_iter
        if 'k1' not in loss_all_dict_m1:
            loss_all_dict_m1['k1'] = 0
        if 'k2' not in loss_all_dict_m1:
            loss_all_dict_m1['k2'] = 0

        for key in loss_all_dict_m2.keys():
            loss_all_dict_m2[key] /= num_iter
        if 'k1' not in loss_all_dict_m2:
            loss_all_dict_m2['k1'] = 0
        if 'k2' not in loss_all_dict_m2:
            loss_all_dict_m2['k2'] = 0

        save_result_stat(loss_all_dict_m1, config, info='modality1-epoch[%2d]'%(epoch))
        print(loss_all_dict_m1)
        save_result_stat(loss_all_dict_m2, config, info='modality2-epoch[%2d]'%(epoch))
        print(loss_all_dict_m2)

        # validation
        # pdb.set_trace()
        stat_m1, stat_m2 = evaluate(phase='val', set='val', save_res=False, epoch=epoch)
        # monitor_metric = stat['all']
        monitor_metric = stat_m1['recon']
        scheduler.step(monitor_metric)
        save_result_stat(stat_m1, config, info='modality1-val')
        print(stat_m1)
        save_result_stat(stat_m2, config, info='modality2-val')
        print(stat_m2)

        # save ckp
        # is_best = False
        # if monitor_metric <= monitor_metric_best:
        #     is_best = True
        #     monitor_metric_best = monitor_metric if is_best == True else monitor_metric_best
    state = {'epoch': epoch, 'monitor_metric': monitor_metric, 'stat_m1': stat_m1, 'stat_m2': stat_m2, \
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), \
            'model_m1': model_m1.state_dict(), 'model_m2': model_m2.state_dict()}
    print(optimizer.param_groups[0]['lr'])
    save_checkpoint(state, True, config['ckpt_path'])

def evaluate(phase='val', set='val', save_res=True, info='', epoch=0):
    model_m1.eval()
    model_m2.eval()

    if phase == 'val':
        loader = valDataLoader
    else:
        if set == 'train':
            loader = trainDataLoader
        elif set == 'val':
            loader = valDataLoader
        elif set == 'test':
            loader = testDataLoader
        else:
            raise ValueError('Undefined loader')

    res_path = os.path.join(config['ckpt_path'], 'result_'+set)
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    path = os.path.join(res_path, 'results_all'+info+'.h5')
    if os.path.exists(path):
        os.remove(path)

    loss_all_dict_m1 = {'all': 0, 'recon': 0., 'recon_zq': 0., 'som': 0., 'commit': 0., 'dir': 0, 'multimodal': 0}
    loss_all_dict_m2 = {'all': 0, 'recon': 0., 'recon_zq': 0., 'som': 0., 'commit': 0., 'dir': 0, 'multimodal': 0}
    csv_idx1_list = []
    csv_idx2_list = []
    interval_list = []
    ze_m1t1_list = []
    ze_m1t2_list = []
    ze_m2t1_list = []
    ze_m2t2_list = []
    ze_diff_m1_list = []
    ze_diff_m2_list = []
    k_m1t1_list = []
    k_m1t2_list = []
    k_m2t1_list = []
    k_m2t2_list = []
    sim_m1t1_list = []
    sim_m1t2_list = []
    sim_m2t1_list = []
    sim_m2t2_list = []
    # subj_id_list = []
    # case_order_list = []

    with torch.no_grad():
        # recon_emb = model.recon_embeddings()

        for iter, sample in enumerate(loader, 0):
            # if iter > 10:
            #     break

            feat_m1t1, feat_m2t1, csv_idx1, feat_m1t2, feat_m2t2, csv_idx2, interval = sample
            interval = interval.to(config['device'], dtype=torch.float)

            # run moddality 1
            feat_m1t1 = feat_m1t1.to(config['device'], dtype=torch.float)
            feat_m1t2 = feat_m1t2.to(config['device'], dtype=torch.float)
            recons_m1, zs_m1 = model_m1.forward_pair_z(feat_m1t1, feat_m1t2, interval)

            # run moddality 2
            feat_m2t1 = feat_m2t1.to(config['device'], dtype=torch.float)
            feat_m2t2 = feat_m2t2.to(config['device'], dtype=torch.float)
            recons_m2, zs_m2 = model_m2.forward_pair_z(feat_m2t1, feat_m2t2, interval)

            def compute_losses(model, feat1, feat2, recons, zs, interval, loss_all_dict, modality="m1"):
                recon_ze1, recon_ze2 = recons[0][0], recons[0][1]
                recon_zq1, recon_zq2 = recons[1][0], recons[1][1]
                z_e1, z_e2, z_e_diff = zs[0][0], zs[0][1], zs[0][2]
                z_q1, z_q2 = zs[1][0], zs[1][1]
                k1, k2 = zs[2][0], zs[2][1]
                sim1, sim2 = zs[3][0], zs[3][1]

                if modality == "m1":
                    lambda_commit = config['lambda_commit_m1']
                    lambda_som = config['lambda_som_m1']
                    lambda_dir = config['lambda_dir_m1']
                else:
                    lambda_commit = config['lambda_commit_m2']
                    lambda_som = config['lambda_som_m2']
                    lambda_dir = config['lambda_dir_m2']

                # loss
                loss = 0
                if config['lambda_recon'] > 0:
                    loss_recon = 0.5 * (model.compute_recon_loss(feat1, recon_ze1) + model.compute_recon_loss(feat2, recon_ze2))
                    loss += config['lambda_recon'] * loss_recon
                else:
                    loss_recon = torch.tensor(0.)

                if config['lambda_recon_zq'] > 0 and epoch >= config['warmup_epochs']:
                    loss_recon_zq = 0.5 * (model.compute_recon_loss(feat1, recon_zq1) + model.compute_recon_loss(feat2, recon_zq2))
                    loss += config['lambda_recon_zq'] * loss_recon_zq
                else:
                    loss_recon_zq = torch.tensor(0.)

                if lambda_commit > 0 and epoch >= config['warmup_epochs']:
                    loss_commit = 0.5 * (model.compute_commit_loss(z_e1, z_q1) + model.compute_commit_loss(z_e2, z_q2))
                    loss += lambda_commit * loss_commit
                else:
                    loss_commit = torch.tensor(0.)

                if lambda_som > 0 and epoch >= config['warmup_epochs']:
                    loss_som = 0.5 * (model.compute_som_loss(z_e1, k1) + model.compute_som_loss(z_e2, k2))
                    loss += lambda_som * loss_som
                else:
                    loss_som = torch.tensor(0.)

                if lambda_dir > 0 and epoch >= config['warmup_epochs']:
                    loss_dir = model.compute_direction_loss(sim1, sim2, alpha=config['dir_thres'])
                    loss += lambda_dir * loss_dir
                else:
                    loss_dir = torch.tensor(0.)

                loss_all_dict['all'] += loss.item()
                loss_all_dict['recon'] += loss_recon.item()
                loss_all_dict['recon_zq'] += loss_recon_zq.item()
                loss_all_dict['commit'] += loss_commit.item()
                loss_all_dict['som'] += loss_som.item()
                loss_all_dict['dir'] += loss_dir.item()
                return loss


            loss_m1 = compute_losses(model_m1, feat_m1t1, feat_m1t2, recons_m1, zs_m1, interval, loss_all_dict_m1)
            # cross-modal loss
            if config['lambda_multimodal'] > 0 and epoch >= config['warmup_epochs']:
                loss_multimodal = 0.5 * (model_m1.compute_direction_loss(zs_m1[3][0], zs_m2[3][0], alpha=config['multimodal_thres']) + \
                                    model_m1.compute_direction_loss(zs_m1[3][1], zs_m2[3][1], alpha=config['multimodal_thres']))
                loss_m1 += config['lambda_multimodal'] * loss_multimodal
                loss_all_dict_m1['multimodal'] += loss_multimodal.item()

            if config['jointly_train']:
                loss_m2 = compute_losses(model_m2, feat_m2t1, feat_m2t2, recons_m2, zs_m2, interval, loss_all_dict_m2, modality="m2")

            if phase == 'test' and save_res:
                ze_m1t1_list.append(zs_m1[0][0].detach().cpu().numpy())
                ze_m1t2_list.append(zs_m1[0][1].detach().cpu().numpy())
                ze_diff_m1_list.append(zs_m1[0][2].detach().cpu().numpy())
                sim_m1t1_list.append(zs_m1[3][0].detach().cpu().numpy())
                sim_m1t2_list.append(zs_m1[3][1].detach().cpu().numpy())
                ze_m2t1_list.append(zs_m2[0][0].detach().cpu().numpy())
                ze_m2t2_list.append(zs_m2[0][1].detach().cpu().numpy())
                ze_diff_m2_list.append(zs_m2[0][2].detach().cpu().numpy())
                sim_m2t1_list.append(zs_m2[3][0].detach().cpu().numpy())
                sim_m2t2_list.append(zs_m2[3][1].detach().cpu().numpy())
                interval_list.append(interval.detach().cpu().numpy())
                csv_idx1_list.append(csv_idx1.numpy())
                csv_idx2_list.append(csv_idx2.numpy())

            k_m1t1_list.append(zs_m1[2][0].detach().cpu().numpy())
            k_m1t2_list.append(zs_m1[2][1].detach().cpu().numpy())
            k_m2t1_list.append(zs_m2[2][0].detach().cpu().numpy())
            k_m2t2_list.append(zs_m2[2][1].detach().cpu().numpy())

        for key in loss_all_dict_m1.keys():
            loss_all_dict_m1[key] /= (iter + 1)
        for key in loss_all_dict_m2.keys():
            loss_all_dict_m2[key] /= (iter + 1)

        if phase == 'test' and save_res:
            csv_idx1_list = np.concatenate(csv_idx1_list, axis=0)
            csv_idx2_list = np.concatenate(csv_idx2_list, axis=0)
            interval_list = np.concatenate(interval_list, axis=0)
            ze_m1t1_list = np.concatenate(ze_m1t1_list, axis=0)
            ze_m1t2_list = np.concatenate(ze_m1t2_list, axis=0)
            ze_diff_m1_list = np.concatenate(ze_diff_m1_list, axis=0)
            k_m1t1_list = np.concatenate(k_m1t1_list, axis=0)
            k_m1t2_list = np.concatenate(k_m1t2_list, axis=0)
            sim_m1t1_list = np.concatenate(sim_m1t1_list, axis=0)
            sim_m1t2_list = np.concatenate(sim_m1t2_list, axis=0)

            ze_m2t1_list = np.concatenate(ze_m2t1_list, axis=0)
            ze_m2t2_list = np.concatenate(ze_m2t2_list, axis=0)
            ze_diff_m2_list = np.concatenate(ze_diff_m2_list, axis=0)
            k_m2t1_list = np.concatenate(k_m2t1_list, axis=0)
            k_m2t2_list = np.concatenate(k_m2t2_list, axis=0)
            sim_m2t1_list = np.concatenate(sim_m2t1_list, axis=0)
            sim_m2t2_list = np.concatenate(sim_m2t2_list, axis=0)

            h5_file = h5py.File(path, 'w')
            # h5_file.create_dataset('subj_id', data=subj_id_list)
            # h5_file.create_dataset('case_order', data=case_order_list)
            h5_file.create_dataset('csv_idx1', data=csv_idx1_list)
            h5_file.create_dataset('csv_idx2', data=csv_idx2_list)
            h5_file.create_dataset('interval', data=interval_list)
            h5_file.create_dataset('ze_m1t1', data=ze_m1t1_list)
            h5_file.create_dataset('ze_m1t2', data=ze_m1t2_list)
            h5_file.create_dataset('ze_diff_m1', data=ze_diff_m1_list)
            h5_file.create_dataset('k_m1t1', data=k_m1t1_list)
            h5_file.create_dataset('k_m1t2', data=k_m1t2_list)
            h5_file.create_dataset('sim_m1t1', data=sim_m1t1_list)
            h5_file.create_dataset('sim_m1t2', data=sim_m1t2_list)
            h5_file.create_dataset('embeddings_m1', data=model_m1.embeddings.detach().cpu().numpy())

            h5_file.create_dataset('ze_m2t1', data=ze_m2t1_list)
            h5_file.create_dataset('ze_m2t2', data=ze_m2t2_list)
            h5_file.create_dataset('ze_diff_m2', data=ze_diff_m2_list)
            h5_file.create_dataset('k_m2t1', data=k_m2t1_list)
            h5_file.create_dataset('k_m2t2', data=k_m2t2_list)
            h5_file.create_dataset('sim_m2t1', data=sim_m2t1_list)
            h5_file.create_dataset('sim_m2t2', data=sim_m2t2_list)
            h5_file.create_dataset('embeddings_m2', data=model_m2.embeddings.detach().cpu().numpy())
            # h5_file.create_dataset('recon_emb', data=recon_emb.detach().cpu().numpy())
        else:
            k_m1t1_list = np.concatenate(k_m1t1_list, axis=0)
            k_m1t2_list = np.concatenate(k_m1t2_list, axis=0)
            k_m2t1_list = np.concatenate(k_m2t1_list, axis=0)
            k_m2t2_list = np.concatenate(k_m2t2_list, axis=0)

        print('modality 1, Number of used embeddings:', np.unique(k_m1t1_list).shape, np.unique(k_m1t2_list).shape)
        loss_all_dict_m1['k1'] = np.unique(k_m1t1_list).shape[0]
        loss_all_dict_m1['k2'] = np.unique(k_m1t2_list).shape[0]
        print('modality 2, Number of used embeddings:', np.unique(k_m2t1_list).shape, np.unique(k_m2t2_list).shape)
        loss_all_dict_m2['k1'] = np.unique(k_m2t1_list).shape[0]
        loss_all_dict_m2['k2'] = np.unique(k_m2t2_list).shape[0]

    return loss_all_dict_m1, loss_all_dict_m2

if config['phase'] == 'train':
    train()

stat = evaluate(phase='test', set='test', save_res=True)
print('Test')
print(stat)
stat = evaluate(phase='test', set='train', save_res=True)
print('Train')
print(stat)
