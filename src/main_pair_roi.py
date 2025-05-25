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

_, config = load_config_yaml('config_pair_roi.yaml')
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
    print('Load config ', os.path.join(config['ckpt_path'], 'config_pair_clean.yaml'))
    flag, config_load = load_config_yaml(os.path.join(config['ckpt_path'], 'config_pair_clean.yaml'))
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
        cols_sel = [col for col in df if 'ST' in col and col != 'STATUS' and '_' not in col and len(col)<8]
    elif mode == 'Amyloid':
        cols_sel = [col for col in df if '_SUVR' in col]
    print('Num. of image features', len(cols_sel))
    return cols_sel

cols_sel = select_feature_columns(config['image_type'])
config['num_features'] = len(cols_sel)
features =  np.array(df_sel[cols_sel])
days_after_baseline = np.array(df_sel['days_after_bl'])

train_feat_1 = features[train_idx_list_1]
train_feat_2 = features[train_idx_list_2]
train_interval = (days_after_baseline[train_idx_list_2] - days_after_baseline[train_idx_list_1]) / 365.0

val_feat_1 = features[val_idx_list_1]
val_feat_2 = features[val_idx_list_2]
val_interval = (days_after_baseline[val_idx_list_2] - days_after_baseline[val_idx_list_1]) / 365.0

test_feat_1 = features[test_idx_list_1]
test_feat_2 = features[test_idx_list_2]
test_interval = (days_after_baseline[test_idx_list_2] - days_after_baseline[test_idx_list_1]) / 365.0

# define loader
# the five tensors are:
# features of timestep 1,
# index of timestep 1 (not used)
# features of timestep 2,
# index of timestep 2 (not used)
# time interval between timestep 1 and 2
trainDataLoader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.tensor(train_feat_1), torch.tensor(train_idx_list_1), torch.tensor(train_feat_2), torch.tensor(train_idx_list_2), torch.tensor(train_interval)),
    batch_size=config['batch_size'],
    shuffle=True,
)
valDataLoader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.tensor(val_feat_1), torch.tensor(val_idx_list_1), torch.tensor(val_feat_2), torch.tensor(val_idx_list_2), torch.tensor(val_interval)),
    batch_size=config['batch_size'],
    shuffle=False,
)
testDataLoader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.tensor(test_feat_1), torch.tensor(test_idx_list_1), torch.tensor(test_feat_2), torch.tensor(test_idx_list_2), torch.tensor(test_interval)),
    batch_size=config['batch_size'],
    shuffle=False,
)
iter_max = len(trainDataLoader) * config['epochs']


# define model
if config['model_name'] == 'SOMPairVisitDirection':
    model = SOMPairVisitDirection(config=config).to(config['device'])
else:
    raise ValueError('Not support other models yet!')

# define optimizer
optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999), weight_decay=1e-5, amsgrad=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-5)

# load pretrained model
if config['continue_train'] or config['phase'] == 'test':
    [optimizer, scheduler, model], start_epoch = load_checkpoint_by_key([optimizer, scheduler, model], config['ckpt_path'], ['optimizer', 'scheduler', 'model'], config['device'], config['ckpt_name'])
    # [optimizer, model], start_epoch = load_checkpoint_by_key([optimizer, model], config['ckpt_path'], ['optimizer', 'model'], config['device'], config['ckpt_name'])
    # [model], start_epoch = load_checkpoint_by_key([model], config['ckpt_path'], ['model'], config['device'], config['ckpt_name'])
    print('starting lr:', optimizer.param_groups[0]['lr'])
else:
    start_epoch = -1

def train():
    global_iter = len(trainDataLoader) * (start_epoch+1)
    monitor_metric_best = 100
    start_time = time.time()

    # print(stat)
    for epoch in range(start_epoch+1, config['epochs']):
        print('Epoch', epoch)
        model.train()
        loss_all_dict = {'all': 0, 'recon': 0., 'recon_zq': 0., 'som': 0., 'commit': 0., 'dir': 0.}
        global_iter0 = global_iter

        # init emb
        if config['warmup_epochs'] == epoch and (config['init_emb'] == 'pretrained-kmeans' or config['init_emb'] == 'pretrained-kmeans-switch'):
            if os.path.exists(os.path.join(config['ckpt_path'], 'init_emb_weights.npz')):
                data = np.load(os.path.join(config['ckpt_path'], 'init_emb_weights.npz'))
                init = data['emb']
                print('Load pre-saved initialization!')
            else:
                with torch.no_grad():
                    z_e_list = []
                    delta_z_list = []
                    for _, sample in enumerate(trainDataLoader, 0):
                        feat1, _, feat2, _, interval = sample
                        feat1 = feat1.to(config['device'], dtype=torch.float)
                        feat2 = feat2.to(config['device'], dtype=torch.float)
                        interval = interval.to(config['device'], dtype=torch.float)
                        feat = torch.cat([feat1, feat2], dim=0)

                        z_e, _ = model.encoder(feat)
                        z_e_list.append(z_e.detach().cpu().numpy())

                        bs = feat1.shape[0]
                        delta_z_list.append(((z_e[bs:]-z_e[:bs]) / interval.unsqueeze(1)).detach().cpu().numpy())

                    z_e_list = np.concatenate(z_e_list, 0)
                    delta_z_list = np.concatenate(delta_z_list, 0)
                    kmeans = sklearn.cluster.KMeans(n_clusters=config['embedding_size'][0]*config['embedding_size'][1]).fit(z_e_list)
                    init = kmeans.cluster_centers_.reshape(config['embedding_size'][0], config['embedding_size'][1], -1)
                    idx = kmeans.predict(z_e_list[:z_e_list.shape[0]//2])
                    init_dz = np.concatenate([delta_z_list[idx==k].mean(0) for k in range(config['embedding_size'][0]*config['embedding_size'][1])], axis=0).reshape(config['embedding_size'][0], config['embedding_size'][1], -1)
                np.savez(os.path.join(config['ckpt_path'], 'init_emb_weights.npz'), emb=init, emb_dz=init_dz)
            model.init_embeddings_ep_weight(init)
            # model.init_embeddings_dz_ema_weight(init_dz)
            model.train()

        for iter, sample in enumerate(trainDataLoader, 0):
            global_iter += 1

            feat1, _, feat2, _, interval = sample
            feat1 = feat1.to(config['device'], dtype=torch.float)
            feat2 = feat2.to(config['device'], dtype=torch.float)
            interval = interval.to(config['device'], dtype=torch.float)

            # run model
            recons, zs = model.forward_pair_z(feat1, feat2, interval)

            recon_ze1, recon_ze2 = recons[0][0], recons[0][1]
            recon_zq1, recon_zq2 = recons[1][0], recons[1][1]
            z_e1, z_e2, z_e_diff = zs[0][0], zs[0][1], zs[0][2]
            z_q1, z_q2 = zs[1][0], zs[1][1]
            k1, k2 = zs[2][0], zs[2][1]
            sim1, sim2 = zs[3][0], zs[3][1]

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

            if config['lambda_commit'] > 0 and epoch >= config['warmup_epochs']:
                loss_commit = 0.5 * (model.compute_commit_loss(z_e1, z_q1) + model.compute_commit_loss(z_e2, z_q2))
                loss += config['lambda_commit'] * loss_commit
            else:
                loss_commit = torch.tensor(0.)

            if config['lambda_som'] > 0 and epoch >= config['warmup_epochs']:
                loss_som = 0.5 * (model.compute_som_loss(z_e1, k1, global_iter-config['warmup_epochs']*len(trainDataLoader), iter_max) + \
                                model.compute_som_loss(z_e2, k2, global_iter-config['warmup_epochs']*len(trainDataLoader), iter_max))
                loss += config['lambda_som'] * loss_som
            else:
                loss_som = torch.tensor(0.)

            if config['lambda_dir'] > 0 and epoch >= config['warmup_epochs']:
                loss_dir = model.compute_direction_loss(sim1, sim2, alpha=config['dir_thres'])
                loss += config['lambda_dir'] * loss_dir
            else:
                loss_dir = torch.tensor(0.)


            loss_all_dict['all'] += loss.item()
            loss_all_dict['recon'] += loss_recon.item()
            loss_all_dict['recon_zq'] += loss_recon_zq.item()
            loss_all_dict['commit'] += loss_commit.item()
            loss_all_dict['som'] += loss_som.item()
            loss_all_dict['dir'] += loss_dir.item()

            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for name, param in model.named_parameters():
                try:
                    if not torch.isfinite(param.grad).all():
                        pdb.set_trace()
                except:
                    continue

            optimizer.step()
            optimizer.zero_grad()

            # add this for ROI
            with torch.no_grad():
                model.embeddings.data.div_(torch.norm(model.embeddings.data, dim=2, keepdim=True))


        # save train result
        num_iter = global_iter - global_iter0
        for key in loss_all_dict.keys():
            loss_all_dict[key] /= num_iter
        if 'k1' not in loss_all_dict:
            loss_all_dict['k1'] = 0
        if 'k2' not in loss_all_dict:
            loss_all_dict['k2'] = 0

        save_result_stat(loss_all_dict, config, info='epoch[%2d]'%(epoch))
        print(loss_all_dict)

        # validation
        # pdb.set_trace()
        stat = evaluate(phase='val', set='val', save_res=False, epoch=epoch)
        # monitor_metric = stat['all']
        monitor_metric = stat['recon']
        scheduler.step(monitor_metric)
        save_result_stat(stat, config, info='val')
        print(stat)

        # save ckp
        # is_best = False
        # if monitor_metric <= monitor_metric_best:
        #     is_best = True
        #     monitor_metric_best = monitor_metric if is_best == True else monitor_metric_best
    state = {'epoch': epoch, 'monitor_metric': monitor_metric, 'stat': stat, \
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), \
            'model': model.state_dict()}
    print(optimizer.param_groups[0]['lr'])
    save_checkpoint(state, True, config['ckpt_path'])

def evaluate(phase='val', set='val', save_res=True, info='', epoch=0):
    model.eval()
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

    loss_all_dict = {'all': 0, 'recon': 0., 'recon_zq': 0., 'som': 0., 'commit': 0., 'dir': 0}
    input1_list = []
    input2_list = []
    csv_idx1_list = []
    csv_idx2_list = []
    interval_list = []
    recon1_list = []
    recon2_list = []
    recon_zq_list = []
    ze1_list = []
    ze2_list = []
    ze_diff_list = []
    zq_list = []
    k1_list = []
    k2_list = []
    sim1_list = []
    sim2_list = []
    subj_id_list = []
    case_order_list = []

    with torch.no_grad():
        # recon_emb = model.recon_embeddings()

        for iter, sample in enumerate(loader, 0):
            # if iter > 10:
            #     break

            feat1, csv_idx1, feat2, csv_idx2, interval = sample
            feat1 = feat1.to(config['device'], dtype=torch.float)
            feat2 = feat2.to(config['device'], dtype=torch.float)
            interval = interval.to(config['device'], dtype=torch.float)

            # run model
            recons, zs = model.forward_pair_z(feat1, feat2, interval)

            recon_ze1, recon_ze2 = recons[0][0], recons[0][1]
            recon_zq1, recon_zq2 = recons[1][0], recons[1][1]
            z_e1, z_e2, z_e_diff = zs[0][0], zs[0][1], zs[0][2]
            z_q1, z_q2 = zs[1][0], zs[1][1]
            k1, k2 = zs[2][0], zs[2][1]
            sim1, sim2 = zs[3][0], zs[3][1]

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

            if config['lambda_commit'] > 0 and epoch >= config['warmup_epochs']:
                loss_commit = 0.5 * (model.compute_commit_loss(z_e1, z_q1) + model.compute_commit_loss(z_e2, z_q2))
                loss += config['lambda_commit'] * loss_commit
            else:
                loss_commit = torch.tensor(0.)

            if config['lambda_som'] > 0 and epoch >= config['warmup_epochs']:
                loss_som = 0.5 * (model.compute_som_loss(z_e1, k1) + model.compute_som_loss(z_e2, k2))
                loss += config['lambda_som'] * loss_som
            else:
                loss_som = torch.tensor(0.)

            if config['lambda_dir'] > 0 and epoch >= config['warmup_epochs']:
                # pdb.set_trace()
                loss_dir = model.compute_direction_loss(sim1, sim2, alpha=config['dir_thres'])
                # loss_dir = model.compute_direction_loss(z_e1, z_e2)
                loss += config['lambda_dir'] * loss_dir
            else:
                loss_dir = torch.tensor(0.)


            loss_all_dict['all'] += loss.item()
            loss_all_dict['recon'] += loss_recon.item()
            loss_all_dict['recon_zq'] += loss_recon_zq.item()
            loss_all_dict['commit'] += loss_commit.item()
            loss_all_dict['som'] += loss_som.item()
            loss_all_dict['dir'] += loss_dir.item()

            if phase == 'test' and save_res:
                input1_list.append(feat1.detach().cpu().numpy())
                input2_list.append(feat2.detach().cpu().numpy())
                recon1_list.append(recon_ze1.detach().cpu().numpy())
                recon2_list.append(recon_ze2.detach().cpu().numpy())
                recon_zq_list.append(recon_zq1.detach().cpu().numpy())
                ze1_list.append(z_e1.detach().cpu().numpy())
                ze2_list.append(z_e2.detach().cpu().numpy())
                ze_diff_list.append(z_e_diff.detach().cpu().numpy())
                interval_list.append(interval.detach().cpu().numpy())
                csv_idx1_list.append(csv_idx1.numpy())
                csv_idx2_list.append(csv_idx2.numpy())
                zq_list.append(z_q1.detach().cpu().numpy())
                sim1_list.append(sim1.detach().cpu().numpy())
                sim2_list.append(sim2.detach().cpu().numpy())
            k1_list.append(k1.detach().cpu().numpy())
            k2_list.append(k2.detach().cpu().numpy())

        for key in loss_all_dict.keys():
            loss_all_dict[key] /= (iter + 1)

        if phase == 'test' and save_res:
            input1_list = np.concatenate(input1_list, axis=0)
            input2_list = np.concatenate(input2_list, axis=0)
            csv_idx1_list = np.concatenate(csv_idx1_list, axis=0)
            csv_idx2_list = np.concatenate(csv_idx2_list, axis=0)
            interval_list = np.concatenate(interval_list, axis=0)
            recon1_list = np.concatenate(recon1_list, axis=0)
            recon2_list = np.concatenate(recon2_list, axis=0)
            recon_zq_list = np.concatenate(recon_zq_list, axis=0)
            ze1_list = np.concatenate(ze1_list, axis=0)
            ze2_list = np.concatenate(ze2_list, axis=0)
            ze_diff_list = np.concatenate(ze_diff_list, axis=0)
            zq_list = np.concatenate(zq_list, axis=0)
            k1_list = np.concatenate(k1_list, axis=0)
            k2_list = np.concatenate(k2_list, axis=0)
            sim1_list = np.concatenate(sim1_list, axis=0)
            sim2_list = np.concatenate(sim2_list, axis=0)

            h5_file = h5py.File(path, 'w')
            # h5_file.create_dataset('subj_id', data=subj_id_list)
            h5_file.create_dataset('case_order', data=case_order_list)
            h5_file.create_dataset('input1', data=input1_list)
            h5_file.create_dataset('input2', data=input2_list)
            h5_file.create_dataset('csv_idx1', data=csv_idx1_list)
            h5_file.create_dataset('csv_idx2', data=csv_idx2_list)
            h5_file.create_dataset('interval', data=interval_list)
            h5_file.create_dataset('recon1', data=recon1_list)
            h5_file.create_dataset('recon2', data=recon2_list)
            h5_file.create_dataset('recon_zq', data=recon_zq_list)
            h5_file.create_dataset('ze1', data=ze1_list)
            h5_file.create_dataset('ze2', data=ze2_list)
            h5_file.create_dataset('ze_diff', data=ze_diff_list)
            h5_file.create_dataset('zq', data=zq_list)
            h5_file.create_dataset('k1', data=k1_list)
            h5_file.create_dataset('k2', data=k2_list)
            h5_file.create_dataset('sim1', data=sim1_list)
            h5_file.create_dataset('sim2', data=sim2_list)
            h5_file.create_dataset('embeddings', data=model.embeddings.detach().cpu().numpy())
            # h5_file.create_dataset('recon_emb', data=recon_emb.detach().cpu().numpy())
        else:
            k1_list = np.concatenate(k1_list, axis=0)
            k2_list = np.concatenate(k2_list, axis=0)

        print('Number of used embeddings:', np.unique(k1_list).shape, np.unique(k2_list).shape)
        loss_all_dict['k1'] = np.unique(k1_list).shape[0]
        loss_all_dict['k2'] = np.unique(k2_list).shape[0]

    return loss_all_dict

if config['phase'] == 'train':
    train()

stat = evaluate(phase='test', set='test', save_res=True)
print(stat)
stat = evaluate(phase='test', set='train', save_res=True)
print(stat)
