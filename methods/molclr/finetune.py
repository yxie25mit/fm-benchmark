import os
import shutil
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

from dataset.dataset_test import MolTestDatasetWrapper


apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False


def _save_config_file(model_checkpoints_folder, config_path):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    shutil.copy(
        config_path,
        os.path.join(model_checkpoints_folder, 'config_finetune.yaml')
    )


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """
        tensor is (N, T) for multi-target (T columns) or (N,) for single-target.
        MT patch: compute per-column mean/std with NaN-aware reduction so QM datasets
        with multiple targets and missing entries normalize correctly. (torch.nanmean
        not available in older torch; compute manually.)
        """
        if tensor.dim() == 1:
            valid = ~torch.isnan(tensor)
            x = torch.where(valid, tensor, torch.zeros_like(tensor))
            n = valid.float().sum().clamp(min=1.0)
            self.mean = x.sum() / n
            var = ((x - self.mean) ** 2 * valid.float()).sum() / n
            self.std = var.sqrt().clamp(min=1e-6)
        else:
            valid = ~torch.isnan(tensor)
            x = torch.where(valid, tensor, torch.zeros_like(tensor))
            n = valid.float().sum(dim=0).clamp(min=1.0)
            self.mean = x.sum(dim=0) / n                    # (T,)
            mean_b = self.mean.unsqueeze(0)
            var = ((x - mean_b) ** 2 * valid.float()).sum(dim=0) / n
            self.std = var.sqrt().clamp(min=1e-6)           # (T,)

    def norm(self, tensor):
        return (tensor - self.mean.to(tensor.device)) / self.std.to(tensor.device)

    def denorm(self, normed_tensor):
        return normed_tensor * self.std.to(normed_tensor.device) + self.mean.to(normed_tensor.device)

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class FineTune(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        # MT patch: target may be a list; build a string tag for log dir.
        target_field = config['dataset']['target']
        target_tag = target_field if isinstance(target_field, str) else f"mt{len(target_field)}"
        dir_name = current_time + '_' + config['task_name'] + '_' + target_tag
        log_dir = config.get('log_dir') or os.path.join('finetune', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.dataset = dataset
        # MT patch: replaced nn.CrossEntropyLoss / nn.L1Loss / nn.MSELoss with NaN-masked
        # BCEWithLogits (cls) or MSE (reg). Same MSE for QM datasets (was L1 in original
        # MolCLR) -- matches chemprop / MolFormer / chemprop forks for cross-method fairness.
        self.task_type = 'cls' if config['dataset']['task'] == 'classification' else 'reg'
        self.num_tasks = len(target_field) if not isinstance(target_field, str) else 1

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step(self, model, data, n_iter):
        # MT patch: pred is (B, num_tasks); y is (B, num_tasks) with NaN for missing.
        # Loss = NaN-masked BCEWithLogits (cls) or NaN-masked MSE (reg).
        # Replaces the original CrossEntropyLoss(pred, data.y.flatten()) for cls
        # and MSELoss/L1Loss(pred, data.y) for reg.
        __, pred = model(data)
        y = data.y.view(pred.shape[0], -1).to(pred.dtype)
        mask = ~torch.isnan(y)
        y_filled = torch.where(mask, y, torch.zeros_like(y))
        if self.task_type == 'cls':
            per = F.binary_cross_entropy_with_logits(pred, y_filled, reduction='none')
        else:
            if self.normalizer is not None:
                y_filled = self.normalizer.norm(y_filled)
            per = (pred - y_filled).pow(2)
        return (per * mask.float()).sum() / mask.float().sum().clamp(min=1.0)

    def train(self):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

        self.normalizer = None
        # MT patch: original `for d, __ in train_loader` was a tuple-unpack bug
        # (DataLoader yields one Data object). Also added qm8. Each d.y is (B, T);
        # cat to (N, T). Normalizer above is NaN-aware per-column.
        if self.config["task_name"] in ['qm7', 'qm8', 'qm9']:
            labels = torch.cat([d.y.view(-1, self.num_tasks) for d in train_loader], dim=0)
            self.normalizer = Normalizer(labels)
            print(self.normalizer.mean, self.normalizer.std, labels.shape)

        if self.config['model_type'] == 'gin':
            from models.ginet_finetune import GINet
            # MT patch: pass num_tasks so pred_head has correct out_dim.
            model = GINet(self.config['dataset']['task'], num_tasks=self.num_tasks,
                          **self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
        elif self.config['model_type'] == 'gcn':
            from models.gcn_finetune import GCN
            model = GCN(self.config['dataset']['task'], num_tasks=self.num_tasks,
                        **self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)

        layer_list = []
        for name, param in model.named_parameters():
            if 'pred_head' in name:
                print(name, param.requires_grad)
                layer_list.append(name)

        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

        optimizer = torch.optim.Adam(
            [{'params': base_params, 'lr': self.config['init_base_lr']}, {'params': params}],
            self.config['init_lr'], weight_decay=eval(self.config['weight_decay'])
        )

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level='O2', keep_batchnorm_fp32=True
            )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(
            model_checkpoints_folder,
            self.config.get('config_path', './config_finetune.yaml')
        )

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_rgr = np.inf
        best_valid_cls = 0

        for epoch_counter in range(self.config['epochs']):
            for bn, data in enumerate(train_loader):
                optimizer.zero_grad()

                data = data.to(self.device)
                loss = self._step(model, data, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print(epoch_counter, bn, loss.item())

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                if self.config['dataset']['task'] == 'classification':
                    valid_loss, valid_cls = self._validate(model, valid_loader)
                    if valid_cls > best_valid_cls:
                        # save the model weights
                        best_valid_cls = valid_cls
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                elif self.config['dataset']['task'] == 'regression':
                    valid_loss, valid_rgr = self._validate(model, valid_loader)
                    if valid_rgr < best_valid_rgr:
                        # save the model weights
                        best_valid_rgr = valid_rgr
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

        # Val metric at the checkpointed (val-best) epoch — same per-target AM
        # aggregation as _test. Used for HP selection (must not leak test).
        self.best_val_metric = (best_valid_cls
                                if self.config['dataset']['task'] == 'classification'
                                else best_valid_rgr)
        self._test(model, test_loader)
        # Save the val-best model's VAL predictions so pick_best_hp can re-score with
        # the dataset's prescribed metric (TDC). _test just reloaded model.pth (val-best).
        try:
            vp, vl, _ = self._collect_preds(model, valid_loader)
            vp = np.asarray(vp, dtype=np.float64)
            if self.task_type == 'cls' and vp.ndim == 2 and vp.shape[1] == 2:
                vp = vp[:, 1]                      # positive-class score for AUROC/AUPRC
            self.val_predictions = vp.reshape(-1, 1)
            self.val_labels = np.asarray(vl, dtype=np.float64).reshape(-1, 1)
        except Exception:
            self.val_predictions = self.val_labels = None

    def _load_pre_trained_weights(self, model):
        # Any failure here raises: silently falling back to a random init would turn a
        # pretrained-foundation-model benchmark into a from-scratch baseline unnoticed.
        # Set fine_tune_from to None/'' to request a scratch run explicitly.
        fine_tune_from = self.config.get('fine_tune_from')
        if not fine_tune_from:
            print("fine_tune_from is unset: training from scratch (explicitly requested).")
            return model

        # Resolved against this module's directory, not the cwd: the workers are launched
        # from the pipeline root, which has no ./ckpt.
        if os.path.isabs(fine_tune_from):
            checkpoints_folder = os.path.join(fine_tune_from, 'checkpoints')
        else:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoints_folder = os.path.join(module_dir, 'ckpt', fine_tune_from, 'checkpoints')
        checkpoint_path = os.path.join(checkpoints_folder, 'model.pth')
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                'MolCLR pre-trained checkpoint not found: {} '
                '(fine_tune_from={!r})'.format(checkpoint_path, fine_tune_from)
            )

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        # Filter out pred_head weights to avoid shape mismatches and ensure fresh init for finetuning
        state_dict = {k: v for k, v in state_dict.items() if 'pred_head' not in k}

        own_state = model.state_dict()
        matched_names = [name for name in state_dict if name in own_state]
        if not matched_names:
            raise RuntimeError(
                'MolCLR pre-trained checkpoint {} shares no parameter names with the '
                'model; nothing would be loaded.'.format(checkpoint_path)
            )
        model.load_my_state_dict(state_dict)
        print('Loaded pre-trained model with success: {}/{} tensors from {}'.format(
            len(matched_names), len(own_state), checkpoint_path))

        return model

    # MT patch: per-target metric helper (NaN-aware), arithmetic mean across targets.
    # Replaces the original single-AUC / single-RMSE / single-MAE in _validate/_test.
    # Pattern matches chemprop_v1_backup/train/evaluate.py (filter NaN per task; skip
    # all-0/all-1 for AUC). Aggregation is arithmetic for ALL metrics for consistency
    # with MolCLR/MolFormer/MoleculeNet convention (chemprop's geometric mean for
    # RMSE/MAE diverges from the literature; we don't follow it).
    def _mt_metric(self, predictions, labels):
        per = []
        for t in range(labels.shape[1]):
            yt, yp = labels[:, t], predictions[:, t]
            valid = ~np.isnan(yt)
            if valid.sum() == 0:
                per.append(None); continue
            yt_v, yp_v = yt[valid], yp[valid]
            try:
                if self.task_type == 'cls':
                    if yt_v.min() == yt_v.max():
                        per.append(None); continue
                    per.append(float(roc_auc_score(yt_v, yp_v)))
                else:
                    # report MAE for QM, RMSE for the rest (matches chemprop reporting +
                    # MoleculeNet); training loss is MSE for all reg (Option B).
                    if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                        per.append(float(mean_absolute_error(yt_v, yp_v)))
                    else:
                        per.append(float(mean_squared_error(yt_v, yp_v, squared=False)))
            except Exception:
                per.append(None)
        valid_vals = [v for v in per if v is not None]
        agg = float(np.mean(valid_vals)) if valid_vals else None
        return agg, per

    def _collect_preds(self, model, loader):
        # MT patch: stack (B, num_tasks) batches into (n, num_tasks). Original code
        # `extend(pred.cpu().numpy())` + `flatten()` collapsed multi-target predictions.
        # Sigmoid (single-logit BCE) replaces softmax (which assumed 2-class out_dim).
        preds, labs = [], []
        total_loss, num = 0.0, 0
        model.eval()
        with torch.no_grad():
            for bn, data in enumerate(loader):
                data = data.to(self.device)
                _, pred = model(data)
                loss = self._step(model, data, bn)
                total_loss += loss.item() * data.y.size(0)
                num += data.y.size(0)
                if self.task_type == 'reg' and self.normalizer is not None:
                    pred = self.normalizer.denorm(pred)
                if self.task_type == 'cls':
                    pred = torch.sigmoid(pred)
                y = data.y.view(pred.shape[0], -1)
                preds.append(pred.detach().cpu().numpy())
                labs.append(y.detach().cpu().numpy())
        model.train()
        return np.concatenate(preds, axis=0), np.concatenate(labs, axis=0), total_loss / max(num, 1)

    def _validate(self, model, valid_loader):
        predictions, labels, valid_loss = self._collect_preds(model, valid_loader)
        agg, _ = self._mt_metric(predictions, labels)
        print(f'Validation loss: {valid_loss:.4f}  agg_metric: {agg}')
        return valid_loss, agg

    def _test(self, model, test_loader):
        model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("Loaded trained model with success.")
        predictions, labels, test_loss = self._collect_preds(model, test_loader)
        agg, per = self._mt_metric(predictions, labels)
        print(f'Test loss: {test_loss:.4f}  agg_metric: {agg}  per_target: {per}')
        # Stash for caller (worker) to save predictions matrix + per-target metrics.
        self.test_metric = agg
        self.test_per_target = per
        self.test_predictions = predictions
        self.test_labels = labels
        # back-compat scalars (some callers read these names):
        if self.task_type == 'cls':
            self.roc_auc = agg
        else:
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                self.mae = agg
            else:
                self.rmse = agg


def main(config):
    dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'])

    fine_tune = FineTune(dataset, config)
    fine_tune.train()
    
    if config['dataset']['task'] == 'classification':
        return fine_tune.roc_auc
    if config['dataset']['task'] == 'regression':
        if config['task_name'] in ['qm7', 'qm8', 'qm9']:
            return fine_tune.mae
        else:
            return fine_tune.rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add this argument so we can pass unique config files
    parser.add_argument('--config', type=str, default='config_finetune.yaml', help='Path to config file')
    args = parser.parse_args()

    # Load the config from the argument, not the hardcoded string
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    config['config_path'] = args.config
    #config = yaml.load(open("config_finetune.yaml", "r"), Loader=yaml.FullLoader)

    if config['task_name'] == 'BBBP':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/bbbp/BBBP.csv'
        target_list = ["p_np"]

    elif config['task_name'] == 'Tox21':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/tox21/tox21.csv'
        target_list = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", 
            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]

    elif config['task_name'] == 'ClinTox':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/clintox/clintox.csv'
        target_list = ['CT_TOX', 'FDA_APPROVED']

    elif config['task_name'] == 'HIV':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/hiv/HIV.csv'
        target_list = ["HIV_active"]

    elif config['task_name'] == 'BACE':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/bace/bace.csv'
        target_list = ["Class"]

    elif config['task_name'] == 'SIDER':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/sider/sider.csv'
        target_list = [
            "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues", 
            "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders", 
            "Gastrointestinal disorders", "Social circumstances", "Immune system disorders", 
            "Reproductive system and breast disorders", 
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)", 
            "General disorders and administration site conditions", "Endocrine disorders", 
            "Surgical and medical procedures", "Vascular disorders", 
            "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders", 
            "Congenital, familial and genetic disorders", "Infections and infestations", 
            "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders", 
            "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions", 
            "Ear and labyrinth disorders", "Cardiac disorders", 
            "Nervous system disorders", "Injury, poisoning and procedural complications"
        ]
    
    elif config['task_name'] == 'MUV':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/muv/muv.csv'
        target_list = [
            'MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644', 'MUV-548', 'MUV-852',
            'MUV-600', 'MUV-810', 'MUV-712', 'MUV-737', 'MUV-858', 'MUV-713', 'MUV-733',
            'MUV-652', 'MUV-466', 'MUV-832'
        ]

    elif config['task_name'] == 'FreeSolv':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/freesolv/freesolv.csv'
        target_list = ["expt"]
    
    elif config["task_name"] == 'ESOL':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/esol/esol.csv'
        target_list = ["measured log solubility in mols per litre"]

    elif config["task_name"] == 'Lipo':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/lipophilicity/Lipophilicity.csv'
        target_list = ["exp"]
    
    elif config["task_name"] == 'qm7':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/qm7/qm7.csv'
        target_list = ["u0_atom"]

    elif config["task_name"] == 'qm8':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/qm8/qm8.csv'
        target_list = [
            "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", 
            "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM","f2-CAM"
        ]
    
    elif config["task_name"] == 'qm9':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/qm9/qm9.csv'
        target_list = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv']

    else:
        raise ValueError('Undefined downstream task!')

    print(config)

    results_list = []
    for target in target_list:
        config['dataset']['target'] = target
        result = main(config)
        results_list.append([target, result])

    os.makedirs('experiments', exist_ok=True)
    df = pd.DataFrame(results_list)
    df.to_csv(
        'experiments/{}_{}_finetune.csv'.format(config['fine_tune_from'], config['task_name']), 
        mode='a', index=False, header=False
    )