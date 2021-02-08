# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2021/1/8

import torch
import torch.nn.functional as F

'''
1) 一般的自动编码器
2）使用卷积的自动编码器
3）变分自动编码器

https://www.jeremyjordan.me/autoencoders/

VAE：
https://zhuanlan.zhihu.com/p/151587288
https://zhuanlan.zhihu.com/p/55557709
https://www.kaggle.com/hone5com/fraud-detection-with-variational-autoencoder
https://ravirajag.dev/machine%20learning/data%20science/deep%20learning/generative/neural%20network/encoder/variational%20autoencoder/2019/02/09/vanillavae.html
https://github.com/ethanluoyc/pytorch-vae/blob/master/vae.py
https://www.cnblogs.com/jiangkejie/p/11179901.html
https://wmathor.com/index.php/archives/1407/
'''


class AutoEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout=0.5):
        super(AutoEncoder, self).__init__()
        layers, _input_dim = [], input_dim
        for h_dim in hidden_layers:
            layers.append(torch.nn.Linear(input_dim, h_dim))
            layers.append(torch.nn.BatchNorm1d(h_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = h_dim
        self.encoder = torch.nn.Sequential(*layers)
        layers, de_hidden_layers = [], hidden_layers[:-1][::-1] + [_input_dim]
        for h_dim in de_hidden_layers:
            layers.append(torch.nn.Linear(input_dim, h_dim))
            layers.append(torch.nn.BatchNorm1d(h_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = h_dim
        self.decoder = torch.nn.Sequential(*layers)

    def forward(self, x):
        x_encode = self.encoder(x)
        x_decode = self.decoder(x_encode)
        return x_decode



class VariationAutoEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout=0.5):
        super(VariationAutoEncoder, self).__init__()
        layers, _input_dim = [], input_dim
        for h_dim in hidden_layers:
            layers.append(torch.nn.Linear(input_dim, h_dim))
            layers.append(torch.nn.BatchNorm1d(h_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = h_dim
        self.encoder = torch.nn.Sequential(*layers)
        layers = []
        de_input_dim = hidden_layers[-1] // 2
        de_hidden_layers = hidden_layers[:-1][::-1] + [_input_dim]
        for h_dim in de_hidden_layers:
            layers.append(torch.nn.Linear(de_input_dim, h_dim))
            layers.append(torch.nn.BatchNorm1d(h_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            de_input_dim = h_dim
        self.decoder = torch.nn.Sequential(*layers)

    def forward(self, x):
        x_encode = self.encoder(x)
        mu, sigma = x_encode.chunk(2, dim=1)
        z = mu + sigma * torch.rand_like(sigma)
        x_decode = self.decoder(z)
        return x_decode, mu, sigma




if __name__ == '__main__':

    import pandas as pd
    import numpy as np
    from datetime import datetime
    import argparse
    import os
    import json
    import torch
    import torchsummary
    import tqdm
    from torch.utils.tensorboard import SummaryWriter
    from torch.utils.data import DataLoader

    from src.utils import setup_seed, EarlyStopper
    from src.dataset import CardFraudDataSet
    from src.nn.autoencoders import AutoEncoder

    setup_seed(2020)

    PATH = "/home/mi/PycharmProjects/ml-mining"
    MODEL_PATH = lambda x: PATH + "/resources/ae_files/" + str(x)
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    def train_ae(model, optimizer, data_loader, criterion, device, log_interval=0):
        model.train()
        total_loss = 0
        tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
        for i, (X_batch, y_batch) in enumerate(tk0):
            # 无监督学习
            X_batch = X_batch.float().to(device)
            X_out = model(X_batch)
            loss = criterion(X_batch, X_out.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if log_interval:
                if (i + 1) % log_interval == 0:
                    tk0.set_postfix(loss=total_loss / log_interval)
                    total_loss = 0
        return total_loss / len(data_loader)


    def test_ae(model, data_loader, criterion, device):
        model.eval()
        total_mse_loss, total_mae_loss = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.float().to(device)
                X_out = model(X_batch)
                loss = criterion(X_batch, X_out.float())
                total_mse_loss += loss.item()
                total_mae_loss += reconstruct_error(X_batch.numpy(), X_out.numpy(), 'mae', True)
        return total_mse_loss / len(data_loader), total_mae_loss / len(data_loader)


    def reconstruct_error(X_org, X_new, method='mse', reduce=False):
        '''MSE'''
        if method == 'mse':
            error = np.mean(np.power(X_org - X_new, 2), axis=1)
        elif method == 'mae':
            error = np.mean(np.abs(X_org - X_new), axis=1)
        else:
            raise ValueError
        if reduce:
            return np.mean(error)
        return error


    def get_reconstruct_error(model, data_loader, device):

        output_X, mse_error, mae_error = [], [], []
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(device)
                X_out = model(X_batch.float())
                X_batch, X_out = X_batch.numpy(), X_out.numpy()
                mse_out = reconstruct_error(X_batch, X_out, 'mse')
                mae_out = reconstruct_error(X_batch, X_out, 'mae')
                output_X.append(X_out)
                mse_error.append(mse_out)
                mae_error.append(mae_out)
        return np.concatenate(output_X, axis=0), np.concatenate(mse_error, axis=0), np.concatenate(mae_error, axis=0)


    def main(args):
        MODE = args.mode
        MODE_TYPE = args.model_type
        BATCH_SIZE = args.batch_size
        EPOCHS = args.epochs

        NUM_TRIALS = args.num_trials
        INPUT_DIM = args.input_dim
        HIDDEN_LAYERS = eval(args.hidden_layers)
        DROPOUT = args.dropout
        LR = args.lr
        WEIGHT_DECAY = args.weight_decay

        TAG = args.tag if args.tag else datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
        MODEL_NAME = '{0}_{1}.pt'.format(TAG, MODE_TYPE)
        MODEL_PARAMS = args.__dict__.copy()

        if MODE == 'train':
            train = pd.read_csv(PATH + '/data/credit_card_train.csv')
            valid = pd.read_csv(PATH + '/data/credit_card_valid.csv')
            train_ds = CardFraudDataSet(train, label_name='Class', feature_names=[f'V{i + 1}' for i in range(28)])
            valid_ds = CardFraudDataSet(valid, label_name='Class', feature_names=[f'V{i + 1}' for i in range(28)])
            train_datalodaer = DataLoader(train_ds, shuffle=True, batch_size=BATCH_SIZE)
            valid_datalodaer = DataLoader(valid_ds, shuffle=True, batch_size=BATCH_SIZE)

            writer = SummaryWriter(PATH + '/logs/ae/{}'.format(TAG.replace(' ', '').replace('-', '').replace(':', '')))

            model = AutoEncoder(input_dim=INPUT_DIM, hidden_layers=HIDDEN_LAYERS, dropout=DROPOUT).to(DEVICE)

            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

            early_stopper = EarlyStopper(num_trials=NUM_TRIALS, save_path=MODEL_PATH(MODEL_NAME), is_better=False)
            best_loss = np.inf
            for epoch in range(EPOCHS):
                train_loss_ = train_ae(model, optimizer, train_datalodaer, criterion, device=DEVICE)
                train_mse_loss, train_mae_loss = test_ae(model, train_datalodaer, criterion, device=DEVICE)
                valid_mse_loss, valid_mae_loss = test_ae(model, valid_datalodaer, criterion, device=DEVICE)
                print(
                    f'epoch {epoch + 1}, train loss {round(train_mse_loss, 4)} - {round(train_mae_loss, 4)}, valid loss {round(valid_mse_loss, 4)} - {round(valid_mae_loss, 4)} \n')
                writer.add_scalars('mse_loss', {'train': train_mse_loss, 'valid': valid_mse_loss}, global_step=epoch)
                writer.add_scalars('mae_loss', {'train': train_mae_loss, 'valid': valid_mae_loss}, global_step=epoch)
                if NUM_TRIALS > 0:
                    if not early_stopper.is_continuable(model, valid_mae_loss,
                                                        losses=(train_mae_loss, valid_mae_loss, None), epoch=epoch + 1):
                        print('early stop. \n')
                        best_valid_losses = early_stopper.get_losses()
                        MODEL_PARAMS.update({'best_losses': best_valid_losses,
                                             'best_epoch': early_stopper.best_epoch})
                        break
                else:
                    if valid_mae_loss < best_loss:
                        torch.save(model, MODEL_PATH(MODEL_NAME))
                        MODEL_PARAMS.update({'best_losses': [train_mae_loss, valid_mae_loss, None],
                                             'best_epoch': epoch + 1})

            writer.close()

            with open(MODEL_PATH(TAG + '_params.json'), 'w', encoding='utf8') as f:
                MODEL_PARAMS.update({'model_name': MODEL_NAME, 'tag': TAG})
                json.dump(MODEL_PARAMS, f)

        elif MODE == 'inference':
            INFER_OUTPUT_PATH = args.infer_output_path
            INFER_MODEL_PATH = args.infer_model_path
            INFER_MODEL_TAG = args.infer_model_tag
            INFER_DATA_PATH = args.infer_data_path

            with open(INFER_MODEL_PATH + f'{INFER_MODEL_TAG}_params.json', 'r', encoding='utf8') as f:
                infer_model_params = json.load(f)

            infer_data = pd.read_csv(INFER_DATA_PATH)
            infer_data_ds = CardFraudDataSet(infer_data, 'Class', [f'V{i + 1}' for i in range(28)])
            infer_dataloader = DataLoader(infer_data_ds, shuffle=False, batch_size=BATCH_SIZE)

            model = torch.load(INFER_MODEL_PATH + f"{infer_model_params['model_name']}")

            reconstruct_X, mse_error, mae_error = get_reconstruct_error(model, infer_dataloader, device=DEVICE)

            infer_data['mse'] = mse_error
            infer_data['mae'] = mae_error
            infer_data[['Class', 'mse', 'mae']].to_csv(
                INFER_OUTPUT_PATH + f"{infer_model_params['model_name']}_error.csv", index=False)

        elif MODE == 'test':
            import matplotlib.pyplot as plt
            from sklearn.metrics import roc_curve, precision_recall_curve, auc
            mse_data = pd.read_csv("")

            markers = ['o', '^']
            colors = ['dodgerblue', 'coral']
            labels = ['Non-fraud', 'Fraud']

            plt.figure(figsize=(12, 8))
            plt.subplot(1, 3, 1)
            for flag in [0, 1]:
                tmp = mse_data[mse_data['Class'] == flag]
                plt.scatter(tmp.index, tmp['mae'], alpha=0.7, marker=markers[flag], c=colors[flag], label=labels[flag])
            plt.title('reconstruct mse')
            plt.ylabel('mse')
            plt.xlabel('index')

            plt.subplot(1, 3, 2)
            fpr, tpr, _ = roc_curve(mse_data['Class'], mse_data['mae'])
            plt.plot(fpr, tpr, c='darkorange')
            plt.plot([0, 1], [0, 1], c='dodgerblue', ls='--')
            auc_score = auc(fpr, tpr)
            plt.title('ROC-AUC score=%.2f' % auc_score)
            plt.xlabel('tpr')
            plt.ylabel('fpr')

            plt.subplot(1, 3, 3)
            precision, recall, _ = precision_recall_curve(mse_data['Class'], mse_data['mae'])
            pr_auc_score = auc(recall, precision)
            plt.plot(recall, precision, c='darkorange')

            plt.title('PR-AUC score=%.2f' % pr_auc_score)
            plt.xlabel('recall')
            plt.ylabel('precision')

            plt.show()


    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="track behavior sequence NN model script")
        parser.add_argument("--mode", type=str, default='test')
        parser.add_argument("--tag", type=str, default=None)
        parser.add_argument("--batch_size", type=int, default=1024)
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--num_trials", type=int, default=5)
        parser.add_argument("--lr", type=float, default=0.001)

        parser.add_argument("--dropout", type=float, default=0.5)

        parser.add_argument("--model_type", type=str, default='AE')
        parser.add_argument("--input_dim", type=int, default=28)
        parser.add_argument("--hidden_layers", type=str, default='[48, 16]')
        parser.add_argument("--weight_decay", type=float, default=0)

        parser.add_argument('--infer_output_path', type=str, default="")
        parser.add_argument('--infer_model_path', type=str,
                            default="")
        parser.add_argument('--infer_model_tag', type=str, default="2021-01-10 12:27:09")
        # TODO 使用新修改的时间差文件打分
        parser.add_argument('--infer_data_path', type=str,
                            default="")

        args = parser.parse_args()
        main(args)