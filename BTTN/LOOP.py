from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from LOSS import gaussian_nll, kl_divergence


class Module(nn.Module):
    def __init__(self, model, prior, lr, n_epochs, device):
        super(Module, self).__init__()
        self.device = device
        self.model = model.to(device)
        self.prior = prior.to(device)
        self.n_epochs = n_epochs

        # Optimizer
        self.model_optimizer = optim.Adam(model.parameters(), lr=lr)
        self.prior_optimizer = optim.Adam(prior.parameters(), lr=lr)

    def train(self, trn_loader, val_loader):
        trn_nll_loss_list, trn_kl_loss_list = [], []
        val_nll_loss_list, val_kl_loss_list = [], []

        for epoch in range(self.n_epochs):
            # Train
            trn_nll_loss, trn_kl_loss = self._train_epoch(trn_loader, epoch)
            trn_nll_loss_list.append(trn_nll_loss)
            trn_kl_loss_list.append(trn_kl_loss)
            print(f"TRN TASK LOSS: {trn_nll_loss:.4f},\tTRN KL LOSS: {trn_kl_loss:.4f}")

            # Validation
            val_nll_loss, val_kl_loss = self._val_epoch(val_loader, epoch)
            val_nll_loss_list.append(val_nll_loss)
            val_kl_loss_list.append(val_kl_loss)
            print(f"TRN TASK LOSS: {val_nll_loss:.4f},\tTRN KL LOSS: {val_kl_loss:.4f}")

        history = dict(
            trn=(trn_nll_loss_list, trn_kl_loss_list),
            val=(val_nll_loss_list, val_kl_loss_list)
            )

        return history

    def predict(self, tst_loader, n_samples):
        self.model.eval()

        Q_idx_list, V_idx_list, target_list, pred_list = [], [], [], []

        iter_obj = tqdm(
            iterable=tst_loader, 
            desc=f"TST"
            )

        for Q_idx_batch, V_idx_batch, target_batch in iter_obj:
            Q_idx_batch = Q_idx_batch.to(self.device)
            V_idx_batch = V_idx_batch.to(self.device)
            target_batch = target_batch.to(self.device)

            # predict
            pred_batch = self.model.predict(Q_idx_batch, V_idx_batch, n_samples)

            # to cpu & save
            Q_idx_list.extend(Q_idx_batch.cpu().tolist())
            V_idx_list.extend(V_idx_batch.cpu().tolist())
            target_list.extend(target_batch.cpu().tolist())
            pred_list.extend(pred_batch.cpu().tolist())


        df_true = pd.DataFrame(
            {
                "Query": Q_idx_list,
                "Value": V_idx_list,
                "Target": target_list,
                }
            )
        df_pred = pd.DataFrame(
            {
                "Query": Q_idx_list,
                "Value": V_idx_list,
                "Predict": pred_list,
                }
            )
        result = dict(
            true=df_true,
            pred=df_pred
        )

        return result

    def _train_epoch(self, trn_loader, epoch):
        self.prior.train()
        self.model.train()

        epoch_nll_loss = 0.0
        epoch_kl_loss = 0.0

        # forward pass of prior
        all_k_idx = torch.arange(self.model.bttn.n_key)
        all_k_embed = self.model.bttn.key(all_k_idx)
        prior_shape, prior_scale = self.prior(all_k_embed)

        iter_obj = tqdm(
            iterable=trn_loader, 
            desc=f"Epoch {epoch+1}/{self.n_epochs} TRN"
            )

        for Q_idx_batch, V_idx_batch, target_batch in iter_obj:
            Q_idx_batch = Q_idx_batch.to(self.device)
            V_idx_batch = V_idx_batch.to(self.device)
            target_batch = target_batch.to(self.device)

            # forward pass of model
            mu_batch, logvar_batch = self.model(
                Q_idx=Q_idx_batch, 
                K_idx=V_idx_batch, 
                V_idx=V_idx_batch
                )

            # sampler prarms
            sampler_shape = self.model.bttn.sampler.sampler_shape
            sampler_scale = self.model.bttn.sampler.sampler_scale
            sampler_type = self.model.sampler_type

            # compute loss
            batch_nll_loss = gaussian_nll(mu_batch, logvar_batch, target_batch)
            batch_kl_loss = kl_divergence(
                prior_shape=prior_shape[Q_idx_batch, V_idx_batch], 
                prior_scale=prior_scale[Q_idx_batch, V_idx_batch], 
                sampler_shape=sampler_shape[Q_idx_batch, V_idx_batch], 
                sampler_scale=sampler_scale[Q_idx_batch, V_idx_batch], 
                sampler_type=sampler_type
                )

            # accumulate loss
            epoch_nll_loss += batch_nll_loss.item()
            epoch_kl_loss += batch_kl_loss.item()

            # backward pass of model
            batch_nll_loss.backward(retain_graph=True)
            batch_kl_loss.backward()
            self.model_optimizer.step()
            self.model_optimizer.zero_grad()

        # backward pass of prior
        self.prior_optimizer.step()
        self.prior_optimizer.zero_grad()

        return epoch_nll_loss/len(trn_loader), epoch_kl_loss/len(trn_loader)

    def _val_epoch(self, val_loader, epoch):
        self.prior.eval()
        self.model.eval()

        epoch_nll_loss = 0.0
        epoch_kl_loss = 0.0

        # forward pass of prior
        with torch.no_grad():
            all_k_idx = torch.arange(self.model.bttn.n_key)
            all_k_embed = self.model.bttn.key(all_k_idx)
            prior_shape, prior_scale = self.prior(all_k_embed)

        iter_obj = tqdm(
            iterable=val_loader, 
            desc=f"Epoch {epoch+1}/{self.n_epochs} VAL"
            )

        # forward pass of model
        with torch.no_grad():
            for Q_idx_batch, V_idx_batch, target_batch in iter_obj:
                Q_idx_batch = Q_idx_batch.to(self.device)
                V_idx_batch = V_idx_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                # forward pass
                mu_batch, logvar_batch = self.model(
                    Q_idx=Q_idx_batch, 
                    K_idx=V_idx_batch, 
                    V_idx=V_idx_batch
                    )

                # sampler prarms
                sampler_shape = self.model.bttn.sampler.sampler_shape
                sampler_scale = self.model.bttn.sampler.sampler_scale
                sampler_type = self.model.sampler_type

                # compute loss
                batch_nll_loss = gaussian_nll(mu_batch, logvar_batch, target_batch)
                batch_kl_loss = kl_divergence(
                    prior_shape=prior_shape[Q_idx_batch, V_idx_batch], 
                    prior_scale=prior_scale[Q_idx_batch, V_idx_batch], 
                    sampler_shape=sampler_shape[Q_idx_batch, V_idx_batch], 
                    sampler_scale=sampler_scale[Q_idx_batch, V_idx_batch],
                    sampler_type=sampler_type
                    )

                # accumulate loss
                epoch_nll_loss += batch_nll_loss.item()
                epoch_kl_loss += batch_kl_loss.item()

        return epoch_nll_loss/len(val_loader), epoch_kl_loss/len(val_loader)