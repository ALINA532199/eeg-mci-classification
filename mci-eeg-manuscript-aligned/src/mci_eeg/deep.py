"""Compact EEGNet-style network; fresh supervised fitting for every train partition.

A stratified subject-level validation subset monitors early stopping. Subject-balanced trial
sampling prevents participants with many retained trials dominating the loss.
"""
import random
import copy
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split


class EEGNet(nn.Module):
    def __init__(self, channels, f1=8, depth=2, f2=16, kernel=32, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel), padding=(0, kernel//2), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1*depth, (channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1*depth), nn.ELU(), nn.AvgPool2d((1,4)), nn.Dropout(dropout),
            nn.Conv2d(f1*depth, f1*depth, (1,16), padding=(0,8), groups=f1*depth, bias=False),
            nn.Conv2d(f1*depth, f2, 1, bias=False), nn.BatchNorm2d(f2), nn.ELU(),
            nn.AvgPool2d((1,8)), nn.Dropout(dropout), nn.AdaptiveAvgPool2d((1,4)), nn.Flatten())
        self.head = nn.Linear(f2*4, 1)

    def forward(self, x):
        z = self.features(x)
        return self.head(z).squeeze(1), z


class DeepFit:
    def __init__(self, cfg, seed):
        self.cfg, self.seed = cfg, seed

    def fit(self, epochs, y):
        y=np.asarray(y)
        all_indices=np.arange(len(y))
        if self.cfg.get("early_stopping",True):
            train_indices,validation_indices=train_test_split(all_indices,
                test_size=self.cfg.get("validation_fraction",0.2),stratify=y,random_state=self.seed)
        else:
            train_indices,validation_indices=all_indices,np.array([],int)
        training_epochs=[epochs[i] for i in train_indices]
        training_y=y[train_indices]
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.cfg.get("threads", 1))
        torch.use_deterministic_algorithms(True)
        self.device = torch.device(self.cfg.get("device", "cpu"))
        # Equal subject weighting for channel statistics, training subjects only.
        self.mean = np.mean([e.mean(axis=(0,2)) for e in training_epochs], axis=0)[None,:,None]
        second = np.mean([(e.astype(float)**2).mean(axis=(0,2)) for e in training_epochs], axis=0)[None,:,None]
        self.sd = np.sqrt(np.maximum(second-self.mean**2, 1e-12))
        self.model = EEGNet(epochs[0].shape[1], **self.cfg.get("network", {})).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.get("learning_rate", 0.001))
        rng = np.random.default_rng(self.seed)
        best_loss,best_state,waiting=float("inf"),None,0
        history=[]
        for epoch in range(self.cfg["epochs"]):
            arrays, labels = [], []
            for e, label in zip(training_epochs, training_y):
                n = self.cfg.get("trials_per_subject", 30)
                ix = rng.choice(len(e), n, replace=len(e)<n)
                arrays.append((e[ix]-self.mean)/self.sd)
                labels.extend([label]*n)
            x = np.concatenate(arrays).astype(np.float32)
            labels = np.asarray(labels, np.float32)
            order = rng.permutation(len(x))
            self.model.train()
            for start in range(0, len(order), self.cfg.get("batch_size", 32)):
                ix = order[start:start+self.cfg.get("batch_size",32)]
                tx = torch.as_tensor(x[ix,None], device=self.device)
                ty = torch.as_tensor(labels[ix], device=self.device)
                opt.zero_grad()
                logits, _ = self.model(tx)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, ty)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite EEGNet loss")
                loss.backward()
                opt.step()
            if len(validation_indices):
                probability,_=self.transform([epochs[i] for i in validation_indices])
                probability=np.clip(probability,1e-7,1-1e-7)
                vy=y[validation_indices]
                val_loss=float(-np.mean(vy*np.log(probability)+(1-vy)*np.log(1-probability)))
                history.append(val_loss)
                if val_loss<best_loss-self.cfg.get("min_delta",0.0):
                    best_loss=val_loss;best_state=copy.deepcopy(self.model.state_dict());waiting=0
                else:waiting+=1
                if waiting>=self.cfg.get("patience",10):break
        if best_state is not None:self.model.load_state_dict(best_state)
        self.training_audit={"train_subject_positions":train_indices.tolist(),
            "validation_subject_positions":validation_indices.tolist(),"epochs_run":epoch+1,
            "validation_loss":history,"scaler_fit_on":"training_subset_only"}
        return self

    def transform(self, epochs):
        probabilities, embeddings = [], []
        self.model.eval()
        with torch.no_grad():
            for e in epochs:
                ps, zs = [], []
                for start in range(0,len(e),64):
                    x = ((e[start:start+64]-self.mean)/self.sd).astype(np.float32)
                    logits, z = self.model(torch.as_tensor(x[:,None], device=self.device))
                    ps.append(torch.sigmoid(logits).cpu().numpy())
                    zs.append(z.cpu().numpy())
                probabilities.append(np.concatenate(ps).mean())
                embeddings.append(np.concatenate(zs).mean(axis=0))
        return np.asarray(probabilities), np.asarray(embeddings)
