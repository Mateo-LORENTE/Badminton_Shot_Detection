import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from extract_sons import *
import sys
from sklearn.metrics import confusion_matrix


import pandas as pd




class MelSpecWithFeatDataset(Dataset):
    """
    Dataset PyTorch combinant des spectrogrammes Mel et des features tabulaires
    pour chaque frame vidéo, avec labels associés.

    Fonctionnalités :
      - Charge un ensemble de fichiers mel_specs_*.npy, leurs labels correspondants
        et des features depuis des fichiers CSV.
      - Normalise les features (StandardScaler) et applique éventuellement une PCA.
      - Permet un split train/val fixe avec fraction configurable.
      - Retourne pour chaque élément : (mels [1, T, M], feats [F], label).

    Args:
        spec_dir (str): dossier contenant les spectrogrammes et labels.
        feat_dir (str): dossier contenant les CSV de features.
        split (str): 'train', 'val' ou None pour tout le dataset.
        val_fraction (float): proportion utilisée pour la validation.
        pca_n (int): nombre de composantes PCA à conserver (0 ou None pour désactiver).
        seed (int): graine pour le split aléatoire.
        keep_vids (list|set): identifiants de vidéos à exclure du dataset.
    """
    def __init__(self,
                 spec_dir,
                 feat_dir,
                 split=None,
                 val_fraction=0.2,
                 pca_n=30,
                 seed=42,
                 keep_vids=None):
    
        keep_vids = set(str(v) for v in keep_vids) if keep_vids else set()
        # 1) repérage des spectrogrammes

        #ALIGNEMENT AVEC LE TEST DE LILA BAPTISTE
        desired_order = [
            "mel_specs_37159",
            "mel_specs_22088",
            "mel_specs_49766",
            "mel_specs_49887",
            "mel_specs_47563",
            "mel_specs_87177",
            "mel_specs_42354",
            "mel_specs_21588",
        ]

        # Ajoute l'extension et le chemin complet
        desired_paths = [os.path.join(spec_dir, name + ".npy") for name in desired_order]

        # Vérifie que les fichiers existent
        spec_paths = [path for path in desired_paths if os.path.exists(path)]

        # spec_paths = sorted(glob.glob(os.path.join(spec_dir, "mel_specs_*.npy")))
        # if not spec_paths:
        #     raise ValueError(f"Aucun 'mel_specs_*.npy' dans {spec_dir!r}")

        all_specs, all_labels, all_feats = [], [], []

        for sp in spec_paths:
            vid = os.path.basename(sp).replace("mel_specs_","").replace(".npy","")
            
            if vid not in keep_vids:
                
                lp  = os.path.join(spec_dir, f"labels_{vid}.npy")
                fp  = os.path.join(feat_dir, f"{vid}_features.csv")   # on cherche vid.csv dans feat_dir
                # vérifications
                if not os.path.isfile(lp):
                    raise FileNotFoundError(f"Labels manquant pour {vid}")
                if not os.path.isfile(fp):
                    raise FileNotFoundError(f"Features manquant pour {vid} ({fp})")

                # chargement spectrogrammes et labels comme avant
                specs  = np.load(sp, allow_pickle=True)
                
                labels = np.load(lp)
                if specs.dtype == object:
                    specs = np.stack(specs, axis=0)

                # --- nouveau : chargement des features depuis le CSV ---
                df_feat = pd.read_csv(fp)
                # on suppose que df_feat a les mêmes lignes que labels, et que
                # les colonnes utiles sont toutes sauf Frame, Time (s), PEAK_HIT
                drop_cols = ['Frame', 'Time (s)', 'PEAK_HIT']
                feat_cols = [c for c in df_feat.columns if c not in drop_cols]
                feats     = df_feat[feat_cols].to_numpy(dtype=np.float32)
                

                scaler = StandardScaler(with_mean=True, with_std=True)
                feats = scaler.fit_transform(feats)
                # collecte
                all_specs .append(specs.astype(np.float32))
                all_labels.append(labels.astype(np.int64))
                all_feats .append(feats)

                
            
                
        # concaténations
        specs  = np.concatenate(all_specs,  axis=0)  # (N, T, M)
        labels = np.concatenate(all_labels, axis=0)  # (N,)
        feats  = np.concatenate(all_feats,   axis=0)  # (N, F)
        
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        if pca_n is not None and pca_n > 0:
            pca = PCA(n_components=pca_n, random_state=seed)
            feats = pca.fit_transform(feats)

        # conversion en tensors
        self.mels   = torch.from_numpy(specs).unsqueeze(1)  # (N,1,T,M)
        self.labels = torch.from_numpy(labels)
        self.feats  = torch.from_numpy(feats)

         # 5) split éventuel corrigé
        if split in ('train', 'val'):
            N = len(self.labels)
            # génération d'une permutation aléatoire fixe
            perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed)).tolist()
            n_train = int((1 - val_fraction) * N)
            train_idx = perm[:n_train]
            val_idx   = perm[n_train:]

            idxs = train_idx if split=='train' else val_idx

            # on indexe directement nos tensors
            self.mels   = self.mels  [idxs]
            self.labels = self.labels[idxs]
            self.feats  = self.feats [idxs]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.mels[i], self.feats[i], self.labels[i]


class CenteredFrameSequenceDataset(Dataset):
    """
    Dataset PyTorch générant des séquences centrées sur un frame cible
    avec un contexte fixe avant et après, pour entraîner des modèles séquentiels (ex. CRNN).

    À partir d'un `base_dataset` contenant :
      - mels : spectrogrammes (N, 1, F, T)
      - feats : features tabulaires (N, feat_dim)
      - labels : labels entiers (N,)

    Args:
        base_dataset (Dataset): dataset source fournissant mels, feats et labels.
        context (int): nombre de frames avant et après à inclure.
    
    Retourne :
        tuple: (x_seq, f_seq, y_target)
            - x_seq : séquence de mels (seq_len, 1, F, T)
            - f_seq : séquence de features tabulaires (seq_len, feat_dim)
            - y_target : label de la frame centrale.
    """
    def __init__(self, base_dataset, context=2):
        """
        Dataset qui retourne une séquence centrée sur un point, avec `context` exemples avant et après utiliser pour CRNN.
        """
        self.mels = base_dataset.mels
        self.feats = base_dataset.feats
        self.labels = base_dataset.labels
        self.context = context
        self.seq_len = 2 * context + 1

    def __len__(self):
        # On évite les bords où il n'y a pas assez de contexte
        return len(self.labels) - 2 * self.context

    def __getitem__(self, idx):
        # Décalage pour centrer la séquence sur le bon exemple
        start = idx
        end = idx + self.seq_len
        x_seq = self.mels[start:end]         # (seq_len, 1, F, T)
        f_seq = self.feats[start:end]        # (seq_len, feat_dim)
        y_target = self.labels[idx + self.context]  # label du centre
        return x_seq, f_seq, y_target

import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNNBiGRUWithFeat(nn.Module):
    """
    Modèle CRNN (CNN + BiGRU) fusionnant des spectrogrammes et des features tabulaires.

    Architecture :
      - CNN 2D sur la séquence de spectrogrammes (par frame) pour extraire des features audio.
      - MLP sur les features tabulaires pour en extraire un embedding.
      - Fusion (concaténation) des deux représentations par frame.
      - BiGRU pour modéliser le contexte temporel sur la séquence.
      - Classification finale sur la frame centrale.

    Args:
        feat_dim (int): dimension des features tabulaires d'entrée.
        deep (int): nombre de couches CNN utilisées.
        in_channels (int): nombre de canaux d'entrée (1 pour un spectrogramme mono).
        conv_channels (list): liste du nombre de filtres pour chaque couche CNN.
        feat_emb (int): dimension de l'embedding des features tabulaires.
        rnn_hidden (int): taille de l'état caché du GRU.
        rnn_layers (int): nombre de couches GRU.
        num_classes (int): nombre de classes en sortie.

    Input:
        x_seq (Tensor): (B, S, 1, F, T) séquence de spectrogrammes centrée sur la frame cible.
        feat_seq (Tensor): (B, S, feat_dim) séquence de features tabulaires correspondantes.

    Output:
        logits (Tensor): (B, num_classes) scores de classification pour la frame centrale.
    """
    def __init__(self, 
                 feat_dim,
                 deep,
                 in_channels: int = 1, 
                 conv_channels: list = [32, 64, 128, 128, 256, 256, 512, 512],
                 feat_emb: int = 128,
                 rnn_hidden: int = 256, 
                 rnn_layers: int = 1, 
                 num_classes: int = 2):
        super().__init__()

        self.deep = deep
        self.rnn_hidden = rnn_hidden
        self.rnn_layers = rnn_layers
        self._rnn_initialized = False  # flag pour initialiser dynamiquement le GRU

        # --- CNN ---
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        in_ch = in_channels
        for out_ch in conv_channels:
            self.convs.append(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False))
            self.bns.append(nn.BatchNorm2d(out_ch))
            in_ch = out_ch

        self.pools = nn.ModuleList([
            nn.MaxPool2d((1, 1)),
            nn.MaxPool2d((1, 2)),
            nn.MaxPool2d((2, 1)),
            nn.MaxPool2d((2, 1)),
            nn.MaxPool2d((1, 2)),
        ])

        # --- Features MLP ---
        self.feat_mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_emb),
            nn.BatchNorm1d(feat_emb),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(2 * rnn_hidden, rnn_hidden),
            nn.ReLU(),
            nn.Linear(rnn_hidden, num_classes)
        )

    def forward(self, x_seq, feat_seq):
        # x_seq: (B, S, 1, F, T), feat_seq: (B, S, feat_dim)
        B, S, C, F, T = x_seq.shape
        x_seq = x_seq.view(B * S, C, F, T)

        pool_idx = 0
        for i in range(self.deep):
            x_seq = torch.nn.functional.relu(self.bns[i](self.convs[i](x_seq)))
            if i in {1, 3, 5, 7, 9} and pool_idx < len(self.pools):
                x_seq = self.pools[pool_idx](x_seq)
                pool_idx += 1

        # Réduction sur la dimension fréquence, reshape
        x_seq = x_seq.mean(dim=2).view(B, S, -1)

        # Embedding des features
        feat_seq = feat_seq.view(B * S, -1)
        feat_emb = self.feat_mlp(feat_seq)
        feat_emb = feat_emb.view(B, S, -1)

        # Concaténation CNN + features
        x_cat = torch.cat([x_seq, feat_emb], dim=2)  # (B, S, C+feat_emb)

        # Initialisation dynamique du GRU si pas encore fait
        if not self._rnn_initialized:
            input_size = x_cat.shape[-1]
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=self.rnn_hidden,
                num_layers=self.rnn_layers,
                batch_first=True,
                bidirectional=True
            ).to(x_cat.device)
            self._rnn_initialized = True

        rnn_out, _ = self.rnn(x_cat)

        center_idx = S // 2
        pooled = rnn_out[:, center_idx, :]  # Sortie centrale

        logits = self.classifier(pooled)
        return logits



class CNN(nn.Module):
    """
    CNN fusionnant spectrogrammes et features tabulaires pour classification frame par frame.

    Architecture :
      - CNN 2D sur l'entrée spectrogramme (réduction adaptative de l'axe fréquence).
      - MLP sur les features tabulaires pour obtenir un embedding.
      - Concaténation des deux représentations.
      - Classifieur fully connected.

    Args:
        feat_dim (int): dimension des features tabulaires d'entrée.
        deep (int): nombre de couches convolutionnelles à utiliser.
        in_channels (int): nombre de canaux d'entrée (1 pour spectrogrammes mono).
        conv_channels (list[int]): nombre de filtres par couche convolutionnelle.
        feat_emb (int): dimension de l'embedding des features tabulaires.
        num_classes (int): nombre de classes de sortie.

    Input:
        x_seq (Tensor): 
            - (B, S, 1, F, T) pour séquences
            - (B, 1, F, T) pour frames isolées
        feat_seq (Tensor): (B, S, feat_dim) ou (B, feat_dim)

    Output:
        logits (Tensor): 
            - (B, num_classes) si S == 1
            - (B, S, num_classes) sinon
    """
    def __init__(self, 
                 feat_dim,
                 deep,
                 in_channels: int = 1, 
                 conv_channels: list = [32, 64, 128, 128, 256, 256, 512, 512],
                 feat_emb: int = 128, 
                 num_classes: int = 2):
        super().__init__()

        assert deep <= len(conv_channels), "Parameter 'deep' exceeds number of convolution layers"

        self.deep = deep

        # --- CNN ---
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        in_ch = in_channels
        for out_ch in conv_channels:
            self.convs.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False))
            self.bns.append(nn.BatchNorm2d(out_ch))
            in_ch = out_ch

        # Pooling every 2 convs, except last if odd number
        self.pool_every = 2

        # --- Adaptive reduction (freq axis) ---
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))  # Reduce Freq to 1, keep Time

        # --- Features MLP ---
        self.feat_mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_emb),
            nn.BatchNorm1d(feat_emb),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(conv_channels[deep - 1] + feat_emb, conv_channels[deep - 1] + feat_emb),
            nn.ReLU(),
            nn.Linear(conv_channels[deep - 1] + feat_emb, num_classes)
        )

    def forward(self, x_seq, feat_seq):
        # x_seq: (B, S, 1, F, T), feat_seq: (B, S, feat_dim)
        if x_seq.ndim == 5:
            B, S, C, F, T = x_seq.shape
            x_seq = x_seq.view(B * S, C, F, T)
        elif x_seq.ndim == 4:
            B, C, F, T = x_seq.shape
            S = 1
            x_seq = x_seq  # déjà bon format
            x_seq = x_seq.view(B, S, C, F, T).view(B * S, C, F, T)
        else:
            raise ValueError(f"Unexpected x_seq shape: {x_seq.shape}")
        # CNN forward
        for i in range(self.deep):
            x_seq = torch.nn.functional.relu(self.bns[i](self.convs[i](x_seq)))
            if (i + 1) % self.pool_every == 0:
                x_seq = torch.nn.functional.max_pool2d(x_seq, kernel_size=2)

        # Reduce frequency dimension to 1 (adaptively)
        x_seq = self.adaptive_pool(x_seq)  # shape: (B*S, C, 1, T')
        x_seq = x_seq.squeeze(2)  # shape: (B*S, C, T')
        x_seq = x_seq.mean(dim=2)  # Global average over time → shape: (B*S, C)

        x_seq = x_seq.view(B, S, -1)

        # Feature embedding
        feat_seq = feat_seq.view(B * S, -1)
        feat_emb = self.feat_mlp(feat_seq)
        feat_emb = feat_emb.view(B, S, -1)

        # Concat CNN + Features
        x_cat = torch.cat([x_seq, feat_emb], dim=2)  # (B, S, C+feat_emb)

        logits = self.classifier(x_cat)  # (B, S, num_classes)
        return logits.squeeze(1) if S == 1 else logits 



#LOSS

class FocalLoss(nn.Module):
    """
    Implémentation de la Focal Loss pour classification binaire/multiclasse.

    Réduit l'impact des exemples faciles et focalise l'entraînement sur 
    les exemples difficiles via un facteur (1 - p_t)^gamma.

    Args:
        alpha (float): facteur d'équilibrage entre classes.
        gamma (float): facteur de focalisation (γ > 0 augmente la focalisation).
        reduction (str): 'mean' ou 'sum' pour l'agrégation des pertes.

    Inputs:
        logits (Tensor): (B, C) scores non normalisés.
        targets (Tensor): (B,) labels entiers dans [0, C-1].

    Output:
        loss (Tensor): scalaire (si reduction='mean' ou 'sum').
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, targets):
        # logits: (B,2), targets: (B,) dans {0,1}
        probs = F.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = - self.alpha * (1 - p_t)**self.gamma * torch.log(p_t + 1e-8)
        return loss.mean() if self.reduction=='mean' else loss.sum()


def train_model(data_dir, feat_dir, deep=5, batch_size=64, lr=1e-3, epochs=20, split_ratio=0.8, context=1,rnn_layer=1,keep_list=None,model_type="rnn"):
    """
    Entraîne un modèle audio+features pour détecter un coup (binaire) et choisit
    le seuil optimal sur le set de validation.

    Ce que fait la fonction :
      - Charge le dataset (Mel-spectrogrammes + features tabulaires), construit séquences centrées si model_type='rnn'.
      - Split train/val/test (10% test à la fin, val = (1 − split_ratio) du reste).
      - Initialise et entraîne soit un CRNN BiGRU, soit un CNN pur.
      - Optimise le seuil de décision via F1 sur la validation, garde les meilleurs poids.
      - Évalue sur le test set (F1 + matrice de confusion) et sauvegarde le modèle.

    Args:
        data_dir (str): dossier des spectrogrammes/labels. 
        feat_dir (str): dossier des CSV de features.
        model_type (str): 'rnn' (CRNNBiGRUWithFeat) ou 'cnn' (CNN).
    
    Returns:
        (torch.nn.Module, float): le meilleur modèle chargé et le seuil optimal (best_t).
    """
    
    full_ds = MelSpecWithFeatDataset(
        data_dir,
        feat_dir,
        split=None,
        val_fraction=0.0,
        pca_n=30,
        seed=42,
        keep_vids=keep_list
    )

    if model_type == "rnn":
        dataset = CenteredFrameSequenceDataset(full_ds, context=context)
    elif model_type == "cnn":
        dataset = full_ds
    else:
        raise ValueError("model_type must be 'rnn' or 'cnn'")



     # --- Split dataset
    # N = len(dataset)
    # test_size = int(0.1 * N)
    # indices = list(range(N))
    # test_indices = indices[:test_size]
    # train_val_indices = indices[test_size:]
    # n_train = int(split_ratio * len(train_val_indices))
    # train_indices = train_val_indices[:n_train]
    # val_indices = train_val_indices[n_train:]

    N = len(dataset)
    test_size = int(0.1 * N)
    indices = list(range(N))
    test_indices = indices[-test_size:]  # <-- derniers éléments pour le test
    train_val_indices = indices[:-test_size]  # tout sauf les derniers

    n_train = int(split_ratio * len(train_val_indices))
    train_indices = train_val_indices[:n_train]
    val_indices = train_val_indices[n_train:]

    from torch.utils.data import Subset
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    test_ds = Subset(dataset, test_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,drop_last=True)

    # --- Récupère feat_dim pour initialiser le modèle
    x_seq, f_seq, _ = next(iter(train_loader))
    feat_dim = f_seq.shape[-1]

    # Sécurité : vérifie la forme en fonction du modèle
    if model_type == "rnn":
        assert x_seq.ndim == 5, f"RNN model requires x_seq to be 5D (B, S, 1, F, T), got {x_seq.shape}"
    elif model_type == "cnn":
        assert x_seq.ndim == 4, f"CNN model requires x_seq to be 4D (B, 1, F, T), got {x_seq.shape}"
        
    # --- Choix du modèle
    if model_type == "rnn":
        model = CRNNBiGRUWithFeat(
            feat_dim=feat_dim,
            deep=deep,
            rnn_layers=rnn_layer
        ).to(device)
    else:
        model = CNN(
            feat_dim=feat_dim,
            deep=deep
        ).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    labels_all = torch.cat([y for _, _, y in train_loader])
    class_counts = torch.bincount(labels_all)
    class_weights = (1.0 / class_counts.float()).to(device)
    criterion = FocalLoss(alpha=1, gamma=2)


    best_model_state = None
    best_val_f1 = 0.0
    best_t = 0.5
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0
        for specs, feats, labels in train_loader:
            specs, feats, labels = specs.to(device), feats.to(device), labels.to(device)
            logits = model(specs, feats)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * specs.size(0)
        train_loss = running_loss / len(train_ds)

        model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for specs, feats, labels in val_loader:
                specs, feats = specs.to(device), feats.to(device)
                logits = model(specs, feats)
                probs = F.softmax(logits, dim=1)[:, 1]
                all_scores.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())
        thresholds = np.linspace(0.01, 0.99, 99)
        f1s = [f1_score(all_labels, (np.array(all_scores) >= t).astype(int)) for t in thresholds]
        idx = int(np.argmax(f1s))
        best_f1, best_t = f1s[idx], thresholds[idx]
        if best_f1 > best_val_f1:
            best_val_f1 = best_f1
            best_model_state = model.state_dict()  # Sauvegarde les poids du meilleur modèle
            best_t = thresholds[idx]  # Met à jour le meilleur threshold

        #print(f"Epoch {epoch}/{epochs} — Loss {train_loss:.4f} — F1 {best_f1:.4f}@{best_t:.2f}")
        if epoch % 1 == 0:
            print(f"Epoch {epoch}/{epochs} — Loss {train_loss:.4f} — F1 {best_f1:.4f}@{best_t:.2f}")
    model.load_state_dict(best_model_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for specs, feats, labels in test_loader:
            specs, feats = specs.to(device), feats.to(device)
            logits = model(specs, feats)
            best_threshold = best_t
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= best_threshold).astype(int)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    test_f1 = f1_score(all_labels, all_preds)
    print("F1 sur le test set :", test_f1)
    cm = confusion_matrix(all_labels, all_preds)
    print("\n📊 Matrice de confusion :\n")
    print(cm)

    torch.save(model, os.path.join(data_dir, 'strike_cnn_feat_new.pt'))
    return model , best_t


import tempfile
import subprocess

# SI UTILISATION D'ONYXIA

def get_video_path(path):
    """
    Retourne un chemin vidéo local, en téléchargeant depuis S3 si nécessaire.

    - Si `path` commence par "s3/", lit le fichier depuis MinIO/S3 avec `mc cat`
      et l'enregistre dans un fichier temporaire .mp4.
    - Sinon, retourne le chemin local inchangé.

    Args:
        path (str): chemin local ou chemin S3 (ex. "s3/bucket/video.mp4").

    Returns:
        str: chemin local vers la vidéo (temporaire si source S3).
    """
    if path.startswith("s3/"):
        # Chemin S3 : lire avec `mc cat` et stocker temporairement
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        with open(temp_file.name, "wb") as f:
            subprocess.run(["mc", "cat", path], stdout=f, check=True)
        return temp_file.name
    else:
        # Chemin local classique
        return path

def is_s3_path(path):
    """
    Indique si un chemin correspond à un chemin S3/MinIO.

    Args:
        path (str): chemin à vérifier.

    Returns:
        bool: True si le chemin commence par "s3/", False sinon.
    """
    return path.startswith("s3/")

def stream_s3_file_to_tempfile(s3_path):
    """
    Télécharge un fichier depuis S3/MinIO dans un fichier temporaire local .mp4.

    Args:
        s3_path (str): chemin S3 (ex. "s3/bucket/video.mp4").

    Returns:
        str: chemin local vers le fichier temporaire.
    """
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    with open(temp.name, "wb") as f:
        subprocess.run(["mc", "cat", s3_path], stdout=f, check=True)
    return temp.name




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Complete Data Extractor Script')
    parser.add_argument('--input_video',  type=str, required=True,
                        help='Dossier contenant les .mp4')
    parser.add_argument('--input_sons',   type=str, required=True,
                        help='Dossier où écrire/charger les .wav')
    parser.add_argument('--outputs_spec', type=str, required=True,
                        help='Dossier de sortie pour spectrogrammes .npy')
    parser.add_argument('--outputs_feat', type=str, default=True,
                        help='(optionnel) Dossier de sortie pour features .csv')
    parser.add_argument('--overwrite', type=bool, default=False,
                        help='(optionnel) booléen rewrite or not')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--epochs',     type=int, default=10)

    args = parser.parse_args()

    global_peak_dir = '/home/onyxia/work/DATA/peak'
    os.makedirs(global_peak_dir, exist_ok=True)

    # Liste pour accumuler tous les df_peaks
    all_peaks = []
    output_dir = '/home/onyxia/work/DATA/pkl_merged_peak_test'
    os.makedirs(output_dir, exist_ok=True)

    folder_df = '/home/onyxia/work/DATA/pkl_merged'
    
    # Si aucun dossier de sortie features n'est précisé,
    # on le crée à l'intérieur de outputs_spec/features
    feat_out = args.outputs_feat or os.path.join(args.outputs_spec, 'features')
    os.makedirs(feat_out, exist_ok=True)
    os.makedirs(args.outputs_spec, exist_ok=True)

    # Balaye tous les .mp4 du dossier
    if is_s3_path(args.input_video):
        # Lister les fichiers mp4 dans le S3 path via mc
        result = subprocess.run(["mc", "ls", args.input_video], capture_output=True, text=True)
        videos = [line.split()[-1] for line in result.stdout.splitlines() if line.endswith(".mp4")]
    else:
        videos = sorted([f for f in os.listdir(args.input_video) if f.lower().endswith('.mp4')])


    if not videos:
        print("⚠️ Aucun fichier .mp4 trouvé dans", args.input_video)
        sys.exit(1)

    for fname in videos:
        nom, _ = os.path.splitext(fname)
        print(f"\n▶ Traitement de la vidéo {nom}")
        if nom == "5684":
            # 1) Extraction des pics
            if is_s3_path(args.input_video):
                chemin_vid = stream_s3_file_to_tempfile(f"{args.input_video}/{fname}")
            else:
                chemin_vid = os.path.join(args.input_video, fname)
            chemin_son = os.path.join(args.input_sons,  f"{nom}.wav")
            
            
            chemin_df = os.path.join(folder_df, f"{nom}_df_predict_merged.pkl")

            feat_csv_out = os.path.join(feat_out, f"{nom}_features.csv")
            write_features = args.overwrite or not os.path.isfile(feat_csv_out)
            write_features = 1
            check = 0

            if not write_features:
                print(f"ℹ️ Le fichier {feat_csv_out} existe déjà et overwrite=False, je passe.")
            else:
                try:
                    df_peaks, fps, sr, y = peak_extractor(
                        chemin_audio=chemin_son,
                        chemin_video=chemin_vid,
                        chemin_df=chemin_df,
                        nom_video=nom
                    )
                    check = 1

                    output_path = os.path.join(output_dir, f"{nom}_df_merged_peak.pkl")
        
                    # Sauvegarde du DataFrame des pics
                    df_peaks.to_pickle(output_path)
                    print(f"✅ Sauvegardé : {output_path}")


                except Exception as e:
                    print(f"❌ Erreur à l’extraction des pics pour {nom} :", e)
                    continue
        
            all_peaks.append(df_peaks)
            os.remove(chemin_son)
            df_peak_features = features_extractor(
                df_peaks=df_peaks,
                fps=fps, sr=sr, y=y,
                chemin_sauvegarde_csv=feat_csv_out
            )


    concatenated = pd.concat(all_peaks, ignore_index=True)
    out_path = os.path.join(global_peak_dir, 'all_peaks.csv')
    concatenated.to_csv(out_path, index=False)
    print(f"\n✅ Tous les df_peaks ont été concaténés ({len(all_peaks)} fichiers) et sauvés dans {out_path}")
        # 3) Extraction des spectrogrammes et labels
    spec_path  = os.path.join(args.outputs_spec, f"mel_specs_{nom}.npy")
    label_path = os.path.join(args.outputs_spec, f"labels_{nom}.npy")

    write_spectres = args.overwrite or not (os.path.isfile(spec_path) and os.path.isfile(label_path))
    if write_spectres:

        if check == 0:
            df_peaks, fps, sr, y = peak_extractor(
                chemin_audio=chemin_son,
                chemin_video=chemin_vid,
                chemin_df=chemin_df,
                nom_video=nom
            )

        spectre_extractor(
            y=y, sr=sr,
            df_peaks=df_peaks,
            save_dir=args.outputs_spec,
            nom_video=nom
        )
        
    else:
        print(f"ℹ️ spectres et labels pour {nom} existent déjà → skip spectres")

    print("\n✅ Tous les fichiers ont été traités.")

    print("============================= ENTRAINEMENT =======================")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    keep_list = []
    train_model(
        data_dir   = args.outputs_spec,
        feat_dir = args.outputs_feat ,
        deep = 8,
        batch_size = args.batch_size,
        lr         = args.lr,
        epochs     = args.epochs,
        keep_list=keep_list,
        rnn_layer = 3,
        context = 3,
        model_type="cnn"
    )
            
