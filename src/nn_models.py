"""
nn_models.py
PyTorch based neural network (Feature attention + encoder) and training loop.
"""
import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from sklearn.metrics import roc_auc_score
from .utils import log

class FeatureAttention(nn.Module):
    def __init__(self, n_features: int, n_steps: int = 3, n_da: int = 64):
        super().__init__()
        self.steps = n_steps
        self.fc_att = nn.Linear(n_features, n_features * n_steps)
        self.bn = nn.BatchNorm1d(n_features)

    def forward(self, x):
        B, F = x.shape
        attn = self.fc_att(x).view(B, self.steps, F)
        attn = torch.softmax(attn, dim=-1).mean(dim=1)
        return x * attn, attn

class EarlyRiskNet(nn.Module):
    def __init__(self, n_features: int, noise_std: float = 0.05):
        super().__init__()
        self.noise_std = noise_std
        self.attention = FeatureAttention(n_features, n_steps=3, n_da=64)
        def block(in_dim, out_dim, p_drop=0.3):
            return nn.Sequential(nn.Linear(in_dim, out_dim, bias=False), nn.BatchNorm1d(out_dim), nn.GELU(), nn.Dropout(p=p_drop))
        self.encoder = nn.Sequential(block(n_features, 256, 0.35), block(256,128,0.35), block(128,64,0.30), block(64,32,0.25))
        self.head = nn.Linear(32,1)
        self.res_proj = nn.Linear(n_features, 32)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        x_att, _ = self.attention(x)
        enc = self.encoder(x_att)
        res = self.res_proj(x_att)
        out = self.head(enc + res)
        return torch.sigmoid(out).squeeze(-1)

def focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.75, gamma: float = 2.0) -> torch.Tensor:
    bce = F.binary_cross_entropy(pred, target.float(), reduction="none")
    pt = torch.where(target == 1, pred, 1 - pred)
    loss = alpha * (1 - pt)**gamma * bce
    return loss.mean()

def train_nn(X_tr, y_tr, X_val, y_val, n_features, epochs=80, batch_size=256):
    if not TORCH_AVAILABLE:
        log.warning("PyTorch unavailable — skipping NN training.")
        return None, []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EarlyRiskNet(n_features).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    def make_loader(X,y,shuffle=True):
        ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y.astype(np.float32)))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=False)
    tr_loader = make_loader(X_tr, y_tr, shuffle=True)
    val_loader = make_loader(X_val, y_val, shuffle=False)
    best_auc, best_state = 0.0, None
    patience, ctr = 20, 0
    history = []
    for epoch in range(epochs):
        model.train()
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(Xb)
            loss = focal_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        # validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                val_preds.append(model(Xb.to(device)).cpu().numpy())
                val_labels.append(yb.numpy())
        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        auc = roc_auc_score(val_labels, val_preds)
        history.append(auc)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
            ctr = 0
        else:
            ctr += 1
        if ctr >= patience:
            log.info(f"Early stopping at epoch {epoch+1} (best AUC {best_auc:.4f})")
            break
    if best_state:
        model.load_state_dict(best_state)
    return model.eval().to("cpu"), history
