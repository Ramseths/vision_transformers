import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class ViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000, dim=768, depth=12, heads=12, mlp_dim=3072, dropout=0.1):
        super().__init__()
        assert image_size % patch_size == 0, "La imagen no es divisible por el tamaño del parche"
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2  # 3 canales de imagen
        
        # Embedding de parches
        self.patch_embedding = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        
        # Posición embebida
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        # Codificador
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout) 
            for _ in range(depth)])
        
        # Capa MLP para la salida del clasificador
        self.classifier = MLP(dim, mlp_dim, num_classes, dropout)
        
    def forward(self, x):
        x = self.patch_embedding(x)  # [B, C, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x = torch.cat([self.positional_embedding[:, 0:1, :].repeat(x.shape[0], 1, 1), 
                       self.positional_embedding[:, 1:, :]], dim=1)
        for layer in self.encoder:
            x = layer(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x
