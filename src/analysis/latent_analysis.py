"""차원 축소 및 잠재 공간 분석을 위한 모듈."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap.umap_ as umap
except ImportError:
    logger.warning(
        "UMAP not available. Please install with: pip install umap-learn"
    )
    umap = None
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, MISSING
from dataclasses import dataclass
import os
import json

logger = logging.getLogger(__name__)

@dataclass
class LatentAnalysisConfig:
    """잠재 요인 분석 설정."""
    
    n_components: int = 2
    random_state: int = 42
    
    vae: Dict[str, Any] = MISSING
    umap: Dict[str, Any] = MISSING
    visualization: Dict[str, Any] = MISSING
    save: Dict[str, bool] = MISSING


class VAE(nn.Module):
    """Variational Autoencoder 모델."""
    
    def __init__(
        self, 
        input_dim: int, 
        latent_dim: int = 2, 
        hidden_dims: list = None,
        beta: float = 1.0
    ) -> None:
        """VAE 모델을 초기화합니다."""
        super().__init__()
        
        self.latent_dim = latent_dim
        self.beta = beta
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
            
        # 인코더
        modules = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU()
            ))
            in_dim = h_dim
            
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # 디코더
        modules = []
        hidden_dims.reverse()
        in_dim = latent_dim
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU()
            ))
            in_dim = h_dim
            
        self.decoder = nn.Sequential(
            *modules,
            nn.Linear(hidden_dims[-1], input_dim)
        )
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """입력을 잠재 공간으로 인코딩"""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """잠재 벡터를 원본 공간으로 디코딩"""
        return self.decoder(z)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """재매개화 트릭"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
        
    def loss_function(
        self, 
        recon_x: torch.Tensor, 
        x: torch.Tensor, 
        mu: torch.Tensor, 
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """VAE 손실 함수"""
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kld_loss


class LatentAnalyzer:
    """잠재 공간 분석 클래스."""

    def __init__(self, config: DictConfig):
        """
        잠재 공간 분석기를 초기화합니다.
        
        Args:
            config: 설정 객체
        """
        self.config = config
        
        # VAE 기본 설정
        if not hasattr(self.config.analysis, 'vae'):
            self.config.analysis.vae = {
                'latent_dim': 2,
                'hidden_dims': [128, 64, 32],
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 50,
                'beta': 1.0,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }
        
        # PCA 초기화
        self.pca = PCA(
            n_components=self.config.analysis.pca.n_components,
            random_state=42
        )
        
        # t-SNE 초기화
        self.tsne = TSNE(
            n_components=self.config.analysis.tsne.n_components,
            perplexity=self.config.analysis.tsne.perplexity,
            random_state=42
        )
        
        # UMAP 초기화 (사용 가능한 경우에만)
        self.umap = None
        if umap is not None:
            self.umap = umap.UMAP(
                n_components=self.config.analysis.umap.n_components,
                n_neighbors=self.config.analysis.umap.n_neighbors,
                min_dist=self.config.analysis.umap.min_dist,
                random_state=42
            )
        else:
            logger.warning("UMAP is not available, skipping UMAP initialization")
        
        self.vae = None  # 동적 초기화

    def _validate_config(self, config: DictConfig) -> None:
        """설정 유효성 검사"""
        required_fields = ['n_components', 'random_state', 'vae', 'umap', 'visualization', 'save']
        for field in required_fields:
            if not hasattr(config.analysis, field):
                raise ValueError(f"Missing required config field: {field}")
                
        if config.analysis.n_components < 2:
            raise ValueError("n_components must be >= 2")

    def _init_vae(self, input_dim: int):
        """VAE 모델 초기화"""
        self.vae = VAE(
            input_dim=input_dim,
            latent_dim=self.config.analysis.vae.latent_dim,
            hidden_dims=self.config.analysis.vae.hidden_dims,
            beta=self.config.analysis.vae.beta
        )
    
    def _train_vae(self, X: torch.Tensor) -> np.ndarray:
        """VAE 학습 및 잠재 벡터 추출"""
        optimizer = torch.optim.Adam(
            self.vae.parameters(), 
            lr=self.config.analysis.vae.learning_rate
        )
        
        self.vae.train()
        for epoch in range(self.config.analysis.vae.n_epochs):
            total_loss = 0
            for i in range(0, len(X), self.config.analysis.vae.batch_size):
                batch = X[i:i + self.config.analysis.vae.batch_size]
                
                optimizer.zero_grad()
                recon_batch, mu, logvar = self.vae(batch)
                loss = self.vae.loss_function(recon_batch, batch, mu, logvar)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{self.config.analysis.vae.n_epochs}, Loss: {total_loss/len(X):.4f}')
                
        # 잠재 벡터 추출
        self.vae.eval()
        with torch.no_grad():
            mu, _ = self.vae.encode(X)
        return mu.numpy()
    
    def fit_transform_all(
        self, 
        X: np.ndarray,
        labels: np.ndarray = None
    ) -> Dict[str, Any]:
        """모든 잠재 요인 분석 수행"""
        results = {}
        
        # PCA
        pca_result = self.pca.fit_transform(X)
        results['pca'] = {
            'embeddings': pca_result,
            'explained_variance_ratio': self.pca.explained_variance_ratio_
        }
        
        # t-SNE
        tsne_result = self.tsne.fit_transform(X)
        results['tsne'] = {
            'embeddings': tsne_result
        }
        
        # UMAP (사용 가능한 경우에만)
        if self.umap is not None:
            umap_result = self.umap.fit_transform(X)
            results['umap'] = {
                'embeddings': umap_result
            }
        
        # VAE
        if self.vae is None:
            self._init_vae(X.shape[1])
            
        X_tensor = torch.FloatTensor(X)
        vae_result = self._train_vae(X_tensor)
        results['vae'] = {
            'embeddings': vae_result,
            'model': self.vae
        }
        
        return results
    
    def visualize_results(
        self,
        results: Dict[str, np.ndarray],
        labels: List[int],
        save_path: str
    ) -> None:
        """잠재 공간 분석 결과를 시각화합니다."""
        try:
            # 저장 경로의 디렉토리 생성
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)
            
            # 감정 레이블 매핑
            emotion_map = {
                0: "neutral", 1: "calm", 2: "happy", 3: "sad",
                4: "angry", 5: "fearful", 6: "disgust", 7: "surprised"
            }
            
            fig, axes = plt.subplots(
                2, 2, 
                figsize=self.config.analysis.visualization.figsize
            )
            methods = ['pca', 'tsne', 'umap', 'vae']
            
            # 색상 팔레트 설정
            colors = plt.cm.tab10(np.linspace(0, 1, len(set(labels))))
            
            for ax, method in zip(axes.ravel(), methods):
                embeddings = results[method]['embeddings']
                
                if labels is not None:
                    scatter = ax.scatter(
                        embeddings[:, 0],
                        embeddings[:, 1],
                        c=labels,
                        cmap='tab10',
                        alpha=0.6,
                        s=100
                    )
                    
                    # 범례 추가
                    legend_elements = [
                        plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=colors[i], label=emotion_map[i],
                                  markersize=10)
                        for i in sorted(set(labels))
                    ]
                    ax.legend(handles=legend_elements, loc='center left', 
                             bbox_to_anchor=(1, 0.5))
                else:
                    ax.scatter(embeddings[:, 0], embeddings[:, 1])
                    
                ax.set_title(f'{method.upper()} Projection', fontsize=12, pad=10)
                ax.grid(True, linestyle='--', alpha=0.3)
                
            plt.tight_layout()
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to: {save_path}")
            plt.close()
            
        except Exception as e:
            logger.error(f"Error visualizing results: {e}")

    def save_analysis_results(
        self,
        results: Dict[str, Any],
        save_dir: str
    ) -> None:
        """분석 결과 저장"""
        if not self.config.analysis.save.embeddings:
            return
            
        os.makedirs(save_dir, exist_ok=True)
        
        # 임베딩 저장
        for method, result in results.items():
            np.save(
                os.path.join(save_dir, f"{method}_embeddings.npy"),
                result['embeddings']
            )
            
        # 메트릭 저장
        if self.config.analysis.save.metrics:
            metrics = {
                method: {
                    k: v for k, v in result.items() 
                    if k != 'embeddings' and k != 'model'
                }
                for method, result in results.items()
            }
            with open(os.path.join(save_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4) 