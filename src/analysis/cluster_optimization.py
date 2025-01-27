"""클러스터 수 최적화를 위한 모듈."""

from typing import List, Tuple, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
import json
from sklearn.manifold import TSNE
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class ClusterOptimizer:
    """클러스터 수 최적화 클래스."""
    
    CLUSTERING_METHODS = {
        'kmeans': KMeans,
        'dbscan': DBSCAN,
        'hierarchical': AgglomerativeClustering,
        'gmm': GaussianMixture
    }
    
    def __init__(
        self,
        n_clusters_range: Tuple[int, int] = (4, 12),
        metrics: List[str] = None,
        method: str = 'kmeans',
        method_params: Dict = None,
        visualization_params: Dict = None
    ):
        """
        클러스터 최적화기를 초기화합니다.
        
        Args:
            n_clusters_range: (최소 클러스터 수, 최대 클러스터 수)
            metrics: 사용할 메트릭 리스트
            method: 클러스터링 방법
            method_params: 클러스터링 방법의 파라미터
            visualization_params: 시각화 파라미터
        """
        self.min_clusters = n_clusters_range[0]
        self.max_clusters = n_clusters_range[1]
        self.metrics = metrics or ['silhouette', 'nmi', 'ari']
        self.method = method
        self.method_params = method_params or {}
        self.viz_params = visualization_params or {}
        
    def optimize(
        self,
        embeddings: np.ndarray,
        ground_truth: List[str],
        save_dir: Optional[str] = None
    ) -> Tuple[int, float]:
        """최적의 클러스터 수를 찾습니다."""
        try:
            if len(embeddings) < 2:
                raise ValueError("Not enough samples for clustering")
            
            metrics_results = {
                'internal': {
                    'silhouette': [],
                    'calinski_harabasz': [],
                    'davies_bouldin': []
                },
                'external': {
                    'ari': [],
                    'nmi': []
                }
            }
            
            n_clusters_range = range(
                min(self.min_clusters, len(embeddings) - 1),
                min(self.max_clusters + 1, len(embeddings))
            )
            
            if not n_clusters_range:
                raise ValueError(
                    f"Invalid cluster range: min={self.min_clusters}, "
                    f"max={self.max_clusters}, samples={len(embeddings)}"
                )
            
            # 각 클러스터 수에 대해 메트릭 계산
            for n_clusters in n_clusters_range:
                labels = self._cluster(embeddings, n_clusters)
                unique_labels = np.unique(labels)
                
                if len(unique_labels) < 2:
                    logger.warning(
                        f"Skipping n_clusters={n_clusters} "
                        f"(produced only {len(unique_labels)} unique clusters)"
                    )
                    # 메트릭에 0 값 추가
                    for metric_list in metrics_results['internal'].values():
                        metric_list.append(0.0)
                    for metric_list in metrics_results['external'].values():
                        metric_list.append(0.0)
                    continue
                
                try:
                    # Internal metrics
                    metrics_results['internal']['silhouette'].append(
                        silhouette_score(embeddings, labels)
                    )
                    metrics_results['internal']['calinski_harabasz'].append(
                        calinski_harabasz_score(embeddings, labels)
                    )
                    metrics_results['internal']['davies_bouldin'].append(
                        davies_bouldin_score(embeddings, labels)
                    )
                    
                    # External metrics
                    metrics_results['external']['ari'].append(
                        adjusted_rand_score(ground_truth, labels)
                    )
                    metrics_results['external']['nmi'].append(
                        normalized_mutual_info_score(ground_truth, labels)
                    )
                except Exception as e:
                    logger.error(f"Error calculating metrics for n_clusters={n_clusters}: {e}")
                    # 실패한 경우 0 값 추가
                    for metric_list in metrics_results['internal'].values():
                        metric_list.append(0.0)
                    for metric_list in metrics_results['external'].values():
                        metric_list.append(0.0)
            
            # 결과 시각화 및 저장
            if save_dir:
                self._plot_optimization_metrics(
                    metrics_results,
                    n_clusters_range,
                    save_dir
                )
            
            # 최적의 클러스터 수 선택
            best_n = self._select_best_n_clusters(
                metrics_results,
                n_clusters_range
            )
            
            return best_n, metrics_results['external']['nmi'][best_n - n_clusters_range[0]]
            
        except Exception as e:
            logger.error(f"Error in optimization: {e}")
            return self.min_clusters, 0.0
    
    def _cluster(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """주어진 클러스터 수로 클러스터링을 수행합니다."""
        try:
            # 임베딩 정규화
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings)
            
            if self.method == 'dbscan':
                # DBSCAN 파라미터 설정
                params = self.method_params.copy()
                eps = params.pop('eps', 0.15)
                min_samples = params.pop('min_samples', 3)
                
                clusterer = DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                    **params
                )
                labels = clusterer.fit_predict(embeddings_scaled)
                labels[labels == -1] = labels.max() + 1
                return labels
                
            elif self.method == 'kmeans':
                # KMeans 기본 파라미터
                kmeans_params = {
                    'n_clusters': n_clusters,
                    'n_init': 10,
                    'max_iter': 300,
                    'random_state': 42
                }
                
                # 사용자 정의 파라미터로 업데이트 (중복 파라미터 제거)
                user_params = self.method_params.copy()
                for param in ['n_clusters', 'n_init', 'max_iter', 'random_state']:
                    user_params.pop(param, None)
                    
                kmeans_params.update(user_params)
                
                clusterer = KMeans(**kmeans_params)
                return clusterer.fit_predict(embeddings_scaled)
                
            elif self.method == 'gmm':
                clusterer = GaussianMixture(
                    n_components=n_clusters,
                    random_state=42,
                    **self.method_params
                )
                return clusterer.fit_predict(embeddings_scaled)
                
            elif self.method == 'hierarchical':
                clusterer = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    **self.method_params
                )
                return clusterer.fit_predict(embeddings_scaled)
                
            else:
                raise ValueError(f"Unsupported clustering method: {self.method}")
            
        except Exception as e:
            logger.error(f"Error in clustering: {e}")
            return np.zeros(len(embeddings))
    
    def _plot_optimization_metrics(
        self,
        metrics_results: Dict[str, Dict[str, List[float]]],
        n_clusters_range: range,
        save_dir: str
    ) -> None:
        """클러스터링 최적화 메트릭을 시각화합니다."""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Internal metrics
            for metric, values in metrics_results['internal'].items():
                ax1.plot(
                    n_clusters_range,
                    values,
                    marker='o',
                    label=metric.replace('_', ' ').title()
                )
            ax1.set_title('Internal Validation Metrics')
            ax1.set_xlabel('Number of Clusters')
            ax1.set_ylabel('Score')
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend()
            
            # External metrics
            for metric, values in metrics_results['external'].items():
                ax2.plot(
                    n_clusters_range,
                    values,
                    marker='o',
                    label=metric.upper()
                )
            ax2.set_title('External Validation Metrics')
            ax2.set_xlabel('Number of Clusters')
            ax2.set_ylabel('Score')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend()
            
            plt.tight_layout()
            
            # 결과 저장
            save_path = Path(save_dir) / "optimization_metrics.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved optimization metrics plot to: {save_path}")
            plt.close()
            
            # 수치 결과를 JSON으로 저장할 때 float32를 float로 변환
            json_results = {
                'n_clusters_range': list(n_clusters_range),
                'metrics': {
                    category: {
                        metric: [float(val) for val in values]  # float32를 float로 변환
                        for metric, values in metrics.items()
                    }
                    for category, metrics in metrics_results.items()
                }
            }
            
            metrics_path = Path(save_dir) / "optimization_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(json_results, f, indent=4)
            logger.info(f"Saved optimization metrics data to: {metrics_path}")
            
        except Exception as e:
            logger.error(f"Error plotting optimization metrics: {e}")

    def _select_best_n_clusters(
        self,
        metrics_results: Dict[str, Dict[str, List[float]]],
        n_clusters_range: range
    ) -> int:
        """최적의 클러스터 수를 선택합니다."""
        best_n = None
        best_score = -float('inf')
        
        for n_clusters in n_clusters_range:
            score = metrics_results['external']['nmi'][n_clusters - n_clusters_range[0]]
            if score > best_score:
                best_score = score
                best_n = n_clusters
        
        return best_n 