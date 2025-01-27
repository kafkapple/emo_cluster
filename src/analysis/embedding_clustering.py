"""임베딩 클러스터링을 위한 모듈."""

import logging
import os
from typing import Dict, List, Any
import json
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class Clustering:
    """임베딩 클러스터링을 수행하는 클래스."""

    @staticmethod
    def perform_clustering(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """클러스터링을 수행합니다."""
        try:
            # 임베딩 정규화
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings)
            
            # KMeans 클러스터링
            clustering = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,  # 여러 번 시도
                max_iter=300  # 더 많은 반복
            )
            labels = clustering.fit_predict(embeddings_scaled)
            
            # 클러스터 크기 확인
            unique, counts = np.unique(labels, return_counts=True)
            for cluster_id, count in zip(unique, counts):
                logger.info(f"Cluster {cluster_id}: {count} samples")
            
            return labels
        
        except Exception as e:
            logger.error(f"Error performing clustering: {e}")
            return np.array([])

    @staticmethod
    def evaluate_clustering(labels: np.ndarray, ground_truth: List[str]) -> Dict[str, Any]:
        """클러스터링 결과를 평가합니다."""
        # 클러스터 분포 계산
        pred_dist = Counter(labels)
        true_dist = Counter(ground_truth)
        
        # 메트릭 계산
        ari = adjusted_rand_score(ground_truth, labels)
        nmi = normalized_mutual_info_score(ground_truth, labels)
        
        # NumPy int32를 Python int로 변환
        pred_dist_converted = {int(k): int(v) for k, v in pred_dist.items()}
        
        # 결과를 딕셔너리로 반환
        return {
            'ari': float(ari),
            'nmi': float(nmi),
            'n_samples': int(len(ground_truth)),
            'n_pred_clusters': int(len(set(labels))),
            'n_true_clusters': int(len(set(ground_truth))),
            'pred_cluster_distribution': pred_dist_converted,
            'true_cluster_distribution': dict(true_dist)
        }

    @staticmethod
    def save_results(
        audio_metrics: Dict[str, Any],
        text_metrics: Dict[str, Any],
        fusion_metrics: Dict[str, Any],
        save_dir: str
    ) -> str:
        """
        클러스터링 결과를 저장합니다.

        Args:
            audio_metrics: 오디오 클러스터링 메트릭
            text_metrics: 텍스트 클러스터링 메트릭
            fusion_metrics: 퓨전 클러스터링 메트릭
            save_dir: 저장 디렉토리

        Returns:
            str: 저장된 파일 경로
        """
        os.makedirs(save_dir, exist_ok=True)
        results_path = os.path.join(save_dir, "clustering_results.json")
        
        results = {
            'audio': audio_metrics,
            'text': text_metrics,
            'fusion': fusion_metrics
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        return results_path
