"""임베딩 캐시 관리를 위한 모듈."""

import os
import json
from typing import Dict, Tuple, List, Optional
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
from omegaconf import DictConfig
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingInfo:
    """임베딩 메타데이터."""
    
    shape: Tuple[int, ...]
    model_name: str
    created_at: str
    config_hash: str
    version: str
    modality: str


class EmbeddingCache:
    """임베딩 캐시를 관리하는 클래스."""
    
    def __init__(self, cfg: DictConfig):
        """
        임베딩 캐시 초기화.
        
        Args:
            cfg: 설정 객체
        """
        self.config = cfg  # config_hash 계산을 위해 필요
        self.enabled = cfg.cache.enabled
        self.root_dir = Path(cfg.cache.root_dir)
        self.version = cfg.cache.version
        self.compression = cfg.dataset.cache.embeddings.compression
        
        # 캐시 디렉토리 생성
        self.cache_dir = self.root_dir / "embeddings" / self.version
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
        self.logger = logging.getLogger(__name__)
    
    def _get_cache_path(self, file_id: str, modality: str) -> Tuple[Path, Path]:
        """
        캐시 파일 경로를 반환합니다.
        
        Returns:
            Tuple[Path, Path]: (임베딩 경로, 메타데이터 경로)
        """
        base_name = Path(file_id).stem
        emb_path = self.cache_dir / f"{base_name}_{modality}.npy"
        meta_path = self.cache_dir / f"{base_name}_{modality}.json"
        return emb_path, meta_path

    def _get_config_hash(self, modality: str) -> str:
        """설정의 해시값을 계산합니다."""
        relevant_config = {
            "audio": {
                "model": self.config.model.audio_model,
                "sample_rate": self.config.dataset.preprocess.audio_sample_rate
            },
            "text": {
                "model": self.config.model.text_model
            }
        }
        return str(hash(str(relevant_config[modality])))

    def load_embedding(self, file_id: str, modality: str) -> Optional[np.ndarray]:
        """캐시에서 임베딩을 로드합니다."""
        if not self.enabled:
            return None
            
        emb_path, meta_path = self._get_cache_path(file_id, modality)
        if not (emb_path.exists() and meta_path.exists()):
            return None
            
        try:
            # 메타데이터 확인
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            if meta['config_hash'] != self._get_config_hash(modality):
                return None
            
            # 임베딩 로드
            if self.compression:
                with np.load(emb_path) as data:
                    return data['embedding']
            return np.load(emb_path)
            
        except Exception as e:
            self.logger.warning(f"Error loading cache for {file_id}: {e}")
            return None

    def save_embedding(
        self,
        file_id: str,
        embedding: np.ndarray,
        modality: str,
        model_name: str
    ) -> bool:
        """임베딩을 캐시에 저장합니다."""
        if not self.enabled:
            return False
            
        try:
            emb_path, meta_path = self._get_cache_path(file_id, modality)
            
            # 임베딩 저장
            if self.compression:
                np.savez_compressed(emb_path, embedding=embedding)
            else:
                np.save(emb_path, embedding)
            
            # 메타데이터 저장
            meta = EmbeddingInfo(
                shape=embedding.shape,
                model_name=model_name,
                created_at=datetime.now().isoformat(),
                config_hash=self._get_config_hash(modality),
                version=self.version,
                modality=modality
            )
            
            with open(meta_path, 'w') as f:
                json.dump(asdict(meta), f, indent=2)
                
            return True
            
        except Exception as e:
            self.logger.warning(f"Error saving cache for {file_id}: {e}")
            return False

    def clear_cache(self, older_than_days: Optional[int] = None) -> None:
        """캐시를 정리합니다."""
        if not self.enabled:
            return
            
        try:
            now = datetime.now()
            for cache_file in self.cache_dir.glob("*.*"):
                if older_than_days is not None:
                    mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if (now - mtime).days <= older_than_days:
                        continue
                try:
                    cache_file.unlink()
                except Exception as e:
                    self.logger.warning(f"Error removing {cache_file}: {e}")
                    
            self.logger.info("Cache cleared successfully")
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}") 