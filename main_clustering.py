"""
감정 클러스터링 메인 스크립트.

오디오와 텍스트 임베딩을 생성하고 클러스터링을 수행합니다.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
import torch

from src.analysis.cluster_optimization import ClusterOptimizer
from src.analysis.embedding_clustering import Clustering
from src.analysis.latent_analysis import LatentAnalyzer
from src.data.download import DatasetDownloader
from src.data.embedding_cache import EmbeddingCache
from src.data.preprocess import (
    AudioPreprocessor,
    TextPreprocessor,
    WHISPER_AVAILABLE,
    TranscriptionConfig,
    load_text_data,
    load_dataset,
    process_text_embeddings
)
from src.data.metadata import DatasetMetadataManager
from src.data.isear_dataset import ISEARDataset
from src.data.dataset_analyzer import DatasetAnalyzer

logger = logging.getLogger(__name__)

# 감정 레이블 매핑
EMOTION_MAP: Dict[str, int] = {
    # ISEAR 감정
    "joy": 0,
    "fear": 1,
    "anger": 2,
    "sadness": 3,
    "disgust": 4,
    "shame": 5,
    "guilt": 6,
    # RAVDESS 감정
    "neutral": 7,
    "calm": 8,
    "happy": 9,
    "sad": 10,
    "angry": 11,
    "fearful": 12,
    "surprised": 13
}


def setup_logging(cfg: DictConfig) -> None:
    """로깅 설정을 초기화합니다."""
    # 로그 디렉토리 생성
    log_dir = os.path.dirname(cfg.logging.handlers.file.filename)
    os.makedirs(log_dir, exist_ok=True)
    
    # 핸들러 리스트 생성
    handlers = []
    
    # 콘솔 핸들러
    if cfg.logging.handlers.console.enabled:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(cfg.logging.handlers.console.level)
        console_handler.setFormatter(
            logging.Formatter(cfg.logging.format)
        )
        handlers.append(console_handler)
    
    # 파일 핸들러
    if cfg.logging.handlers.file.enabled:
        try:
            file_handler = logging.FileHandler(
                cfg.logging.handlers.file.filename,
                mode=cfg.logging.handlers.file.mode
            )
            file_handler.setFormatter(
                logging.Formatter(cfg.logging.format)
            )
            handlers.append(file_handler)
        except Exception as e:
            print(f"Warning: Could not create file handler: {e}")
            
    # 기본 로깅 설정
    logging.basicConfig(
        level=cfg.logging.level,
        format=cfg.logging.format,
        handlers=handlers
    )
    
    # Whisper 관련 로깅
    if not WHISPER_AVAILABLE:
        logging.warning(
            "Whisper not available. Speech transcription will be disabled. "
            "To enable, install with: pip install --upgrade openai-whisper"
        )


def process_audio_embeddings(
    cfg: DictConfig,
    audio_preprocessor: AudioPreprocessor,
    file_ids: List[str],
    cache: EmbeddingCache
) -> np.ndarray:
    """오디오 임베딩을 생성하거나 캐시에서 로드합니다."""
    embeddings = []
    
    for file_id in tqdm(file_ids, desc="Processing audio"):
        try:
            # 캐시에서 임베딩 확인
            cached_emb = cache.load_embedding(file_id, modality="audio")
            if cached_emb is not None:
                embeddings.append(cached_emb)
                continue
            
            # 오디오 파일 경로
            audio_path = os.path.join(cfg.dataset.download_path, file_id)
            if not os.path.exists(audio_path):
                logger.warning(f"Audio file not found: {audio_path}")
                continue
                
            # 오디오 전처리 및 임베딩 생성
            emb = audio_preprocessor.preprocess(audio_path)
            if emb is not None:
                emb_np = emb.numpy()
                embeddings.append(emb_np)
                # 임베딩 캐시 저장
                cache.save_embedding(
                    file_id,
                    emb_np,
                    modality="audio",
                    model_name=cfg.model.audio.name  # 모델 이름 직접 전달
                )
                
        except Exception as e:
            logger.error(f"Error processing audio file {file_id}: {str(e)}")
            continue
            
    if not embeddings:
        logger.error("No audio embeddings were generated")
        return np.array([])
        
    return np.stack(embeddings)


def process_text(cfg: DictConfig) -> Optional[np.ndarray]:
    """텍스트 데이터를 처리하고 임베딩을 생성합니다."""
    try:
        # 텍스트 데이터 로드
        text_data = load_text_data(cfg.dataset.download_path, cfg)
        if not text_data:
            logger.error("No text data found")
            return None
            
        # 텍스트 전처리 및 임베딩 생성
        preprocessor = TextPreprocessor(
            model_name=cfg.model.text.name,
            max_length=cfg.model.text.max_length
        )
        
        # 텍스트 처리
        texts = []
        for idx in sorted(text_data.keys()):  # 정렬된 순서로 처리
            text = text_data[idx]
            if text:  # 빈 텍스트 제외
                texts.append(text)
            else:
                logger.warning(f"Empty text for ID: {idx}")
        
        if not texts:
            logger.error("No valid texts found")
            return None
        
        # 임베딩 생성
        embeddings = preprocessor.preprocess_batch(texts)
        if embeddings is None:
            logger.error("No text embeddings were generated")
            return None
            
        return embeddings.numpy()
        
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        return None


def process_embeddings(
    cfg: DictConfig,
    audio_preprocessor: Optional[AudioPreprocessor],
    text_preprocessor: Optional[TextPreprocessor],
    file_ids: List[str],
    labels_data: Dict[str, str],
    cache: EmbeddingCache
) -> Tuple[np.ndarray, np.ndarray]:
    """임베딩을 생성하거나 캐시에서 로드합니다."""
    
    available_modalities = cfg.dataset.modalities
    primary_modality = cfg.dataset.primary_modality
    
    # RAVDESS 데이터셋 처리
    if cfg.dataset.name == "ravdess":
        audio_paths, transcriptions = load_dataset(cfg)
        
        # 오디오 임베딩 처리
        audio_embeddings = []
        if "audio" in available_modalities and audio_preprocessor:
            for audio_path in tqdm(
                audio_paths, 
                desc="Processing audio"
            ):
                # 캐시 확인
                file_id = str(Path(audio_path).relative_to(cfg.dataset.data_path))
                cached_emb = cache.load_embedding(file_id, modality="audio")
                
                if cached_emb is not None:
                    audio_embeddings.append(cached_emb)
                else:
                    # 새로운 임베딩 생성
                    emb = audio_preprocessor.preprocess(audio_path)
                    if emb is not None:
                        emb_np = emb.numpy()
                        audio_embeddings.append(emb_np)
                        cache.save_embedding(
                            file_id,
                            emb_np,
                            modality="audio",
                            model_name=cfg.model.audio.name
                        )
            
            audio_embeddings = (
                np.stack(audio_embeddings) 
                if audio_embeddings 
                else np.array([])
            )
        
        # 텍스트 임베딩 처리
        text_embeddings = []
        if ("text" in available_modalities and 
            text_preprocessor and 
            transcriptions):
            text_embeddings = process_text_embeddings(
                cfg, text_preprocessor, transcriptions, cache
            )
        else:
            text_embeddings = np.array([])
        
        # 모달리티별 임베딩 반환
        if primary_modality == "audio":
            return audio_embeddings, text_embeddings
        else:
            return text_embeddings, audio_embeddings
    
    # ISEAR 데이터셋 처리 (기존 로직)
    else:
        # 텍스트 전용 데이터셋인 경우
        if len(available_modalities) == 1 and "text" in available_modalities:
            text_embeddings = process_text(cfg)
            return text_embeddings, np.array([])  # 빈 secondary 임베딩
        
        # 멀티모달 데이터셋인 경우 기존 로직 수행
        audio_embeddings = np.array([])
        if "audio" in available_modalities:
            if not audio_preprocessor:
                raise ValueError("Audio preprocessor required but not provided")
            audio_embeddings = process_audio_embeddings(
                cfg, audio_preprocessor, file_ids, cache
            )
        
        text_embeddings = np.array([])
        if "text" in available_modalities:
            if not text_preprocessor:
                raise ValueError("Text preprocessor required but not provided")
            text_embeddings = process_text(cfg)
        
        # 모달리티별 임베딩 반환
        if primary_modality == "audio":
            return audio_embeddings, text_embeddings
        else:  # text
            return text_embeddings, audio_embeddings


def analyze_dataset_with_metadata(metadata_path: str) -> Dict:
    """메타데이터를 사용하여 데이터셋을 분석합니다."""
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # 감정별 분포 분석
        emotion_dist = {}
        gender_dist = {'male': 0, 'female': 0}
        intensity_dist = {'normal': 0, 'strong': 0}
        
        for file_info in metadata.values():
            emotion = file_info['emotion']
            gender = file_info['gender']
            intensity = file_info['intensity']
            
            emotion_dist[emotion] = emotion_dist.get(emotion, 0) + 1
            gender_dist[gender] += 1
            intensity_dist[intensity] += 1
            
        return {
            'total_files': len(metadata),
            'emotion_distribution': emotion_dist,
            'gender_distribution': gender_dist,
            'intensity_distribution': intensity_dist,
            'class_weights': {
                emotion: 1.0 / count 
                for emotion, count in emotion_dist.items()
            }
        }
    except Exception as e:
        logger.error(f"Error analyzing metadata: {e}")
        return {}


def optimize_clusters(
    cfg: DictConfig,
    embeddings: np.ndarray,
    ground_truth: List[str],
    modality: str,
    dataset_stats: Optional[Dict] = None
) -> Tuple[int, float]:
    """클러스터 수를 최적화합니다."""
    try:
        save_dir = os.path.join(
            cfg.general.output_dir,
            f"cluster_optimization_{modality}"
        )
        os.makedirs(save_dir, exist_ok=True)
        
        # 클러스터 최적화기 초기화
        optimizer = ClusterOptimizer(
            n_clusters_range=(
                cfg.clustering.optimization.n_clusters_range.min,
                cfg.clustering.optimization.n_clusters_range.max
            ),
            method=cfg.clustering.method,
            method_params=cfg.clustering.methods[cfg.clustering.method],
            # 공통 시각화 설정 사용
            visualization_params=cfg.clustering.visualization
        )
        
        # 최적화 수행
        best_n, best_score = optimizer.optimize(
            embeddings,
            ground_truth,
            save_dir=save_dir
        )
        
        logger.info(f"\nOptimization results for {modality}:")
        logger.info(f"Best number of clusters: {best_n}")
        logger.info(f"Best NMI score: {best_score:.3f}")
        
        return best_n, best_score
        
    except Exception as e:
        logger.error(f"Error in optimization: {e}")
        return cfg.clustering.n_clusters, 0.0


def analyze_dataset(
    audio_preprocessor: AudioPreprocessor,
    dataset_path: str
) -> Dict[str, Any]:
    """데이터셋을 분석하고 통계를 출력합니다."""
    dataset_stats = audio_preprocessor.analyze_dataset(dataset_path)
    
    logger.info("Dataset Statistics:")
    logger.info(f"Total files: {dataset_stats['total_files']}")
    logger.info(f"Valid files: {dataset_stats['valid_files']}")
    logger.info(f"Average duration: {dataset_stats['avg_duration']:.2f}s")
    logger.info(f"Min duration: {dataset_stats['min_duration']:.2f}s")
    logger.info(f"Max duration: {dataset_stats['max_duration']:.2f}s")
    logger.info(f"Duration std: {dataset_stats['std_duration']:.2f}s")
    logger.info(f"Sample rates: {dataset_stats['sample_rates']}")
    logger.info(f"Channels: {dataset_stats['channels']}")
    
    return dataset_stats


def print_clustering_metrics(
    metrics: Dict[str, Any],
    modality: str
) -> None:
    """클러스터링 메트릭을 출력합니다."""
    logger.info(f"\n{modality.title()} Clustering Metrics:")
    logger.info(f"- ARI: {metrics['ari']:.3f}")
    logger.info(f"- NMI: {metrics['nmi']:.3f}")
    logger.info(f"- Samples: {metrics['n_samples']}")
    logger.info(f"- Predicted clusters: {metrics['n_pred_clusters']}")
    logger.info(f"- True emotion classes: {metrics['n_true_clusters']}")
    
    logger.info("\nPredicted cluster distribution:")
    for cluster, count in metrics['pred_cluster_distribution'].items():
        logger.info(f"  Cluster {cluster}: {count} samples")
        
    logger.info("\nTrue emotion distribution:")
    for emotion, count in metrics['true_cluster_distribution'].items():
        logger.info(f"  {emotion}: {count} samples")


def process_missing_files(
    cfg: DictConfig,
    audio_preprocessor: AudioPreprocessor,
    text_preprocessor: TextPreprocessor,
    file_ids: List[str],
    missing_audio: List[str],
    missing_text: List[str],
    embedding_cache: EmbeddingCache,
    transcriptions_dir: str
) -> Dict[str, Any]:
    """누락된 파일들을 처리합니다."""
    new_audio_embeddings = []
    new_text_embeddings = []
    new_audio_files = []
    new_text_files = []
    processed_files = []
    
    for file_id in tqdm(file_ids):
        try:
            full_audio_path = os.path.join(cfg.dataset.download_path, file_id)
            if not os.path.exists(full_audio_path):
                logger.warning(f"Audio file not found: {full_audio_path}")
                continue
            
            success = False
            
            # 오디오 임베딩 처리
            if file_id in missing_audio:
                audio_emb = audio_preprocessor.preprocess(full_audio_path)
                if audio_emb is not None:
                    new_audio_embeddings.append(audio_emb.numpy())
                    new_audio_files.append(file_id)
                    success = True
            
            # 텍스트 임베딩 처리
            if file_id in missing_text:
                # Whisper 모델 사용 가능 여부 확인
                if (audio_preprocessor.whisper_model is None or 
                    not audio_preprocessor.transcription_config or 
                    not audio_preprocessor.transcription_config.enabled):
                    logger.warning("Whisper transcription is not available or disabled")
                    continue
                    
                # 트랜스크립션 시도
                try:
                    transcription = audio_preprocessor.transcribe(
                        full_audio_path,
                        transcriptions_dir
                    )
                    if transcription:
                        logger.debug(
                            f"Transcription for {file_id}: {transcription[:50]}..."
                        )
                        text_emb = text_preprocessor.preprocess(transcription)
                        if text_emb is not None:
                            new_text_embeddings.append(text_emb.numpy())
                            new_text_files.append(file_id)
                            success = True
                    else:
                        logger.warning(f"Transcription failed for {file_id}")
                except Exception as e:
                    logger.error(f"Transcription error for {file_id}: {str(e)}")
                    continue
            
            if success:
                processed_files.append(file_id)
            
        except Exception as e:
            logger.error(f"Error processing {file_id}: {str(e)}")
            continue
    
    return {
        'audio_embeddings': new_audio_embeddings,
        'text_embeddings': new_text_embeddings,
        'audio_files': new_audio_files,
        'text_files': new_text_files,
        'processed_files': processed_files
    }


def update_embeddings(
    audio_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    new_embeddings: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """기존 임베딩을 새로운 임베딩으로 업데이트합니다."""
    # 오디오 임베딩 업데이트
    if new_embeddings['audio_embeddings']:
        new_audio = np.stack(new_embeddings['audio_embeddings'])
        audio_embeddings = (
            np.concatenate([audio_embeddings, new_audio])
            if len(audio_embeddings) > 0
            else new_audio
        )
    
    # 텍스트 임베딩 업데이트
    if new_embeddings['text_embeddings']:
        new_text = np.stack(new_embeddings['text_embeddings'])
        text_embeddings = (
            np.concatenate([text_embeddings, new_text])
            if len(text_embeddings) > 0
            else new_text
        )
    
    return audio_embeddings, text_embeddings


def check_dataset_exists(download_path: str) -> bool:
    """데이터셋이 이미 존재하는지 확인합니다."""
    download_path = Path(download_path)
    metadata_path = download_path / "metadata.json"
    labels_path = download_path / "labels.json"
    
    # 디렉토리 구조 로깅
    logger.debug("Checking paths:")
    logger.debug(
        f"Download path: {download_path} (exists: {download_path.exists()})"
    )
    
    # Actor 폴더들이 존재하는지 직접 확인
    actor_dirs = list(download_path.glob("Actor_*"))
    if actor_dirs:
        # WAV 파일 검색
        wav_files = []
        for actor_dir in actor_dirs:
            wav_files.extend(actor_dir.glob("*.wav"))
        file_count = len(wav_files)
        
        if file_count > 0:
            logger.info(
                f"Found {file_count} WAV files in {len(actor_dirs)} actor dirs"
            )
            
            # 메타데이터 매니저 초기화
            metadata_manager = DatasetMetadataManager(str(download_path))
            
            # 메타데이터 확인 및 생성
            if not metadata_path.exists():
                logger.info("Creating dataset metadata...")
                metadata_manager.create_metadata()
            else:
                logger.info("Using existing metadata")
            
            # 레이블 파일 확인 및 생성
            if not labels_path.exists():
                logger.info("Creating labels file...")
                DatasetDownloader._create_ravdess_labels(
                    str(download_path), 
                    metadata_manager
                )
            else:
                logger.info("Using existing labels")
                
            return True
            
        else:
            logger.warning("Actor directories exist but contain no WAV files")
    else:
        logger.debug("No Actor directories found")
            
    return False


def perform_fusion(
    primary_embeddings: np.ndarray,
    secondary_embeddings: np.ndarray,
    fusion_config: DictConfig
) -> np.ndarray:
    """임베딩 퓨전을 수행합니다."""
    try:
        if fusion_config.method == "early":
            # 단순 연결
            fused_embeddings = np.concatenate(
                [primary_embeddings, secondary_embeddings],
                axis=1
            )
        elif fusion_config.method == "weighted":
            # 가중치 합
            alpha = fusion_config.get('alpha', 0.5)
            fused_embeddings = (
                alpha * primary_embeddings + 
                (1 - alpha) * secondary_embeddings
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_config.method}")
            
        logger.info(f"Fused embeddings shape: {fused_embeddings.shape}")
        return fused_embeddings
        
    except Exception as e:
        logger.error(f"Error in fusion: {e}")
        return primary_embeddings  # 실패 시 primary 임베딩만 반환


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """메인 실행 함수."""
    try:
        setup_logging(cfg)
        
        # 데이터셋 모달리티 설정
        primary_modality = cfg.dataset.primary_modality
        modalities = cfg.dataset.modalities
        
        # 데이터셋 다운로드
        downloader = DatasetDownloader(cfg)
        if not downloader.check_dataset_exists():
            logger.info("Downloading dataset...")
            if cfg.dataset.name == "ravdess":
                downloader.download_ravdess()
            elif cfg.dataset.name == "isear":
                downloader.download_isear()
        else:
            logger.info("Dataset already exists, skipping download")
            
        # 데이터셋 분석기 초기화
        analyzer = DatasetAnalyzer(cfg)
        
        # ISEAR 데이터셋 분석
        if cfg.dataset.name == "isear":
            try:
                dataset = ISEARDataset(cfg)
                
                # 레이블 파일이 없으면 생성
                if not Path(cfg.dataset.labels_path).exists():
                    logger.info("Creating labels file...")
                    dataset.create_labels()
                
                df = dataset.load_data()
                
                # 데이터셋 분석
                stats = analyzer.analyze_isear(df)
                
                # 전처리 권장사항 확인
                recommendations = analyzer.get_preprocessing_recommendations()
                logger.info("\nPreprocessing Recommendations:")
                for category, items in recommendations.get("isear", {}).items():
                    if isinstance(items, (list, tuple)) and items:
                        logger.info(f"\n{category.replace('_', ' ').title()}:")
                        for item in items:
                            logger.info(f"  - {item}")
                    elif isinstance(items, dict) and items:
                        logger.info(f"\n{category.replace('_', ' ').title()}:")
                        for key, value in items.items():
                            logger.info(f"  - {key}: {value}")
                
                # 문제가 있는 샘플 처리
                if stats["potential_issues"]["invalid_emotions"]:
                    invalid_indices = stats["potential_issues"]["invalid_emotions"]
                    n_invalid = len(invalid_indices)
                    logger.warning(f"Found {n_invalid} invalid emotion labels")
                    df = df.drop(invalid_indices)
                
                if stats["potential_issues"]["very_short_texts"]:
                    short_indices = stats["potential_issues"]["very_short_texts"]
                    n_short = len(short_indices)
                    logger.warning(f"Found {n_short} very short texts")
                    df = df.drop(short_indices)
                
                # 통계 저장 (데이터셋별로 구분)
                stats_path = Path(cfg.general.output_dir) / f"{cfg.dataset.name}_stats.json"
                with open(stats_path, "w") as f:
                    # DictConfig를 일반 dict로 변환
                    serializable_stats = json.loads(
                        json.dumps(
                            stats,
                            default=lambda x: (
                                x.item() if hasattr(x, 'item') else str(x)
                            )
                        )
                    )
                    json.dump(serializable_stats, f, indent=2)
                logger.info(f"Saved dataset statistics to {stats_path}")
                
            except Exception as e:
                logger.error(f"Error processing ISEAR dataset: {e}")
                raise
            
        # RAVDESS 데이터셋 분석
        elif cfg.dataset.name == "ravdess":
            try:
                audio_paths, transcriptions = load_dataset(cfg)
                
                # 데이터셋 분석
                stats = analyzer.analyze_ravdess(audio_paths)
                
                # 전처리 권장사항 확인
                recommendations = analyzer.get_preprocessing_recommendations()
                logger.info("\nPreprocessing Recommendations:")
                for category, items in recommendations.get("ravdess", {}).items():
                    if isinstance(items, (list, tuple)) and items:
                        logger.info(f"\n{category.replace('_', ' ').title()}:")
                        for item in items:
                            logger.info(f"  - {item}")
                    elif isinstance(items, dict) and items:
                        logger.info(f"\n{category.replace('_', ' ').title()}:")
                        for key, value in items.items():
                            logger.info(f"  - {key}: {value}")
                
                # 문제가 있는 파일 제외
                if stats["potential_issues"]["corrupted_files"]:
                    corrupted_files = stats["potential_issues"]["corrupted_files"]
                    n_corrupted = len(corrupted_files)
                    logger.warning(f"Excluding {n_corrupted} corrupted files")
                    # audio_paths와 transcriptions 동기화
                    valid_indices = [
                        i for i, p in enumerate(audio_paths)
                        if p not in corrupted_files
                    ]
                    audio_paths = [audio_paths[i] for i in valid_indices]
                    if transcriptions:
                        transcriptions = [transcriptions[i] for i in valid_indices]
                
                # 통계 저장 (데이터셋별로 구분)
                stats_path = Path(cfg.general.output_dir) / f"{cfg.dataset.name}_stats.json"
                with open(stats_path, "w") as f:
                    # DictConfig를 일반 dict로 변환
                    serializable_stats = json.loads(
                        json.dumps(
                            stats,
                            default=lambda x: (
                                x.item() if hasattr(x, 'item') else str(x)
                            )
                        )
                    )
                    json.dump(serializable_stats, f, indent=2)
                logger.info(f"Saved dataset statistics to {stats_path}")
                
            except Exception as e:
                logger.error(f"Error processing RAVDESS dataset: {e}")
                raise
        
        # 전처리기 초기화
        audio_preprocessor = None
        text_preprocessor = None
        
        # 오디오 전처리기 초기화
        if "audio" in modalities:
            # Whisper 설정 생성
            transcription_config = TranscriptionConfig(
                enabled=cfg.model.transcription.enabled,
                model_size=cfg.model.transcription.model_size,
                device=cfg.model.transcription.device,
                model_dir=cfg.model.transcription.model_dir,
                batch_size=cfg.model.transcription.batch_size,
                options={
                    'language': cfg.model.transcription.options.language,
                    'task': cfg.model.transcription.options.task,
                    'beam_size': cfg.model.transcription.options.beam_size,
                    'best_of': cfg.model.transcription.options.best_of,
                    'fp16': torch.cuda.is_available(),
                    'condition_on_previous_text': (
                        cfg.model.transcription.options.condition_on_previous_text
                    ),
                    'temperature': cfg.model.transcription.options.temperature
                }
            )
            
            audio_preprocessor = AudioPreprocessor(
                model_name=cfg.model.audio.name,
                sample_rate=cfg.model.audio.sample_rate,
                verbose=True,
                transcription_config=transcription_config
            )
        
        # 텍스트 전처리기 초기화
        if "text" in modalities:
            text_preprocessor = TextPreprocessor(
                model_name=cfg.model.text.name,
                max_length=cfg.model.text.max_length,
                verbose=False
            )
            logger.info(f"Initialized text preprocessor with model: {cfg.model.text.name}")
        
        # 데이터셋 분석
        logger.info("\n=== Analyzing Dataset ===")
        if audio_preprocessor:
            dataset_stats = analyze_dataset(audio_preprocessor, cfg.dataset.download_path)
            
            # 분석 결과에 따라 transcription 설정 조정
            if dataset_stats['avg_duration'] > 30:
                logger.warning("Audio files are too long, adjusting transcription settings")
                cfg.model.transcription.options.chunk_length = 30
            elif dataset_stats['avg_duration'] < 1:
                logger.warning("Audio files are too short, disabling transcription")
                cfg.model.transcription.enabled = False
        else:
            dataset_stats = {}
            logger.info("Skipping audio analysis (text-only dataset)")

        logger.info("=== Loading Labels ===")
        # 레이블 로드
        ground_truth = []
        try:
            with open(cfg.dataset.labels_path, "r") as f:
                labels_data = json.load(f)
                # labels 필드에서 값들만 추출
                if "labels" not in labels_data:
                    raise KeyError("Labels field not found in labels file")
                labels = labels_data["labels"]
                # 정렬된 순서로 레이블 로드
                sorted_ids = sorted(labels.keys(), key=int)
                ground_truth = [labels[id_] for id_ in sorted_ids]
            logger.info(f"Loaded {len(ground_truth)} labels")
            logger.info(f"Unique emotions: {set(ground_truth)}")
        except Exception as e:
            logger.error(f"Error loading labels: {e}")
            return

        # 파일 ID 목록 생성 (정렬된 순서)
        file_ids = sorted_ids  # 위에서 정렬된 ID 목록 재사용

        logger.info("\n=== Initializing Preprocessors ===")
        logger.info("Preprocessors initialized successfully")

        logger.info("\n=== Processing Audio and Text Data ===")
        # transcriptions 디렉토리 설정
        transcriptions_dir = os.path.join(cfg.dataset.download_path, "transcriptions")
        os.makedirs(transcriptions_dir, exist_ok=True)
        
        # 임베딩 캐시 초기화
        embedding_cache = EmbeddingCache(cfg)
        
        # 파일 ID 목록 생성
        file_ids = list(labels.keys())
        primary_embeddings, secondary_embeddings = process_embeddings(
            cfg, audio_preprocessor, text_preprocessor,
            file_ids, labels, embedding_cache
        )
        
        if len(primary_embeddings) == 0 or len(ground_truth) == 0:
            logger.error("No files were processed successfully")
            return

        # 임베딩 데이터 디버그 정보
        logger.info("\n=== Embeddings Debug Info ===")
        logger.info(f"Primary embeddings shape: {primary_embeddings.shape}")
        logger.info(f"Primary embeddings stats:")
        logger.info(f"- Mean: {primary_embeddings.mean():.4f}")
        logger.info(f"- Std: {primary_embeddings.std():.4f}")
        logger.info(f"- Min: {primary_embeddings.min():.4f}")
        logger.info(f"- Max: {primary_embeddings.max():.4f}")
        logger.info(f"- NaN values: {np.isnan(primary_embeddings).sum()}")
        logger.info(f"- Inf values: {np.isinf(primary_embeddings).sum()}")

        # 레이블 데이터 디버그 정보
        logger.info("\n=== Labels Debug Info ===")
        unique_labels = set(ground_truth)
        logger.info(f"Number of unique labels: {len(unique_labels)}")
        logger.info("Label distribution:")
        for label in sorted(unique_labels):
            count = ground_truth.count(label)
            percentage = (count / len(ground_truth)) * 100
            logger.info(f"- {label}: {count} samples ({percentage:.2f}%)")

        logger.info(f"Number of processed files: {len(primary_embeddings)}")
        logger.info(f"Number of ground truth labels: {len(ground_truth)}")
        
        # 임베딩 형태 확인
        logger.info(f"Primary embeddings shape: {primary_embeddings.shape}")
        if len(secondary_embeddings) > 0:
            logger.info(f"Secondary embeddings shape: {secondary_embeddings.shape}")
        else:
            logger.warning("No secondary embeddings available - using primary only mode")
            secondary_embeddings = np.zeros_like(primary_embeddings)

        logger.info("\n=== Performing Clustering ===")
        # 클러스터링 수행
        logger.info("Processing primary embeddings...")
        
        # 클러스터링 디버그 정보
        logger.info("\n=== Clustering Debug Info ===")
        logger.info(f"Number of clusters: {cfg.clustering.n_clusters}")
        logger.info(f"Clustering method: {cfg.clustering.method}")
        logger.info("Method parameters:")
        for param, value in cfg.clustering.methods[cfg.clustering.method].items():
            logger.info(f"- {param}: {value}")
        
        primary_labels = Clustering.perform_clustering(
            primary_embeddings,
            cfg.clustering.n_clusters
        )
        
        # 클러스터링 결과 디버그 정보
        unique_clusters = set(primary_labels)
        logger.info("\n=== Clustering Results Debug Info ===")
        logger.info(f"Number of unique clusters: {len(unique_clusters)}")
        logger.info("Cluster size distribution:")
        for cluster in sorted(unique_clusters):
            count = list(primary_labels).count(cluster)
            percentage = (count / len(primary_labels)) * 100
            logger.info(f"- Cluster {cluster}: {count} samples ({percentage:.2f}%)")

        # 보조 모달리티 클러스터링 (있는 경우)
        secondary_labels = None
        if not np.array_equal(secondary_embeddings, np.zeros_like(primary_embeddings)):
            logger.info("Processing secondary embeddings...")
            secondary_labels = Clustering.perform_clustering(
                secondary_embeddings,
                cfg.clustering.n_clusters
            )

        # 퓨전 클러스터링
        fused_labels = None
        if cfg.model.fusion == "early":
            logger.info("Using early fusion method")
            fused_embeddings = np.concatenate(
                [primary_embeddings, secondary_embeddings],
                axis=1
            )
            fused_labels = Clustering.perform_clustering(
                fused_embeddings,
                cfg.clustering.n_clusters
            )
        elif cfg.model.fusion == "weighted":
            logger.info("Using weighted fusion method")
            alpha = 0.5
            fused_embeddings = (alpha * primary_embeddings +
                              (1 - alpha) * secondary_embeddings)
            fused_labels = Clustering.perform_clustering(
                fused_embeddings,
                cfg.clustering.n_clusters
            )

        # 결과 평가
        logger.info("\n=== Evaluating Results ===")
        primary_metrics = Clustering.evaluate_clustering(primary_labels, ground_truth)
        
        if secondary_labels is not None:
            secondary_metrics = Clustering.evaluate_clustering(secondary_labels, ground_truth)
        else:
            secondary_metrics = {
                'ari': 0.0,
                'nmi': 0.0,
                'n_samples': len(ground_truth),
                'n_pred_clusters': cfg.clustering.n_clusters,
                'n_true_clusters': len(set(ground_truth)),
                'pred_cluster_distribution': {},
                'true_cluster_distribution': {}
            }

        if fused_labels is not None:
            fusion_metrics = Clustering.evaluate_clustering(fused_labels, ground_truth)
        else:
            fusion_metrics = primary_metrics.copy()

        # 결과 출력
        logger.info("\n=== Final Results ===")
        print_clustering_metrics(primary_metrics, "primary")
        print_clustering_metrics(secondary_metrics, "secondary")
        print_clustering_metrics(fusion_metrics, "fusion")

        # 결과 저장
        results_file = Clustering.save_results(
            primary_metrics,
            secondary_metrics,
            fusion_metrics,
            save_dir=Path(cfg.general.output_dir)
        )
        logger.info(f"\nResults saved to: {results_file}")

        # 잠재 분석 시 숫자 레이블 사용
        numeric_labels = [EMOTION_MAP[label] for label in ground_truth]
        
        logger.info("\n=== Performing Latent Analysis ===")
        latent_analyzer = LatentAnalyzer(cfg)  # config 전달
        
        # 결과 저장 디렉토리 생성
        results_dir = Path(cfg.general.output_dir) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 오디오 임베딩 분석
        logger.info("Analyzing primary embeddings...")
        primary_latent_results = latent_analyzer.fit_transform_all(
            primary_embeddings,
            labels=numeric_labels
        )
        
        # 결과 시각화 및 저장
        latent_analyzer.visualize_results(
            primary_latent_results,
            labels=numeric_labels,
            save_path=str(results_dir / "primary_latent_analysis.png")
        )

        # 텍스트 임베딩 분석 (있는 경우에만)
        if not np.array_equal(secondary_embeddings, np.zeros_like(primary_embeddings)):
            logger.info("Analyzing secondary embeddings...")
            secondary_latent_results = latent_analyzer.fit_transform_all(
                secondary_embeddings,
                labels=numeric_labels
            )
            
            latent_analyzer.visualize_results(
                secondary_latent_results,
                labels=numeric_labels,
                save_path=str(results_dir / "secondary_latent_analysis.png")
            )
        else:
            logger.info("Skipping secondary embeddings analysis (primary only mode)")

        # 메타데이터 분석
        metadata_path = os.path.join(cfg.dataset.download_path, "metadata.json")
        dataset_stats = analyze_dataset_with_metadata(metadata_path)
        
        if dataset_stats:
            logger.info("\n=== Dataset Statistics from Metadata ===")
            logger.info(f"Total files: {dataset_stats['total_files']}")
            logger.info("\nEmotion distribution:")
            for emotion, count in dataset_stats['emotion_distribution'].items():
                logger.info(f"  {emotion}: {count}")
            logger.info("\nGender distribution:")
            for gender, count in dataset_stats['gender_distribution'].items():
                logger.info(f"  {gender}: {count}")
            logger.info("\nIntensity distribution:")
            for intensity, count in dataset_stats['intensity_distribution'].items():
                logger.info(f"  {intensity}: {count}")
        
        # 클러스터 최적화 시 메타데이터 활용
        logger.info("\n=== Optimizing Number of Clusters ===")
        best_n_primary, criterion_primary = optimize_clusters(
            cfg, primary_embeddings, ground_truth, "primary", dataset_stats
        )
        
        if len(secondary_embeddings) > 0:
            secondary_modality = "text" if primary_modality == "audio" else "audio"
            logger.info(f"\n=== Optimizing Number of Clusters for {secondary_modality} ===")
            best_n_secondary, criterion_secondary = optimize_clusters(
                cfg, secondary_embeddings, ground_truth, secondary_modality, dataset_stats
            )
            
            # 퓨전 클러스터링 수행
            if cfg.clustering.fusion.enabled:
                logger.info("\n=== Performing Fusion Clustering ===")
                fusion_embeddings = perform_fusion(
                    primary_embeddings,
                    secondary_embeddings,
                    cfg.clustering.fusion
                )
                optimize_clusters(
                    cfg, fusion_embeddings, ground_truth, "fusion", dataset_stats
                )

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
