"""오디오 및 텍스트 전처리를 위한 모듈."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, List
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
import torchaudio
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
from transformers.modeling_outputs import BaseModelOutput
from dotenv import load_dotenv
import pandas as pd
from omegaconf import DictConfig
import json
from tqdm import tqdm
from src.data.embedding_cache import EmbeddingCache

logger = logging.getLogger(__name__)

# .env 파일 로드
load_dotenv()

# Whisper 모델 import 및 타입 정의
WhisperModel = Any  # 타입 힌트용 플레이스홀더
WHISPER_AVAILABLE = False

try:
    import whisper
    WhisperModel = type(whisper.load_model("tiny"))  # 실제 Whisper 타입 가져오기
    WHISPER_AVAILABLE = True
except ImportError:
    logger.warning(
        "Failed to import whisper. "
        "Speech transcription will be disabled. "
        "To enable, install with: pip install --upgrade openai-whisper"
    )
except Exception as e:
    logger.warning(f"Error initializing Whisper: {str(e)}")

@dataclass
class AudioConfig:
    """오디오 처리 설정."""
    sample_rate: int = 16000
    normalize: bool = True
    chunk_duration: float = 30.0  # 초 단위
    min_duration: float = 1.0
    max_duration: float = 30.0
    batch_size: int = 32

@dataclass
class TranscriptionConfig:
    """음성-텍스트 변환 설정."""
    enabled: bool = True
    model_size: str = "tiny"
    device: str = "cpu"
    batch_size: int = 1
    model_dir: str = "models/whisper"
    options: Dict[str, Any] = None

@dataclass
class TextConfig:
    """텍스트 처리 설정."""
    max_length: int = 512
    padding: bool = True
    truncation: bool = True
    return_tensors: str = "pt"
    batch_size: int = 32

class AudioPreprocessor:
    """오디오 전처리 및 임베딩 생성 클래스."""

    def __init__(
        self,
        model_name: str,
        sample_rate: Optional[int] = None,
        config: Optional[AudioConfig] = None,
        transcription_config: Optional[TranscriptionConfig] = None,
        verbose: bool = True
    ) -> None:
        """오디오 전처리기를 초기화합니다."""
        if sample_rate is not None:
            config = config or AudioConfig()
            config.sample_rate = sample_rate
        
        self.config = config or AudioConfig()
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델 초기화
        self.model = self._initialize_audio_model(model_name)
        self.whisper_model: Optional[WhisperModel] = None
        self.transcription_config = transcription_config  # 설정 저장
        
        if transcription_config and transcription_config.enabled:
            self.whisper_model = self._initialize_whisper(transcription_config)
        
    def _initialize_audio_model(self, model_name: str) -> PreTrainedModel:
        """오디오 임베딩 모델을 초기화합니다."""
        try:
            # 모델 캐시 디렉토리 설정
            cache_dir = os.path.join("models", "audio")
            os.makedirs(cache_dir, exist_ok=True)
            
            # 모델 로드 시 캐시 디렉토리 지정
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=False  # 필요한 경우 다운로드 허용
            ).to(self.device)
            
            model.eval()
            if self.verbose:
                logger.info(f"Loaded {model_name} on {self.device}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading audio model {model_name}: {str(e)}")
            logger.info("Attempting to load from local cache...")
            try:
                # 로컬 캐시에서 재시도
                model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=os.path.join("models", "audio"),
                    local_files_only=True
                ).to(self.device)
                model.eval()
                return model
            except Exception as e2:
                logger.error(f"Failed to load from cache: {str(e2)}")
                raise

    def _initialize_whisper(
        self,
        config: TranscriptionConfig
    ) -> Optional[WhisperModel]:
        """Whisper 모델을 초기화합니다."""
        if not WHISPER_AVAILABLE:
            return None
            
        try:
            os.makedirs(config.model_dir, exist_ok=True)
            
            # GPU 사용 가능 여부 확인
            device = "cuda" if torch.cuda.is_available() else "cpu"
            config.device = device  # 설정 업데이트
            
            # 기본 옵션 설정
            if config.options is None:
                config.options = {}
            
            # 모델 로드
            model = whisper.load_model(
                config.model_size,
                device=device,
                download_root=config.model_dir
            )
            
            if self.verbose:
                logger.info(f"Loaded Whisper {config.model_size} model on {device}")
                logger.info("Whisper options:")
                for k, v in config.options.items():
                    logger.info(f"  {k}: {v}")
                
            return model
            
        except Exception as e:
            logger.error(f"Failed to initialize Whisper: {str(e)}")
            return None

    def preprocess(self, audio_path: str) -> Optional[Tensor]:
        """단일 오디오 파일을 처리합니다."""
        try:
            # 오디오 로드 및 리샘플링
            waveform, sample_rate = torchaudio.load(audio_path)
            if sample_rate != self.config.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.config.sample_rate
                )
                waveform = resampler(waveform)
            
            # 정규화
            if self.config.normalize:
                waveform = torch.nn.functional.normalize(waveform, dim=1)
            
            # 모델 입력 형식으로 변환
            with torch.no_grad(), torch.cuda.amp.autocast():
                inputs = waveform.to(self.device)
                if self.verbose:
                    logger.debug(f"Input shape: {inputs.shape}")
                
                outputs = self.model(inputs)
                
                # Wav2Vec2 모델 출력 처리
                if hasattr(outputs, 'last_hidden_state'):
                    # 시퀀스 차원에 대해 평균을 계산하여 고정된 크기의 임베딩 생성
                    embeddings = outputs.last_hidden_state
                    if self.verbose:
                        logger.debug(f"Last hidden state shape: {embeddings.shape}")
                    
                    # 시퀀스 차원에 대해 평균 계산
                    embeddings = embeddings.mean(dim=1)  # [batch, sequence, features] -> [batch, features]
                else:
                    embeddings = outputs
                    if self.verbose:
                        logger.debug(f"Output shape before processing: {embeddings.shape}")
                
                # 차원 정규화
                embeddings = embeddings.squeeze()  # 모든 싱글톤 차원 제거
                if self.verbose:
                    logger.debug(f"Shape after squeeze: {embeddings.shape}")
                
                # 항상 2D 텐서로 변환
                if embeddings.dim() == 1:
                    embeddings = embeddings.unsqueeze(0)  # [features] -> [1, features]
                elif embeddings.dim() > 2:
                    # 마지막 차원을 제외한 모든 차원에 대해 평균
                    dims_to_reduce = tuple(range(0, embeddings.dim() - 1))
                    embeddings = embeddings.mean(dim=dims_to_reduce).unsqueeze(0)
                
                if self.verbose:
                    logger.debug(f"Final embedding shape: {embeddings.shape}")
                
                # 최종 shape 검증
                if embeddings.dim() != 2:
                    raise ValueError(f"Expected 2D tensor, got shape: {embeddings.shape}")
                
                return embeddings.cpu()
                
        except Exception as e:
            logger.error(f"Error preprocessing audio {audio_path}: {str(e)}")
            if self.verbose:
                logger.exception("Detailed error:")
            return None

    def transcribe(
        self,
        audio_path: str,
        cache_dir: Optional[str] = None
    ) -> Optional[str]:
        """오디오를 텍스트로 변환합니다."""
        if not WHISPER_AVAILABLE or self.whisper_model is None:
            logger.warning("Whisper is not available or not initialized")
            return None
        
        try:
            # 캐시 확인
            if cache_dir:
                cache_path = Path(cache_dir) / f"{Path(audio_path).stem}.txt"
                if cache_path.exists():
                    cached_text = cache_path.read_text().strip()
                    if cached_text:  # 빈 문자열이 아닌 경우에만 반환
                        return cached_text
            
            # 오디오 로드 및 전처리
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            
            # 오디오 길이 확인
            if len(audio) < 1600:  # 최소 0.1초
                logger.warning(f"Audio too short: {audio_path}")
                return None
            
            # mel 스펙트로그램 계산
            mel = whisper.log_mel_spectrogram(audio).to(self.whisper_model.device)
            
            # 트랜스크립션 수행
            options = {
                'language': 'en',
                'task': 'transcribe',
                'beam_size': 5,  # 증가
                'best_of': 5,    # 증가
                'fp16': False if self.whisper_model.device == "cpu" else True,
                'condition_on_previous_text': False,
                'temperature': [0.0, 0.2, 0.4, 0.6, 0.8]  # 다양한 temperature 시도
            }
            
            # 설정된 옵션으로 업데이트
            if self.transcription_config and self.transcription_config.options:
                options.update(self.transcription_config.options)
            
            result = self.whisper_model.transcribe(
                audio,
                **options
            )
            
            text = result["text"].strip()
            logger.info(f"Transcribed text for {audio_path}: {text}")
            
            # 결과 캐시 (빈 문자열이 아닌 경우에만)
            if cache_dir and text:
                os.makedirs(cache_dir, exist_ok=True)
                cache_path.write_text(text)
            
            return text
            
        except Exception as e:
            logger.error(f"Transcription error for {audio_path}: {str(e)}")
            return None

    def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        오디오 파일의 특성을 분석합니다.

        Args:
            audio_path: 오디오 파일 경로

        Returns:
            Dict[str, Any]: 오디오 특성 정보
        """
        try:
            waveform, sr = torchaudio.load(audio_path)
            duration = waveform.shape[1] / sr  # 초 단위 길이
            
            info = {
                'duration': duration,
                'sample_rate': sr,
                'channels': waveform.shape[0],
                'samples': waveform.shape[1],
                'min_value': float(waveform.min()),
                'max_value': float(waveform.max()),
                'mean_value': float(waveform.mean()),
                'std_value': float(waveform.std())
            }
            return info
            
        except Exception as e:
            logger.error(f"Error analyzing audio file {audio_path}: {str(e)}")
            return {}

    def analyze_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        데이터셋의 오디오 파일들을 분석합니다.

        Args:
            dataset_path: 데이터셋 경로

        Returns:
            Dict[str, Any]: 데이터셋 통계 정보
        """
        stats = {
            'total_files': 0,
            'valid_files': 0,
            'durations': [],
            'sample_rates': set(),
            'channels': set()
        }
        
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.wav'):
                    stats['total_files'] += 1
                    audio_path = os.path.join(root, file)
                    
                    info = self.analyze_audio(audio_path)
                    if info:
                        stats['valid_files'] += 1
                        stats['durations'].append(info['duration'])
                        stats['sample_rates'].add(info['sample_rate'])
                        stats['channels'].add(info['channels'])
        
        if stats['durations']:
            stats['avg_duration'] = np.mean(stats['durations'])
            stats['min_duration'] = np.min(stats['durations'])
            stats['max_duration'] = np.max(stats['durations'])
            stats['std_duration'] = np.std(stats['durations'])
        
        return stats

    def preprocess_batch(
        self,
        audio_paths: List[str],
        batch_size: Optional[int] = None
    ) -> Optional[Tensor]:
        """오디오 배치를 처리합니다."""
        if not audio_paths:
            return None
            
        batch_size = batch_size or self.config.batch_size
        all_embeddings = []
        
        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i + batch_size]
            batch_embeddings = []
            
            for path in batch_paths:
                emb = self.preprocess(path)
                if emb is not None:
                    batch_embeddings.append(emb)
                    
            if batch_embeddings:
                all_embeddings.extend(batch_embeddings)
                
        if not all_embeddings:
            return None
            
        return torch.stack(all_embeddings)

    def check_memory_usage(self) -> bool:
        """메모리 사용량을 확인합니다."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = memory_info.rss / psutil.virtual_memory().total
            
            if memory_percent > 0.8:  # 80% 이상 사용
                logger.warning(f"High memory usage: {memory_percent:.1%}")
                return False
            return True
        except ImportError:
            return True  # psutil 없으면 항상 True 반환


class TextPreprocessor:
    """텍스트 전처리 및 임베딩 생성 클래스."""

    def __init__(
        self,
        model_name: str,
        verbose: bool = False,
        max_length: Optional[int] = None  # max_length를 선택적 인자로 변경
    ):
        """텍스트 전처리기 초기화."""
        self.model_name = model_name
        self.verbose = verbose
        self.max_length = max_length or 512  # 기본값 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 토크나이저와 모델 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def preprocess(self, text: str) -> Optional[Tensor]:
        """단일 텍스트를 처리합니다."""
        if not text:
            return None
            
        try:
            # 토크나이징
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # 임베딩 생성
            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰
                
            return embeddings.cpu().squeeze(0)  # 배치 차원 제거
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            return None
    
    def preprocess_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> Optional[Tensor]:
        """텍스트 배치를 처리합니다."""
        if not texts:
            return None
            
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                # 텍스트 정규화
                batch = [text.lower().strip() for text in batch]
                
                # 토크나이징
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                # 임베딩 생성 (CLS 토큰 + 평균 풀링)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    outputs = self.model(**inputs)
                    # CLS 토큰과 평균 풀링 결합
                    cls_embeddings = outputs.last_hidden_state[:, 0, :]
                    mean_embeddings = outputs.last_hidden_state.mean(dim=1)
                    embeddings = (cls_embeddings + mean_embeddings) / 2
                    all_embeddings.append(embeddings.cpu())
                    
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                continue
                
        if not all_embeddings:
            return None
            
        return torch.cat(all_embeddings, dim=0)

def load_text_data(dataset_path: str, config: DictConfig) -> Dict[str, str]:
    """CSV 파일에서 텍스트 데이터를 로드합니다."""
    try:
        # ISEAR: 원본 CSV에서 텍스트 컬럼 직접 사용
        if config.dataset.name == "isear":
            isear_csv = Path(dataset_path) / "ISEAR.csv"
            if isear_csv.exists():
                isear_df = pd.read_csv(isear_csv, sep='|', encoding='utf-8')
                text_col = config.dataset.columns.text
                return dict(zip(isear_df.index.astype(str), isear_df[text_col]))
            else:
                logger.error(f"ISEAR CSV file not found: {isear_csv}")
        else:
            # RAVDESS: dataset.csv에서 텍스트 컬럼 사용
            csv_path = Path(dataset_path) / "dataset.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                return dict(zip(df['id'], df['text']))
            else:
                logger.error(f"Dataset CSV file not found: {csv_path}")
        
    except Exception as e:
        logger.error(f"Error loading text data: {e}")
    return {}

def update_text_data(dataset_path: str, text_dict: Dict[str, str], config: DictConfig) -> None:
    """CSV 파일의 텍스트 데이터를 업데이트합니다."""
    try:
        csv_path = Path(dataset_path) / "dataset.csv"
        df = pd.read_csv(csv_path)
        
        # 텍스트 업데이트
        for idx, text in text_dict.items():
            df.loc[df['id'] == idx, 'text'] = text
        
        df.to_csv(csv_path, index=False)
        logger.info(f"Updated text data in: {csv_path}")
        
        # RAVDESS: transcription을 별도 CSV에도 저장
        if config.dataset.name == "ravdess":
            transcriptions_df = pd.DataFrame({
                'id': text_dict.keys(),
                'text': text_dict.values()
            })
            trans_path = Path(dataset_path) / "transcriptions.csv"
            transcriptions_df.to_csv(trans_path, index=False)
            logger.info(f"Saved transcriptions to: {trans_path}")
            
    except Exception as e:
        logger.error(f"Error updating text data: {e}")

def load_dataset(config: DictConfig) -> Tuple[List[str], List[str]]:
    """데이터셋을 로드합니다."""
    try:
        if config.dataset.name == "ravdess":
            return load_ravdess_dataset(config)
        elif config.dataset.name == "isear":
            return load_isear_dataset(config)
        else:
            raise ValueError(f"Unknown dataset: {config.dataset.name}")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return [], []

def load_ravdess_dataset(config: DictConfig) -> Tuple[List[str], List[str]]:
    """RAVDESS 데이터셋을 로드합니다."""
    try:
        # 메타데이터 로드
        metadata_path = Path(config.dataset.data_path) / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        with open(metadata_path) as f:
            metadata = json.load(f)
            
        # 오디오 파일 경로와 텍스트 데이터 준비
        audio_paths = []
        transcriptions = []
        
        # 디버그 로깅 추가
        base_path = Path(config.dataset.data_path)
        logger.debug(f"Audio base path: {base_path} (exists: {base_path.exists()})")
        
        # 모든 WAV 파일 찾기 (Actor_XX 디렉토리 내부)
        wav_files = []
        for actor_dir in base_path.glob("Actor_*"):
            wav_files.extend(actor_dir.glob("*.wav"))
        logger.debug(f"Found {len(wav_files)} WAV files")
        
        # 메타데이터의 파일 ID와 실제 파일 매핑
        for file_id in sorted(metadata.keys()):
            audio_file = base_path / file_id
            logger.debug(f"Checking audio file: {audio_file} (exists: {audio_file.exists()})")
            
            if audio_file.exists():
                audio_paths.append(str(audio_file))
                
                # 전사 텍스트 파일 경로
                trans_file = base_path / "transcriptions" / f"{file_id}.txt"
                if trans_file.exists():
                    with open(trans_file, 'r') as f:
                        transcriptions.append(f.read().strip())
                else:
                    transcriptions.append("")
        
        logger.info(f"Loaded {len(audio_paths)} audio files from RAVDESS")
        if transcriptions:
            logger.info(f"Found {len([t for t in transcriptions if t])} transcriptions")
            
        return audio_paths, transcriptions
        
    except Exception as e:
        logger.error(f"Error loading RAVDESS dataset: {e}")
        logger.exception("Detailed error:")  # 상세 오류 출력
        return [], []

def load_isear_dataset(config: DictConfig) -> Tuple[List[str], List[str]]:
    """ISEAR 데이터셋을 로드합니다."""
    try:
        csv_path = Path(config.dataset.data_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV file not found: {csv_path}")
            
        df = pd.read_csv(
            csv_path,
            sep='|',
            encoding='utf-8',
            on_bad_lines='skip'
        )
        
        texts = df[config.dataset.columns.text].fillna("").tolist()
        logger.info(f"Loaded {len(texts)} text samples from ISEAR")
        
        return [], texts  # ISEAR는 오디오가 없으므로 빈 리스트 반환
        
    except Exception as e:
        logger.error(f"Error loading ISEAR dataset: {e}")
        return [], []

def process_text_embeddings(
    cfg: DictConfig,
    text_preprocessor: TextPreprocessor,
    texts: List[str],
    cache: EmbeddingCache
) -> np.ndarray:
    """텍스트 임베딩을 생성하거나 캐시에서 로드합니다."""
    try:
        embeddings = []
        
        for idx, text in enumerate(tqdm(texts, desc="Processing text")):
            if not text:  # 빈 텍스트 건너뛰기
                continue
                
            # 캐시 확인
            cache_key = f"text_{idx}"
            cached_emb = cache.load_embedding(cache_key, modality="text")
            
            if cached_emb is not None:
                embeddings.append(cached_emb)
            else:
                # 새로운 임베딩 생성
                emb = text_preprocessor.preprocess(text)
                if emb is not None:
                    emb_np = emb.numpy()
                    embeddings.append(emb_np)
                    # 임베딩 캐시 저장
                    cache.save_embedding(
                        cache_key,
                        emb_np,
                        modality="text",
                        model_name=cfg.model.text.name
                    )
        
        if not embeddings:
            logger.error("No text embeddings were generated")
            return np.array([])
            
        return np.stack(embeddings)
        
    except Exception as e:
        logger.error(f"Error processing text embeddings: {e}")
        return np.array([])

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
            for audio_path in tqdm(audio_paths, desc="Processing audio"):
                try:
                    # 캐시 확인
                    # 수정: data_path가 아닌 download_path 사용
                    file_id = str(Path(audio_path).relative_to(
                        Path(cfg.dataset.download_path) / "audio_speech_actors_01-24"
                    ))
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
                except Exception as e:
                    logger.error(f"Error processing audio file {audio_path}: {e}")
                    continue
            
            audio_embeddings = np.stack(audio_embeddings) if audio_embeddings else np.array([])
        
        # 나머지 코드는 동일...
