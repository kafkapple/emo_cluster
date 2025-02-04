"""데이터셋 메타데이터 관리를 위한 모듈."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import json
import os
import logging
from pathlib import Path
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

@dataclass
class RAVDESSMetadata:
    """RAVDESS 파일 메타데이터."""
    
    file_id: str
    modality: str
    vocal_channel: str
    emotion: str
    intensity: str
    statement: str
    repetition: int
    actor_id: int
    gender: str
    
    # 메타데이터 매핑
    MODALITY_MAP = {
        "01": "full-AV",
        "02": "video-only",
        "03": "audio-only"
    }
    
    VOCAL_CHANNEL_MAP = {
        "01": "speech",
        "02": "song"
    }
    
    EMOTION_MAP = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised"
    }
    
    INTENSITY_MAP = {
        "01": "normal",
        "02": "strong"
    }
    
    STATEMENT_MAP = {
        "01": "Kids are talking by the door",
        "02": "Dogs are sitting by the door"
    }
    
    @classmethod
    def from_filename(cls, filename: str) -> 'RAVDESSMetadata':
        """파일명에서 메타데이터 생성."""
        parts = Path(filename).stem.split("-")
        if len(parts) != 7:
            raise ValueError(f"Invalid filename format: {filename}")
            
        actor_id = int(parts[6])
        return cls(
            file_id=filename,
            modality=cls.MODALITY_MAP.get(parts[0], "unknown"),
            vocal_channel=cls.VOCAL_CHANNEL_MAP.get(parts[1], "unknown"),
            emotion=cls.EMOTION_MAP.get(parts[2], "unknown"),
            intensity=cls.INTENSITY_MAP.get(parts[3], "unknown"),
            statement=cls.STATEMENT_MAP.get(parts[4], "unknown"),
            repetition=int(parts[5]),
            actor_id=actor_id,
            gender="female" if actor_id % 2 == 0 else "male"
        )
    
    def to_dict(self) -> Dict:
        """메타데이터를 딕셔너리로 변환."""
        return {
            "file_id": self.file_id,
            "modality": self.modality,
            "vocal_channel": self.vocal_channel,
            "emotion": self.emotion,
            "intensity": self.intensity,
            "statement": self.statement,
            "repetition": self.repetition,
            "actor_id": self.actor_id,
            "gender": self.gender
        }


class DatasetMetadataManager:
    """데이터셋 메타데이터 관리자."""
    
    def __init__(self, dataset_path: str):
        """메타데이터 관리자 초기화."""
        self.dataset_path = Path(dataset_path)
        self.metadata_path = self.dataset_path / "metadata.json"
        
    def create_metadata(self) -> None:
        """데이터셋의 메타데이터를 생성하고 저장."""
        raise NotImplementedError("Subclasses must implement create_metadata")
        
    def load_metadata(self) -> Dict:
        """메타데이터를 로드합니다."""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
            
        with open(self.metadata_path, 'r') as f:
            return json.load(f)

@dataclass
class ISEARMetadata:
    """ISEAR 파일 메타데이터."""
    
    text_id: str
    text: str
    emotion: str
    intensity: int
    gender: str
    age: int
    country: str
    
    # 메타데이터 매핑
    EMOTION_MAP = {
        '1': 'joy',
        '2': 'fear',
        '3': 'anger',
        '4': 'sadness',
        '5': 'disgust',
        '6': 'shame',
        '7': 'guilt'
    }
    
    @classmethod
    def from_row(cls, row: pd.Series) -> 'ISEARMetadata':
        """데이터프레임 행에서 메타데이터 생성."""
        emotion_code = str(row['EMOT']).strip()
        if emotion_code not in cls.EMOTION_MAP:
            raise ValueError(f"Invalid emotion code: {emotion_code}")
            
        return cls(
            text_id=str(row.name),
            text=str(row['SIT']),
            emotion=cls.EMOTION_MAP[emotion_code],
            intensity=int(row['INTS']),
            gender='male' if row['SEX'] == '1' else 'female',
            age=int(row['AGE']),
            country=str(row['COUN'])
        )
    
    def to_dict(self) -> Dict:
        """메타데이터를 딕셔너리로 변환."""
        return {
            "text_id": self.text_id,
            "text": self.text,
            "emotion": self.emotion,
            "intensity": self.intensity,
            "gender": self.gender,
            "age": self.age,
            "country": self.country
        }

class ISEARMetadataManager(DatasetMetadataManager):
    """ISEAR 데이터셋 메타데이터 관리자."""
    
    def __init__(self, dataset_path: str, config: DictConfig):
        """메타데이터 관리자 초기화."""
        super().__init__(dataset_path)
        self.config = config
    
    def create_metadata(self) -> None:
        """데이터셋의 메타데이터를 생성하고 저장."""
        try:
            # 원본 ISEAR CSV 읽기
            df = pd.read_csv(
                self.dataset_path / "ISEAR.csv",
                sep='|',
                encoding='utf-8',
                on_bad_lines='skip'
            )
            
            # 언어 감지 및 필터링
            if self.config.dataset.language.enabled:
                from langdetect import detect
                
                def detect_language(text):
                    try:
                        return detect(str(text))
                    except:
                        return 'unknown'
                
                logger.info("Detecting languages and filtering...")
                df['language'] = df[self.config.dataset.columns.text].apply(detect_language)
                df = df[df['language'] == self.config.dataset.language.target]
                logger.info(f"Filtered to {len(df)} {self.config.dataset.language.target} entries")
            
            # 표준 형식으로 변환
            dataset_df = pd.DataFrame({
                'id': df.index.astype(str),
                'text': df[self.config.dataset.columns.text].astype(str),  # 문자열로 변환
                'emotion': df[self.config.dataset.columns.emotion].astype(str).map(ISEARMetadata.EMOTION_MAP),
                'intensity': df[self.config.dataset.columns.intensity].astype(int),
                'gender': df[self.config.dataset.columns.gender].astype(str).map({'1': 'male', '2': 'female'}),
                'age': df[self.config.dataset.columns.age].astype(int),
                'source': 'text'
            })
            
            # 표준화된 CSV 저장
            csv_path = self.dataset_path / "dataset.csv"
            dataset_df.to_csv(csv_path, index=False)
            
            # 메타데이터 생성
            metadata = {
                str(idx): {
                    'text_id': str(idx),
                    'text': row['text'],
                    'emotion': row['emotion'],
                    'intensity': row['intensity'],
                    'gender': row['gender'],
                    'age': row['age'],
                    'source': row['source']
                }
                for idx, row in dataset_df.iterrows()
            }
            
            # 메타데이터 JSON 저장
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
                
            logger.info(f"Dataset CSV saved to: {csv_path}")
            logger.info(f"Metadata saved to: {self.metadata_path}")
            self._log_dataset_statistics(metadata)
            
        except Exception as e:
            logger.error(f"Error creating metadata: {e}")

    def _log_dataset_statistics(self, metadata: Dict) -> None:
        """데이터셋 통계 정보를 로깅."""
        stats = {
            "total_samples": len(metadata),
            "emotions": {},
            "gender_distribution": {"male": 0, "female": 0},
            "age_distribution": {
                "under_20": 0,
                "20_30": 0,
                "30_40": 0,
                "over_40": 0
            },
            "intensity_distribution": {}
        }
        
        for meta in metadata.values():
            # 감정 분포
            stats["emotions"][meta["emotion"]] = stats["emotions"].get(meta["emotion"], 0) + 1
            
            # 성별 분포
            stats["gender_distribution"][meta["gender"].lower()] += 1
            
            # 나이 분포
            age = meta["age"]
            if age < 20:
                stats["age_distribution"]["under_20"] += 1
            elif age < 30:
                stats["age_distribution"]["20_30"] += 1
            elif age < 40:
                stats["age_distribution"]["30_40"] += 1
            else:
                stats["age_distribution"]["over_40"] += 1
                
            # 강도 분포
            intensity = meta["intensity"]
            stats["intensity_distribution"][intensity] = \
                stats["intensity_distribution"].get(intensity, 0) + 1
        
        # 통계 출력
        logger.info("\nDataset Statistics:")
        logger.info(f"Total samples: {stats['total_samples']}")
        
        logger.info("\nEmotion distribution:")
        for emotion, count in stats["emotions"].items():
            logger.info(f"  {emotion}: {count}")
            
        logger.info("\nGender distribution:")
        for gender, count in stats["gender_distribution"].items():
            logger.info(f"  {gender}: {count}")
            
        logger.info("\nAge distribution:")
        for age_group, count in stats["age_distribution"].items():
            logger.info(f"  {age_group}: {count}")
            
        logger.info("\nIntensity distribution:")
        for intensity, count in sorted(stats["intensity_distribution"].items()):
            logger.info(f"  Level {intensity}: {count}")

class RAVDESSMetadataManager(DatasetMetadataManager):
    """RAVDESS 메타데이터 관리자."""
    
    def __init__(self, dataset_path: str, config: DictConfig):
        """메타데이터 관리자 초기화."""
        super().__init__(dataset_path)
        self.config = config
    
    # RAVDESS 파일명 형식: 03-01-06-02-02-02-07.wav
    # 03: 음성 전용
    # 01: 발화 채널
    # 06: 감정 (01=중립, 02=차분, 03=행복, 04=슬픔, 05=화남, 06=공포, 07=혐오, 08=놀람)
    # 02: 감정 강도
    # 02: 발화문
    # 02: 반복
    # 07: 배우 ID
    
    EMOTION_MAP = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised"
    }
    
    def create_metadata(self) -> None:
        """메타데이터를 생성합니다."""
        try:
            base_path = Path(self.config.dataset.data_path)
            metadata = {}
            labels = {}
            file_index = 0  # 파일 인덱스 추가
            
            # Actor_XX 디렉토리 검색
            for actor_dir in sorted(base_path.glob("Actor_*")):  # 정렬된 순서로 처리
                try:
                    # WAV 파일 처리
                    for wav_file in sorted(actor_dir.glob("*.wav")):  # 정렬된 순서로 처리
                        try:
                            # 파일명에서 정보 추출
                            filename = wav_file.name
                            parts = filename.split("-")
                            
                            # 상대 경로로 저장 (Actor_XX/filename.wav 형식)
                            rel_path = str(wav_file.relative_to(base_path))
                            file_id = rel_path.replace("\\", "/")  # Windows 경로 처리
                            
                            # 감정 코드 추출 및 매핑
                            emotion_code = parts[2]
                            emotion = self.EMOTION_MAP.get(emotion_code, "unknown")
                            
                            # 메타데이터 생성
                            metadata[file_id] = {
                                "emotion": emotion,
                                "intensity": "normal" if parts[3] == "01" else "strong",
                                "gender": "female" if int(parts[6].split(".")[0]) % 2 == 0 else "male",
                                "actor_id": int(parts[6].split(".")[0])
                            }
                            
                            # 레이블 저장 (인덱스 기반)
                            labels[str(file_index)] = emotion
                            file_index += 1
                            
                        except Exception as e:
                            logger.warning(f"Error processing file {filename}: {e}")
                            continue
                            
                except Exception as e:
                    logger.warning(f"Error processing directory {actor_dir}: {e}")
                    continue
            
            # 메타데이터 저장
            metadata_path = os.path.join(self.dataset_path, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            
            # 레이블 저장 (새로운 형식)
            labels_path = os.path.join(self.dataset_path, "labels.json")
            with open(labels_path, "w") as f:
                json.dump({"labels": labels}, f, indent=4)  # labels 필드 추가
            
            logger.info(f"Created metadata for {len(metadata)} files")
            logger.info(f"Metadata saved to: {metadata_path}")
            logger.info(f"Labels saved to: {labels_path}")
            
        except Exception as e:
            logger.error(f"Error creating metadata: {e}")
            raise 