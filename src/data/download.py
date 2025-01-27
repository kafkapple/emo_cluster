"""데이터셋 다운로드 및 전처리를 위한 모듈."""

import os
import json
import requests
import zipfile
from pathlib import Path
import logging
from typing import Dict, Optional
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig
from src.data.metadata import DatasetMetadataManager, RAVDESSMetadataManager, ISEARMetadataManager
from langdetect import detect

logger = logging.getLogger(__name__)

class DatasetDownloader:
    """데이터셋 다운로드 및 전처리를 담당하는 클래스."""
    
    def __init__(self, cfg: DictConfig):
        """초기화."""
        self.config = cfg
    
    def download_ravdess(self) -> None:
        """RAVDESS 데이터셋을 다운로드합니다."""
        download_path = Path(self.config.dataset.download_path)
        logger.info(f"Checking RAVDESS dataset in: {download_path}")
        
        # 데이터셋 구조 검증
        audio_path = download_path / "audio_speech_actors_01-24"
        if audio_path.exists():
            wav_files = list(audio_path.rglob("*.wav"))
            logger.info(f"Found {len(wav_files)} WAV files in {audio_path}")
            
            # 메타데이터 매니저 초기화 및 파일 생성
            metadata_manager = RAVDESSMetadataManager(
                dataset_path=str(download_path),
                config=self.config
            )
            
            # 메타데이터 확인 및 생성
            metadata_path = download_path / "metadata.json"
            labels_path = download_path / "labels.json"
            
            if not metadata_path.exists() or not labels_path.exists():
                logger.info("Creating metadata and labels...")
                try:
                    metadata_manager.create_metadata()
                    logger.info("Metadata and labels created successfully")
                    return
                except Exception as e:
                    logger.error(f"Error creating metadata: {e}")
            else:
                logger.info("Using existing metadata and labels")
                return
        
        # 데이터셋이 없는 경우 다운로드
        logger.info("Downloading RAVDESS dataset...")
        
        # 다운로드 및 압축 해제
        if not self.download_and_extract(
            self.config.dataset.download_url,
            str(download_path)
        ):
            return
            
        # 메타데이터 생성
        metadata_manager = RAVDESSMetadataManager(
            dataset_path=str(download_path),
            config=self.config
        )
        logger.info("Creating dataset metadata...")
        metadata_manager.create_metadata()
    
    def download_isear(self) -> None:
        """ISEAR 데이터셋을 다운로드합니다."""
        download_path = Path(self.config.dataset.download_path)
        download_path.mkdir(parents=True, exist_ok=True)
        
        # CSV 파일 다운로드
        csv_path = download_path / "ISEAR.csv"
        if not csv_path.exists():
            logger.info("Downloading ISEAR dataset...")
            try:
                response = requests.get(self.config.dataset.download_url)
                response.raise_for_status()
                
                # 다운로드한 내용을 ISEAR.csv로 저장
                with open(csv_path, 'wb') as f:
                    f.write(response.content)
                logger.info("ISEAR dataset downloaded successfully")
                
                # 데이터 검증 및 전처리
                df = pd.read_csv(
                    csv_path,
                    sep='|',
                    encoding='utf-8',
                    on_bad_lines='skip'
                )
                
                # 필수 컬럼 확인
                required_columns = ['SIT', 'EMOT', 'INTS', 'SEX', 'AGE', 'COUN']
                if not all(col in df.columns for col in required_columns):
                    raise ValueError("Downloaded file is missing required columns")
                
                # 언어 감지 및 영어 텍스트 필터링
                def detect_language(text):
                    try:
                        return detect(str(text))
                    except:
                        return 'unknown'
                
                logger.info("Detecting languages and filtering English text...")
                df['language'] = df['SIT'].apply(detect_language)
                df = df[df['language'] == 'en']
                
                # 필터링된 데이터 저장
                df.to_csv(csv_path, sep='|', index=False)
                logger.info(f"Filtered dataset saved with {len(df)} English entries")
                
            except Exception as e:
                logger.error(f"Failed to download ISEAR dataset: {e}")
                if csv_path.exists():
                    csv_path.unlink()
                return
        else:
            logger.info("ISEAR dataset already exists")
        
        # 메타데이터 및 레이블 생성
        metadata_manager = ISEARMetadataManager(download_path, self.config)
        if not (download_path / "metadata.json").exists():
            logger.info("Creating dataset metadata...")
            metadata_manager.create_metadata()
            
        if not (download_path / "labels.json").exists():
            logger.info("Creating labels file...")
            self._create_isear_labels(str(download_path))

    @staticmethod
    def download_file(url: str, save_path: str) -> bool:
        """파일을 다운로드합니다."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            
            with open(save_path, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    pbar.update(size)
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            return False

    @staticmethod
    def download_and_extract(url: str, download_path: str) -> bool:
        """데이터셋을 다운로드하고 필요한 경우 압축을 해제합니다."""
        try:
            os.makedirs(download_path, exist_ok=True)
            
            # URL에서 파일명 추출
            filename = url.split('/')[-1]
            file_path = os.path.join(download_path, filename)
            
            logger.info(f"Downloading dataset from {url}")
            if not DatasetDownloader.download_file(url, file_path):
                return False
                
            # 파일 확장자 확인
            if filename.endswith('.zip'):
                logger.info("Extracting dataset...")
                try:
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(download_path)
                    os.remove(file_path)  # 압축 파일 삭제
                except Exception as e:
                    logger.error(f"Extraction failed: {str(e)}")
                    return False
            elif filename.endswith('.csv'):
                logger.info("Downloaded CSV file successfully")
                # CSV 파일은 압축 해제가 필요 없음
                pass
            else:
                logger.warning(f"Unknown file format: {filename}")
                
            return True
            
        except Exception as e:
            logger.error(f"Error during download and extraction: {str(e)}")
            return False

    @staticmethod
    def _create_ravdess_labels(
        download_path: str,
        metadata_manager: DatasetMetadataManager
    ) -> None:
        """RAVDESS 데이터셋의 레이블 파일을 생성합니다."""
        try:
            metadata = metadata_manager.load_metadata()
            labels = {
                file_id: meta['emotion']
                for file_id, meta in metadata.items()
            }
            
            labels_path = os.path.join(download_path, 'labels.json')
            with open(labels_path, 'w') as f:
                json.dump(labels, f, indent=4)
                
            logger.info(f"Labels saved to: {labels_path}")
            
        except Exception as e:
            logger.error(f"Error creating labels: {str(e)}")

    @staticmethod
    def _create_isear_labels(download_path: str) -> None:
        """ISEAR 데이터셋의 레이블 파일을 생성합니다."""
        try:
            csv_path = os.path.join(download_path, 'ISEAR.csv')
            df = pd.read_csv(
                csv_path,
                sep='|',
                encoding='utf-8',
                on_bad_lines='skip'
            )
            
            # 감정 레이블 매핑
            emotion_map = {
                '1': 'joy',
                '2': 'fear',
                '3': 'anger',
                '4': 'sadness',
                '5': 'disgust',
                '6': 'shame',
                '7': 'guilt'
            }
            
            # 레이블 생성
            labels = {}
            for idx, row in df.iterrows():
                try:
                    emotion_code = str(row['EMOT']).strip()
                    if emotion_code in emotion_map:
                        labels[str(idx)] = emotion_map[emotion_code]
                except Exception as e:
                    logger.warning(f"Error processing row {idx}: {e}")
            
            # 레이블 저장
            labels_path = os.path.join(download_path, 'labels.json')
            with open(labels_path, 'w') as f:
                json.dump(labels, f, indent=4)
            
            logger.info(f"Labels saved to: {labels_path}")
            logger.info(f"Created labels for {len(labels)} samples")
            
        except Exception as e:
            logger.error(f"Error creating ISEAR labels: {e}")

    def check_dataset_exists(self) -> bool:
        """데이터셋이 이미 존재하는지 확인합니다."""
        try:
            download_path = Path(self.config.dataset.download_path)
            logger.info(f"Checking dataset in: {download_path}")
            
            # 공통 필수 파일 체크
            common_files = [
                download_path / "metadata.json",
                download_path / "labels.json"
            ]
            
            for f in common_files:
                logger.debug(f"Checking file: {f} (exists: {f.exists()})")
            
            if not all(f.exists() for f in common_files):
                logger.info("Missing metadata or label files")
                return False
            
            if self.config.dataset.name == "ravdess":
                # RAVDESS 전용 체크
                audio_path = download_path / "audio_speech_actors_01-24"
                if not audio_path.exists():
                    logger.info(f"Missing RAVDESS audio directory: {audio_path}")
                    return False
                    
                wav_files = list(audio_path.rglob("*.wav"))
                if not wav_files:
                    logger.info("No WAV files found in RAVDESS directory")
                    return False
                    
                logger.info(f"Found existing RAVDESS dataset with {len(wav_files)} WAV files")
                return True
            
            elif self.config.dataset.name == "isear":
                csv_exists = (download_path / "ISEAR.csv").exists()
                logger.info(f"ISEAR CSV file exists: {csv_exists}")
                return csv_exists
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking dataset: {e}")
            return False