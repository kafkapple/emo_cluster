import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import json
from collections import Counter
import librosa
import textstat
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DatasetAnalyzer:
    """데이터셋 분석 및 전처리를 위한 클래스"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.stats: Dict = {}
        self.data_path = Path(config.dataset.data_path)
        self.stats_path = Path(config.general.output_dir) / "dataset_stats.json"
        
    def analyze_isear(self, df: pd.DataFrame) -> Dict:
        """ISEAR 텍스트 데이터셋 분석"""
        stats = {
            "total_samples": len(df),
            "missing_values": df.isnull().sum().to_dict(),
            "emotion_distribution": df['emotion_label'].value_counts().to_dict(),
            "text_stats": {
                "avg_length": df['text'].str.len().mean(),
                "min_length": df['text'].str.len().min(),
                "max_length": df['text'].str.len().max(),
                "empty_texts": (df['text'].str.strip() == "").sum(),
                "readability_scores": [],
                "language_distribution": df['language'].value_counts().to_dict(),
                "word_counts": {
                    "mean": df['text'].str.split().str.len().mean(),
                    "std": df['text'].str.split().str.len().std(),
                    "min": df['text'].str.split().str.len().min(),
                    "max": df['text'].str.split().str.len().max()
                },
                "sentence_counts": {
                    "mean": df['text'].str.count('[.!?]+').mean(),
                    "std": df['text'].str.count('[.!?]+').std()
                },
                "punctuation_stats": {
                    "questions": df['text'].str.count(r'\?').sum(),
                    "exclamations": df['text'].str.count(r'!').sum()
                }
            },
            "potential_issues": {
                "very_short_texts": [],
                "very_long_texts": [],
                "invalid_emotions": [],
                "non_english_texts": []
            }
        }
        
        # 텍스트 품질 분석
        for idx, row in tqdm(df.iterrows(), desc="Analyzing text quality"):
            text = row['text']
            
            # 텍스트 길이 체크
            if len(text.strip()) < 10:
                stats["potential_issues"]["very_short_texts"].append(idx)
            elif len(text.strip()) > 1000:
                stats["potential_issues"]["very_long_texts"].append(idx)
                
            # 가독성 점수 계산
            try:
                readability = textstat.flesch_reading_ease(text)
                stats["text_stats"]["readability_scores"].append(readability)
            except:
                logger.warning(f"Could not calculate readability for text at index {idx}")
                
            # 감정 레이블 유효성 검사
            if row['emotion_label'] not in self.config.dataset.emotions.classes:
                stats["potential_issues"]["invalid_emotions"].append(idx)
                
            # 영어 텍스트 확인 (기본 휴리스틱)
            if not all(ord(c) < 128 for c in text):
                stats["potential_issues"]["non_english_texts"].append(idx)
        
        # 통계 요약
        stats["text_stats"]["avg_readability"] = np.mean(stats["text_stats"]["readability_scores"])
        stats["text_stats"]["readability_std"] = np.std(stats["text_stats"]["readability_scores"])
        
        # 감정별 텍스트 길이 분포
        stats["emotion_text_lengths"] = {
            emotion: {
                "mean": group['text'].str.len().mean(),
                "std": group['text'].str.len().std()
            }
            for emotion, group in df.groupby('emotion_label')
        }
        
        # 로깅에 추가 정보 포함
        logger.info("\n=== ISEAR Dataset Analysis ===")
        logger.info(f"Total samples: {stats['total_samples']}")
        logger.info("\nEmotion distribution:")
        for emotion, count in stats["emotion_distribution"].items():
            logger.info(f"  {emotion}: {count}")
        logger.info("\nPotential issues:")
        logger.info(f"  Very short texts: {len(stats['potential_issues']['very_short_texts'])}")
        logger.info(f"  Very long texts: {len(stats['potential_issues']['very_long_texts'])}")
        logger.info(f"  Invalid emotions: {len(stats['potential_issues']['invalid_emotions'])}")
        logger.info(f"  Non-English texts: {len(stats['potential_issues']['non_english_texts'])}")
        logger.info("\nText Statistics:")
        logger.info(f"  Average words per text: {stats['text_stats']['word_counts']['mean']:.1f}")
        logger.info(f"  Average sentences per text: {stats['text_stats']['sentence_counts']['mean']:.1f}")
        logger.info("\nEmotion-specific text lengths:")
        for emotion, lengths in stats["emotion_text_lengths"].items():
            logger.info(f"  {emotion}: {lengths['mean']:.1f} chars (±{lengths['std']:.1f})")
        
        self.stats["isear"] = stats
        return stats
    
    def analyze_ravdess(self, audio_paths: List[str]) -> Dict:
        """RAVDESS 오디오 데이터셋 분석"""
        stats = {
            "total_files": len(audio_paths),
            "audio_stats": {
                "durations": [],
                "sample_rates": set(),
                "channels": set(),
                "file_sizes": [],
                "mean_amplitudes": [],
                "spectral_features": {
                    "centroids": [],
                    "rolloffs": [],
                    "zero_crossing_rates": []
                },
                "silence_stats": {
                    "total_silence": [],
                    "silence_ratio": []
                }
            },
            "potential_issues": {
                "corrupted_files": [],
                "very_short_audio": [],
                "very_long_audio": [],
                "low_volume": [],
                "clipping": []
            }
        }
        
        for audio_path in tqdm(audio_paths, desc="Analyzing audio files"):
            try:
                # 오디오 로드
                y, sr = librosa.load(audio_path, sr=None)
                
                # 기본 통계
                duration = librosa.get_duration(y=y, sr=sr)
                stats["audio_stats"]["durations"].append(duration)
                stats["audio_stats"]["sample_rates"].add(sr)
                stats["audio_stats"]["channels"].add(librosa.get_samplerate(audio_path))
                stats["audio_stats"]["file_sizes"].append(Path(audio_path).stat().st_size)
                
                # 볼륨 레벨
                mean_amplitude = np.abs(y).mean()
                stats["audio_stats"]["mean_amplitudes"].append(mean_amplitude)
                
                # 잠재적 문제 체크
                if duration < 0.5:  # 0.5초 미만
                    stats["potential_issues"]["very_short_audio"].append(audio_path)
                elif duration > 20:  # 20초 초과
                    stats["potential_issues"]["very_long_audio"].append(audio_path)
                    
                if mean_amplitude < 0.01:  # 낮은 볼륨
                    stats["potential_issues"]["low_volume"].append(audio_path)
                    
                if np.any(np.abs(y) > 0.99):  # 클리핑
                    stats["potential_issues"]["clipping"].append(audio_path)
                    
                # 스펙트럴 특성 분석
                centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
                zcr = librosa.feature.zero_crossing_rate(y).mean()
                
                stats["audio_stats"]["spectral_features"]["centroids"].append(float(centroid))
                stats["audio_stats"]["spectral_features"]["rolloffs"].append(float(rolloff))
                stats["audio_stats"]["spectral_features"]["zero_crossing_rates"].append(float(zcr))
                
                # 무음 구간 분석
                intervals = librosa.effects.split(y, top_db=20)
                total_silence = (len(y) - sum(i[1] - i[0] for i in intervals)) / sr
                silence_ratio = total_silence / (len(y) / sr)
                
                stats["audio_stats"]["silence_stats"]["total_silence"].append(total_silence)
                stats["audio_stats"]["silence_stats"]["silence_ratio"].append(silence_ratio)
                
            except Exception as e:
                logger.error(f"Error processing {audio_path}: {e}")
                stats["potential_issues"]["corrupted_files"].append(audio_path)
        
        # 통계 계산
        stats["audio_stats"]["avg_duration"] = np.mean(stats["audio_stats"]["durations"])
        stats["audio_stats"]["std_duration"] = np.std(stats["audio_stats"]["durations"])
        stats["audio_stats"]["avg_file_size"] = np.mean(stats["audio_stats"]["file_sizes"])
        stats["audio_stats"]["avg_amplitude"] = np.mean(stats["audio_stats"]["mean_amplitudes"])
        
        # 통계 요약 업데이트
        for feature_type in ["centroids", "rolloffs", "zero_crossing_rates"]:
            values = stats["audio_stats"]["spectral_features"][feature_type]
            if values:
                stats["audio_stats"]["spectral_features"].update({
                    f"{feature_type}_mean": float(np.mean(values)),
                    f"{feature_type}_std": float(np.std(values))
                })
        
        # 무음 통계 요약
        stats["audio_stats"]["silence_stats"].update({
            "avg_silence_ratio": float(np.mean(stats["audio_stats"]["silence_stats"]["silence_ratio"])),
            "std_silence_ratio": float(np.std(stats["audio_stats"]["silence_stats"]["silence_ratio"]))
        })
        
        # 추가 로깅
        logger.info("\n=== RAVDESS Dataset Analysis ===")
        logger.info(f"Total files: {stats['total_files']}")
        logger.info(f"Average duration: {stats['audio_stats']['avg_duration']:.2f}s")
        logger.info(f"Sample rates: {stats['audio_stats']['sample_rates']}")
        logger.info("\nPotential issues:")
        logger.info(f"  Corrupted files: {len(stats['potential_issues']['corrupted_files'])}")
        logger.info(f"  Very short audio: {len(stats['potential_issues']['very_short_audio'])}")
        logger.info(f"  Very long audio: {len(stats['potential_issues']['very_long_audio'])}")
        logger.info(f"  Low volume: {len(stats['potential_issues']['low_volume'])}")
        logger.info(f"  Clipping: {len(stats['potential_issues']['clipping'])}")
        logger.info("\nAudio Feature Statistics:")
        logger.info(f"  Average silence ratio: {stats['audio_stats']['silence_stats']['avg_silence_ratio']:.2%}")
        logger.info(f"  Average spectral centroid: {stats['audio_stats']['spectral_features']['centroids_mean']:.2f} Hz")
        
        self.stats["ravdess"] = stats
        return stats
    
    def save_stats(self) -> None:
        """분석 통계 저장"""
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"Dataset statistics saved to {self.stats_path}")
    
    def get_preprocessing_recommendations(self) -> Dict:
        """데이터 품질 문제에 대한 전처리 권장사항 반환"""
        recommendations = {
            "isear": {
                "text_cleaning": [],
                "label_handling": [],
                "language_filtering": [],
                "length_normalization": []
            },
            "ravdess": {
                "audio_preprocessing": [],
                "file_handling": [],
                "feature_engineering": []
            }
        }
        
        if "isear" in self.stats:
            isear_stats = self.stats["isear"]
            
            # 텍스트 관련 권장사항
            if len(isear_stats["potential_issues"]["very_short_texts"]) > 0:
                recommendations["isear"]["text_cleaning"].append(
                    "Remove or review very short texts (length < 10 characters)"
                )
            
            if len(isear_stats["potential_issues"]["non_english_texts"]) > 0:
                recommendations["isear"]["language_filtering"].append(
                    "Apply language detection and filter non-English texts"
                )
                
            # 레이블 관련 권장사항
            if len(isear_stats["potential_issues"]["invalid_emotions"]) > 0:
                recommendations["isear"]["label_handling"].append(
                    "Handle invalid emotion labels through correction or removal"
                )
            
            # 텍스트 길이 정규화 권장사항
            word_counts = isear_stats["text_stats"]["word_counts"]
            if word_counts["std"] / word_counts["mean"] > 0.5:  # 높은 변동성
                recommendations["isear"]["length_normalization"].append(
                    "Consider text length normalization due to high variance in word counts"
                )
            
            # 문장 구조 관련 권장사항
            if isear_stats["text_stats"]["sentence_counts"]["mean"] > 3:
                recommendations["isear"]["text_cleaning"].append(
                    "Consider splitting long texts into separate samples"
                )
        
        if "ravdess" in self.stats:
            ravdess_stats = self.stats["ravdess"]
            
            # 오디오 관련 권장사항
            if len(ravdess_stats["potential_issues"]["low_volume"]) > 0:
                recommendations["ravdess"]["audio_preprocessing"].append(
                    "Apply volume normalization to low volume files"
                )
                
            if len(ravdess_stats["potential_issues"]["clipping"]) > 0:
                recommendations["ravdess"]["audio_preprocessing"].append(
                    "Apply peak normalization to prevent clipping"
                )
                
            if len(ravdess_stats["potential_issues"]["corrupted_files"]) > 0:
                recommendations["ravdess"]["file_handling"].append(
                    "Remove or repair corrupted audio files"
                )
            
            # 특성 엔지니어링 권장사항
            if ravdess_stats["audio_stats"]["silence_stats"]["avg_silence_ratio"] > 0.2:
                recommendations["ravdess"]["feature_engineering"].append(
                    "Consider silence removal or voice activity detection"
                )
            
            # 스펙트럴 특성 기반 권장사항
            if any(np.std(ravdess_stats["audio_stats"]["spectral_features"][feat]) > 1000 
                   for feat in ["centroids", "rolloffs"]):
                recommendations["ravdess"]["audio_preprocessing"].append(
                    "Consider spectral normalization due to high feature variance"
                )
        
        return recommendations 