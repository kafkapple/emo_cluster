import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging
import json

logger = logging.getLogger(__name__)

class ISEARDataset:
    """ISEAR 데이터셋을 로드하고 전처리하는 클래스"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_path = Path(config.dataset.data_path)
        self.labels_path = Path(config.dataset.labels_path)
        
        # 감정 레이블 매핑
        self.emotion_map = config.dataset.emotions.classes
        self.n_classes = config.dataset.emotions.n_classes
        
        # 데이터 저장용
        self.df: Optional[pd.DataFrame] = None
        self.labels: Optional[Dict] = None
        
    def load_data(self) -> pd.DataFrame:
        """데이터 로드 및 기본 전처리"""
        logger.info(f"Loading data from {self.data_path}")
        
        # CSV 파일 로드
        self.df = pd.read_csv(self.data_path, sep='|')
        
        # 필요한 컬럼만 선택
        cols = [
            self.config.dataset.columns.text,  # SIT
            'Field1',  # 감정 레이블
            'EMOT',    # 감정 숫자
            'language' # 언어
        ]
        self.df = self.df[cols]
        
        # 컬럼명 변경
        self.df = self.df.rename(columns={
            self.config.dataset.columns.text: 'text',
            'Field1': 'emotion_label',
            'EMOT': 'emotion_id',
            'language': 'language'
        })
        
        # 데이터 타입 변환
        self.df['emotion_id'] = self.df['emotion_id'].astype(int)
        self.df['emotion_label'] = self.df['emotion_label'].astype(str)
        self.df['text'] = self.df['text'].astype(str)
        self.df['language'] = self.df['language'].astype(str)
        
        # 결측치 제거
        self.df = self.df.dropna()
        
        logger.info(f"Loaded {len(self.df)} samples")
        return self.df
    
    def create_labels(self) -> Dict:
        """레이블 정보 생성"""
        if self.df is None:
            self.load_data()
            
        labels = {
            'num_samples': len(self.df),
            'num_classes': self.n_classes,
            'class_distribution': self.df['emotion_label'].value_counts().to_dict(),
            'class_mapping': self.emotion_map
        }
        
        # 레이블 파일 저장
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.labels_path, 'w') as f:
            json.dump(labels, f, indent=2)
            
        logger.info(f"Labels saved to: {self.labels_path}")
        self.labels = labels
        return labels 