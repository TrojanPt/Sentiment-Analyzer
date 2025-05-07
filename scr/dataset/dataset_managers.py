from typing import Dict, List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
import torch


class SentimentDataset:
    """情感分析数据集类"""
    
    def __init__(
            self, 
            file_path: str, 
            label, 
            device: str,
            dtype: torch.dtype,
            val_size: float = 0.1, 
            test_size: float = 0.2
            ):
        """
        Args:
            file_path (str): 数据文件路径
            label (List[str]): 标签列表
            device (str): 设备
            dtype (torch.dtype): 数据类型
            val_size (float): 验证集比例
            test_size (float): 测试集比例
        """
        self.dim = len(label)

        self.device = device
        self.dtype = dtype

        self.label_map = self._label_embedding_map(label)
        print(f"标签映射: {self.label_map}")

        self.data = pd.read_csv(file_path)
        self._load_data()

        self._split_data(val_size, test_size)

        # 嵌入向量缓存
        self.train_vectors = None
        self.val_vectors = None
        self.test_vectors = None

    def _label_embedding_map(self, label: List[str]) -> Dict[str, torch.Tensor]:
        """
        建立不同标签与相互正交的单位向量间的映射关系
        """
        dict = {}
        for i in range(len(label)):
            dict[label[i]] = torch.zeros(self.dim, dtype=self.dtype).to(self.device)
            dict[label[i]][i] = 1.0
        
        return dict

    def _load_data(self) -> Tuple[List[str], List[torch.Tensor]]:
        """
        加载数据
        """
        texts = self.data['text'].tolist()
        labels = self.data['label'].tolist()
        labels = [self.label_map[label] for label in labels]

        self.texts = texts
        self.labels = labels
        
    def _split_data(
            self, 
            val_size: float = 0.1, 
            test_size: float = 0.2
            ) -> Tuple[Tuple[List[str], List[torch.Tensor]], Tuple[List[str], List[torch.Tensor]], Tuple[List[str], List[torch.Tensor]]]:
        """
        划分训练集、验证集和测试集

        Args:
            val_size (float): 验证集比例
            test_size (float): 测试集比例
        """
        texts = self.texts
        labels = self.labels

        X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=val_size + test_size)
        val_size_adjusted = val_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted)

        self.train_data = (X_train, y_train)
        self.val_data = (X_val, y_val)
        self.test_data = (X_test, y_test)
        
    def cache_vectors(self, split: str, vectors: List[torch.Tensor]):
        """
        缓存预计算的向量
        
        Args:
            split (str): 数据集分割 ('train', 'val', 'test')
            vectors (List[torch.Tensor]): 向量列表
        """
        if split == 'train':
            self.train_vectors = vectors
        elif split == 'val':
            self.val_vectors = vectors
        elif split == 'test':
            self.test_vectors = vectors
        else:
            raise ValueError(f"未知的数据集分割: {split}")
    
    def get_cached_vectors(self, split: str) -> List[torch.Tensor]:
        """
        获取缓存的向量
        
        Args:
            split (str): 数据集分割 ('train', 'val', 'test')
            
        Returns:
            List[torch.Tensor]: 缓存的向量列表
        """
        if split == 'train':
            return self.train_vectors
        elif split == 'val':
            return self.val_vectors
        elif split == 'test':
            return self.test_vectors
        else:
            raise ValueError(f"未知的数据集分割: {split}")