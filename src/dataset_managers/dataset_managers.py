import collections
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch


class SentimentDataset:
    """情感分析数据集类"""
    
    def __init__(
            self, 
            file_path: str, 
            label, 
            device: torch.device,
            dtype: torch.dtype,
            val_size: float = 0.1, 
            test_size: float = 0.2,
            apply_augmentation: bool = True
            ):
        """
        Args:
            file_path (str): 数据文件路径
            label (List[str]): 标签列表
            device (torch.device): 设备类型
            dtype (torch.dtype): 数据类型
            val_size (float): 验证集比例
            test_size (float): 测试集比例
            apply_augmentation (bool): 是否应用数据增强
        """
        self.dim = len(label)
        self.label = label

        self.device = device
        self.dtype = dtype

        self.label_map = self._label_embedding_map(label)
        print(f"Label Map: {self.label_map}")

        self.data = pd.read_csv(file_path)
        self._load_data()

        self.apply_augmentation = apply_augmentation
        if self.apply_augmentation:
            self.augmenter = TextAugmentation(aug_prob=0.3)

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

    def _load_data(self) -> None:
        """
        加载数据
        """
        texts = self.data['text'].tolist()
        labels = self.data['label'].tolist()
        # labels = [self.label_map[label] for label in labels]

        self.texts = texts
        self.labels = labels
        
    def _split_data(
            self, 
            val_size: float = 0.1, 
            test_size: float = 0.2
            ) -> None:
        """
        划分训练集、验证集和测试集

        Args:
            val_size (float): 验证集比例
            test_size (float): 测试集比例
        """
        texts = self.texts
        labels = self.labels

        X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=val_size + test_size)
        test_size_adjusted = test_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size_adjusted)

        # 应用数据增强
        if self.apply_augmentation:
            X_train_aug, y_train_aug = self.augmenter.batch_augment(X_train, y_train)
            print(f"Using data augmentation, train size: {len(X_train)} -> {len(X_train_aug)}")
            X_train, y_train = X_train_aug, y_train_aug

        # 计算训练集的类别权重
        self.train_class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array(self.label),
            y=y_train
        )
        self.train_class_weights = torch.tensor(self.train_class_weights, dtype=self.dtype).to(self.device)
        # 计算验证集的类别权重
        self.val_class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array(self.label),
            y=y_val
        )
        self.val_class_weights = torch.tensor(self.val_class_weights, dtype=self.dtype).to(self.device)
        # 计算测试集的类别权重
        self.test_class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array(self.label),
            y=y_test
        )
        self.test_class_weights = torch.tensor(self.test_class_weights, dtype=self.dtype).to(self.device)

        # 将标签转换为嵌入向量
        y_train = [self.label_map[label] for label in y_train]
        y_val = [self.label_map[label] for label in y_val]
        y_test = [self.label_map[label] for label in y_test]


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
    
    def get_cached_vectors(self, split: str) -> List[torch.Tensor] | None:
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
        

import random
import jieba

class TextAugmentation:
    """文本增强器"""
    
    def __init__(self, aug_prob=0.3):
        """
        初始化文本增强器
        
        Args:
            aug_prob (float): 应用增强的概率
        """
        self.aug_prob = aug_prob
    
    def synonym_replacement(self, text: str, n=1) -> str:
        """
        随机替换n个非停用词为其同义词
        
        Args:
            text (str): 输入文本
            n (int): 替换词的数量
            
        Returns:
            str: 增强后的文本
        """
        words = list(jieba.cut(text))
        if len(words) <= 1:
            return text
            
        n = min(n, max(1, int(len(words) * 0.3)))
        indices = random.sample(range(len(words)), n)

        for idx in indices:
            # 同义词替换
            if len(words[idx]) > 1:
                pass
        
        return "".join(words)
    
    def random_insertion(self, text: str, n=1) -> str:
        """
        随机在文本中插入n个词
        
        Args:
            text (str): 输入文本
            n (int): 插入词的数量
            
        Returns:
            str: 增强后的文本
        """
        words = jieba.lcut(text)
        if len(words) <= 1:
            return text
            
        n = min(n, max(1, int(len(words) * 0.2)))
        
        # 在随机位置插入随机词
        for _ in range(n):
            insert_pos = random.randint(0, len(words))
            # 从原文中随机选择一个词插入
            insert_word = random.choice(words)
            words.insert(insert_pos, insert_word)
            
        return "".join(words)
    
    def random_insert_punctuation(self, text: str, n=2) -> str:
        """
        随机在文本中插入n个标点符号
        
        Args:
            text (str): 输入文本
            n (int): 插入标点的数量
            
        Returns:
            str: 增强后的文本
        """
        words = jieba.lcut(text)
        if len(words) <= 1:
            return text
            
        n = min(n, max(1, int(len(words) * 0.2)))

        punctuation = ['，', '。', '！', '？', '；', '：', '、', '“', '”', '‘', '’', '（', '）', '——', '…', '·', '~']
        
        # 在随机位置插入随机标点
        for _ in range(n):
            insert_pos = random.randint(0, len(words))
            insert_word = random.choice(punctuation)
            words.insert(insert_pos, insert_word)
            
        return "".join(words)
    
    def random_swap(self, text: str, n=1) -> str:
        """
        随机交换文本中的n对词
        
        Args:
            text (str): 输入文本
            n (int): 交换的词对数量
            
        Returns:
            str: 增强后的文本
        """
        words = list(jieba.cut(text))
        if len(words) <= 1:
            return text
            
        n = min(n, max(1, int(len(words) * 0.2)))
        
        # 随机交换n对词
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
            
        return "".join(words)
    
    def random_deletion(self, text: str, p=0.1) -> str:
        """
        以概率p随机删除词
        
        Args:
            text (str): 输入文本
            p (float): 删除词的概率
            
        Returns:
            str: 增强后的文本
        """
        words = list(jieba.cut(text))
        if len(words) <= 1:
            return text
        
        # 保留单词的概率为1-p
        words = [word for word in words if random.random() > p or len(word) <= 1]
        
        if not words:  # 确保不会删除所有词
            return text
            
        return "".join(words)
       
    def augment(self, text: str) -> Tuple[str, str]:
        """
        对文本应用随机增强
        
        Args:
            text (str): 输入文本
            
        Returns:
            Tuple[str, str]: 原文本和增强后的文本
        """
        if random.random() > self.aug_prob:
            return text, text
            
        aug_methods = [
            # lambda t: self.synonym_replacement(t, n=1),
            lambda t: self.random_insertion(t, n=1),
            lambda t: self.random_insert_punctuation(t, n=2),
            lambda t: self.random_swap(t, n=1),
            lambda t: self.random_deletion(t, p=0.1)
        ]
        
        method = random.choice(aug_methods)
        augmented_text = method(text)
        
        return text, augmented_text
    
    def batch_augment(self, texts: List[str], labels: List[Any]) -> Tuple[List[str], List[Any]]:
        """
        对一批文本进行数据增强
        
        Args:
            texts (List[str]): 输入文本列表
            labels (List[Any]): 对应标签列表
            
        Returns:
            Tuple[List[str], List[Any]]: 增强后的文本和标签
        """
        augmented_texts = []
        augmented_labels = []
        
        for text, label in zip(texts, labels):
            # 保留原样本
            augmented_texts.append(text)
            augmented_labels.append(label)
            
            # 使用增强方法生成新样本
            if random.random() < self.aug_prob:
                aug_method = random.choice([
                    # lambda t: self.synonym_replacement(t, n=1),
                    lambda t: self.random_insertion(t, n=1),
                    lambda t: self.random_insert_punctuation(t, n=2),
                    lambda t: self.random_swap(t, n=1),
                    lambda t: self.random_deletion(t, p=0.1)
                ])
                
                augmented_text = aug_method(text)
                augmented_texts.append(augmented_text)
                augmented_labels.append(label)  # 使用相同标签
        
        return augmented_texts, augmented_labels