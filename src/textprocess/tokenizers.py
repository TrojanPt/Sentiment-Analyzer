'''
分词器
'''

from abc import ABC, abstractmethod
from typing import List


class Tokenizer(ABC):
    """分词器基类"""
  
    @abstractmethod
    def list(self, text: str) -> List[str]:
        """
        以列表形式返回分词结果
        Args:
            text (str): 输入文本
        Returns:
            List[str]: 分词后的列表
        """
        pass

    @abstractmethod
    def generator(self, text: str) -> str:
        """
        以生成器形式返回分词结果
        Args:
            text (str): 输入文本
        Yields:
            str: 分词后的token
        """
        pass


class JiebaTokenizer(Tokenizer):
    """基于jieba的分词器"""
    
    def list(self, text: str, cut_all: bool = False) -> List[str]:
        import jieba
        return jieba.lcut(text, cut_all=cut_all)
    
    def generator(self, text: str, cut_all: bool = False):
        import jieba
        for token in jieba.cut(text, cut_all=cut_all):
            yield token

