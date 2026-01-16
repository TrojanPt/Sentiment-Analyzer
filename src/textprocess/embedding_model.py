'''
嵌入式模型
'''
from abc import ABC, abstractmethod
from typing import List

import torch


class EmbeddingModel(ABC):
    """嵌入式模型基类"""
    def __init__(self, embedding_dim: int, device: torch.device, dtype: torch.dtype):
        """
        Args:
            embedding_dim (int): 嵌入维度
            device (torch.device): 设备类型
            dtype (torch.dtype): 数据类型
        """
        self.embedding_dim = embedding_dim

        self.device = device
        self.dtype = dtype
    
    @abstractmethod
    def embed(self, token: str) -> torch.Tensor:
        """
        嵌入方法
        Args:
            token (str): 需要嵌入的对象
        Returns:
            torch.Tensor: 嵌入后的张量
        """
        pass

    @abstractmethod
    def batch_embed(self, tokens: List[str], batch_size: int = 16) -> List[torch.Tensor]:
        """
        批量嵌入方法
        Args:
            tokens (List[str]): 需要嵌入的对象列表
            batch_size (int): 批大小
        Returns:
            List[torch.Tensor]: 嵌入后的张量列表
        """
        pass


class BGEm3EmbeddingModel(EmbeddingModel):
    """基于BGEm3的嵌入式模型"""
    
    def __init__(self, embedding_dim: int, device: torch.device, dtype: torch.dtype):
        """
        Args:
            embedding_dim (int): 嵌入维度
            device (torch.device): 设备类型
            dtype (torch.dtype): 数据类型
        """
        super().__init__(embedding_dim, device, dtype)
        
        self._load_model()
    
    def _load_model(self) -> None:
        """
        加载模型
        Args:
            model_path (str): 模型路径
        """
        from FlagEmbedding import BGEM3FlagModel

        model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True if self.dtype == torch.float16 else False, device=self.device)

        self.model = model
    
    def embed(self, token: str) -> torch.Tensor:
        embedding = self.model.encode(token)['dense_vecs']
        embedding = torch.tensor(embedding, dtype=self.dtype).to(self.device)
        return embedding
    
    def batch_embed(self, tokens: List[str], batch_size: int = 32) -> List[torch.Tensor]:
        """
        批量嵌入方法
        Args:
            tokens (List[str]): 需要嵌入的对象列表
            batch_size (int): 批大小
        Returns:
            List[torch.Tensor]: 嵌入后的张量列表
        """
        embeddings = self.model.encode(tokens, batch_size=batch_size)['dense_vecs']
        embeddings = [torch.tensor(embedding, dtype=self.dtype).to(self.device) for embedding in embeddings]
        return embeddings


class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI嵌入式模型，亦可通过OpenAI兼容的API调用格式调用其它嵌入式模型"""
    
    def __init__(
            self, 
            embedding_dim: int,
            device: torch.device,
            dtype: torch.dtype,
            api_base_url: str = 'https://api.openai.com',
            api_key: str = '',
            model_name: str = 'text-embedding-ada-002'
            ):
        """
        Args:
            embedding_dim (int): 嵌入维度
            device (torch.device): 设备类型
            dtype (torch.dtype): 数据类型
            api_base_url (str): URL
            api_key (str): API密钥
            model_name (str): 模型名称
        """
        super().__init__(embedding_dim, device, dtype)
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.model_name = model_name 
    
    def embed(self, token: str) -> torch.Tensor:
        import requests
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        data = {
            'model': self.model_name,
            'input': token
        }
        response = requests.post(f'{self.api_base_url}/v1/embeddings', headers=headers, json=data)
        if response.status_code == 200:
            embedding = response.json()['data'][0]['embedding']
            return torch.tensor(embedding, dtype=self.dtype).to(self.device)
        else:
            raise Exception(f"API请求失败: {response.status_code}, {response.text}")
        
    def batch_embed(self, tokens: List[str], batch_size: int = 16) -> torch.Tensor:
        raise NotImplementedError("批量嵌入方法尚未实现")