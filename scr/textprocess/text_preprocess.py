'''
文本预处理模块
'''


from collections import defaultdict
from functools import lru_cache
import queue
import threading
from typing import Iterator, List, Optional, Tuple

import concurrent.futures

import torch

from .tokenizers import Tokenizer
from .embedding_model import EmbeddingModel

class TextPreprocessor:
    """文本预处理"""
    
    def __init__(self, tokenizer: Tokenizer, embedding_model: EmbeddingModel):
        """
        Args:
            tokenizer: 分词器
            embedding_model: 嵌入式模型
        """
        self.tokenizer = tokenizer
        self.embed = embedding_model.embed
        self.batch_embed = embedding_model.batch_embed


def text_preprocess(
        device: str,
        textpreprocessor: TextPreprocessor, 
        dataset: Tuple[List[str], List[torch.Tensor]],
        cache_vector: Optional[List[torch.Tensor]] = None,
        max_batch_size: int = 32
        ):
    """
    对数据集进行文本预处理，并返回批
    通过实时返回批次、并行处理、缓存机制提高处理效率

    Args:
        device (str): 设备类型
        textpreprocessor (TextPreprocessor): 文本预处理器
        dataset (Tuple[List[str], List[torch.Tensor]]): 包含文本及对应标签的数据集
        cache_vector (List[torch.Tensor]): 词嵌入缓存
        max_batch_size (int): 批大小
    Yield:
        Tuple[torch.Tensor, torch.Tensor]: 处理后的文本向量和标签
    """
    if cache_vector is None:
        # 如果没有提供缓存向量，则创建一个新的缓存
        texts, labels = dataset
        cache_vector = [None] * len(texts)
        
        # 缓存词嵌入结果以避免重复计算
        @lru_cache(maxsize=10000)
        def cached_embed(token):
            return textpreprocessor.embed(token)
        
        # 创建一个字典来存储不同长度的批次
        length_batches = defaultdict(list)
        length_labels = defaultdict(list)
        
        # 并行处理文本的函数
        def process_text(idx):
            text = texts[idx]
            label = labels[idx]
            
            # 打印进度
            print(f'processing text {idx+1}/{len(texts)}')
            
            # 分词并嵌入
            tokens = textpreprocessor.tokenizer.list(text)
            vector = []
            
            for token in tokens:
                vector.append(cached_embed(token))

            vector = torch.stack(vector, dim=0).to(device) # [seq_len, embedding_dim]
            cache_vector[idx] = vector
            
            return idx, vector, label, vector.size(0)
        
        # 分批处理文本，避免一次性提交所有任务
        batch_size_multiplier = 3  # 控制每次提交的任务数量
        processed_count = 0
        
        while processed_count < len(texts):
            # 计算本批次要处理的文本范围
            start_idx = processed_count
            end_idx = min(start_idx + max_batch_size * batch_size_multiplier, len(texts))
            
            # 使用线程池仅处理当前批次的文本
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                # 只提交当前批次的任务
                future_to_idx = {executor.submit(process_text, i): i for i in range(start_idx, end_idx)}
                
                # 处理当前批次的完成任务
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx, vector, label, length = future.result()
                    
                    # 将处理结果添加到相应长度的批次中
                    length_batches[length].append(vector)
                    length_labels[length].append(label)
                    
                    # 检查是否有任何长度的批次达到了max_batch_size
                    for length, batch_vectors in list(length_batches.items()):
                        if len(batch_vectors) >= max_batch_size:
                            batch_labels = length_labels[length]
                            # 创建并返回批次
                            yield torch.stack(batch_vectors[:max_batch_size]), \
                                torch.stack(batch_labels[:max_batch_size])
                            
                            # 更新剩余数据
                            length_batches[length] = batch_vectors[max_batch_size:]
                            length_labels[length] = batch_labels[max_batch_size:]
            
            # 更新已处理的文本数量
            processed_count = end_idx
            

        # 处理剩余的批次
        for length, batch_vectors in length_batches.items():
            if batch_vectors:  # 确保批次不为空
                batch_labels = length_labels[length]
                yield torch.stack(batch_vectors), \
                    torch.stack(batch_labels)
                
        yield cache_vector
        return
                
    else:
        # 如果提供了缓存向量，则直接返回
        texts, labels = dataset

        length_batches = defaultdict(list)
        length_labels = defaultdict(list)

        for idx, vector in enumerate(cache_vector):

            if vector is None:
                # 如果缓存为空，则使用文本预处理器处理文本
                text = texts[idx]
                tokens = textpreprocessor.tokenizer.list(text)
                vector = []
                
                for token in tokens:
                    vector.append(textpreprocessor.embed(token))

                vector = torch.stack(vector, dim=0).to(device)
            
            length = vector.size(0)

            length_batches[length].append(vector)
            length_labels[length].append(labels[idx])

            # 检查是否有任何长度的批次达到了max_batch_size
            for length, batch_vectors in list(length_batches.items()):
                if len(batch_vectors) >= max_batch_size:
                    batch_labels = length_labels[length]
                    # 创建并返回批次
                    yield torch.stack(batch_vectors[:max_batch_size]), \
                        torch.stack(batch_labels[:max_batch_size])
                    
                    # 更新剩余数据
                    length_batches[length] = batch_vectors[max_batch_size:]
                    length_labels[length] = batch_labels[max_batch_size:]    
        
        # 处理剩余的批次
        for length, batch_vectors in length_batches.items():
            if batch_vectors:  # 确保批次不为空
                batch_labels = length_labels[length]
                yield torch.stack(batch_vectors), \
                    torch.stack(batch_labels)
        
        return


class BatchPreprocessor:
    """并行批处理预加载器"""
    
    def __init__(
            self, 
            device: torch.device,
            textpreprocessor: TextPreprocessor, 
            dataset: Tuple[List[str], List[torch.Tensor]], 
            cache_vector: Optional[List[torch.Tensor]] = None,
            max_batch_size: int = 32,
            queue_size: int = 3
        ):
        """
        初始化批处理预加载器
        
        Args:
            device: 设备
            textpreprocessor: 文本预处理器
            dataset: 包含文本及标签的数据集
            cache_vector: 词嵌入缓存
            max_batch_size: 最大批处理大小
            queue_size: 队列大小，控制预加载的批次数量
        """
        self.device = device
        self.textpreprocessor = textpreprocessor

        self.dataset = dataset
        self.cache_vector = cache_vector

        self.max_batch_size = max_batch_size

        self.batch_queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.thread = None

        self.collected_vectors = []
        
    def start(self):
        """启动预处理线程"""
        if self.thread is not None and self.thread.is_alive():
            return
            
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._preprocess_worker, daemon=True)
        self.thread.start()
        
    def stop(self):
        """停止预处理线程"""
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join(timeout=1.0)
            
    def _preprocess_worker(self):
        """后台工作线程，持续生成批次并放入队列"""
        try:
            for batch in text_preprocess(
                self.device,
                self.textpreprocessor, 
                self.dataset, 
                cache_vector = self.cache_vector,
                max_batch_size = self.max_batch_size
                ):
                if len(batch) == 2:
                    if self.stop_event.is_set():
                        break
                    self.batch_queue.put(batch, block=True)
                else:
                    self.collected_vectors = batch
            
            # 数据集处理完成，放入结束标记
            self.batch_queue.put(None)

        except Exception as e:
            print(f"预处理线程发生错误: {e}")
            # 确保队列中有结束标记
            self.batch_queue.put(None)
            
    def get_batch(self, timeout: Optional[float] = None) -> Optional[Tuple[torch.Tensor, torch.Tensor] | List[torch.Tensor]]:
        """
        获取一个预处理好的批次
        
        Args:
            timeout: 等待超时时间，None表示一直等待
            
        Returns:
            预处理好的批次元组(输入, 标签)，如果没有更多批次则返回None
        """
        try:
            batch = self.batch_queue.get(block=True, timeout=timeout)
            return batch
        except queue.Empty:
            return None
            
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor] | List[torch.Tensor]]:
        """迭代器接口，方便在循环中使用"""
        self.start()
        while True:
            batch = self.get_batch()
            if batch is None:
                break
            yield batch
        self.stop()

    def get_collected_vectors(self):
        """获取收集的向量"""
        return self.collected_vectors
