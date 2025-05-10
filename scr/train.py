from dataclasses import dataclass
import os
from typing import Any, Dict, List, Tuple, TypedDict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from layer import (
    LSTMLayer,
    BiLSTMWrapper,
    StackedLSTM, 
    AttentionLayer, 
    FCLayer, 
    LabelEmbedding,
    LabelSmoothingLoss
)

from dataset import SentimentDataset

from textprocess import (
    TextPreprocessor, 
    BatchPreprocessor, 
    JiebaTokenizer,
    BGEm3EmbeddingModel
)

# 使用huggingface的镜像网站
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


class ModelConfig(TypedDict):
    labels: List[str]
    embedding_dim: int
    hidden_dim: int
    num_layers: int
    bidirectional: bool
    fc_hidden_sizes: List[int]
    output_size: int

class ModelState(TypedDict):
    lstm: Dict[str, Any]
    attention: Dict[str, Any]
    fc: Dict[str, Any]
    label_embedding: Dict[str, Any]
    optimizer: Dict[str, Any]
    config: ModelConfig


class Trainer:
    """模型训练器"""
    
    def __init__(
            self, 

            lstm_layer: LSTMLayer,
            attention_layer: AttentionLayer,
            fc_layer: FCLayer,
            label_embedding: LabelEmbedding,

            optimizer: torch.optim.Optimizer,
            criterion: nn.Module,
            device: str,
            dtype: torch.dtype,

            textpreprocessor: TextPreprocessor,

            dataset: SentimentDataset,

            max_batch_size: int = 24,

            l2_lambda: float = 0.001
            ):
        """
        Args:
            lstm_layer (LSTMLayer): LSTM层
            attention_layer (AttentionLayer): 注意力层
            fc_layer (FCLayer): 全连接层
            label_embedding (LabelEmbedding): 标签嵌入层

            optimizer (torch.optim.Optimizer): 优化器
            criterion (nn.Module): 损失函数
            device (str): 计算设备
            dtype (torch.dtype): 数据类型

            textpreprocessor (TextPreprocessor): 文本预处理器

            dataset (SentimentDataset): 数据集对象

            max_batch_size (int): 批大小

            l2_lambda (float): L2正则化系数
        """
        self.lstm_layer = lstm_layer.to(device, dtype=dtype)
        self.attention_layer = attention_layer.to(device, dtype=dtype)
        self.fc_layer = fc_layer.to(device, dtype=dtype)
        self.label_embedding = label_embedding.to(device, dtype=dtype)

        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.textpreprocessor = textpreprocessor

        self.dataset = dataset
        self.train_dataset = dataset.train_data
        self.val_dataset = dataset.val_data
        self.test_dataset = dataset.test_data

        self.max_batch_size = max_batch_size

        self.l2_lambda = l2_lambda

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        self.is_first_train_epoch = True
        self.is_first_eval_epoch = True
        
    def train_epoch(self) -> Tuple[float, float]:
        """执行一个epoch的训练，返回平均训练损失和准确率"""
        self.lstm_layer.train()
        self.attention_layer.train()
        self.fc_layer.train()
        self.label_embedding.train()

        total_loss = 0.0
        total_samples = 0
        correct = 0

        if not self.is_first_train_epoch:
            cache_vector = self.dataset.get_cached_vectors('train')
        else:
            cache_vector = None
        # 使用BatchPreprocessor进行并行预处理
        preprocessor = BatchPreprocessor(
            self.device,
            self.textpreprocessor, 
            self.train_dataset, 
            cache_vector,
            self.max_batch_size
        )

        total = len(self.train_dataset[0])
        pbar = tqdm(
            total=total,
            desc='Training',
            dynamic_ncols=True
            )
        
        # 使用text_preprocess方法获取批次数据
        for batch in preprocessor:
            inputs, targets = batch
            
            # 确保输入和目标在正确的设备上
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            batch_size = inputs.size(0)
            total_samples += batch_size

            pbar.update(batch_size)
            
            # 清除梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            # LSTM层
            lstm_output, _ = self.lstm_layer(inputs)
            
            # 应用注意力
            context, _ = self.attention_layer(lstm_output.transpose(0, 1))  # 转换维度为[batch_size, seq_len, hidden_size]
            
            # 通过全连接层
            semantic_vector, l2_loss = self.fc_layer(context)

            # 应用标签嵌入计算logits
            logits = self.label_embedding(semantic_vector)
            
            # 计算损失
            loss = self.criterion(logits, targets) + self.l2_lambda * l2_loss
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪以防止梯度爆炸
            nn.utils.clip_grad_norm_(list(self.lstm_layer.parameters()) + 
                                    list(self.attention_layer.parameters()) + 
                                    list(self.fc_layer.parameters()), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            # 累计损失
            total_loss += loss.item() * batch_size
            
            # 计算准确率
            _, predicted = torch.max(logits, 1)
            _, true_labels = torch.max(targets, 1)
            correct += (predicted == true_labels).sum().item()
        
        pbar.close()

        # 计算平均损失和准确率
        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples
        
        # 更新学习率
        self.scheduler.step(avg_loss)

        if self.is_first_train_epoch:
            collected_vectors = preprocessor.get_collected_vectors()
            if collected_vectors is not None:
                self.dataset.cache_vectors('train', collected_vectors)
                self.is_first_train_epoch = False
                print(f"训练数据嵌入向量缓存成功，大小: {len(collected_vectors)}")
            else:
                raise ValueError("没有收集到训练数据的嵌入向量，请检查数据预处理。")
        
        return avg_loss, accuracy

    def evaluate(self, split='val') -> Tuple[float, float]:
        """
        在验证集、测试集上评估模型，返回损失和准确率

        Args:
            dataset: 要评估的数据集 ('val' 或 'test')
        """
        self.lstm_layer.eval()
        self.attention_layer.eval()
        self.fc_layer.eval()
        self.label_embedding.eval()

        if split == 'val':
            dataset = self.val_dataset
        elif split == 'test':
            dataset = self.test_dataset
        else:
            raise ValueError("未知的数据集类型，请选择 'val' 或 'test'。")
        
        total_loss = 0.0
        total_samples = 0
        correct = 0

        if split == 'val' and not self.is_first_eval_epoch:
            cache_vector = self.dataset.get_cached_vectors('val')
        else:
            cache_vector = None

        # 使用BatchPreprocessor进行并行预处理
        preprocessor = BatchPreprocessor(
            device=self.device,
            textpreprocessor = self.textpreprocessor, 
            dataset = dataset, 
            cache_vector = cache_vector, 
            max_batch_size = self.max_batch_size
        )

        total = len(dataset[0])
        pbar = tqdm(
            total=total,
            desc=split,
            dynamic_ncols=True
            )
        
        with torch.no_grad():
            for batch in preprocessor:
                inputs, targets = batch
                
                # 确保输入和目标在正确的设备上
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                batch_size = inputs.size(0)
                total_samples += batch_size

                pbar.update(batch_size)
                
                # 前向传播
                lstm_output, _ = self.lstm_layer(inputs)
                
                # 应用注意力
                context, _ = self.attention_layer(lstm_output.transpose(0, 1))
                
                # 通过全连接层
                semantic_vector, l2_loss = self.fc_layer(context)

                # 应用标签嵌入计算logits
                logits = self.label_embedding(semantic_vector)
                
                # 计算损失
                loss = self.criterion(logits, targets) + self.l2_lambda * l2_loss
                
                # 累计损失
                total_loss += loss.item() * batch_size
                
                # 计算准确率
                _, predicted = torch.max(logits, 1)
                _, true_labels = torch.max(targets, 1)
                correct += (predicted == true_labels).sum().item()
                
        pbar.close()

        # 计算平均损失和准确率
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        accuracy = correct / total_samples if total_samples > 0 else 0

        if split == 'val' and self.is_first_eval_epoch:
            collected_vectors = preprocessor.get_collected_vectors()
            if collected_vectors is not None:
                self.dataset.cache_vectors('val', collected_vectors)
                self.is_first_eval_epoch = False
            else:
                raise ValueError("没有收集到验证数据的嵌入向量，请检查数据预处理。")
        
        return avg_loss, accuracy

    def save_model(self, save_path: str, epoch: int):
        """保存模型"""
        # 确保目录存在
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型组件状态
        model_path = os.path.join(save_path, f"sentiment_model_epoch_{epoch}.pth")
        model_state : ModelState = {
            'lstm': self.lstm_layer.state_dict(),
            'attention': self.attention_layer.state_dict(),
            'fc': self.fc_layer.state_dict(),
            'label_embedding': self.label_embedding.state_dict(),
            # 保存优化器状态
            'optimizer': self.optimizer.state_dict(),
            # 额外保存模型配置信息
            'config': {
                'labels': self.label_embedding.labels,
                'embedding_dim': self.lstm_layer.input_size if hasattr(self.lstm_layer, 'input_size') else 
                            self.lstm_layer.lstm_layers[0].input_size,
                'hidden_dim': self.lstm_layer.hidden_size if hasattr(self.lstm_layer, 'hidden_size') else 
                            self.lstm_layer.lstm_layers[0].hidden_size,
                'num_layers': self.lstm_layer.num_layers if hasattr(self.lstm_layer, 'num_layers') else 
                            len(self.lstm_layer.lstm_layers),
                'bidirectional': self.lstm_layer.bidirectional if hasattr(self.lstm_layer, 'bidirectional') else 
                                isinstance(self.lstm_layer.lstm_layers[0], BiLSTMWrapper),
                'fc_hidden_sizes': [m.out_features for m in self.fc_layer.fc_layers if isinstance(m, nn.Linear)][:-1],
                'output_size': [m.out_features for m in self.fc_layer.fc_layers if isinstance(m, nn.Linear)][-1]
            }
        }
        torch.save(model_state, model_path)
        print(f"模型已保存到 {model_path}")


def visualize_training_history(history):
    """
    可视化训练过程中的损失和准确率变化
    
    参数:
        history: 包含训练记录的字典，结构为:
            {
                'train': [(train_loss, train_acc), ...],
                'val': [(val_loss, val_acc), ...]
            }
    
    返回:
        fig: matplotlib的Figure对象
        axs: 包含两个Axes子图对象的数组
    """
    # 提取训练数据
    train_loss = [epoch[0] for epoch in history['train']]
    train_acc = [epoch[1] for epoch in history['train']]
    
    # 提取验证数据
    val_loss = [epoch[0] for epoch in history['val']]
    val_acc = [epoch[1] for epoch in history['val']]
    
    # 创建画布和子图
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    axs[0].plot(train_loss, label='Training Loss', color='blue')
    axs[0].plot(val_loss, label='Validation Loss', color='orange')
    axs[0].set_title('Loss Curve')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[0].legend()
    
    # 绘制准确率曲线
    axs[1].plot(train_acc, label='Training Accuracy', color='blue')
    axs[1].plot(val_acc, label='Validation Accuracy', color='orange')
    axs[1].set_title('Accuracy Curve')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].grid(True, linestyle='--', alpha=0.5)
    axs[1].legend()
    
    plt.tight_layout()
    return fig, axs


if __name__ == "__main__":
    # 设置参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前设备: {device}")

    sentiment_data_path = r"dataset\test.csv"
    model_save_folder = r"models"
    graph_save_folder = r'graph'

    sentiment_labels: List[str] = ['well', 'normal', 'snicker', 'self-satisfied', 'shocked', 'a bit down', 'laugh', 'angry']
    # sentiment_labels: List[str] = ['positive', 'negative', 'neutral']
    num_classes = len(sentiment_labels)

    dtype = torch.float32

    embedding_dim = 1024
    hidden_dim = 2048
    num_layers = 4


    max_batch_size = 2

    learning_rate = 1e-5
    num_epochs = 5

    bidirectional = True  # 是否使用双向LSTM
    
    # 初始化分词器和嵌入模型
    tokenizer = JiebaTokenizer()
    embedding_model = BGEm3EmbeddingModel(
        embedding_dim=embedding_dim,
        device=device,
        dtype=dtype
        )
    
    # 初始化文本预处理器
    preprocessor = TextPreprocessor(tokenizer, embedding_model)
    
    # 加载数据集
    dataset = SentimentDataset(
        file_path=sentiment_data_path,
        label=sentiment_labels,
        device=device,
        dtype=dtype,
        val_size=0.1,
        test_size=0.2
    )
    
    # 初始化模型
    lstm_layer = StackedLSTM(
        input_size=embedding_dim, 
        hidden_size=hidden_dim,
        num_layers=num_layers,  # 堆叠LSTM层数
        device=device,
        dropout=0.2,
        bidirectional=bidirectional  # 使用双向LSTM
    )

    attention_layer = AttentionLayer(hidden_size=hidden_dim * 2 if bidirectional else hidden_dim)

    fc_layer = FCLayer(
        input_size=hidden_dim * 2 if bidirectional else hidden_dim,
        hidden_sizes=[1024, 1024],
        output_size=embedding_dim,
        dropout_rate=0.3
    )

    # 初始化标签嵌入
    label_embedding = LabelEmbedding(sentiment_labels, embedding_model)
    
    # 优化器
    optimizer = torch.optim.AdamW(
        list(lstm_layer.parameters()) + 
        list(attention_layer.parameters()) + 
        list(fc_layer.parameters()) +
        list(label_embedding.parameters()),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    criterion = LabelSmoothingLoss(classes=num_classes)

    # 初始化训练器
    trainer = Trainer(
        lstm_layer=lstm_layer,
        attention_layer=attention_layer,
        fc_layer=fc_layer,
        label_embedding=label_embedding,

        optimizer=optimizer,
        criterion=criterion,
        device=device,
        dtype=dtype,

        textpreprocessor=preprocessor,

        dataset=dataset,

        max_batch_size=max_batch_size,

        l2_lambda=0.001
    )
    
    # 训练模型
    print("Starting training...")
    best_val_acc = 0
    patience = 5
    counter = 0

    history = {'train': [], 'val': []}
    
    epoch = 0
    while epoch < num_epochs:

        print(f"Epoch [{epoch+1}/{num_epochs}]")

        train_loss, train_acc = trainer.train_epoch()
        history['train'].append((train_loss, train_acc))

        val_loss, val_accuracy = trainer.evaluate()
        history['val'].append((val_loss, val_accuracy))
        
        print(f"train loss: {train_loss:.4f}, "
              f"train accuracy: {train_acc:.4f}, "
              f"val loss: {val_loss:.4f}, "
              f"val accuracy: {val_accuracy:.4f}")
        
        trainer.save_model(model_save_folder, epoch+1)
        
        # 早停机制
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            counter = 0
            # 记录最佳模型
            print(f"Best model saved at epoch {epoch+1}")
            best_model_path = os.path.join(model_save_folder, f"sentiment_model_epoch_{epoch+1}.pth")
        else:
            counter += 1
            if counter >= patience:
                if_early_stop = input(f'The accuracy of verification has not increased in {counter} epochs, stop training?[y/n]: ')
                if if_early_stop == 'y':
                    break
                print('Continue training...')

        epoch += 1

        if epoch == num_epochs:
            figure, axes = visualize_training_history(history)
            figure.savefig(f'{graph_save_folder}/training_history.png', dpi=300, bbox_inches='tight')
            print('training_history.png is saved')

            if_continue = input('Continue Training?[y/n]').lower()
            if if_continue == 'y':
                add_num_epochs = int(input('Add epochs: '))
                max_batch_size = int(input('Max batch size: '))
                trainer.max_batch_size = max_batch_size

                num_epochs += add_num_epochs

    # 在测试集上评估模型
    test_loss, test_accuracy = trainer.evaluate(split='test')
    print(f"test loss: {test_loss:.4f}, test accuracy: {test_accuracy:.4f}")

    print(f'Best model path: {best_model_path}')