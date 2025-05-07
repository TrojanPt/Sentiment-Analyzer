import torch
import torch.nn.functional as F

from layer import StackedLSTM, AttentionLayer, FCLayer, LabelEmbedding

from textprocess import (
    BGEm3EmbeddingModel,
    JiebaTokenizer,
    TextPreprocessor
)

from train import ModelState

def load_model(model_path, device, dtype):
    """加载模型权重"""
    
    # 加载模型状态
    checkpoint: ModelState = torch.load(model_path, map_location=device)
    
    # 获取配置
    config = checkpoint['config']
    
    # 重新创建模型组件
    lstm_layer = StackedLSTM(
        input_size=config['embedding_dim'],
        hidden_size=config['hidden_dim'],
        num_layers=config['num_layers'],
        device=device,
        dropout=0.0,  # 预测时不使用dropout
        bidirectional=config['bidirectional']
    )
    
    # 创建注意力层
    attention_hidden_size = config['hidden_dim'] * 2 if config['bidirectional'] else config['hidden_dim']
    attention_layer = AttentionLayer(hidden_size=attention_hidden_size)
    
    # 创建全连接层
    fc_layer = FCLayer(
        input_size=attention_hidden_size,
        hidden_sizes=config['fc_hidden_sizes'],
        output_size=config['output_size'],
        dropout_rate=0.0  # 预测时不使用dropout
    )
    
    # 创建标签嵌入层
    label_embedding = LabelEmbedding(
        labels=config['labels'],
        embedding_model=BGEm3EmbeddingModel(embedding_dim=config['embedding_dim'], device=device, dtype=dtype)
    )

    # 加载状态
    lstm_layer.load_state_dict(checkpoint['lstm'])
    attention_layer.load_state_dict(checkpoint['attention'])
    fc_layer.load_state_dict(checkpoint['fc'])
    label_embedding.load_state_dict(checkpoint['label_embedding'])

    lstm_layer.to(device, dtype=dtype)
    attention_layer.to(device, dtype=dtype)
    fc_layer.to(device, dtype=dtype)
    label_embedding.to(device, dtype=dtype)
    
    # 将模型设置为评估模式
    lstm_layer.eval()
    attention_layer.eval()
    fc_layer.eval()
    label_embedding.eval()
    
    return lstm_layer, attention_layer, fc_layer, label_embedding, config

def predict(
        text, 
        lstm_layer, 
        attention_layer, 
        fc_layer, 
        label_embedding,
        preprocessor, 
        device,
        label_map=None
        ):
    """
    预测函数，对输入文本进行情感分析
    
    Args:
        text: 输入文本
        lstm_layer: LSTM模型
        attention_layer: 注意力层
        fc_layer: 全连接层
        label_embedding: 标签嵌入层
        preprocessor: 文本预处理器
        device: 计算设备
        label_map: 标签映射字典

    Returns:
        预测标签和置信度
    """
    
    # 设置模型为评估模式
    lstm_layer.eval()
    attention_layer.eval()
    fc_layer.eval()
    label_embedding.eval()
    
    # 预处理文本
    tokens = preprocessor.tokenizer.list(text)
    print(f"分词结果: {tokens[:10]}..." if len(tokens) > 10 else f"分词结果: {tokens}")
    
    # 嵌入分词后的标记
    token_embeddings = []
    for token in tokens:
        embedding = preprocessor.embed(token)
        token_embeddings.append(embedding)
    
    # 确保至少有一个词嵌入
    if not token_embeddings:
        print("警告：文本为空或无法分词")
        return None, 0.0
    
    # 将词嵌入堆叠成一个批次
    token_tensor = torch.stack(token_embeddings).unsqueeze(0).to(device)  # [1, seq_len, embedding_dim]
    
    # 预测
    with torch.no_grad():
        # 前向传播
        lstm_output, _ = lstm_layer(token_tensor)
        
        # 应用注意力
        context, attention_weights = attention_layer(lstm_output.transpose(0, 1))
        
        # 通过全连接层
        semantic_vector, _ = fc_layer(context)

        # 应用标签嵌入
        logits = label_embedding(semantic_vector)
        
        # 获取预测结果
        probabilities = F.softmax(logits, dim=1)[0]
        print(f"预测概率分布: {probabilities}")
        confidence, predicted_idx = torch.max(probabilities, 0)
    
    # 获取预测标签
    if label_map:
        # 反转标签映射以便从索引映射到标签
        idx_to_label = {}
        for label, tensor in label_map.items():
            for i, val in enumerate(tensor):
                if val == 1.0:
                    idx_to_label[i] = label
        
        predicted_label = idx_to_label.get(predicted_idx.item(), "未知")
    else:
        predicted_label = str(predicted_idx.item())
    
    # 打印详细概率分布
    if label_map:
        for i, prob in enumerate(probabilities):
            if i in idx_to_label:
                print(f"{idx_to_label[i]}: {prob.item():.4f}")
    
    # 返回预测标签和置信度
    return predicted_label, confidence.item()

if __name__ == "__main__":
    # 设置设备和数据类型

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 

    best_model_path = "path/to/model.pth"  # 模型路径
    
    lstm_layer, attention_layer, fc_layer, label_embedding, config = load_model(
        model_path = best_model_path,
        device = device,
        dtype = dtype
        )

    # 创建标签到索引的反向映射
    labels = config['labels']

    # 初始化分词器和嵌入模型
    tokenizer = JiebaTokenizer()
    embedding_model = BGEm3EmbeddingModel(embedding_dim=config['embedding_dim'], device=device, dtype=dtype)

    # 初始化文本预处理器
    preprocessor = TextPreprocessor(tokenizer, embedding_model)

    label_map = {}
    for i in range(len(labels)):
        label_map[labels[i]] = torch.zeros(config['embedding_dim'], dtype=dtype).to(device)
        label_map[labels[i]][i] = 1.0

    while True:
        sample_text = input("请输入要预测的文本（输入exit退出）：")
        if sample_text.lower() == "exit":
            break
        
        # 预测
        predicted_label, confidence = predict(
            text = sample_text, 
            lstm_layer = lstm_layer, 
            attention_layer = attention_layer, 
            fc_layer = fc_layer, 
            label_embedding = label_embedding,
            preprocessor = preprocessor,
            device = device,
            label_map = label_map
        )
        
        print(f"预测结果: {predicted_label}, 置信度: {confidence:.4f}")