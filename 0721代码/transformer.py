import torch
import torch.nn as nn
# N: 表示批次大小（batch size），即一次处理的样本数量。
# T: 表示序列长度（sequence length），可以是输入序列的长度，也可以是输出序列的长度。在编码器中，它通常表示输入序列的长度；在解码器中，它表示目标序列的长度。
# E: 表示嵌入维度（embedding size）或特征维度（feature size），即每个向量的大小。

def qkv_attention_value(q, k, v, mask=False):
    """
    计算attention value
    :param q: [N,T1,E] or [N,h,T1,E]
    :param k: [N,T2,E] or [N,h,T2,E]
    :param v: [N,T2,E] or [N,h,T2,E]
    :param mask: True or False or Tensor
    :return: [N,T1,E] or [N,h,T1,E]
    """
    # 2. 计算q和k之间的相关性->F函数
    k = torch.transpose(k, dim0=-2, dim1=-1)  # [??, T2, E] --> [??, E, T2]
    # matmul([??,T1,E], [??,E,T2])
    scores = torch.matmul(q, k)  # [??,T1,T2]

    if isinstance(mask, bool):
        if mask:
            _shape = scores.shape
            mask = torch.ones((_shape[-2], _shape[-1]))
            mask = torch.triu(mask, diagonal=1) * -10000
            mask = mask[None][None]
        else:
            mask = None
    if mask is not None:
        scores = scores + mask

    # 3. 转换为权重
    alpha = torch.softmax(scores, dim=-1)  # [??,T1,T2]

    # 4. 值的合并
    # matmul([??,T1,T2], [??,T2,E])
    v = torch.matmul(alpha, v)  # [??,T1,E]
    return v


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_header):
        super(MultiHeadSelfAttention, self).__init__()
        assert hidden_size % num_header == 0, f"header的数目没办法整除:{hidden_size}, {num_header}"

        self.hidden_size = hidden_size  # 就是向量维度大小，也就是E
        self.num_header = num_header  # 头的数目

        self.wq = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        )
        self.wk = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        )
        self.wv = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        )
        self.wo = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU()
        )
#    在这里通过reshape 将传入的q k v进行分割
    def split(self, vs):
        n, t, e = vs.shape
        vs = torch.reshape(vs, shape=(n, t, self.num_header, e // self.num_header))
        vs = torch.permute(vs, dims=(0, 2, 1, 3))
        return vs

    def forward(self, x, attention_mask=None, **kwargs):
        """
        前向过程
        :param x: [N,T,E] 输入向量
        :param attention_mask: [N,T,T] mask矩阵
        :return: [N,T,E] 输出向量
        """
        # 1. 获取q、k、v
        q = self.wq(x)  # [n,t,e]
        k = self.wk(x)  # [n,t,e]
        v = self.wv(x)  # [n,t,e]
        q = self.split(q)  # [n,t,e] --> [n,h,t,v]  e=h*v h就是head的数目，v就是每个头中self-attention的维度大小
        k = self.split(k)  # [n,t,e] --> [n,h,t,v]  e=h*v
        v = self.split(v)  # [n,t,e] --> [n,h,t,v]  e=h*v

        # 计算attention value
        v = qkv_attention_value(q, k, v, attention_mask)

        # 5. 输出
        v = torch.permute(v, dims=(0, 2, 1, 3))  # [n,h,t,v] --> [n,t,h,v]
        n, t, _, _ = v.shape
        v = torch.reshape(v, shape=(n, t, -1))  # [n,t,h,v] -> [n,t,e]
        v = self.wo(v)  # 多个头之间的特征组合合并
        return v


class MultiHeadEncoderDecoderAttention(nn.Module):
    def __init__(self, hidden_size, num_header):
        super(MultiHeadEncoderDecoderAttention, self).__init__()
        assert hidden_size % num_header == 0, f"header的数目没办法整除:{hidden_size}, {num_header}"

        self.hidden_size = hidden_size  # 就是向量维度大小，也就是E
        self.num_header = num_header  # 头的数目

        self.wo = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU()
        )

    def split(self, vs):
        n, t, e = vs.shape
        vs = torch.reshape(vs, shape=(n, t, self.num_header, e // self.num_header))
        vs = torch.permute(vs, dims=(0, 2, 1, 3))
        return vs

    def forward(self, q, encoder_k, encoder_v, encoder_attention_mask, **kwargs):
        """
        编码器解码器attention
        :param q: [N,T1,E]
        :param encoder_k: [N,T2,E]
        :param encoder_v: [N,T2,E]
        :param encoder_attention_mask: [N,1,T2,T2]
        :return: [N,T1,E]
        """
        q = self.split(q)  # [n,t,e] --> [n,h,t,v]  e=h*v h就是head的数目，v就是每个头中self-attention的维度大小
        k = self.split(encoder_k)  # [n,t,e] --> [n,h,t,v]  e=h*v
        v = self.split(encoder_v)  # [n,t,e] --> [n,h,t,v]  e=h*v

        # 计算attention value
        v = qkv_attention_value(q, k, v, mask=encoder_attention_mask)

        # 5. 输出
        v = torch.permute(v, dims=(0, 2, 1, 3))  # [n,h,t,v] --> [n,t,h,v]
        n, t, _, _ = v.shape
        v = torch.reshape(v, shape=(n, t, -1))  # [n,t,h,v] -> [n,t,e]
        v = self.wo(v)  # 多个头之间的特征组合合并
        return v


class FFN(nn.Module):
    def __init__(self, hidden_size):
        super(FFN, self).__init__()

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )

    def forward(self, x, **kwargs):
        return self.ffn(x)


class ResidualsNorm(nn.Module):
    def __init__(self, block, hidden_size):
        super(ResidualsNorm, self).__init__()
        self.block = block
        self.norm = nn.LayerNorm(normalized_shape=hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x, **kwargs):
        z = self.block(x, **kwargs)
        z = self.relu(x + z)
        z = self.norm(z)
        return z


class TransformerEncoderLayers(nn.Module):
    def __init__(self, hidden_size, num_header, encoder_layers):
        super(TransformerEncoderLayers, self).__init__()

        layers = []
        for i in range(encoder_layers):
            layer = [
                ResidualsNorm(
                    block=MultiHeadSelfAttention(hidden_size=hidden_size, num_header=num_header),
                    hidden_size=hidden_size
                ),
                ResidualsNorm(
                    block=FFN(hidden_size=hidden_size),
                    hidden_size=hidden_size
                )
            ]
            layers.extend(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, x, attention_mask):
        attention_mask = torch.unsqueeze(attention_mask, dim=1)  # 增加header维度
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_header, max_seq_length, encoder_layers):
        super(TransformerEncoder, self).__init__()

        self.input_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.position_emb = nn.Embedding(num_embeddings=max_seq_length, embedding_dim=hidden_size)
        self.layers = TransformerEncoderLayers(hidden_size, num_header, encoder_layers)

    def forward(self, input_token_ids, input_position_ids, input_mask):
        """
        前向过程
        :param input_token_ids: [N,T] long类型的token id
        :param input_position_ids: [N,T] long类型的位置id
        :param input_mask: [N,T,T] float类型的mask矩阵
        :return:
        """
        # 1. 获取token的embedding
        inp_embedding = self.input_emb(input_token_ids)  # [N,T,E]

        # 2. 获取位置embedding
        position_embedding = self.position_emb(input_position_ids)

        # 3. 合并embedding
        emd = inp_embedding + position_embedding

        # 4. 输入到attention提取特征
        feat_emd = self.layers(emd, attention_mask=input_mask)

        return feat_emd


class TransformerDecoderLayers(nn.Module):
    def __init__(self, hidden_size, num_header, decoder_layers):
        super(TransformerDecoderLayers, self).__init__()

        self.wk = nn.Linear(hidden_size, hidden_size)
        self.wv = nn.Linear(hidden_size, hidden_size)

        layers = []
        for i in range(decoder_layers):
            layer = [
                ResidualsNorm(
                    block=MultiHeadSelfAttention(hidden_size=hidden_size, num_header=num_header),
                    hidden_size=hidden_size
                ),
                ResidualsNorm(
                    block=MultiHeadEncoderDecoderAttention(hidden_size=hidden_size, num_header=num_header),
                    hidden_size=hidden_size
                ),
                ResidualsNorm(
                    block=FFN(hidden_size=hidden_size),
                    hidden_size=hidden_size
                )
            ]
            layers.extend(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, x, encoder_outputs=None, encoder_attention_mask=None, attention_mask=None):
        """
        :param x: [N,T2,E]
        :param encoder_outputs: [N,T1,E]
        :param encoder_attention_mask: [N,1,T1]
        :param attention_mask: [N,T2,T2]
        :return:
        """
        attention_mask = torch.unsqueeze(attention_mask, dim=1)  # 增加header维度 [N,T2,T2] -> [N,1,T2,T2]
        encoder_attention_mask = torch.unsqueeze(encoder_attention_mask, dim=1)  # 增加header维度 [N,1,T1] -> [N,1,1,T1]
        k = self.wk(encoder_outputs)  # [N,T1,E]
        v = self.wv(encoder_outputs)  # [N,T1,E]

        for layer in self.layers:
            x = layer(
                x,
                encoder_k=k, encoder_v=v, encoder_attention_mask=encoder_attention_mask,
                attention_mask=attention_mask
            )
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_header, max_seq_length, decoder_layers):
        super(TransformerDecoder, self).__init__()

        self.input_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.position_emb = nn.Embedding(num_embeddings=max_seq_length, embedding_dim=hidden_size)
        self.layers = TransformerDecoderLayers(hidden_size, num_header, decoder_layers)

    def forward(self, input_token_ids, input_position_ids, input_mask, encoder_outputs, encoder_attention_mask):
        """
        前向过程
        :param input_token_ids: [N,T] long类型的token id
        :param input_position_ids: [N,T] long类型的位置id
        :param input_mask: [N,T,T] float类型的mask矩阵
        :param encoder_outputs: [N,T1,E] 编码器的输出状态信息
        :param encoder_attention_mask: [N,T1,T1] 编码器的输入mask信息
        :return:
        """
        if self.training:
            # 1. 获取token的embedding
            inp_embedding = self.input_emb(input_token_ids)  # [N,T,E]

            # 2. 获取位置embedding
            position_embedding = self.position_emb(input_position_ids)

            # 3. 合并embedding
            emd = inp_embedding + position_embedding

            # 4. 输入到attention提取特征
            feat_emd = self.layers(
                emd, encoder_outputs=encoder_outputs,
                encoder_attention_mask=encoder_attention_mask, attention_mask=input_mask
            )

            return feat_emd
        else:
            raise ValueError("当前模拟代码不实现推理过程，仅实现training过程")


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()

    def forward(self, encoder_input_ids, encoder_lengths, label_ids=None, label_lengths=None):
        """
        :param encoder_input_ids: 编码器输入token id: [N,T1]
        :param encoder_lengths: 编码器输入的文本实际长度值, [N,] 也可以直接外部传入mask等信息，但是至少要有代码实现从length到mask的转换
        :param label_ids: 模型预测期望输出的token id: [N,T2]
        :param label_lengths: 模型预测期望输出文本实际长度, [N,]
        :return:
        """
        pass


def t0():
    encoder = TransformerEncoder(vocab_size=1000, hidden_size=512, num_header=8, max_seq_length=1024, encoder_layers=6)
    decoder = TransformerDecoder(vocab_size=1000, hidden_size=512, num_header=8, max_seq_length=1024, decoder_layers=6)

    input_token_ids = torch.tensor([
        [100, 102, 108, 253, 125],  # 第一个样本实际长度为5
        [254, 125, 106, 0, 0]  # 第二个样本实际长度为3
    ])
    input_position_ids = torch.tensor([
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4]
    ])
    input_mask = torch.tensor([
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0, -10000.0, -10000.0],
            [0.0, 0.0, 0.0, -10000.0, -10000.0],
            [0.0, 0.0, 0.0, -10000.0, -10000.0],
            [-10000.0, -10000.0, -10000.0, 0.0, -10000.0],
            [-10000.0, -10000.0, -10000.0, -10000.0, 0.0],
        ],
    ])
    encoder_attention_mask = torch.tensor([
        [
            [0.0, 0.0, 0.0, 0.0, 0.0]  # 表示第一个样本的解码器中第一个时刻和编码器的各个时刻之间的mask值
        ],
        [
            [0.0, 0.0, 0.0, -10000.0, -10000.0]  # 是因为编码器的输入中，最后两个位置是填充
        ],
    ])

    input_decoder_token_ids = torch.tensor([
        [251, 235, 124, 321, 25, 68],
        [351, 235, 126, 253, 0, 0]
    ])
    input_decoder_position_ids = torch.tensor([
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5]
    ])
    input_decoder_mask = torch.tensor([
        [
            [0.0, -10000.0, -10000.0, -10000.0, -10000.0, -10000.0],
            [0.0, 0.0, -10000.0, -10000.0, -10000.0, -10000.0],
            [0.0, 0.0, 0.0, -10000.0, -10000.0, -10000.0],
            [0.0, 0.0, 0.0, 0.0, -10000.0, -10000.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, -10000.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ],
        [
            [0.0, -10000.0, -10000.0, -10000.0, -10000.0, -10000.0],
            [0.0, 0.0, -10000.0, -10000.0, -10000.0, -10000.0],
            [0.0, 0.0, 0.0, -10000.0, -10000.0, -10000.0],
            [0.0, 0.0, 0.0, 0.0, -10000.0, -10000.0],
            [-10000.0, -10000.0, -10000.0, -10000.0, 0.0, -10000.0],
            [-10000.0, -10000.0, -10000.0, -10000.0, -10000.0, 0.0]
        ],
    ])

    encoder_outputs = encoder(input_token_ids, input_position_ids, input_mask)
    print(encoder_outputs.shape)

    decoder_outputs = decoder(
        input_token_ids=input_decoder_token_ids,
        input_position_ids=input_decoder_position_ids,
        input_mask=input_decoder_mask,
        encoder_outputs=encoder_outputs,
        encoder_attention_mask=encoder_attention_mask
    )
    print(decoder_outputs.shape)


if __name__ == '__main__':
    t0()
