from typing import Optional, Any
import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer


def model_factory(config, data):
    task = config['task']
    feat_dim = data.feature_df.shape[1] - 1   # dimensionality of data features
    # data windowing is used when samples don't have a predefined length or the length is too long
    max_seq_len = config['data_window_len'] if config['data_window_len'] is not None else config['max_seq_len']
    if max_seq_len is None:
        try:
            max_seq_len = data.max_seq_len
        except AttributeError as x:
            print("Data class does not define a maximum sequence length, so it must be defined with the script argument `max_seq_len`")
            raise x

    # Pretraining
    if task == "imputation":
        return TSTransformerEncoder(feat_dim, max_seq_len, config['d_model'], config['num_heads'],
                                        config['num_layers'], config['dim_feedforward'], dropout=config['dropout'],
                                        pos_encoding=config['pos_encoding'], activation=config['activation'],
                                        norm=config['normalization_layer'], freeze=config['freeze'])

    if task == "classification":
        num_labels = len(data.class_names)# dimensionality of labels
        return TSTransformerEncoderClassifier(feat_dim, max_seq_len, config['d_model'],
                                                    config['num_heads'],
                                                    config['num_layers'], config['dim_feedforward'],
                                                    num_classes=num_labels,
                                                    dropout=config['dropout'], pos_encoding=config['pos_encoding'],
                                                    activation=config['activation'],
                                                    norm=config['normalization_layer'], freeze=config['freeze'])
    else:
        raise ValueError("Model class for task '{}' does not exist".format(task))


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


from itertools import combinations

def calculate_sleep_distances(X):
    """
    Args:
        X: (batch_size, seq_length, feat_dim) torch tensor of input features
    Returns:
        distances: (seq_length, batch_size, num_combinations + 1) tensor of distances
    """
    batch_size, seq_length, feat_dim = X.shape

    # Pairwise Euclidean distances
    num_combinations = feat_dim * (feat_dim - 1) // 2
    pair_distances = torch.zeros((batch_size, seq_length, num_combinations), device=X.device)
    comb_idx = 0
    for (f1, f2) in combinations(range(feat_dim), 2):
        diff = X[:, :, f1] - X[:, :, f2]
        dist = diff.pow(2)
        pair_distances[:, :, comb_idx] = dist
        comb_idx += 1

    # Sum of all distances
    all_distance = torch.sum(pair_distances, dim=2, keepdim=True)

    # Concatenate pairwise distances and sum of all distances
    distances = torch.cat((pair_distances, all_distance), dim=-1)
    distances = torch.sqrt(distances)
    distances = distances.permute(1, 0, 2)
    return distances

def min_max_scale(tensor): # 정규화 (사람, 피처별)

    seq_len, batch_size, num_features = tensor.size()
    mask = tensor != 0

    feature_mins = torch.where(mask, tensor, torch.tensor(float('inf'))).min(dim=0, keepdim=True).values
    feature_maxs = torch.where(mask, tensor, torch.tensor(float('-inf'))).max(dim=0, keepdim=True).values

    scaled_tensor = (tensor - feature_mins) / (feature_maxs - feature_mins)
    scaled_tensor[tensor == 0] = 0  # 0인 값은 그대로 유지

    return scaled_tensor
    
class SleepEncoding(nn.Module):

    def __init__(self, sleep_feat_dim, d_model, dropout=0, max_len=1024):
        super(SleepEncoding, self).__init__()
        self.sleep = nn.Parameter(torch.empty(1, sleep_feat_dim))   # requires_grad automatically set to True
        self.d_model = d_model
        nn.init.uniform_(self.sleep, -0.02, 0.02)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sleep_feat):
        r"""Inputs of forward function
        Args:
            x: the sequence after positional encoding
            sleep_feat: additional feature tensor
        Shape:
            x: [sequence length, batch size, d_model]
            sleep_feat: [sequence length, batch size, feat_dim]
            output: [sequence length, batch size, d_model]
        """
        # Validate dimensions
        seq_len, batch_size, _ = x.size()
        assert sleep_feat.size(0) == seq_len and sleep_feat.size(1) == batch_size, \
            "Sequence length and batch size must match between x and sleep_feat"

        # Expand self.sleep to match seq_len
        expanded_sleep = self.sleep.expand(seq_len, -1)

        # Compute weighted sum of sleep features
        weighted_sleep_feat = torch.einsum('sf,sbf->sb', expanded_sleep, sleep_feat)

        # Expand to match the last dimension of x
        weighted_sleep_feat = weighted_sleep_feat.unsqueeze(-1).expand(-1, -1, self.d_model)

        # Add to x and apply dropout
        x = x + weighted_sleep_feat
        return self.dropout(x)
    


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src


class TSTransformerEncoder(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoder, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.sleep_enc = SleepEncoding(5, self.d_model, dropout=0, max_len=self.max_len) #dh

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp_t = X.permute(1, 0, 2)
        inp = inp_t[:, :, :-1]
        sleep_time = inp_t[:, :, 3]
        
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding


        # Sleep Encoding
        # create feature
        X = X[:,:,:-1]
        sleep_distances = calculate_sleep_distances(X) # [seq_length, batch_size, sleep_distance_feat = pair_distance + all_distance]
        sleep_distances = min_max_scale(sleep_distances) # min-max scaling 적용
        sleep_time = sleep_time.unsqueeze(-1)
        sleep_feat = torch.cat((sleep_time, sleep_distances), dim=-1)
        # print("확인: ",sleep_feat.size(-1))        
        # sleep_enc = SleepEncoding(sleep_feat.size(-1), self.d_model, dropout=0, max_len=self.max_len)

        # add Sleep encoding
        inp = self.sleep_enc(inp,sleep_feat)

        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)

        return output


class TSTransformerEncoderClassifier(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoderClassifier, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.sleep_enc = SleepEncoding(5, self.d_model, dropout=0, max_len=self.max_len) #dh

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.output_layer = self.build_output_module(d_model, max_len, num_classes)

    def build_output_module(self, d_model, max_len, num_classes):
        output_layer = nn.Linear(d_model * max_len, num_classes)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer


    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp_t = X.permute(1, 0, 2)
        inp = inp_t[:, :, :-1]
        sleep_time = inp_t[:, :, 3]
        
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding


        # Sleep Encoding
        # create feature
        X = X[:,:,:-1]
        sleep_distances = calculate_sleep_distances(X) # [seq_length, batch_size, sleep_distance_feat = pair_distance + all_distance]
        sleep_distances = min_max_scale(sleep_distances) # min-max scaling 적용
        sleep_time = sleep_time.unsqueeze(-1)
        sleep_feat = torch.cat((sleep_time, sleep_distances), dim=-1)
        
        # add Sleep encoding
        inp = self.sleep_enc(inp,sleep_feat)
        # print(inp.size)

        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)

        # Output
        output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.output_layer(output)  # (batch_size, num_classes)

        return output

