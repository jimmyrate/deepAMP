import math
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from configer import BertConfig

logger = logging.getLogger(__name__)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask, d_k):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self,config):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model = config.d_model
        self.d_k = d_k = config.d_k
        self.d_v = d_v = config.d_v
        self.n_heads = n_heads = config.n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        # self.ln = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        d_model = self.d_model
        d_k = self.d_k
        d_v = self.d_v
        n_heads = self.n_heads
        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size, n_heads, seq_len, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context, attention_weight = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask, d_k)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size, seq_len, n_heads, d_v]
        output = self.linear(context)
        # return self.ln(output + residual) # output: [batch_size, seq_len, d_model]
        return output + residual  # output: [batch_size, seq_len, d_model]

    def get_attention(self, Q, K, V, attn_mask):
        d_model = self.d_model
        d_k = self.d_k
        d_v = self.d_v
        n_heads = self.n_heads
        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # v_s: [batch_size, n_heads, seq_len, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context, attention_weight = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask, d_k)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            n_heads * d_v)  # context: [batch_size, seq_len, n_heads, d_v]
        output = self.linear(context)
        # return self.ln(output + residual) # output: [batch_size, seq_len, d_model]
        return output + residual, attention_weight

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,config):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(self.gelu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.enc_self_attn = MultiHeadAttention(config)
        self.pos_ffn = PoswiseFeedForwardNet(config)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_inputs = self.ln1(enc_inputs)
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.ln2(enc_outputs)
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs

    def get_attention(self,  enc_inputs, enc_self_attn_mask):
        enc_inputs = self.ln1(enc_inputs)
        enc_outputs, attention_weight = self.enc_self_attn.get_attention(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.ln2(enc_outputs)
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs, attention_weight


class AMPBERT(nn.Module):
    def __init__(self, config:BertConfig):
        super(AMPBERT, self).__init__()
        self.block_size = config.block_size
        self.tok_embed = nn.Embedding(config.vocab_size, config.n_embd)  # token embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd)) # position embedding
        self.norm = nn.LayerNorm(config.n_embd)
        self.fc = nn.Linear(config.n_embd, config.d_model)

        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.linear = nn.Linear(config.d_model, config.d_model)
        self.activ2 = nn.GELU()
        self.drop = nn.Dropout(0.3)
        # fc2 is shared with embedding layer
        # embed_weight = self.tok_embed.weight
        self.fc2 = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # self.fc2.weight = embed_weight
        self.apply(self._init_weights)

    def forward(self, masked_seqs, masked_pos, masked_tokens=None):
        b, t = masked_seqs.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        embedding = self.tok_embed(masked_seqs) + self.pos_embed[:, :t, :]
        output = self.fc(self.norm(embedding))                                  # [bach_size, seq_len, d_model]
        enc_self_attn_mask = self.get_attn_pad_mask(masked_seqs, masked_seqs) # [batch_size, seq_len, seq_len]
        for layer in self.layers:
            # output: [batch_size, max_len, d_model]
            output = layer(output, enc_self_attn_mask)

        output = self.ln_f(output)
        _, __, d_model = output.size()
        masked_pos = masked_pos[:, :, None].expand(-1, -1, d_model) # [batch_size, max_pred, d_model]
        h_masked = torch.gather(output, 1, masked_pos) # masking position [batch_size, max_pred, d_model]
        h_masked = self.drop(h_masked)
        h_masked = self.activ2(self.linear(h_masked)) # [batch_size, max_pred, d_model]
        logits_lm = self.fc2(h_masked) # [batch_size, max_pred, vocab_size]

        loss = None
        if masked_tokens is not None:
            loss = F.cross_entropy(logits_lm.view(-1, logits_lm.size(-1)), masked_tokens.view(-1))
        return logits_lm, loss

    def get_feature(self, masked_seqs, masked_pos, masked_tokens=None):
        b, t = masked_seqs.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        embedding = self.tok_embed(masked_seqs) + self.pos_embed[:, :t, :]
        output = self.fc(self.norm(embedding))  # [bach_size, seq_len, d_model]
        enc_self_attn_mask = self.get_attn_pad_mask(masked_seqs, masked_seqs)  # [batch_size, seq_len, seq_len]
        all_attention = []
        for layer in self.layers:
            # output: [batch_size, max_len, d_model]
            output, attention_weight = layer.get_attention(output, enc_self_attn_mask)
            all_attention.append(attention_weight)

        output = self.ln_f(output)
        _, __, d_model = output.size()
        masked_pos = masked_pos[:, :, None].expand(-1, -1, d_model)  # [batch_size, max_pred, d_model]
        h_masked = torch.gather(output, 1, masked_pos)  # masking position [batch_size, max_pred, d_model]
        h_masked = self.drop(h_masked)
        h_masked = self.activ2(self.linear(h_masked))  # [batch_size, max_pred, d_model]
        logits_lm = self.fc2(h_masked)  # [batch_size, max_pred, vocab_size]

        loss = None
        if masked_tokens is not None:
            loss = F.cross_entropy(logits_lm.view(-1, logits_lm.size(-1)), masked_tokens.view(-1))
        return output, all_attention


    def get_attn_pad_mask(self, seq_q, seq_k):
        device = seq_q.device
        batch_size, seq_len = seq_q.size()
        # eq(zero) is PAD token
        pad_attn_mask = seq_q.data.eq(0).unsqueeze(1).to(device)  # [batch_size, 1, seq_len]
        return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

