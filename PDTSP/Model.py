import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = self.model_params['embedding_dim']
        self.encoder = Encoder(**model_params)
        self.decoder = Decoder(**model_params)
        self.encoded_node = None
        # shape: (batch, node, embedding)
        self.encoded_pick = None
        # shape: (batch, customer, embedding)
        self.encoded_delivery = None
        # shape: (batch, customer, embedding)


    def pre_forward(self, reset_state, return_h_mean=False):
        depot_x_y = reset_state.depot_x_y
        pick_x_y = reset_state.pick_x_y
        delivery_x_y = reset_state.delivery_x_y
        self.encoded_node = self.encoder(depot_x_y, pick_x_y, delivery_x_y)
        self.decoder.set_k_v(encoded_node=self.encoded_node, reset_state=reset_state)


    def forward(self, state):
        # selected shape: (batch, mt)
        # probability shape: (batch, mt)
        batch_size = state.batch_idx.size(0)
        mt_size = state.mt_idx.size(1)
        if state.selected_count == 0:
            selected = torch.zeros(size=(batch_size, mt_size), dtype=torch.long)
            probability = torch.ones(size=(batch_size, mt_size))
        elif state.selected_count == 1:
            selected = torch.arange(1, mt_size + 1)[None, :].expand(batch_size, mt_size)
            probability = torch.ones(size=(batch_size, mt_size))
        else:
            prob = self.decoder(state)
            # shape: (batch, mt, node)
            if self.training or self.model_params['sample']:
                while True:
                    with torch.no_grad():
                        selected = prob.reshape(batch_size * mt_size, -1). \
                            multinomial(1).squeeze(dim=1).reshape(batch_size, mt_size)
                    probability = prob[state.batch_idx, state.mt_idx, selected].reshape(batch_size, mt_size)
                    if (probability != 0).all():
                        break
            else:
                selected = prob.argmax(dim=2)
                probability = None
        return selected, probability


class Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = self.model_params['embedding_dim']
        self.encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding_depot = nn.Linear(2, self.embedding_dim)
        self.embedding_pick = nn.Linear(2, self.embedding_dim)
        self.embedding_delivery = nn.Linear(2, self.embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(self.encoder_layer_num)])

    def forward(self, depot_x_y, pick_x_y, delivery_x_y):
        # depot_x_y: (batch, depot,2)
        # pick_x_y, delivery_x_y: (batch, customer, 2)
        embedding_depot = self.embedding_depot(depot_x_y)
        embedded_pick = self.embedding_pick(pick_x_y)
        embedded_delivery = self.embedding_delivery(delivery_x_y)
        embedded_node = torch.cat((embedding_depot, embedded_pick, embedded_delivery), dim=1)
        for layer in self.layers:
            embedded_node = layer(embedded_node)

        return embedded_node


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = self.model_params['embedding_dim']
        self.head_num = self.model_params['head_num']
        self.qkv_dim = self.model_params['qkv_dim']

        self.Wq_n = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk_n = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv_n = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wq_p = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk_p = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv_p = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wq_d = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk_d = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv_d = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(self.head_num * self.qkv_dim, self.embedding_dim)

        self.norm1 = Norm(**model_params)
        self.ff = FF(**model_params)
        self.norm2 = Norm(**model_params)

        self.norm_factor = 1 / math.sqrt(self.qkv_dim)

    def forward(self, embedded_node):
        batch = embedded_node.size(0)
        node = embedded_node.size(1)
        pick = (node - 1) // 2
        # shape: (batch, node, embedding)
        q_n = multi_head_qkv(self.Wq_n(embedded_node), head_num=self.head_num)
        k_n = multi_head_qkv(self.Wk_n(embedded_node), head_num=self.head_num)
        v_n = multi_head_qkv(self.Wv_n(embedded_node), head_num=self.head_num)
        # shape: (batch, head, node, qkv)

        embedded_pick = embedded_node[:, 1:1+pick, :].contiguous().view(batch, pick, self.embedding_dim)
        embedded_delivery = embedded_node[:, 1+pick:, :].contiguous().view(batch, pick, self.embedding_dim)

        q_p = multi_head_qkv(self.Wq_p(embedded_pick), head_num=self.head_num)
        k_p = multi_head_qkv(self.Wk_p(embedded_pick), head_num=self.head_num)
        v_p = multi_head_qkv(self.Wv_p(embedded_pick), head_num=self.head_num)
        # shape: (batch, head, pick, qkv)


        q_d = multi_head_qkv(self.Wq_d(embedded_delivery), head_num=self.head_num)
        k_d = multi_head_qkv(self.Wk_d(embedded_delivery), head_num=self.head_num)
        v_d = multi_head_qkv(self.Wv_d(embedded_delivery), head_num=self.head_num)
        # shape: (batch, head, delivery, qkv)

        v_d_addition = torch.cat([  # shape: (batch, head, node, qkv)
            torch.zeros(batch, self.head_num, 1, self.embedding_dim // self.head_num, dtype=v_n.dtype),
            v_d,
            torch.zeros(batch, self.head_num, pick, self.embedding_dim // self.head_num,
                        dtype=v_n.dtype)
        ], dim=2)

        v_p_addition = torch.cat([  # shape: (batch, head, node, qkv)
            torch.zeros(batch, self.head_num, 1, self.embedding_dim // self.head_num, dtype=v_n.dtype),
            torch.zeros(batch, self.head_num, pick, self.embedding_dim // self.head_num,
                        dtype=v_n.dtype),
            v_p
        ], dim=2)

        score_n = self.norm_factor * torch.matmul(q_n, k_n.transpose(2, 3))
        # shape: (batch, head, node, node)
        score_p_d1 = self.norm_factor * torch.sum(q_p * k_d, -1)
        # element_wise, (batch, head, pick)
        score_p = self.norm_factor * torch.matmul(q_p, k_p.transpose(2, 3))
        score_p_d = self.norm_factor * torch.matmul(q_p, k_d.transpose(2, 3))
        # shape: (batch, head, pick, pick)

        score_d_p1 = self.norm_factor * torch.sum(q_d * k_p, -1)
        # element_wise, (batch, head, pick)
        score_d = self.norm_factor * torch.matmul(q_d, k_d.transpose(2, 3))
        score_d_p = self.norm_factor * torch.matmul(q_d, k_p.transpose(2, 3))
        # shape: (batch, head, pick, pick)

        score_p_d1_addition = torch.cat([
            -np.inf * torch.ones(batch, self.head_num, 1, dtype=score_n.dtype),
            score_p_d1,
            -np.inf * torch.ones(batch, self.head_num, pick, dtype=score_n.dtype)
        ], dim=-1).view(batch, self.head_num, node, 1)

        score_p_addition = torch.cat([
            -np.inf * torch.ones(batch, self.head_num, 1, pick, dtype=score_n.dtype),
            score_p,
            -np.inf * torch.ones(batch, self.head_num, pick, pick, dtype=score_n.dtype)
        ], dim=2).view(batch, self.head_num, node, pick)

        score_p_d_addition = torch.cat([
            -np.inf * torch.ones(batch, self.head_num, 1, pick, dtype=score_n.dtype),
            score_p_d,
            -np.inf * torch.ones(batch, self.head_num, pick, pick, dtype=score_n.dtype)
        ], dim=2).view(batch, self.head_num, node, pick)
        score_d_p1_addition = torch.cat([
            -np.inf * torch.ones(batch, self.head_num, 1, dtype=score_n.dtype),
            -np.inf * torch.ones(batch, self.head_num, pick, dtype=score_n.dtype),
            score_d_p1,
        ], dim=-1).view(batch, self.head_num, node, 1)
        score_d_addition = torch.cat([
            -np.inf * torch.ones(batch, self.head_num, 1, pick, dtype=score_n.dtype),
            -np.inf * torch.ones(batch, self.head_num, pick, pick, dtype=score_n.dtype),
            score_d
        ], dim=2).view(batch, self.head_num, node, pick)
        score_d_p_addition = torch.cat([
            -np.inf * torch.ones(batch, self.head_num, 1, pick, dtype=score_n.dtype),
            -np.inf * torch.ones(batch, self.head_num, pick, pick, dtype=score_n.dtype),
            score_d_p
        ], dim=2).view(batch, self.head_num, node, pick)

        score = torch.cat([
            score_n,
            score_p_d1_addition,
            score_p_addition,
            score_p_d_addition,
            score_d_p1_addition,
            score_d_addition,
            score_d_p_addition
        ], dim=-1)
        # shape: (batch, head, node, 3 * node)
        attention = torch.softmax(score, dim=-1)
        sdpa =   torch.matmul(attention[:, :, :, :node], v_n) \
               + attention[:, :, :, node].view(batch, self.head_num, node, 1) * v_d_addition \
               + torch.matmul(attention[:, :, :, 1+node:1+node+pick], v_p) \
               + torch.matmul(attention[:, :, :, 1+node+pick:node*2], v_d) \
               + attention[:, :, :, node*2].view(batch, self.head_num, node, 1) * v_p_addition \
               + torch.matmul(attention[:, :, :, 1+node*2:1+node*2+pick], v_d) \
               + torch.matmul(attention[:, :, :, 1+node*2+pick:], v_p)

        out = self.multi_head_combine(sdpa.permute(0, 2, 1, 3).contiguous().view(batch, node, self.head_num * self.qkv_dim))
        out1 = self.norm1(embedded_node, out)
        out2 = self.ff(out1)
        out3 = self.norm2(out1, out2)
        return out3
        # shape :(batch, node, embedding)


class Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = self.model_params['embedding_dim']
        self.head_num = self.model_params['head_num']
        self.qkv_dim = self.model_params['qkv_dim']
        self.clip = self.model_params['clip']
        self.Wq_n = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk_n = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv_n = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wq_p = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk_p = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wq_d = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk_d = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(self.head_num * self.qkv_dim, self.embedding_dim)
        self.node_key = None
        # shape : (batch, embedding, node)
        self.q_n = None
        self.k_n = None
        self.v_n = None
        # shape : (batch, head, node, qkv)
        self.q_p = None
        self.k_p = None
        # shape: (batch, head, pick, qkv)
        self.q_d = None
        self.k_d = None
        # shape: (batch, head, delivery, qkv)
        self.encoded_node = None
        self.customer_size = None

    def set_k_v(self, encoded_node, reset_state):
        self.customer_size = reset_state.customer_size
        self.encoded_node = encoded_node
        encoded_pick = encoded_node[:, 1:1+self.customer_size, :].clone()
        encoded_delivery = encoded_node[:, 1+self.customer_size:, :].clone()
        self.k_n = multi_head_qkv(self.Wk_n(self.encoded_node), head_num=self.head_num)
        self.v_n = multi_head_qkv(self.Wv_n(self.encoded_node), head_num=self.head_num)
        self.k_p = multi_head_qkv(self.Wk_p(encoded_pick), head_num=self.head_num)
        self.k_d = multi_head_qkv(self.Wk_d(encoded_delivery), head_num=self.head_num)
        self.node_key = encoded_node.transpose(1, 2)

    def forward(self, state):
        q = get_encoding(encoded_node=self.encoded_node, state=state)
        self.q_n = multi_head_qkv(self.Wq_n(q), head_num=self.head_num)
        # attention_node = multi_head_attention(self.q_n, self.k_n, self.v_n, rank3_mask=state.mask)
        # shape: (batch, mt, head * qkv)
        self.q_p = multi_head_qkv(self.Wq_p(q), head_num=self.head_num)
        self.q_d = multi_head_qkv(self.Wq_d(q), head_num=self.head_num)
        attention_node = self.decoder_attention(
            self.q_n, self.k_n, self.v_n, self.q_p, self.k_p, self.q_d, self.k_d, rank3_mask=state.mask
        )
        score = self.multi_head_combine(attention_node)
        # shape: (batch, mt, embedding)
        score_mm = torch.matmul(score, self.node_key)
        # shape:(batch, mt, node)
        scale = self.embedding_dim ** (1 / 2)
        score_scaled = score_mm / scale
        score_clipped = self.clip * torch.tanh(score_scaled)
        score_masked = score_clipped + state.mask
        prob = F.softmax(score_masked, dim=2)
        return prob


    @staticmethod
    def decoder_attention(q_n, k_n, v_n, q_p, k_p, q_d, k_d, rank3_mask=None):
        mask = rank3_mask
        batch_size = q_n.size(0)
        mt = q_n.size(2)
        node = k_n.size(2)
        pick = k_p.size(2)
        delivery = k_d.size(2)
        head = q_n.size(1)
        qkv = q_n.size(3)

        score_n = torch.matmul(q_n, k_n.transpose(2, 3))
        score_n = score_n / torch.sqrt(torch.tensor(qkv, dtype=torch.float))
        # shape :(batch, head, mt, node)

        score_p = torch.matmul(q_p, k_p.transpose(2, 3))
        score_p = score_p / torch.sqrt(torch.tensor(qkv, dtype=torch.float))
        # shape :(batch, head, mt, pick)

        score_d = torch.matmul(q_d, k_d.transpose(2, 3))
        score_d = score_d / torch.sqrt(torch.tensor(qkv, dtype=torch.float))
        # shape :(batch, head, mt, delivery)

        if mask is not None:
            mask_p = mask[:, :, 1:1 + pick].clone()
            mask_d = mask[:, :, 1 + pick:].clone()
            score_n = score_n + mask[:, None, :, :].expand(batch_size, head, mt, node)
            # mask shape: (batch, mt, node)
            score_p = score_p + mask_p[:, None, :, :].expand(batch_size, head, mt, pick)
            # mask_p shape: (batch, mt, pick)
            score_d = score_d + mask_d[:, None, :, :].expand(batch_size, head, mt, delivery)
            # mask_d shape: (batch, mt, delivery)

        # weights_n = nn.Softmax(dim=3)(score_n)
        # shape :(batch, head, mt, node)
        # weights_p = nn.Softmax(dim=3)(score_p)
        # shape :(batch, head, mt, pick)
        # weights_d = nn.Softmax(dim=3)(score_d)
        # shape :(batch, head, mt, delivery)
        # weights = torch.cat([weights_n, weights_p, weights_d], dim=-1)
        # shape :(batch, head, mt, node + pick + delivery)
        score_cat = torch.cat([score_n, score_p, score_d], dim=-1)
        weights = nn.Softmax(dim=3)(score_cat)
        out = torch.matmul(weights[:, :, :, :node], v_n)
        # shape: (batch, head. n1, key)
        out_transposed = out.transpose(1, 2)
        # shape: (batch, n1, head, key)
        out_concat = out_transposed.reshape(batch_size, mt, head * qkv)
        # shape: (batch, n1, head * key)
        return out_concat


class Norm(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(self.embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # shape: (batch, node, embedding)
        input_added = input1 + input2
        # shape: (batch, node, embedding)
        input_transposed = input_added.transpose(1, 2)
        # shape: (batch, embedding, node)
        input_normed = self.norm(input_transposed)
        # shape: (batch, embedding, node)
        output_transposed = input_normed.transpose(1, 2)
        # shape: (batch, node, embedding)
        return output_transposed


class FF(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.embedding_dim = model_params['embedding_dim']
        self.ff_hidden_dim = model_params['ff_hidden_dim']
        self.W1 = nn.Linear(self.embedding_dim, self.ff_hidden_dim)
        self.W2 = nn.Linear(self.ff_hidden_dim, self.embedding_dim)

    def forward(self, input1):
        # shape: (batch, node, embedding)
        return self.W2(F.relu(self.W1(input1)))


def get_encoding(encoded_node, state):
    # encoded_node shape: (batch, node, embedding)
    # index_to_pick shape: (batch, mt)
    index_to_pick = state.current_node
    batch_size = index_to_pick.size(0)
    mt_size = index_to_pick.size(1)
    embedding_dim = encoded_node.size(2)
    index_to_gather = index_to_pick[:, :, None].expand(batch_size, mt_size, embedding_dim)
    # shape: (batch, mt, embedding)
    picked_node = encoded_node.gather(dim=1, index=index_to_gather)
    # shape: (batch, mt, embedding)
    return picked_node


def multi_head_qkv(qkv, head_num):
    # shape: (batch, n, embedding) : n can be 1 or node_size
    batch_size = qkv.size(0)
    n = qkv.size(1)
    qkv_multi_head = qkv.reshape(batch_size, n, head_num, -1)
    qkv_transposed = qkv_multi_head.transpose(1, 2)
    # shape: (batch, head, n, qkv)
    return qkv_transposed


def multi_head_attention(q, k, v, rank2_mask=None, rank3_mask=None):
    # q shape: (batch, head, n1, key)
    # k,v shape: (batch, head_num, n2, key)
    # rank2_mask shape: (batch, node)
    # rank3_mask shape: (batch, mt, node)
    batch_size = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    node_size = k.size(2)
    score = torch.matmul(q, k.transpose(2, 3))
    # shape :(batch, head, n1, n2)
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_mask is not None:
        score_scaled = score_scaled + rank2_mask[:, None, None, :].expand(batch_size, head_num, n, node_size)
    if rank3_mask is not None:
        score_scaled = score_scaled + rank3_mask[:, None, :, :].expand(batch_size, head_num, n, node_size)
    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch_size, head_num, n, node_size)
    out = torch.matmul(weights, v)
    # shape: (batch_size, head_num. n, key_dim)
    out_transposed = out.transpose(1, 2)
    # shape: (batch_size, n, head_num, key_dim)
    out_concat = out_transposed.reshape(batch_size, n, head_num * key_dim)
    # shape: (batch_size, n, head_num * key_dim)
    return out_concat

