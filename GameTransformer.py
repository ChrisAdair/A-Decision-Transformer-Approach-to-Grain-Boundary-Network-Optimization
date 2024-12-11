import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import Tensor
from typing import Mapping, Optional, Tuple
import numpy as np
import scipy

from multigame_dt_utils import (
    accuracy,
    autoregressive_generate,
    cross_entropy,
    decode_return,
    encode_return,
    encode_reward,
    sample_from_logits,
    variance_scaling_,
)

#From "Multi-Game Decision Transformers", 2022
class MLP(nn.Module):
    r"""A 2-layer MLP which widens then narrows the input."""

    def __init__(
        self,
        in_dim: int,
        init_scale: float,
        widening_factor: int = 4,
    ):
        super().__init__()
        self._init_scale = init_scale
        self._widening_factor = widening_factor

        self.fc1 = nn.Linear(in_dim, self._widening_factor * in_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(self._widening_factor * in_dim, in_dim)

        self.reset_parameters()

    def reset_parameters(self):
        variance_scaling_(self.fc1.weight, scale=self._init_scale)
        nn.init.zeros_(self.fc1.bias)
        variance_scaling_(self.fc2.weight, scale=self._init_scale)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

#Modified from "Multi-Game Decision Transformers", 2022
class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        w_init_scale: Optional[float] = None,
        qkv_bias: bool = True,
        proj_bias: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.w_init_scale = w_init_scale

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.reset_parameters()

    def reset_parameters(self):
        variance_scaling_(self.qkv.weight, scale=self.w_init_scale)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        variance_scaling_(self.proj.weight, scale=self.w_init_scale)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x, mask: Optional[Tensor] = None) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask_value = -torch.finfo(attn.dtype).max  # max_neg_value
            attn = attn.masked_fill(~mask.to(dtype=torch.bool), mask_value)

        self.attnScores = attn
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x

#Modified from "Multi-Game Decision Transformers", 2022
class CausalSelfAttention(Attention):
    r"""Self attention with a causal mask applied."""

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        custom_causal_mask: Optional[Tensor] = None,
        prefix_length: Optional[int] = 0,
    ) -> Tensor:
        if x.ndim != 3:
            raise ValueError("Expect queries of shape [B, T, D].")

        seq_len = x.shape[1]
        # If custom_causal_mask is None, the default causality assumption is
        # sequential (a lower triangular causal mask).
        causal_mask = custom_causal_mask
        if causal_mask is None:
            device = x.device
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
        causal_mask = causal_mask[None, None, :, :]

        # Similar to T5, tokens up to prefix_length can all attend to each other.
        causal_mask[:, :, :, :prefix_length] = 1
        mask = mask * mask.permute(0,1,3,2) * causal_mask if mask is not None else causal_mask #Change from original: Testing a different masking for grain numbers

        return super().forward(x, mask)

#From "Multi-Game Decision Transformers", 2022
class Block(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, init_scale: float, dropout_rate: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads=num_heads, w_init_scale=init_scale)
        self.dropout_1 = nn.Dropout(dropout_rate)

        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, init_scale)
        self.dropout_2 = nn.Dropout(dropout_rate)

    def forward(self, x, **kwargs):
        x = x + self.dropout_1(self.attn(self.ln_1(x), **kwargs))
        x = x + self.dropout_2(self.mlp(self.ln_2(x)))
        return x

#From "Multi-Game Decision Transformers", 2022
class Transformer(nn.Module):
    r"""A transformer stack."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float,
    ):
        super().__init__()
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._dropout_rate = dropout_rate

        init_scale = 2.0 / self._num_layers
        self.layers = nn.ModuleList([])
        for _ in range(self._num_layers):
            block = Block(embed_dim, num_heads, init_scale, dropout_rate)
            self.layers.append(block)
        self.norm_f = nn.LayerNorm(embed_dim)

    def forward(
        self,
        h: Tensor,
        mask: Optional[Tensor] = None,
        custom_causal_mask: Optional[Tensor] = None,
        prefix_length: Optional[int] = 0,
    ) -> Tensor:
        r"""Connects the transformer.

        Args:
        h: Inputs, [B, T, D].
        mask: Padding mask, [B, T].
        custom_causal_mask: Customized causal mask, [T, T].
        prefix_length: Number of prefix tokens that can all attend to each other.

        Returns:
        Array of shape [B, T, D].
        """
        if mask is not None:
            # Make sure we're not passing any information about masked h.
            h = h * mask[:, :, None]
            mask = mask[:, None, None, :]

        for block in self.layers:
            h = block(
                h,
                mask=mask,
                custom_causal_mask=custom_causal_mask,
                prefix_length=prefix_length,
            )
        h = self.norm_f(h)
        return h

def pos_encoding(quaternions, dual_laplacian, num_encode, n_grains): #Change this function to take a sequence of GBNs

    #weighting function

    #construct laplacians


    #extract first 4 eigenvectors as position encoding
    [], encoding = torch.linalg.eigh(dual_laplacian)
    encoding = encoding[:,1:num_encode]
    return encoding

#Model for Grain Boundary Network optimization
class GameTransformer(nn.Module):

    def __init__(
            self,
            max_grains,
            act_dim,
            hidden_size:int,
            max_ep_length,
            n_eigs,
            **kwargs
    ):
        super().__init__()

        self.max_grains = max_grains
        self.act_dim = act_dim
        self.state_dim = 4
        self.max_ep_length = max_ep_length
        self.hidden_size = hidden_size
        self.n_eigs = n_eigs
        self.tokens_per_timestep = max_grains*2+1 #action for each grain means n_grains times 2
        if 'stateLoss' in kwargs:
            self.state_loss = True
        else:
            self.state_loss = False
        # Laplacian embedding for state input
        self.embed_pos = nn.Linear(n_eigs,hidden_size)
        self.attention = Transformer(embed_dim=hidden_size, num_heads=1, num_layers=1, dropout_rate=0.1)
        # State is already in [Batch x Timestep x Grain x Dimensionality] form, where D is the quaternion embedding of the rotation

        #Taken from decision transformer
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_action = nn.Embedding(self.act_dim, hidden_size)
        self.embed_timestep = nn.Embedding(max_ep_length, hidden_size)
        self.embed_state = nn.Linear(4,hidden_size)
        self.embed_eig = nn.Linear(n_eigs,hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        #self.predict_action = nn.Sequential(
        #    *([nn.Linear(hidden_size, self.act_dim)]+[nn.Tanh()])
        #)
        self.predict_action = nn.Linear(hidden_size, act_dim)
        self.predict_state = nn.Linear(hidden_size, self.state_dim)
        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, actions, rewards, posEncode, timesteps, masks=None, attention_mask=None, n_g=None):

        batch_size, num_steps = rewards.shape[0], rewards.shape[1]
        if n_g is not None:
            n_grains = n_g
        else:
            n_grains = states.shape[2]
        #pad_size = self.max_grains - n_grains
        #assert pad_size >= 0
        # Pad the inputs to the max grain number (Prepend) MOVE TO BATCHING AS WELL AS MASKS
        #states = torch.cat([torch.zeros(states.shape[0], states.shape[1], pad_size, 4, device=states.device), states],dim=2).to(dtype=torch.float32)
        #actions = torch.cat([torch.zeros(states.shape[0], states.shape[1], pad_size, device=actions.device), actions], dim=2).to(dtype=torch.int)
        #posEncode = torch.cat([torch.zeros(states.shape[0], states.shape[1], pad_size, 4, device=states.device), posEncode],dim=2).to(dtype=torch.float32)
        # Embed the imputs
        
        returns_embeddings = self.embed_return(rewards)
        time_embeddings = self.embed_timestep(timesteps)

        pos_embeddings = self.embed_pos(posEncode)

        action_embeddings = self.embed_action(actions) + pos_embeddings
        state_embeddings = self.embed_state(states) + pos_embeddings
        tokens_per_step = state_embeddings.shape[2]*2 + 1
        num_obs_tokens = state_embeddings.shape[2]
        state_emb = torch.reshape(state_embeddings, state_embeddings.shape[:2] + (-1,))
        act_emb = torch.reshape(action_embeddings, action_embeddings.shape[:2] + (-1,))
        token_emb = torch.cat([state_emb, returns_embeddings, act_emb], dim=-1)
        device = state_embeddings.device
        # sequence is [obs ret act ... obs ret act]
        token_emb = torch.reshape(token_emb, [batch_size, tokens_per_step * num_steps, self.hidden_size])
        # Create time embeddings
        token_emb = token_emb + time_embeddings
        # Mask padding
        batch_size = token_emb.shape[0]
        #obs_mask = np.concatenate([np.zeros([states.shape[0], states.shape[1], pad_size], dtype=bool), np.ones([batch_size, num_steps, n_grains], dtype=bool)], axis=2) 
        #ret_mask = np.ones([batch_size, num_steps, 1], dtype=bool)
        #act_mask = np.concatenate([np.zeros([actions.shape[0], actions.shape[1], pad_size], dtype=bool), np.ones([batch_size, num_steps, n_grains], dtype=bool)], axis=2)

        #mask = [obs_mask, ret_mask, act_mask]
        #mask = np.concatenate(mask, axis=-1)
        #mask = np.reshape(mask, [batch_size, tokens_per_step * num_steps])
        #mask = torch.tensor(mask, dtype=torch.bool, device=device)

        # Attention mask needs block attention since all observations in a timestep should attend each other
        seq_len = token_emb.shape[1]
        sequential_causal_mask = np.tril(np.ones((seq_len, seq_len)))

        num_timesteps = seq_len // tokens_per_step
        num_non_obs_tokens = tokens_per_step - num_obs_tokens
        diag = [
            np.array(0) if i % 3 == 1 else np.ones((num_obs_tokens, num_obs_tokens)) for i in range(num_timesteps * 3)
        ]
        
        block_diag = scipy.linalg.block_diag(*diag)
        custom_causal_mask = np.logical_or(sequential_causal_mask, block_diag)
        custom_causal_mask = torch.tensor(custom_causal_mask, dtype=torch.bool, device=device)
        #masks[:,-self.max_grains:] = 0 # Testing if masking the last action input helps training
        # Send inputs to attention block
        output_emb = self.attention(token_emb, masks, custom_causal_mask)

        #De-interleaving the outputs into state, return, actions
        output_emb = torch.reshape(output_emb, [batch_size, -1, tokens_per_step, self.hidden_size])
        #Get next step predictions
        return_preds = output_emb[:, :, num_obs_tokens-1, :] #This I'm unsure about but follows the logic of multigame_dt
        state_preds = output_emb[:, :, num_obs_tokens+1:, :] #Predicts next state from next action
        action_preds = output_emb[:, :, :num_obs_tokens, :] #Predicts from tokens of past state
        
        # Project to appropriate dimensionality.
        return_preds = self.predict_return(return_preds)
        action_preds = self.predict_action(action_preds)
        state_preds = self.predict_state(state_preds)

        #Take the n_grains out of padding
        seq_loss = self.sequence_loss(actions, action_preds, rewards, return_preds, states, state_preds, n_grains)
        seq_acc = self.sequence_accuracy(actions, action_preds, rewards, return_preds, states, state_preds, n_grains)
        return state_preds, action_preds, return_preds, seq_loss, seq_acc

    def sequence_loss(self, act_in, act_pred, return_in, return_pred, state_in, state_pred, n_grains) -> Tensor:
        r"""Compute the loss on data wrt model outputs."""
        act_target = act_in
        ret_target = return_in
        act_logits = act_pred
        ret_logits = return_pred
        #if self.state_loss:
        #    stateLoss = nn.L1Loss()
        act_loss = []
        ret_loss = []
        for i in range(len(n_grains)):
            act_loss.append(cross_entropy(act_logits[i,:,-n_grains[i]:,:], act_target[i,:,-n_grains[i]:]))
        act_loss = torch.sum(torch.stack(act_loss))
        ret_loss = torch.mean((ret_target-ret_logits)**2)
        totalLoss = act_loss + ret_loss
        return totalLoss

    def sequence_accuracy(self, act_in, act_pred, return_in, return_pred, state_in, state_pred, n_grains) -> Tensor:
        r"""Compute the accuracy on data wrt model outputs."""
        act_target = act_in
        ret_target = return_in
        act_logits = act_pred
        ret_logits = return_pred
        obj_pairs = [(act_logits[i,:,-n_grains[i]:,:], act_target[i,:,-n_grains[i]:]) for i in range(len(n_grains))]
        if self.state_loss:
            obj_pairs.append((state_in,state_pred))
        obj = [accuracy(logits, target) for logits, target in obj_pairs]
        #obj.append(torch.clamp(1-torch.sum(torch.square(ret_logits-ret_target))/torch.sum(torch.square(torch.mean(ret_target)-ret_target)),min=0)) #Wrong accuracy type
        obj.append(torch.mean(1-torch.abs((ret_target-ret_logits)/ret_target)))
        return sum(obj) / len(obj)
    

