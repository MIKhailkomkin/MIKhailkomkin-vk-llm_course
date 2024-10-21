import torch
import torch.nn.functional as F
import numpy as np


def compute_attention(queries, keys, values) -> torch.Tensor:
    """
    queries- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    keys- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    values- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    """
    BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM = queries.shape

    d = HIDDEN_DIM ** 0.5
    
    
    A = torch.bmm(queries, keys.transpose(1, 2)) / d  
    A = torch.softmax(A, dim=-1)  
    

    H = torch.bmm(A, values)  
    return H


def compute_multihead_attention(queries, keys, values, projection_matrix) -> torch.Tensor:
    """
    queries- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    keys- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    values- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    projection_matrix- (N_HEADS*DIM_PER_HEAD, N_HEADS*DIM_PER_HEAD)
    """
    BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD = queries.shape

    torch_list = []
    for i in range(N_HEADS):
        H = compute_attention(queries[:, i, :, :], 
                              keys[:, i, :, :], 
                              values[:, i, :, :])
        torch_list.append(H)
      
    multihead_attention = torch.cat(torch_list, dim=2)
    

    multihead_attention = torch.matmul(multihead_attention, projection_matrix.T)
    return multihead_attention


def compute_rotary_embeddings(x: torch.Tensor) -> torch.Tensor:
    """
    x - (BATCH_SIZE, SEQ_LENGTH, N_HEADS, DIM_PER_HEAD)
    Применение RoPE к тензору x
    """
    BATCH_SIZE, SEQ_LENGTH, N_HEADS, DIM_PER_HEAD = x.shape

    theta = torch.arange(0, DIM_PER_HEAD // 2, dtype=torch.float32) 
    theta = 1.0 / (10000 ** (theta / (DIM_PER_HEAD // 2)))

    seq_idx = torch.arange(SEQ_LENGTH, dtype=torch.float32)  
    angles = torch.einsum('i,j->ij', seq_idx, theta)  

    cos_emb = torch.cos(angles).unsqueeze(0).unsqueeze(2)  
    sin_emb = torch.sin(angles).unsqueeze(0).unsqueeze(2) 

    x1 = x[..., ::2]  
    x2 = x[..., 1::2]  

    rotated_x1 = x1 * cos_emb - x2 * sin_emb
    rotated_x2 = x1 * sin_emb + x2 * cos_emb
    result = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)

    return result
