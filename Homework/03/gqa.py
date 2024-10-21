import torch
import torch.nn.functional as F

def scaled_dot_product_gqa(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, is_causal: bool = True, need_weights: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product attention in a grouped manner.
    Args:
        query (torch.Tensor): Query tensor of shape [batch size; seq len; num heads; hidden dim]
        key (torch.Tensor): Key tensor of shape [batch size; kv seq len; num kv heads; hidden dim]
        value (torch.Tensor): Value tensor of shape [batch size; kv seq len; num kv heads; hidden dim]
        is_causal (bool): Whether a causal mask of attention should be used
        need_weights (bool): Whether attention weights should be returned

    Returns:
        2-tuple of torch.Tensor:
            - Attention output with shape [batch size; seq len; num heads; hidden dim]
            - (Optional) Attention weights with shape [batch size; num heads; seq len; kv seq len].
                Only returned if 'need_weights' is True.
    """
    BATCH_SIZE, SEQ_LENGTH, N_HEADS, DIM_PER_HEAD = query.shape
    _, KV_SEQ_LEN, NUM_KV_HEADS, _ = key.shape

    if N_HEADS % NUM_KV_HEADS != 0:
        raise ValueError()

    scale = DIM_PER_HEAD ** -0.5
    query = query * scale

    head_ratio = N_HEADS // NUM_KV_HEADS
    if head_ratio > 1:
        key = key.repeat_interleave(head_ratio, dim=2)
        value = value.repeat_interleave(head_ratio, dim=2)

    attn_weights = torch.einsum("bqhd,bkhd->bhqk", query, key)

    if is_causal:
        mask = torch.tril(torch.ones(SEQ_LENGTH, KV_SEQ_LEN))
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

    attn_probs = F.softmax(attn_weights, dim=-1)

    attn_output = torch.einsum("bhqk,bkhd->bqhd", attn_probs, value)

    if need_weights:
        return attn_output, attn_probs
    else:
        return attn_output, None
