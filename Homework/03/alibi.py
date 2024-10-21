import torch
import numpy as np

def compute_alibi(num_heads: int, seq_len: int) -> torch.Tensor:

    m_0 = 2.0 ** (-8.0 / num_heads)

    m = np.power(m_0, np.arange(1, 1 + num_heads))

    alibi = torch.zeros(num_heads, seq_len, seq_len)
    
    for head in range(num_heads):

        M = m[head]

        for i in range(seq_len):

            for j in range(seq_len):

                if i < j:
                    alibi[head, i, j] = (j - i) * M
                    alibi[head, j, i] = -alibi[head, i, j] 

    return alibi


if __name__ == "__main__":
    bias = compute_alibi(4, 4)
    print(bias)
