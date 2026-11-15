import math

import torch
import triton.language as tl

CUDA = "cuda"
# Every score and squared distance is at most (2 * norm) ** 2, which this bound keeps inside float32.
MAX_ROW_NORM = math.sqrt(torch.finfo(torch.float32).max / 4)
MATRIX_RANK = 2
SELECT_ROWS = 32
PAIR_ROWS = 64
MEMBER_TILE = 64
BUCKET_BLOCK = 1024
REDUCE_BLOCK = 1024
REFRESH_ROWS = 32
LLOYD_ROWS = 32
GROUP_BLOCK = 16
ACCUMULATE_ROWS = 32
EMA_ROWS = 32
DIM_CHUNK = 64
MIN_TILE = 16
MAX_GROUP_SPAN = 512
INT32_MAX = 2**31 - 1
UNREACHABLE = tl.constexpr(INT32_MAX)
RELAXED = tl.constexpr("relaxed")
