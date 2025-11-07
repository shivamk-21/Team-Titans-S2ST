# Ablation Study Results

## For Hindi

| Encoder Layers | Attention Heads | Precision | BLEU Score |
|----------------|-----------------|-----------|------------|
| 6              | 2               | Mixed     | 17.412 `Original` |
| 4              | 2               | Mixed     | 15.900 |
| 8 | 2 | Mixed | 16.200 |
| 6 | 2 | F32 | OOM Crash |

## For Malayalam

| Encoder Layers | Attention Heads | Precision | BLEU Score |
|----------------|-----------------|-----------|------------|
| 6 | 2 | Mixed | 13.456 `Original`|
| 4 | 2 | Mixed | 12.100 |

## For Marathi

| Encoder Layers | Attention Heads | Precision | BLEU Score |
|----------------|-----------------|-----------|------------|
| 6 | 2 | Mixed | 15.196 `Original`|
| 4 | 2 | Mixed | 13.462 |