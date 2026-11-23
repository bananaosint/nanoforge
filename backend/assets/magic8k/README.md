---
language:
- en
license: apache-2.0
library_name: transformers
tags:
- tokenizer
- bpe
- binary-analysis
- file-type-detection
- magic-bert
---

# Magic-BERT Tokenizer (8K vocabulary)

A byte-level BPE tokenizer trained on binary file data for the Magic-BERT project.
This tokenizer is designed for binary file classification and analysis tasks.

## Features

- **Vocabulary Size:** 8,192 tokens
- **Type:** Byte-level BPE (Byte Pair Encoding)
- **Training Data:** Binary files from diverse sources (executables, documents, archives, media files, etc.)
- **Encoding:** Latin-1 (ISO-8859-1) - treats each byte as a character

## Special Tokens

| Token | ID | Purpose |
|-------|-----|---------|
| `<\|start\|>` | 0 | Beginning of sequence (BOS) |
| `<\|end\|>` | 1 | End of sequence (EOS) |
| `<\|pad\|>` | 2 | Padding |
| `<\|unk\|>` | 3 | Unknown token |
| `<\|cls\|>` | 4 | Classification token |
| `<\|sep\|>` | 5 | Separator token |
| `<\|mask\|>` | 6 | Mask token (for MLM) |

## Usage

```python
from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("mjbommar/magic-bert-tokenizer-8k")

# Tokenize binary data (read as latin-1)
with open("some_file.bin", "rb") as f:
    binary_data = f.read()
text = binary_data.decode("latin-1")

# Encode
tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
print(tokens)
```

## Training Details

This tokenizer was trained using the HuggingFace `tokenizers` library with:
- Byte-level BPE algorithm
- Training on ~100K+ binary file samples
- Diverse file types: executables, documents, archives, media, source code, etc.

## Model

This tokenizer is designed to be used with Magic-BERT models for binary file
MIME type classification. See the main model repository for more details.

## License

Apache 2.0
