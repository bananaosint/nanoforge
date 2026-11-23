# Tokenizers

Choose a tokenizer before creating a new experiment:

- **Character-level** builds and saves the selected corpus vocabulary.
- **GPT-2 BPE** uses tiktoken's 50,257-token vocabulary; its assets are bundled.
- **Magic BPE · 8,192 tokens** uses `mjbommar/magic-bert-tokenizer-8k`.

Despite its repository name, Magic's actual tokenizer JSON defines a byte-level **BPE** model, with no normalizer and no BERT post-processor. NanoForge uses that exact JSON through Hugging Face's lightweight `tokenizers` runtime, without Transformers, pretrained model weights or automatic CLS/SEP injection. Tokenization calls `add_special_tokens=False`; decoding retains explicit special tokens. Original character and GPT-2 checkpoint formats remain supported.

Magic assets are bundled at revision `b00d91a6be28c97dea4d11255de3bc95dd8c6aa7`. The asset's source, SHA-256 and Apache 2.0 license are under `backend/assets/magic8k/`. No network request is needed to create or restore a Magic tokenizer. Each checkpoint embeds its entire tokenizer JSON, so future asset changes cannot silently relabel an older model's token IDs.

Magic's smaller embedding matrix reduces parameter count compared with GPT-2 BPE at the same hidden width. It changes the vocabulary, not the decoder-only GPT architecture. Inspect the status bar for the actual parameter count. Preset estimates remain estimates for a small character vocabulary.

All three tokenizers implement encode, decode and per-token display. A byte-level token can be only part of a UTF-8 character; its individual display can contain a replacement character while decoding the complete token sequence correctly reconstructs Unicode. Candidate probabilities correspond to the generated token IDs.

Caches use uint16 for vocabularies of at most 65,536 IDs and uint32 otherwise. A custom vocabulary exceeding that boundary is not truncated. BPE cache preparation uses bounded text chunks; merges at a chunk boundary may differ from tokenizing a complete multi-GB string at once, but no text is dropped. JSONL formatting preserves complete prompt/completion delimiters per record.
