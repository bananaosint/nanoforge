from typing import Literal
from pydantic import BaseModel, Field, model_validator, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Source(StrictModel):
    name: str
    weight: float = Field(ge=0, le=100)


class Dimensions(StrictModel):
    n_layer: int | None = Field(default=None, ge=1, le=12)
    n_head: int | None = Field(default=None, ge=1, le=16)
    n_embd: int | None = Field(default=None, ge=16, le=768)
    block_size: int | None = Field(default=None, ge=8, le=512)
    dropout: float | None = Field(default=None, ge=0, le=0.8)

    @model_validator(mode='after')
    def heads_divide_width(self):
        if self.n_embd is not None and self.n_head is not None and self.n_embd % self.n_head:
            raise ValueError('Embedding width must be divisible by heads.')
        return self


class Training(StrictModel):
    mode: Literal['pretrain', 'sft', 'dpo'] = 'pretrain'
    precision: Literal['auto','float32','float16'] = 'auto'
    datasets: list[Source] = Field(default_factory=lambda: [Source(name='starter-sample.txt', weight=100)])
    max_steps: int = Field(default=250, ge=1, le=1000000)
    learning_rate: float = Field(default=0.0003, gt=0, le=0.1)
    batch_size: int = Field(default=8, ge=1, le=128)
    eval_interval: int = Field(default=50, ge=1, le=10000)
    gradient_accumulation: int = Field(default=1, ge=1, le=64)
    base_mix: float = Field(default=0.1, ge=0, le=0.2)
    beta: float = Field(default=0.1, gt=0, le=2)


class NewRun(Training):
    preset: Literal['Nano', 'Micro', 'Mini'] = 'Nano'
    tokenizer: Literal['char', 'bpe', 'magic8k'] = 'char'
    model: Dimensions | None = None


class Generation(StrictModel):
    prompt: str = Field(min_length=1, max_length=20000)
    max_new_tokens: int = Field(default=64, ge=1, le=256)
    temperature: float = Field(default=0.8, ge=0.01, le=3)
    top_k: int = Field(default=40, ge=0, le=50257)
    seed: int = Field(default=42, ge=0, le=2147483647)
    telemetry: bool = True
    layer: int | None = Field(default=None, ge=0)
    head: int | None = Field(default=None, ge=0)


class Arena(Generation):
    checkpoint_a: str | None = None
    checkpoint_b: str | None = None
    seed_b: int = Field(default=123, ge=0, le=2147483647)


class Feedback(StrictModel):
    kind: Literal['sft', 'preference']
    prompt: str = Field(min_length=1, max_length=20000)
    original_output: str = Field(default='', max_length=50000)
    corrected_output: str = Field(default='', max_length=50000)
    response_a: str = Field(default='', max_length=50000)
    response_b: str = Field(default='', max_length=50000)
    winner: Literal['a', 'b'] | None = None
    checkpoint_a: str | None = None
    checkpoint_b: str | None = None

    @model_validator(mode='after')
    def valid_pair(self):
        if not self.prompt.strip():
            raise ValueError('Prompt cannot be blank.')
        if self.kind == 'sft' and not self.corrected_output.strip():
            raise ValueError('A corrected completion is required.')
        if self.kind == 'preference' and (not self.response_a.strip() or not self.response_b.strip() or not self.winner):
            raise ValueError('Both responses and a winner are required.')
        return self


class CheckpointSelection(StrictModel):
    name: str


class Prompts(StrictModel):
    prompts: list[str] = Field(min_length=3, max_length=3)

    @model_validator(mode='after')
    def validate_prompts(self):
        if any(not p.strip() or len(p) > 2000 for p in self.prompts):
            raise ValueError('Provide three nonempty prompts of at most 2000 characters.')
        return self


class HubImportRequest(StrictModel):
    dataset_id: str = 'all'
    dataset_ids: list[str] | None = Field(default=None, min_length=1, max_length=32)
    max_records: int = Field(default=0, ge=0, le=1000000)
