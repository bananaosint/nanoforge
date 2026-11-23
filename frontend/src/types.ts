export type Source = {name:string;weight:number}
export type ModelConfig = {n_layer:number;n_head:number;n_embd:number;block_size:number;dropout:number;vocab_size?:number}
export type Metric = {step:number;train_loss:number;val_loss:number|null;steps_per_sec:number;tokens_per_sec:number;mode:string}
export type Status = {imported_pairs?:Record<string,{sft:number;dpo:number}>;state:string;device:string;device_name?:string;precision?:string;step:number;target_step?:number|null;mode:string;run_id:string|null;checkpoint:string|null;error:string|null;val_loss:number|null;parameters:number;model_config:ModelConfig|null;config:Record<string,unknown>;metrics:Metric[];prompts:string[]}
export type Checkpoint = {name:string;step:number;val_loss:number|null;mode:string;parameters:number;created_at:string;version_tag:string}
export type Feedback = {id:number;kind:'sft'|'preference';prompt:string;original_output:string;corrected_output:string;response_a:string;response_b:string;winner:'a'|'b'|null;checkpoint_a:string|null;checkpoint_b:string|null;created_at:string}
export type Generation = {text:string;tokens:string[];steps:{token_id:number;token:string;top5:{id:number;token:string;probability:number}[]}[];attention:number[][][][];context_tokens:string[];checkpoint:string|null;seed:number}
export type Timeline = {id:number;run_id:string;step:number;prompt:string;output:string}
export type Dataset = {name:string;bytes:number}

export type Runtime = {environment:'local'|'colab';label:string;transport:'websocket'|'poll';defaults:Record<string,unknown>|null}
