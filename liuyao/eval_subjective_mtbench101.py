from opencompass.models import OpenAI, VLLM, VLLMwithChatTemplate
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import MTBench101Summarizer
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from mmengine.config import read_base
with read_base():
    from ..configs.datasets.subjective.multiround.mtbench101_judge import subjective_datasets


api_meta_template = dict(
    round=[
        dict(role='SYSTEM', api_role='SYSTEM'),
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]
)
qwen_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='<|im_start|>user\n', end='<|im_end|>\n'),
        dict(role="BOT", begin="<|im_start|>assistant\n", end='<|im_end|>\n', generate=True),
    ],
    eos_token_id=151645,
)
karakuri_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='[INST] ', end=' [ATTR] helpfulness: 4 correctness: 4 coherence: 4 complexity: 4 verbosity: 4 quality: 4 toxicity: 0 humor: 0 creativity: 0 [/ATTR] [/INST]'),
        dict(role="BOT", begin="", end='</s>', generate=True),
    ],
    eos_token_id=2,
)
suzume_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='<|start_header_id|>user<|end_header_id|>\n\n', end='<|eot_id|>'),
        dict(role="BOT", begin="<|start_header_id|>assistant<|end_header_id|>\n\n", end='<|eot_id|>', generate=True),
    ],
    eos_token_id=128009,
)
GPU_NUMS = 2


prefix = '/maindata/data/user/ai_story/yao.liu/multilingual/Japanese'
v1_6_path = f'{prefix}/Qwen2-57B-A14B-Instruct_SFT_SEQ4096_LR5e-6_EP4_GBS32x1x1_NEFT0_20240609_synthetic0530_common2/checkpoints/checkpoint-260'
v1_5_path = f'{prefix}/Qwen2-57B-A14B-Instruct_SFT_SEQ4096_LR5e-6_EP4_GBS32x1x1_NEFT0_20240609_synthetic0530_common1/checkpoints/checkpoint-260'
v1_3_path = f'{prefix}/Qwen2-57B-A14B-Instruct_SFT_SEQ4096_LR5e-6_EP4_GBS48x1x1_NEFT0_20240609_synthetic0530/checkpoints/checkpoint-1733'
v1_0_path = f'{prefix}/karakuri-lm-8x7b-chat-v0.1_SFT_SEQ4096_LR5e-6_EP4_GBS32x1x1_NEFT0_20240530_synthetic_0530/checkpoints/checkpoint-2597'
v0_3_path = f'{prefix}/suzume-llama-3-8B-japanese_SFT_SEQ4096_LR5e-6_EP4_GBS8x1x2_NEFT0_20240519_synthetic_0516/checkpoints/checkpoint-3146'


models = [
    dict(
        abbr='ja_v1_6',
        type=VLLM,
        path=v1_6_path,
        model_kwargs=dict(tensor_parallel_size=GPU_NUMS),
        meta_template=qwen_meta_template,
        generation_kwargs=dict(do_sample=True),         # For subjective evaluation, we often set do sample for models
        max_seq_len=4096,
        max_out_len=4096,
        batch_size=4,
        stop_words=['<|im_end|>'],
        run_cfg=dict(num_gpus=GPU_NUMS, num_procs=1),
    ),
    dict(
        abbr='ja_v1_5',
        type=VLLM,
        path=v1_5_path,
        model_kwargs=dict(tensor_parallel_size=GPU_NUMS),
        meta_template=qwen_meta_template,
        generation_kwargs=dict(do_sample=True),
        max_seq_len=4096,
        max_out_len=4096,
        batch_size=4,
        stop_words=['<|im_end|>'],
        run_cfg=dict(num_gpus=GPU_NUMS, num_procs=1),
    ),
    dict(
        abbr='ja_v1_3',
        type=VLLM,
        path=v1_3_path,
        model_kwargs=dict(tensor_parallel_size=GPU_NUMS),
        meta_template=qwen_meta_template,
        generation_kwargs=dict(do_sample=True),
        max_seq_len=4096,
        max_out_len=4096,
        batch_size=4,
        stop_words=['<|im_end|>'],
        run_cfg=dict(num_gpus=GPU_NUMS, num_procs=1),
    ),
    dict(
        abbr='ja_v1_0',
        type=VLLM,
        path=v1_0_path,
        model_kwargs=dict(tensor_parallel_size=GPU_NUMS),
        meta_template=karakuri_meta_template,
        generation_kwargs=dict(do_sample=True),
        max_seq_len=4096,
        max_out_len=4096,
        batch_size=4,
        stop_words=['</s>'],
        run_cfg=dict(num_gpus=GPU_NUMS, num_procs=1),
    ),
    dict(
        abbr='ja_v0_3',
        type=VLLM,
        path=v0_3_path,
        model_kwargs=dict(tensor_parallel_size=GPU_NUMS),
        meta_template=suzume_meta_template,
        generation_kwargs=dict(do_sample=True),
        max_seq_len=4096,
        max_out_len=4096,
        batch_size=4,
        stop_words=['<|eot_id|>'],
        run_cfg=dict(num_gpus=GPU_NUMS, num_procs=1),
    )
]
datasets = [x for x in subjective_datasets if x['abbr'] in ['mtbench101_ja']]


judge_models = [dict(
    abbr='GPT4-Turbo',
    type=OpenAI,
    path='gpt-4-1106-preview',
    key='xxx',
    meta_template=api_meta_template,
    query_per_second=16,
    max_out_len=4096,
    max_seq_len=4096,
    batch_size=8,
    temperature=0.8,
)]

infer = dict(
    partitioner=dict(
        type=SizePartitioner,
        max_task_size=10000
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=32,
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(
        type=SubjectiveSizePartitioner,
        max_task_size=10000,
        mode='singlescore',
        models=models,
        judge_models=judge_models
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,
        task=dict(type=SubjectiveEvalTask)
    ),
)

summarizer = dict(type=MTBench101Summarizer, judge_type='single')

work_dir = 'outputs/mtbench101/'