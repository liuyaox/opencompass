from opencompass.models import OpenAI, VLLM, VLLMwithChatTemplate
from opencompass.models.template import api_meta_template, qwen_meta_template
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import MTBench101Summarizer
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from mmengine.config import read_base
with read_base():
    from ..configs.datasets.subjective.multiround.mtbench101_judge import subjective_datasets


work_dir = 'outputs/mtbench101_id/'
GPU_NUMS = 1
GPU_NUMS2 = 2

prefix = '/maindata/data/user/ai_story/yao.liu/multilingual/Indonesian'
v0_3_ep1 = f'{prefix}/Sailor-7B-Chat_SFT_SEQ4096_LR5e-6_EP4_GBS8x1x2_20240628_expand_trans_write_chatedit/checkpoints/checkpoint-471'   # 最小loss
v0_3_ep2 = f'{prefix}/Sailor-7B-Chat_SFT_SEQ4096_LR5e-6_EP4_GBS8x1x2_20240628_expand_trans_write_chatedit/checkpoints/checkpoint-943'

v0_4_ep1 = f'{prefix}/Sailor-7B-Chat_SFT_SEQ4096_LR5e-6_EP4_GBS8x1x2_20240628_trans151_write0422_chatedit0508/checkpoints/checkpoint-13'
v0_4_ep2 = f'{prefix}/Sailor-7B-Chat_SFT_SEQ4096_LR5e-6_EP4_GBS8x1x2_20240628_trans151_write0422_chatedit0508/checkpoints/checkpoint-27'    # 最小loss
v0_4_ep3 = f'{prefix}/Sailor-7B-Chat_SFT_SEQ4096_LR5e-6_EP4_GBS8x1x2_20240628_trans151_write0422_chatedit0508/checkpoints/checkpoint-40'
v0_4_ep4 = f'{prefix}/Sailor-7B-Chat_SFT_SEQ4096_LR5e-6_EP4_GBS8x1x2_20240628_trans151_write0422_chatedit0508/checkpoints/checkpoint-52'

v0_5_ep1 = f'{prefix}/Sailor-7B-Chat_SFT_SEQ4096_LR1e-5_EP4_GBS8x1x2_20240628_trans151_write0422_chatedit0508/checkpoints/checkpoint-13'
v0_5_ep2 = f'{prefix}/Sailor-7B-Chat_SFT_SEQ4096_LR1e-5_EP4_GBS8x1x2_20240628_trans151_write0422_chatedit0508/checkpoints/checkpoint-27'    # 最小loss
v0_5_ep3 = f'{prefix}/Sailor-7B-Chat_SFT_SEQ4096_LR1e-5_EP4_GBS8x1x2_20240628_trans151_write0422_chatedit0508/checkpoints/checkpoint-40'
v0_5_ep4 = f'{prefix}/Sailor-7B-Chat_SFT_SEQ4096_LR1e-5_EP4_GBS8x1x2_20240628_trans151_write0422_chatedit0508/checkpoints/checkpoint-52'

v0_6_ep1 = f'{prefix}/Sailor-7B-Chat_SFT_SEQ4096_LR1e-5_EP4_GBS8x1x2_20240628_write0422_chatedit0508/checkpoints/checkpoint-10'
v0_6_ep2 = f'{prefix}/Sailor-7B-Chat_SFT_SEQ4096_LR1e-5_EP4_GBS8x1x2_20240628_write0422_chatedit0508/checkpoints/checkpoint-20'     # 最小loss
v0_6_ep3 = f'{prefix}/Sailor-7B-Chat_SFT_SEQ4096_LR1e-5_EP4_GBS8x1x2_20240628_write0422_chatedit0508/checkpoints/checkpoint-30'
v0_6_ep4 = f'{prefix}/Sailor-7B-Chat_SFT_SEQ4096_LR1e-5_EP4_GBS8x1x2_20240628_write0422_chatedit0508/checkpoints/checkpoint-40'

v0_7_ep1 = f'{prefix}/Sailor-7B-Chat_SFT_SEQ4096_LR1e-5_EP4_GBS8x1x2_20240628_write0422/checkpoints/checkpoint-7'   # 最小loss
v0_7_ep2 = f'{prefix}/Sailor-7B-Chat_SFT_SEQ4096_LR1e-5_EP4_GBS8x1x2_20240628_write0422/checkpoints/checkpoint-14'
v0_7_ep3 = f'{prefix}/Sailor-7B-Chat_SFT_SEQ4096_LR1e-5_EP4_GBS8x1x2_20240628_write0422/checkpoints/checkpoint-21'
v0_7_ep4 = f'{prefix}/Sailor-7B-Chat_SFT_SEQ4096_LR1e-5_EP4_GBS8x1x2_20240628_write0422/checkpoints/checkpoint-28'

v0_8_ep1 = f'{prefix}/Sailor-7B-Chat_SFT_SEQ4096_LR1e-5_EP4_GBS8x1x2_20240628_chatedit0508/checkpoints/checkpoint-3'    # 最小loss
v0_8_ep3 = f'{prefix}/Sailor-7B-Chat_SFT_SEQ4096_LR1e-5_EP4_GBS8x1x2_20240628_chatedit0508/checkpoints/checkpoint-10'

v1_1_ep1 = f'{prefix}/Qwen2-57B-A14B-Instruct_SFT_SEQ4096_LR1e-5_EP4_GBS32x1x1_20240628_trans151_write0422_chatedit0508/checkpoints/checkpoint-7'
v1_1_ep2 = f'{prefix}/Qwen2-57B-A14B-Instruct_SFT_SEQ4096_LR1e-5_EP4_GBS32x1x1_20240628_trans151_write0422_chatedit0508/checkpoints/checkpoint-14'  # 最小loss
v1_1_ep3 = f'{prefix}/Qwen2-57B-A14B-Instruct_SFT_SEQ4096_LR1e-5_EP4_GBS32x1x1_20240628_trans151_write0422_chatedit0508/checkpoints/checkpoint-21'
v1_1_ep4 = f'{prefix}/Qwen2-57B-A14B-Instruct_SFT_SEQ4096_LR1e-5_EP4_GBS32x1x1_20240628_trans151_write0422_chatedit0508/checkpoints/checkpoint-28'

v1_2_ep1 = f'{prefix}/Qwen2-57B-A14B-Instruct_SFT_SEQ4096_LR5e-6_EP4_GBS32x1x1_20240628_expand_trans_write_chatedit//checkpoints/checkpoint-236'    # 最小loss
v1_2_ep2 = f'{prefix}/Qwen2-57B-A14B-Instruct_SFT_SEQ4096_LR5e-6_EP4_GBS32x1x1_20240628_expand_trans_write_chatedit//checkpoints/checkpoint-472'
v1_2_ep3 = f'{prefix}/Qwen2-57B-A14B-Instruct_SFT_SEQ4096_LR5e-6_EP4_GBS32x1x1_20240628_expand_trans_write_chatedit//checkpoints/checkpoint-708'


models = [
    dict(
        abbr='id_v0_3_ep1',
        type=VLLM,
        path=v0_3_ep1,
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
        abbr='id_v0_3_ep2',
        type=VLLM,
        path=v0_3_ep2,
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
        abbr='id_v0_4_ep1',
        type=VLLM,
        path=v0_4_ep1,
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
        abbr='id_v0_4_ep2',
        type=VLLM,
        path=v0_4_ep2,
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
        abbr='id_v0_4_ep3',
        type=VLLM,
        path=v0_4_ep3,
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
        abbr='id_v0_4_ep4',
        type=VLLM,
        path=v0_4_ep4,
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
        abbr='id_v0_5_ep1',
        type=VLLM,
        path=v0_5_ep1,
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
        abbr='id_v0_5_ep2',
        type=VLLM,
        path=v0_5_ep2,
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
        abbr='id_v0_5_ep3',
        type=VLLM,
        path=v0_5_ep3,
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
        abbr='id_v0_5_ep4',
        type=VLLM,
        path=v0_5_ep4,
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
        abbr='id_v0_6_ep1',
        type=VLLM,
        path=v0_6_ep1,
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
        abbr='id_v0_6_ep2',
        type=VLLM,
        path=v0_6_ep2,
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
        abbr='id_v0_6_ep3',
        type=VLLM,
        path=v0_6_ep3,
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
        abbr='id_v0_6_ep4',
        type=VLLM,
        path=v0_6_ep4,
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
        abbr='id_v0_7_ep1',
        type=VLLM,
        path=v0_7_ep1,
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
        abbr='id_v0_7_ep2',
        type=VLLM,
        path=v0_7_ep2,
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
        abbr='id_v0_7_ep3',
        type=VLLM,
        path=v0_7_ep3,
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
        abbr='id_v0_7_ep4',
        type=VLLM,
        path=v0_7_ep4,
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
        abbr='id_v0_8_ep1',
        type=VLLM,
        path=v0_8_ep1,
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
        abbr='id_v0_8_ep3',
        type=VLLM,
        path=v0_8_ep3,
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
        abbr='id_v1_1_ep1',
        type=VLLM,
        path=v1_1_ep1,
        model_kwargs=dict(tensor_parallel_size=GPU_NUMS2),
        meta_template=qwen_meta_template,
        generation_kwargs=dict(do_sample=True),
        max_seq_len=4096,
        max_out_len=4096,
        batch_size=4,
        stop_words=['<|im_end|>'],
        run_cfg=dict(num_gpus=GPU_NUMS2, num_procs=1),
    ),
    dict(
        abbr='id_v1_1_ep2',
        type=VLLM,
        path=v1_1_ep2,
        model_kwargs=dict(tensor_parallel_size=GPU_NUMS2),
        meta_template=qwen_meta_template,
        generation_kwargs=dict(do_sample=True),
        max_seq_len=4096,
        max_out_len=4096,
        batch_size=4,
        stop_words=['<|im_end|>'],
        run_cfg=dict(num_gpus=GPU_NUMS2, num_procs=1),
    ),
    dict(
        abbr='id_v1_1_ep3',
        type=VLLM,
        path=v1_1_ep3,
        model_kwargs=dict(tensor_parallel_size=GPU_NUMS2),
        meta_template=qwen_meta_template,
        generation_kwargs=dict(do_sample=True),
        max_seq_len=4096,
        max_out_len=4096,
        batch_size=4,
        stop_words=['<|im_end|>'],
        run_cfg=dict(num_gpus=GPU_NUMS2, num_procs=1),
    ),
    dict(
        abbr='id_v1_1_ep4',
        type=VLLM,
        path=v1_1_ep4,
        model_kwargs=dict(tensor_parallel_size=GPU_NUMS2),
        meta_template=qwen_meta_template,
        generation_kwargs=dict(do_sample=True),
        max_seq_len=4096,
        max_out_len=4096,
        batch_size=4,
        stop_words=['<|im_end|>'],
        run_cfg=dict(num_gpus=GPU_NUMS2, num_procs=1),
    ),
    dict(
        abbr='id_v1_2_ep1',
        type=VLLM,
        path=v1_2_ep1,
        model_kwargs=dict(tensor_parallel_size=GPU_NUMS2),
        meta_template=qwen_meta_template,
        generation_kwargs=dict(do_sample=True),
        max_seq_len=4096,
        max_out_len=4096,
        batch_size=4,
        stop_words=['<|im_end|>'],
        run_cfg=dict(num_gpus=GPU_NUMS2, num_procs=1),
    ),
    dict(
        abbr='id_v1_2_ep2',
        type=VLLM,
        path=v1_2_ep2,
        model_kwargs=dict(tensor_parallel_size=GPU_NUMS2),
        meta_template=qwen_meta_template,
        generation_kwargs=dict(do_sample=True),
        max_seq_len=4096,
        max_out_len=4096,
        batch_size=4,
        stop_words=['<|im_end|>'],
        run_cfg=dict(num_gpus=GPU_NUMS2, num_procs=1),
    ),
    dict(
        abbr='id_v1_2_ep3',
        type=VLLM,
        path=v1_2_ep3,
        model_kwargs=dict(tensor_parallel_size=GPU_NUMS2),
        meta_template=qwen_meta_template,
        generation_kwargs=dict(do_sample=True),
        max_seq_len=4096,
        max_out_len=4096,
        batch_size=4,
        stop_words=['<|im_end|>'],
        run_cfg=dict(num_gpus=GPU_NUMS2, num_procs=1),
    )
]
# models_target = ['id_v0_3_ep1', 'id_v0_3_ep2', 'id_v0_4_ep1', 'id_v0_4_ep2', 'id_v0_4_ep3', 'id_v0_4_ep4', 'id_v0_5_ep1', 'id_v0_5_ep2', 'id_v0_5_ep3', 'id_v0_5_ep4']  # , 'id_v0_6_ep1', 'id_v0_6_ep2', 'id_v0_6_ep3', 'id_v0_6_ep4'
# models = [x for x in models if x['abbr'] in models_target]
datasets = [x for x in subjective_datasets if x['abbr'] in ['mtbench101_id']]

judge_models = [dict(
    abbr='GPT4-Turbo',
    type=OpenAI,
    path='gpt-4-1106-preview',
    key='',
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
        max_num_workers=32,                 # TODO 尝试下更多workers，也尝试下更大num_procs和batch_size，看看速度有没有提升
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
