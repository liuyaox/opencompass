
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
