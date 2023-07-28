# Chain of Thought

## Background

During the process of reasoning, CoT (Chain of Thought) method is an efficient way to help LLMs deal complex questions, for example: math problem and relation inference. In OpenCompass, we support multiple types of CoT method.

## 1. Zero Shot CoT

You can change the `PromptTemplate` of the dataset config, by simply add *Let's think step by step* to realize a Zero-Shot CoT prompt for your evaluation:

```python
qa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="Answer the question:\nQ: {question}?\nLet's think step by step:\n"
    ),
    retriever=dict(type=ZeroRetriever)
)
```

## 2. Few Shot CoT

Few-shot CoT can make LLMs easy to follow your instructions and get better answers. For few-shot CoT, add your CoT template to `PromptTemplate` like following config to create a one-shot prompt:

```python
qa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=
'''Question: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?
Let's think step by step
Answer:
Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers.
His team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers
They scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws.
All together his team scored 50+24+10= 84 points
Mark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers.
His opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers.
They also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws.
All together Mark's opponents scored 100+12+5=117 points
The total score for the game is both team's scores added together, so it is 84+117=201 points
The answer is 201

Question: {question}\nLet's think step by step:\n{answer}
'''),
    retriever=dict(type=ZeroRetriever)
)
```

## 3. Self-Consistency

The SC (Self-Consistency) method is proposed in [this paper](https://arxiv.org/abs/2203.11171), which will sample multiple reasoning paths for the question, and make majority voting to the generated answers for LLMs. This method displays remarkable proficiency among reasoning tasks with high accuracy but may consume more time and resources when inferencing, because of the majority voting strategy. In OpenCompass, you can simply set SC method in the dataset config like:

```python
gsm8k_infer_cfg = dict(
    inferencer=dict(
        type=SCInferencer,
        generation_kwargs=dict(do_sample=True, temperature=0.7, top_k=40),  # Set sample parameters to make sure model generate various output
        infer_type='SC',
        sc_size = SAMPLE_SIZE
    )
)
gsm8k_eval_cfg = dict(sc_size=SAMPLE_SIZE)
```

```{note}
注意，OpenCompass 默认使用默认使用 argmax 的方式采样下一个 token，因此若不指定采样参数，模型每次的推理结果将会是完全一致的，多轮评测将会失效。
```

Where `SAMPLE_SIZE` is the number of reasoning paths in Self-Consistency, higher value usually outcome higher performance. The following figure from the paper demonstrates the relation between reasoning paths and performance in several reasoning tasks:
![image](https://github.com/InternLM/opencompass/assets/28834990/05c7d850-7076-43ca-b165-e6251f9b3001)
From the figure, it can be seen that in different reasoning tasks, performance tends to improve as the number of reasoning paths increases. However, for some tasks, increasing the number of reasoning paths may reach a limit, and further increasing the number of paths may not bring significant performance improvement. Therefore, it is necessary to conduct experiments and adjustments on specific tasks to find the optimal number of reasoning paths that best suit the task.
