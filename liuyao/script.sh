export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=1
# 添加以上两行，否则报错：[[0m Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp-a34b3233.so.1 library.

# Japanese
/maindata/data/user/ai_story/zhigong.wang/miniconda3/envs/opencompass/bin/python run.py liuyao/eval_subjective_mtbench101.py --reuse 20240625_105451
python run.py liuyao/eval_subjective_mtbench101_ja.py --debug
python run.py liuyao/eval_subjective_mtbench101_ja.py --reuse 20240624_000906 --debug
python run.py liuyao/eval_subjective_mtbench101_ja.py --mode infer --debug
python run.py liuyao/eval_subjective_mtbench101_ja.py --mode eval --reuse 20240625_145959
python run.py liuyao/eval_subjective_mtbench101_ja.py --mode viz --reuse 20240625_145959

# Indonesian
python run.py liuyao/eval_subjective_mtbench101_id.py --reuse --debug
python run.py liuyao/eval_subjective_mtbench101_id.py --mode infer --reuse
python run.py liuyao/eval_subjective_mtbench101_id.py --mode eval --reuse


# TODO 数据集里的问题有些多，要不要删减一些，以节省GPT4 token

