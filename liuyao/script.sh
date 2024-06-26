export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=1
# 添加以上两行，否则报错：[[0m Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp-a34b3233.so.1 library.

python run.py liuyao/eval_subjective_mtbench101.py --debug
/maindata/data/user/ai_story/zhigong.wang/miniconda3/envs/opencompass/bin/python run.py liuyao/eval_subjective_mtbench101.py --reuse 20240625_105451
python run.py liuyao/eval_subjective_mtbench101.py --reuse 20240624_000906 --debug
python run.py liuyao/eval_subjective_mtbench101.py --mode infer --debug
python run.py liuyao/eval_subjective_mtbench101.py --mode eval --reuse 20240625_145959
