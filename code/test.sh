export JT_SYNC=1
export trace_py_var=3
export HF_ENDPOINT="https://hf-mirror.com" 
CUDA_VISIBLE_DEVICES='0' python test.py --choice=1 --mid=1 --output='results/prompt_1_1500_16' --style_path='ckpt/style_1_1000_8' --inference_step=30 --postfix='style' --guided_scale=7.5
