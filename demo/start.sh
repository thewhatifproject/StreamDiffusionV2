#!/bin/bash
cd frontend
npm install
npm run build
if [ $? -eq 0 ]; then
    echo -e "\033[1;32m\nfrontend build success \033[0m"
else
    echo -e "\033[1;31m\nfrontend build failed\n\033[0m" >&2  exit 1
fi
cd ../
NUM_GPUS="${NUM_GPUS:-2}"
GPU_IDS="${GPU_IDS:-0,1}"
python main.py --port 7860 --host 0.0.0.0 --num_gpus "${NUM_GPUS}" --step 2 --gpu_ids "${GPU_IDS}" --model_type T2V-1.3B
