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
MODEL_TYPE="${MODEL_TYPE:-T2V-14B}"
CHECKPOINT_FOLDER="${CHECKPOINT_FOLDER:-../ckpts/wan_causal_dmd_v2v_14b}"

# Abilita tunnel Cloudflare impostando ENABLE_TUNNEL=1
ENABLE_TUNNEL="${ENABLE_TUNNEL:-0}"
TUNNEL_URL="${TUNNEL_URL:-http://localhost:7860}"

if [ "${ENABLE_TUNNEL}" = "1" ]; then
    if ! command -v cloudflared >/dev/null 2>&1; then
        echo -e "\033[1;33m\ncloudflared non trovato: avvio senza tunnel\033[0m"
        python main.py --port 7860 --host 0.0.0.0 --num_gpus "${NUM_GPUS}" --step 2 --gpu_ids "${GPU_IDS}" --model_type "${MODEL_TYPE}" --checkpoint_folder "${CHECKPOINT_FOLDER}"
        exit 0
    fi

    python main.py --port 7860 --host 0.0.0.0 --num_gpus "${NUM_GPUS}" --step 2 --gpu_ids "${GPU_IDS}" --model_type "${MODEL_TYPE}" --checkpoint_folder "${CHECKPOINT_FOLDER}" &
    APP_PID=$!

    # aspetta che il server risponda
    for _ in {1..30}; do
        if command -v curl >/dev/null 2>&1; then
            curl -fsS "http://localhost:7860" >/dev/null 2>&1 && break
        fi
        sleep 1
    done

    cloudflared tunnel --url "${TUNNEL_URL}" &
    TUNNEL_PID=$!

    trap 'kill ${TUNNEL_PID} ${APP_PID} 2>/dev/null || true' EXIT
    wait "${APP_PID}"
else
    python main.py --port 7860 --host 0.0.0.0 --num_gpus "${NUM_GPUS}" --step 2 --gpu_ids "${GPU_IDS}" --model_type "${MODEL_TYPE}" --checkpoint_folder "${CHECKPOINT_FOLDER}"
fi
