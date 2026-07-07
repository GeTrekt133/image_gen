# Единая точка входа в окружение пода. Source это перед любыми командами.
source /venv/main/bin/activate
set -a
source /workspace/secrets.env
export HF_HOME=/workspace/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1
set +a
