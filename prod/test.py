import os
from transformers import AutoConfig

model_name = "BAAI/bge-m3"
# 로컬 캐시 경로를 명시적으로 확인
config = AutoConfig.from_pretrained(model_name)
full_path = getattr(config, "_name_or_path", "Not Found")

print(f"📍 현재 설정된 경로: {full_path}")

# 만약 경로가 여전히 'BAAI/bge-m3'로 나온다면, 실제 캐시 루트를 뒤집니다.
import huggingface_hub
cache_dir = huggingface_hub.constants.HF_HUB_CACHE
print(f"📂 HF 캐시 루트 디렉토리: {cache_dir}")

# 실제 모델 폴더 리스트 확인
models_dir = os.path.join(cache_dir, "models--BAAI--bge-m3", "snapshots")
if os.path.exists(models_dir):
    sub_dir = os.listdir(models_dir)[0]
    print(f"🚀 Docker에 사용할 최종 절대 경로: {os.path.join(models_dir, sub_dir)}")