from FlagEmbedding import BGEM3FlagModel

# 모델 로드 (이미 클래스 내부에 선언하셨으니 해당 인스턴스의 tokenizer를 쓰셔도 됩니다)
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
tokenizer = model.tokenizer

text = """나 밥 먹었어"""

# 1. 토큰화 (텍스트 -> 토큰 ID)
tokens = tokenizer.encode(text, add_special_tokens=True)

# 2. 토큰 ID를 다시 읽을 수 있는 단어 조각으로 변환
token_pieces = [tokenizer.decode([t]) for t in tokens]

print(f"원문: {text}")
print("-" * 30)
print(f"토큰 ID 리스트: {tokens}")
print(f"토큰 조각 매칭: {token_pieces}")
print(f"토큰 개수: {len(tokens)}")