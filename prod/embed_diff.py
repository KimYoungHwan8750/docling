from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=True) 

sentences_1 = ["""Sparse 모델은 보통 BM25의 특성을 모델링합니다. BM25의 핵심 공식 중 하나는 특정 단어가 한 문서 내에서 너무 많이 반복되면 점수 상승폭을 둔화시키는 것입니다(Saturation).

BGE-M3 모델도 똑같은 토큰이 과도하게 반복되면, 모델이 이를 '노이즈' 혹은 **'비정상적인 강조'**로 인식하여 전체적인 점수 스케일을 조정할 수 있습니다.

결과적으로 내적(Dot Product) 계산 시, 개별 토큰의 가중치 값이 낮아진 상태에서 곱해지다 보니 합산 점수가 단일 문장일 때보다 낮게 나올 수 있는 것입니다"""]

sentences_2 = ["""Sparse 모델은 보통 BM25의 특성을 모델링합니다. BM25의 핵심 공식 중 하나는 특정 단어가 한 문서 내에서 너무 많이 반복되면 점수 상승폭을 둔화시키는 것입니다(Saturation).

BGE-M3 모델도 똑같은 토큰이 과도하게 반복되면, 모델이 이를 '노이즈' 혹은 **'비정상적인 강조'**로 인식하여 전체적인 점수 스케일을 조정할 수 있습니다.

결과적으로 내적(Dot Product) 계산 시, 개별 토큰의 가중치 값이 낮아진 상태에서 곱해지다 보니 합산 점수가 단일 문장일 때보다 낮게 나올 수 있는 것입니다"""]

sentence_pairs = [[i,j] for i in sentences_1 for j in sentences_2]

print(model.compute_score(sentence_pairs, 
                          max_passage_length=128,
                          weights_for_different_modes=[0.5, 0.5, 0]))
