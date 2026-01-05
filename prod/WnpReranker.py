from FlagEmbedding import FlagReranker

class WnpReranker:
    # 성능 좋은 BGE-Reranker-v2-m3 모델 추천
    _model = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

    @staticmethod
    def rerank(query: str, documents: list[str]):
        # [질문, 문서1], [질문, 문서2] 쌍을 만듭니다.
        pairs = [[query, doc] for doc in documents]
        scores = WnpReranker._model.compute_score(pairs)
        return scores

# 사용 예시
# query = "그리드"
# contents = [hit['_source']['content'] for hit in hits]
# new_scores = WnpReranker.rerank(query, contents)