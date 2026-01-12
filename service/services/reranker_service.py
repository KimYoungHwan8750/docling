from typing import List


class RerankerService:
    """리랭커 서비스"""
    
    def __init__(self, model):
        self.model = model
    
    def rerank(self, query: str, documents: List[str]) -> List[float]:
        """쿼리와 문서들의 관련성 점수를 계산합니다."""
        if not documents:
            return []
        
        # [질문, 문서1], [질문, 문서2] 쌍을 만듭니다.
        pairs = [[query, doc] for doc in documents]
        scores = self.model.compute_score(pairs)
        
        # 단일 값이 반환되는 경우 리스트로 변환
        if isinstance(scores, (int, float)):
            return [float(scores)]
        
        return [float(score) for score in scores]
