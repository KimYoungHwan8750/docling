from typing import Tuple, List, Dict, Optional


class EmbedService:
    """임베딩 서비스"""
    
    def __init__(self, model):
        self.model = model
    
    def embed(self, sentences: List[str]) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        """문장들을 임베딩합니다."""
        if not sentences or len(sentences) == 0:
            return [], []
        
        output = self.model.encode(
            sentences,
            batch_size=12,
            max_length=8192,
            return_dense=True,
            return_sparse=True,
        )
        
        dense_vecs = [vec.tolist() for vec in output['dense_vecs']]
        sparse_vecs = [
            {str(k): float(v) for k, v in sparse.items()} 
            for sparse in output['lexical_weights']
        ]
        
        return dense_vecs, sparse_vecs
    
    def embed_single(self, sentence: str) -> Tuple[List[float], Dict[str, float]]:
        """단일 문장을 임베딩합니다."""
        dense_vecs, sparse_vecs = self.embed([sentence])
        return dense_vecs[0] if dense_vecs else [], sparse_vecs[0] if sparse_vecs else {}
