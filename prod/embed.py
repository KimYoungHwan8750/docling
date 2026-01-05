import time
from FlagEmbedding import BGEM3FlagModel

class WnpEmbedModel:
    _model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)


    @staticmethod
    def getModel() -> BGEM3FlagModel:
        return WnpEmbedModel._model

    @staticmethod
    def embed(sentences: list[str]) -> tuple[list[float], list[float]]:
        if (sentences is None or len(sentences) == 0):
            print("임베딩할 문자열이 없습니다.")
            return None, None
        output = WnpEmbedModel._model.encode(
            sentences,
            batch_size=12,
            max_length=8192,
            return_dense=True,
            return_sparse=True, # 희소 벡터
        )
        dense = output['dense_vecs'][0].tolist()
        sparse = output['lexical_weights'][0]
        clean_sparse = {str(k): float(v) for k, v in sparse.items()}
        return dense, clean_sparse



if __name__ == "__main__":

    sentences = ["그리드 정렬"]
    [dense_vecs, sparse_vecs] = WnpEmbedModel.embed(sentences)
    with open("dense_output.txt", "w") as f:
        f.write(str(dense_vecs))
    with open("sparse_output.txt", "w") as f:
        f.write(str(sparse_vecs))
    time.sleep(1)