import time
from FlagEmbedding import BGEM3FlagModel
from protos import bge_embed_pb2
from protos import bge_embed_pb2_grpc
import grpc
from concurrent import futures

class EmbedServicer(bge_embed_pb2_grpc.BgeEmbedServicer):

    def __init__(self, model):
        self.model = model

    def _embed(self, sentences: list[str]) -> tuple[list[float], list[float]]:
        if (sentences is None or len(sentences) == 0):
            print("임베딩할 문자열이 없습니다.")
            return None, None
        output = self.model.encode(
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

    def Embed(self, request, context):
        sentences = request.texts
        dense, sparse = self._embed(sentences)
        return bge_embed_pb2.EmbedResponse(vectors=[bge_embed_pb2.VectorData(dense=dense, sparse=sparse)])

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    bge_embed_pb2_grpc.add_BgeEmbedServicer_to_server(EmbedServicer(model), server)
    server.add_insecure_port('[::]:50055')
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()