from FlagEmbedding import FlagReranker
from ..protos import bge_rerank_pb2
from ..protos import bge_rerank_pb2_grpc
import grpc
from concurrent import futures

class RerankServicer(bge_rerank_pb2_grpc.BgeRerankServicer):
    def __init__(self, model):
        self.model = model

    def _rerank(self, query: str, documents: list[str]) -> list[float]:
        pairs = [[query, doc] for doc in documents]
        scores = self.model.compute_score(pairs)
        return scores

    def Rerank(self, request, context):
        query = request.query
        documents = request.documents
        scores = self._rerank(query, documents)
        return bge_rerank_pb2.RerankResponse(scores=scores)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
    bge_rerank_pb2_grpc.add_BgeRerankServicer_to_server(RerankServicer(model), server)
    server.add_insecure_port('[::]:50056')
    server.start()
    server.wait_for_termination()
