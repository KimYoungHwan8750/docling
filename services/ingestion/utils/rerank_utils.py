import grpc
from ...protos import bge_rerank_pb2_grpc
from ...protos import bge_rerank_pb2

def get_rerank_client(server_address='localhost:50056'):
    channel = grpc.insecure_channel(server_address)
    return bge_rerank_pb2_grpc.BgeRerankStub(channel)

def rerank_via_grpc(query: str, documents: list[str], client=None) -> list[float]:
    if client is None:
        client = get_rerank_client()
    
    request = bge_rerank_pb2.RerankRequest(query=query, documents=documents)
    response = client.Rerank(request)
    return response.scores
