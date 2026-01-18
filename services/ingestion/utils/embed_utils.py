import grpc
from ...protos import bge_embed_pb2_grpc
from ...protos import bge_embed_pb2

def get_embed_client(server_address='localhost:50055'):
    """gRPC 임베딩 서비스 클라이언트 생성"""
    channel = grpc.insecure_channel(server_address)
    return bge_embed_pb2_grpc.BgeEmbedStub(channel)

def embed_via_grpc(sentences: list[str], client=None) -> tuple[list[float], dict]:
    if client is None:
        client = get_embed_client()
    
    request = bge_embed_pb2.EmbedRequest(texts=sentences)
    response = client.Embed(request)
    
    if len(response.vectors) == 0:
        return None, None
    
    vector_data = response.vectors[0]
    dense = list(vector_data.dense)
    
    sparse = {str(k): float(v) for k, v in vector_data.sparse.items()}
    return dense, sparse
