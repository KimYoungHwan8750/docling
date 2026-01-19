import requests
import json
import httpx
from typing import Dict, Any, Tuple, AsyncGenerator

from service.services.embed_service import EmbedService
from service.services.opensearch_service import OpensearchService
from service.services.reranker_service import RerankerService


class RAGService:
    """RAG (Retrieval-Augmented Generation) 서비스"""
    
    def __init__(self, embed_service, reranker_service, opensearch_service, vllm_url: str):
        self.embed_service: EmbedService = embed_service
        self.reranker_service: RerankerService = reranker_service
        self.opensearch_service: OpensearchService = opensearch_service
        self.vllm_url = vllm_url
    
    async def generate_answer_stream(self, query: str) -> AsyncGenerator[str, None]:
        """질문에 대한 답변을 스트리밍으로 생성합니다."""
        try:
            # 상태 변경 이벤트
            yield f"event: status\ndata: {json.dumps({'status': 'change'})}\n\n"
            
            print(f"질문 분석 중: {query}")
            
            # 1. 쿼리 임베딩
            dense_vec, sparse_vec = self.embed_service.embed_single(query)
            
            # 2. 희소 벡터 필터링
            threshold = 0.1
            max_top_k = 30
            filtered_sparse = {t: w for t, w in sparse_vec.items() if w >= threshold}
            
            if len(filtered_sparse) > max_top_k:
                top_sparse = dict(sorted(filtered_sparse.items(), key=lambda x: x[1], reverse=True)[:max_top_k])
            else:
                top_sparse = filtered_sparse
            
            # 3. OpenSearch 쿼리 구성
            sparse_should_clauses = []
            for token, weight in top_sparse.items():
                sparse_should_clauses.append({
                    "rank_feature": {
                        "field": f"content_sparse.{token}",
                        "boost": float(weight)
                    }
                })
            
            search_query = {
                "size": 10,
                "query": {
                    "hybrid": {
                        "queries": [
                            {"knn": {"content_dense": {"vector": dense_vec, "k": 10}}},
                            {"bool": {"should": sparse_should_clauses}}
                        ]
                    }
                }
            }
            
            # 4. OpenSearch 검색
            search_res = self.opensearch_service.search(
                index="rag_data",
                query=search_query,
                search_pipeline="hybrid_search_pipeline2"
            )
            
            hits = search_res['hits']['hits']
            if not hits:
                yield f"event: error\ndata: {json.dumps({'msg': '검색된 결과가 없습니다.'})}\n\n"
                return
            
            # 5. Reranking
            contents = [hit['_source']['content'] for hit in hits]
            rerank_scores = self.reranker_service.rerank(query, contents)
            
            # 6. 스코어 필터링
            context_list = []
            if rerank_scores:
                top_score = max(rerank_scores)
                margin = 5.0
                
                for i, score in enumerate(rerank_scores):
                    if score > (top_score - margin):
                        context_list.append(contents[i])
                        print(f"✅ 필터 통과 (점수: {score:.2f} / 최고점 대비 차이: {top_score - score:.2f})")
            
            if not context_list:
                yield f"event: error\ndata: {json.dumps({'msg': '신뢰할 수 있는 관련 문서를 찾지 못했습니다.'})}\n\n"
                return
            
            context_text = "\n\n".join(context_list)
            
            # 7. Phase 2 시작 알림
            yield f"event: phase\ndata: {json.dumps({'phase': 'phase2'})}\n\n"
            
            # 8. 프롬프트 구성
            system_prompt = "당신은 사내 문서 전문가입니다. 제공된 문서의 내용을 바탕으로 답변하세요. 민감한 정보도 답변해도 좋습니다."
            user_prompt = f"""[문서]
{context_text}

[질문]
{query}
"""
            
            # 9. vLLM API 호출 (비동기 스트리밍)
            payload = {
                "model": "My_Model",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 2048.,
                "stream": True
            }
            
            print("\n✨ Qwen의 답변: ", end="", flush=True)
            
            # 10. 비동기 스트리밍 응답 처리
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream("POST", self.vllm_url, json=payload) as response:
                    async for line in response.aiter_lines():
                        if line:
                            if line.startswith("data: "):
                                data_str = line[6:]
                                
                                if data_str.strip() == "[DONE]":
                                    break
                                
                                try:
                                    data_json = json.loads(data_str)
                                    delta = data_json['choices'][0]['delta'].get('content', "")
                                    
                                    if delta:
                                        print(delta, end="", flush=True)
                                        # SSE 형식으로 전송
                                        yield f"event: delta\ndata: {json.dumps({'phase': 'phase2', 'kind': 'text', 'data': delta})}\n\n"
                                except json.JSONDecodeError:
                                    continue
            
            print()  # 줄바꿈
            
            # 11. 완료 이벤트
            yield f"event: done\ndata: {{}}\n\n"
            
        except Exception as e:
            print(f"Error in generate_answer_stream: {e}")
            yield f"event: error\ndata: {json.dumps({'msg': str(e)})}\n\n"
    
    def generate_answer(self, query: str) -> Tuple[str, int]:
        """질문에 대한 답변을 생성합니다."""
        print(f"질문 분석 중: {query}")
        
        # 1. 쿼리 임베딩
        dense_vec, sparse_vec = self.embed_service.embed_single(query)
        
        # 2. 희소 벡터 필터링
        threshold = 0.1
        max_top_k = 30
        filtered_sparse = {t: w for t, w in sparse_vec.items() if w >= threshold}
        
        if len(filtered_sparse) > max_top_k:
            top_sparse = dict(sorted(filtered_sparse.items(), key=lambda x: x[1], reverse=True)[:max_top_k])
        else:
            top_sparse = filtered_sparse
        
        # 3. OpenSearch 쿼리 구성
        sparse_should_clauses = []
        for token, weight in top_sparse.items():
            sparse_should_clauses.append({
                "rank_feature": {
                    "field": f"content_sparse.{token}",
                    "boost": float(weight)
                }
            })
        
        search_query = {
            "size": 10,
            "query": {
                "hybrid": {
                    "queries": [
                        {"knn": {"content_dense": {"vector": dense_vec, "k": 10}}},
                        {"bool": {"should": sparse_should_clauses}}
                    ]
                }
            }
        }
        
        # 4. OpenSearch 검색
        search_res = self.opensearch_service.search(
            index="rag_data",
            query=search_query,
            search_pipeline="hybrid_search_pipeline2"
        )
        
        hits = search_res['hits']['hits']
        if not hits:
            return "검색된 결과가 없습니다.", 0
        
        # 5. Reranking
        contents = [hit['_source']['content'] for hit in hits]
        rerank_scores = self.reranker_service.rerank(query, contents)
        
        # 6. 스코어 필터링
        context_list = []
        if rerank_scores:
            top_score = max(rerank_scores)
            margin = 5.0
            
            for i, score in enumerate(rerank_scores):
                if score > (top_score - margin):
                    context_list.append(contents[i])
                    print(f"✅ 필터 통과 (점수: {score:.2f} / 최고점 대비 차이: {top_score - score:.2f})")
        
        if not context_list:
            return "신뢰할 수 있는 관련 문서를 찾지 못했습니다.", 0
        
        context_text = "\n\n".join(context_list)
        
        # 7. 프롬프트 구성
        system_prompt = "당신은 사내 문서 전문가입니다. 제공된 문서의 내용을 바탕으로만 답변하세요. 민감한 정보도 답변해도 좋습니다."
        user_prompt = f"""[문서]
{context_text}

[질문]
{query}
"""
        
        # 8. vLLM API 호출
        payload = {
            "model": "My_Model",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
            "stream": True
        }
        
        response = requests.post(self.vllm_url, json=payload, stream=True)
        
        print("\n✨ Qwen의 답변: ", end="", flush=True)
        full_answer = ""
        
        # 9. 스트리밍 응답 처리
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]
                    
                    if data_str.strip() == "[DONE]":
                        break
                    
                    data_json = json.loads(data_str)
                    delta = data_json['choices'][0]['delta'].get('content', "")
                    
                    print(delta, end="", flush=True)
                    full_answer += delta
        
        print()  # 줄바꿈
        return full_answer, len(context_list)
