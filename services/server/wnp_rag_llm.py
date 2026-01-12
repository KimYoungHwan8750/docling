import requests
import json
from wnp_opensearch import WnpOpensearch
from embed import WnpEmbedModel
from wnp_reranker import WnpReranker

VLLM_URL = "http://localhost:8000/v1/chat/completions"

def generate_answer(query):
    print(f"질문 분석 중: {query}")
    
    dense_vec, sparse_vec = WnpEmbedModel.embed([query])
    threshold = 0.1
    max_top_k = 30

    filtered_sparse = {t: w for t, w in sparse_vec.items() if w >= threshold}

    if len(filtered_sparse) > max_top_k:
        top_sparse = dict(sorted(filtered_sparse.items(), key=lambda x: x[1], reverse=True)[:max_top_k])
    else:
        top_sparse = filtered_sparse

    sparse_should_clauses = []
    for token, weight in top_sparse.items():
        sparse_should_clauses.append({
            "rank_feature": {
                "field": f"content_sparse.{token}",      # BGE-M3 토큰 ID
                "boost": float(weight)    # 가중치를 부스트로 사용
            }
        })
    
    search_query = {
        "size": 10,
        "query": {
            "hybrid": {
                "queries": [
                    # {"match": {"content": query}},
                    {"knn": {"content_dense": {"vector": dense_vec, "k": 10}}},
                    {"bool": {"should": sparse_should_clauses}}
                ]
            }
        }
    }
    
    search_res = WnpOpensearch.get_client().search(
        index="rag_data",
        body=search_query,
        params={"search_pipeline": "hybrid_search_pipeline2"}
    )
    
    hits = search_res['hits']['hits']
    if not hits:
        return "검색된 결과가 없습니다."

    contents = [hit['_source']['content'] for hit in hits]
    rerank_scores = WnpReranker.rerank(query, contents)
    
    context_list = []
    if rerank_scores:
        top_score = max(rerank_scores)
        
        margin = 5.0 
        
        for i, score in enumerate(rerank_scores):
            if score > (top_score - margin):
                context_list.append(contents[i])
                print(f"✅ 필터 통과 (점수: {score:.2f} / 최고점 대비 차이: {top_score - score:.2f})")
    
    if not context_list:
        return "신뢰할 수 있는 관련 문서를 찾지 못했습니다."

    context_text = "\n\n".join(context_list)

    system_prompt = "당신은 사내 문서 전문가입니다. 제공된 [문서]의 내용을 바탕으로만 답변하세요."
    user_prompt = f"""[문서]
{context_text}

[질문]
{query}

[답변]"""

    # 6. vLLM API 호출
    payload = {
        "model": "My_Model",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 2048,
        "stream": True
    }

    response = requests.post(VLLM_URL, json=payload, stream=True)
    
    print("\n✨ Qwen의 답변: ", end="", flush=True)
    full_answer = ""

    # 3. 데이터가 들어오는 대로 실시간 처리
    for line in response.iter_lines():
        if line:
            # "data: " 접두사 제거 후 JSON 파싱
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith("data: "):
                data_str = decoded_line[6:]
                
                # 스트림 끝 표시인 [DONE] 체크
                if data_str.strip() == "[DONE]":
                    break
                
                data_json = json.loads(data_str)
                # content 조각(delta) 추출
                delta = data_json['choices'][0]['delta'].get('content', "")
                
                print(delta, end="", flush=True) # 화면에 즉시 출력
                full_answer += delta

    return full_answer

if __name__ == "__main__":
    query = "what it the password for snack24"
    answer = generate_answer(query)