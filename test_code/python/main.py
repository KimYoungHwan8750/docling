from qwen_client import QwenClient

# 클라이언트 초기화
qwen = QwenClient(ip="192.168.0.99", port="8000")

def main():
    # 1. 텍스트 프롬프트 테스트
    print("--- [텍스트 질문 테스트] ---")
    text_result = qwen.ask_text("사내 RAG 시스템을 구축할 때 가장 중요한 보안 요소 3가지는?")
    print(f"답변: {text_result}\n")

    # 2. 이미지 프롬프트 테스트 (이미지 파일이 있을 때 주석 해제)
    # print("--- [이미지 분석 테스트] ---")
    # img_result = qwen.ask_image("이 문서 안의 표를 마크다운 형식으로 정리해줘.", "document_sample.png")
    # print(f"답변: {img_result}")

if __name__ == "__main__":
    main()