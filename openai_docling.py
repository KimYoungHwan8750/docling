import os
import base64
import mimetypes 
from dotenv import load_dotenv
from openai import OpenAI
from openai import APIError, APIStatusError
import fitz # PyMuPDF 라이브러리 (PDF 처리용)
from PIL import Image # Pillow 라이브러리 (이미지 처리용)

load_dotenv()
API_KEY = os.getenv("OPEN_ROUTER_API_KEY")

# -----------------------------------------------------
# 1. 설정
# -----------------------------------------------------

OPENROUTER_API_KEY = API_KEY
MODEL_NAME = "qwen/qwen3-vl-235b-a22b-instruct" 

# ✅ 입력 파일 (PDF)
pdf_path = "eng_payment_method.pdf" 
# ✅ 임시로 생성할 이미지 파일명 (자동 생성)
temp_image_path = "temp_page_1.png"

# -----------------------------------------------------
# ✅ 2. PDF를 PNG로 변환하는 함수
# -----------------------------------------------------
def convert_pdf_to_png(pdf_path, output_path):
    """PDF의 첫 페이지를 PNG 이미지로 변환합니다."""
    print(f"🔄 PDF 첫 페이지를 PNG로 변환 중...")
    try:
        doc = fitz.open(pdf_path)  # PyMuPDF로 PDF 열기
        page = doc.load_page(0)    # 첫 번째 페이지 로드 (0부터 시작)
        
        # 150 DPI 해상도로 렌더링 (해상도가 높아야 모델이 글자를 잘 읽음)
        pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72)) 
        
        # PIL 이미지로 변환 후 저장
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img.save(output_path)
        doc.close()
        print(f"✅ PNG 파일이 '{output_path}'으로 저장되었습니다.")
        return True
    except Exception as e:
        print(f"❌ PDF 변환 오류: {e}")
        return False

# -----------------------------------------------------
# 3. 로컬 이미지를 Base64 코드로 변환하는 함수 (수정됨)
# -----------------------------------------------------
def encode_image(image_path):
    """지정된 경로의 이미지 파일을 Base64 문자열로 인코딩합니다."""
    try:
        # PNG 형식의 MIME 타입을 사용 (PDF에서 변환된 파일 가정)
        mime_type = "image/png" 
        
        with open(image_path, "rb") as image_file:
            base64_data = base64.b64encode(image_file.read()).decode("utf-8")
        
        return base64_data, mime_type
        
    except FileNotFoundError:
        print(f"❌ 오류: '{image_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return None, None

# -----------------------------------------------------
# 4. 메인 실행 로직
# -----------------------------------------------------

if not os.path.exists(pdf_path):
    print(f"❌ 치명적 오류: '{pdf_path}' 파일이 존재하지 않습니다. 프로그램을 종료합니다.")
    exit()

# PDF를 PNG로 변환
if not convert_pdf_to_png(pdf_path, temp_image_path):
    exit()
    
# 클라이언트 초기화 (이하 동일)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# 이미지를 코드로 변환
print(f"🚀 '{temp_image_path}' 분석 요청 중...")
base64_image, mime_type = encode_image(temp_image_path)

if base64_image is None:
    exit()

try:
    # API 호출 (이미지 + 텍스트)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": "이 문서의 내용을 상세하게 설명해줘. 표나 그래프라면 주요 수치도 요약해줘."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            # ⚠️ 수정: 동적으로 MIME 타입을 사용하도록 수정
                            "url": f"data:{mime_type};base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        stream=False
    )
    
    # ... (이하 결과 출력 및 오류 처리 로직은 동일)
    if response and response.choices:
        print("\n🤖 Qwen VL 답변:")
        print(response.choices[0].message.content)
    else:
        print("\n⚠️ 경고: API는 응답했지만, 유효한 답변(choices)을 포함하지 않았습니다.")

except APIStatusError as e:
    # ... (오류 처리 부분은 그대로 유지)
    print("\n❌ API 통신 오류 발생:")
    print(f"상태 코드: {e.status_code}")
    print(f"오류 메시지: {e.response.text}")
    print("\n[해결 가이드]")
    print("- 잔액 부족 또는 모델 이름 오류를 확인하세요.")
    
except Exception as e:
    print(f"\n❌ 예상치 못한 오류: {type(e).__name__} - {e}")
finally:
    # 임시로 만든 PNG 파일 삭제 (선택 사항)
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)
        print(f"🧹 임시 파일 '{temp_image_path}' 삭제 완료.")