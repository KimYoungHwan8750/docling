const QwenClient = require('./qwenClient');

// 클라이언트 초기화
const qwen = new QwenClient("192.168.0.99", "8000");

async function main() {
    try {
        // 1. 텍스트 프롬프트 테스트
        console.log("--- [텍스트 질문 테스트] ---");
        const textResult = await qwen.askText("Node.js 환경에서 LLM 서버를 연동할 때 주의할 점은?");
        console.log(`답변: ${textResult}\n`);

        // 2. 이미지 프롬프트 테스트 (파일이 있을 경우 주석 해제)
        /*
        console.log("--- [이미지 분석 테스트] ---");
        const imgResult = await qwen.askImage("이 이미지의 내용을 한 문장으로 요약해줘.", "test_image.jpg");
        console.log(`답변: ${imgResult}`);
        */
    } catch (err) {
        console.error("실행 중 오류 발생:", err);
    }
}

main();