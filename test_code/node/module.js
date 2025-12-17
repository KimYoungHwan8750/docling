const { OpenAI } = require('openai');
const axios = require('axios');
const fs = require('fs');

class QwenClient {
    constructor(ip = "192.168.0.99", port = "8000", model = "My_Model") {
        this.apiUrl = `http://${ip}:${port}/v1`;
        this.model = model;
        this.openai = new OpenAI({
            baseURL: this.apiUrl,
            apiKey: 'EMPTY' // vLLM 로컬 서버용
        });
    }

    // 1. 텍스트 전용 질문 메서드
    async askText(prompt, maxTokens = 512) {
        try {
            const response = await this.openai.chat.completions.create({
                model: this.model,
                messages: [{ role: "user", content: prompt }],
                max_tokens: maxTokens,
                temperature: 0.7
            });
            return response.choices[0].message.content;
        } catch (error) {
            console.error("Text API 에러:", error.message);
            throw error;
        }
    }

    // 2. 이미지 분석 질문 메서드 (Requests 대신 Axios 사용)
    async askImage(prompt, imagePath, maxTokens = 1024) {
        try {
            // 이미지를 Base64로 읽기
            const imageBuffer = fs.readFileSync(imagePath);
            const base64Image = imageBuffer.toString('base64');

            const payload = {
                model: this.model,
                messages: [
                    {
                        role: "user",
                        content: [
                            { type: "text", text: prompt },
                            {
                                type: "image_url",
                                image_url: { url: `data:image/jpeg;base64,${base64Image}` }
                            }
                        ]
                    }
                ],
                max_tokens: maxTokens
            };

            const response = await axios.post(`${this.apiUrl}/chat/completions`, payload);
            return response.data.choices[0].message.content;
        } catch (error) {
            console.error("Image API 에러:", error.message);
            throw error;
        }
    }
}

module.exports = QwenClient;