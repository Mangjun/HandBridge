import openai
import os
from typing import List
from pathlib import Path

class TextGenerator:
    def __init__(self):
        try:
            base_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            api_key_path = base_dir / 'api-key-local'
            
            if api_key_path.exists():
                with open(api_key_path, 'r') as f:
                    api_key_line = f.read().strip()
                    openai.api_key = api_key_line.split(':')[1].strip() if ':' in api_key_line else api_key_line.strip()
                    print(f"API key loaded from file successfully")

        except Exception as e:
            print(f"API key file read error: {e}")
            openai.api_key = os.getenv('OPENAI_API_KEY')

        if not openai.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or provide api-key-local file.")

    def generate_sentence(self, sign_words: List[str], emotion: str) -> str:
        try:
            if not sign_words:
                return "입력된 단어가 없습니다"

            words_str = ", ".join(sign_words)
            
            emotion_map = {
                "sad": "슬픈",
                "disgust": "불쾌감을 느끼는",
                "angry": "화가 난",
                "neutral": "중립적인",
                "fear": "두려워하는",
                "surprise": "놀란",
                "happy": "행복한"
            }
            
            if emotion not in emotion_map:
                return "지원하지 않는 감정 상태입니다"
                
            current_emotion = emotion_map[emotion]
            
            prompt = f"""현재 {current_emotion} 상태인 화자가 사용한 다음 단어들로 자연스러운 한국어 문장을 만들어주세요.

                        단어: {words_str}

                        조건:
                        - 모든 단어를 반드시 사용할 것
                        - 문법적으로 올바른 한국어 문장
                        - 일상적이고 자연스러운 표현
                        - 한 문장으로 완성
                        - 문장만 답변하고 다른 설명은 하지 말 것

                        문장:"""

            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "당신은 한국어 문장 생성 전문가입니다. 주어진 단어들을 모두 사용하여 자연스럽고 문법적으로 올바른 한국어 문장을 만드세요. 오직 완성된 문장만 응답하세요."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=80,
                temperature=0.7,
                top_p=1.0
            )
            
            generated_text = response.choices[0].message.content.strip()
            
            generated_text = generated_text.replace("문장:", "").strip()
            generated_text = generated_text.replace("\"", "").strip()
            generated_text = generated_text.replace("답:", "").strip()
            
            prefixes_to_remove = ["답변:", "문장:", "결과:", "생성된 문장:"]
            for prefix in prefixes_to_remove:
                if generated_text.startswith(prefix):
                    generated_text = generated_text[len(prefix):].strip()
            
            if not generated_text or len(generated_text) < 3:
                return "문장 생성에 실패했습니다"
                
            return generated_text
            
        except Exception as e:
            print(f"OpenAI API Error: {str(e)}")
            return "문장 생성에 실패했습니다"