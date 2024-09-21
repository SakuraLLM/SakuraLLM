import requests
import json

prompt = '你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。<|im_end|>\n<|im_start|>user\n将下面的日文文本翻译成中文：女は、きゅっと口角を吊りあげて、にいっと淫靡な笑みを、月に一度の経血さながら滴らせると、挑むように告げた。<|im_end|>\n<|im_start|>assistant\n'

data = {
    'prompt': prompt,
    'max_new_tokens': 1024,
    'stream': True
}
response = requests.post('http://localhost:5000/v1/chat/completions', json=data, stream=True)

if response.status_code == 200:
    for chunk in response.iter_content(chunk_size=None):
        if chunk:
            chunk = chunk.decode('utf-8').strip()
            if chunk.startswith('data: '):
                chunk = chunk[6:]
            try:
                json_data = json.loads(chunk)
                if 'choices' in json_data and len(json_data['choices']) > 0 and 'delta' in json_data['choices'][0]:
                    content = json_data['choices'][0]['delta'].get('content')
                    if content:
                        print(content, end='')
            except json.JSONDecodeError:
                continue
