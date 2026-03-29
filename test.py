from openai import OpenAI
import time
client = OpenAI(
    base_url = 'https://844db0e.r5.cpolar.top/v1',
    api_key = 'Hui1631'
)
star = time.time()
char_star = time.time()
if_one = True
res = ''
for chunk in client.chat.completions.create(
    model = 'gpt-5.4',
    messages = [
        {
            'role': 'user',
            'content': '写一个千字的青春恋爱轻小说'
        }
    ],
    stream = True
):
    char = chunk.choices[0].delta.content
    if char is None:
        break
    print(char, end='', flush=True)
    res = res + char
    if if_one:
        first_char_time = time.time() - char_star
        if_one = False
        char_star = time.time()

end = time.time()
print(f'\n\n总时间: {end - star:.2f} seconds')
print(f'总字符数: {len(res)}')
print(f'首字符时间: {first_char_time:.2f} seconds')
print('-'*10)
print(f'平均每个字符时间: {(end - char_star) / len(res):.4f} seconds')
print(f'平均每秒字符数: {len(res) / (end - char_star):.2f} chars/sec')
print('-'*10)
print(f'总每个字符时间: {(end - star) / len(res):.4f} seconds')
print(f'总每秒字符数: {len(res) / (end - star):.2f} chars/sec')