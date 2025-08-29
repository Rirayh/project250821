# 文件名: test_client.py
import requests
import json

# API 服务器的地址和端口
# 确保这个端口号 (8001) 和您启动 baseline_api_server1.py 时使用的端口号一致
BASE_URL = "http://127.0.0.1:8008"

# 我们将向 Qwen 模型发送请求
# 您也可以将下面的 "qwen" 更改为 "dream" 来测试另一个模型
# QWEN_ENDPOINT = f"{BASE_URL}/v1/qwen/chat/completions"
# QWEN_ENDPOINT = f"{BASE_URL}/v1/dream/chat/completions"
QWEN_ENDPOINT = f"{BASE_URL}/v1/chat/completions"

def ask_question(prompt: str):
    """
    向正在运行的 API 服务器发送一个问题并获取回答。
    """
    # 构建符合服务器要求的请求数据结构
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "model": "Dream-org/Dream-v0-Instruct-7B",#"Qwen/Qwen2.5-7B-Instruct", # 这个模型名称需要和服务端匹配
        "max_gen_toks": 1024, # 你可以按需调整生成文本的长度
        "temperature": 0.0
    }

    headers = {
        "Content-Type": "application/json"
    }

    print(f"正在向 {QWEN_ENDPOINT} 发送请求...")
    print(f"发送的数据: \n{json.dumps(payload, indent=2, ensure_ascii=False)}")

    try:
        # 发送 POST 请求
        response = requests.post(QWEN_ENDPOINT, json=payload, headers=headers)

        # 检查请求是否成功
        response.raise_for_status()

        # 解析并打印返回的 JSON 数据
        print("\n--- 服务器响应 ---")
        response_data = response.json()
        print(json.dumps(response_data, indent=2, ensure_ascii=False))

        # 提取并打印出模型的回答内容
        print("\n--- 模型的回答 ---")
        answer = response_data["choices"][0]["message"]["content"]
        print(answer)

    except requests.exceptions.RequestException as e:
        print(f"\n错误：无法连接到服务器。请确认:")
        print(f"1. baseline_api_server1.py 脚本是否正在运行且没有报错。")
        print(f"2. URL '{BASE_URL}' 是否正确。")
        print(f"错误详情: {e}")

if __name__ == "__main__":
    # 定义一个你想问的问题
    question = r"Tobias went to a swimming pool for 3 hours. Swimming every 100 meters took him 5 minutes, but every 25 minutes he had to take a 5-minute pause. How many meters did Tobias swim during his visit to the swimming pool?"
    # question = r"Janet, a third grade teacher, is picking up the sack lunch order from a local deli for the field trip she is taking her class on. There are 35 children in her class, 5 volunteer chaperones, and herself. She she also ordered three additional sack lunches, just in case there was a problem. Each sack lunch costs $7. How much do all the lunches cost in total?"
    ask_question(question)
#     At 30, Anika is 4/3 the age of Maddie. What would be their average age in 15 years?
# If Anika is 30 now, in 15 years, she'll be 30+15=<<30+15=45>>45 years old.
# At 30, Anika is 4/3 the age of Maddie, meaning Maddie is 4/3*30=<<4/3*30=40>>40 years.
# In 15 years, Maddie will be 40+15=<<40+15=55>>55 years old.
# Their total age in 15 years will be 55+45=<<55+45=100>>100
# Their average age in 15 years will be 100/2=<<100/2=50>>50
# #### 50