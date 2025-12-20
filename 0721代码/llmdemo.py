# 导入OpenAI模块，这个模块用于与OpenAI的API进行交互。
from openai import OpenAI

# 创建一个OpenAI客户端实例，用于访问API。
# api_key是用于验证API请求的密钥。
# base_url是API的基础URL。
client = OpenAI(
    #++++++++++++++++++++++++++++++= 填写你自己的key+++++++++++++++++++++++++++++++++++
    api_key="sk-5xmTYoLGOBZsoRLFQB1ZA6Yl2GAClThy7CCFBxVoVhHxFo2s",
    base_url="https://api.moonshot.cn/v1",
)

# 使用chat.completions.create方法创建一个聊天回复。
# model参数指定使用的模型。
# messages是一个包含聊天消息的列表，每个消息包含role和content。
# temperature是控制生成文本的创造性的参数。
# stream=True表示以流的形式接收回复。
response = client.chat.completions.create(
    model="moonshot-v1-8k",
    messages=[
        # 系统消息，定义了Kimi的角色和行为准则。
        {
            "role": "system",
            "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。",
        },
    
        {"role": "user", "content": "请帮我选择一个深圳的好餐厅。"},
    ],
    temperature=1,
    stream=True,
)

# 初始化一个空列表来收集聊天消息。
collected_messages = []

# 遍历response对象，这个对象是一个生成器，会逐块返回数据。
for idx, chunk in enumerate(response):
    # 检查chunk是否有内容，如果没有则跳过。
    chunk_message = chunk.choices[0].delta
    if not chunk_message.content:
        continue
    # 将当前块的消息添加到collected_messages列表中。
    collected_messages.append(chunk_message)
    # 打印到目前为止收集到的所有消息。
    print(f"#{idx}: {''.join([m.content for m in collected_messages])}")

# 打印完整的对话。
print(f"Full conversation received: {''.join([m.content for m in collected_messages])}")