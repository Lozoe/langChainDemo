from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Tongyi  # 导入通义千问Tongyi模型

# 加载环境变量
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import dashscope
import os

print('DASHSCOPE_API_KEY=' +os.environ.get('DASHSCOPE_API_KEY'))

# 直接设置 API Key
# api_key = os.getenv('DASHSCOPE_API_KEY')
api_key = os.environ.get('DASHSCOPE_API_KEY')
dashscope.api_key = api_key
 
# 加载 Tongyi 模型
llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=api_key)  # 使用通义千问qwen-turbo模型

# 创建Prompt Template
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

# 新推荐用法：将 prompt 和 llm 组合成一个"可运行序列"
chain = prompt | llm

# 使用 invoke 方法传入输入
# import pdb
# pdb.set_trace()  # 程序会在这里停下来，等待你的调试命令
result1 = chain.invoke({"product": "colorful socks"})
print(result1)

result2 = chain.invoke({"product": "广告设计"})
print(result2)

