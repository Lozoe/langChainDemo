"""基于 Assistant 实现的 TXT 文件统计智能助手

这个模块提供了一个智能助手，可以：
1. 统计桌面上的 .txt 文件数量
2. 列出桌面上的 .txt 文件
3. 读取指定的 .txt 文件内容
"""

import os
from typing import Optional
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
# 配置 DashScope
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')  # 从环境变量获取 API Key
dashscope.timeout = 30  # 设置超时时间为 30 秒

def init_agent_service():
    """初始化 TXT 文件统计助手服务
    
    Returns:
        Assistant: 配置好的助手实例
    """
    # LLM 模型配置
    llm_cfg = {
        'model': 'qwen-max',
        'timeout': 30,  # 设置模型调用超时时间
        'retry_count': 3,  # 设置重试次数
    }
    # 系统角色设定
    system = ('你扮演一个桌面TXT文件管理助手，你具有统计、列出以及读取桌面txt文件的能力。'
             '你可以帮助用户了解桌面上有什么txt文件，并读取它们的内容。')
    
    # 获取当前 txt_counter.py 的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    txt_counter_path = os.path.join(current_dir, 'txt_counter.py')

    # MCP 工具配置
    tools = [{
        "mcpServers": {
            "txt-counter-mcp": {
                "command": "python",
                "args": [
                    txt_counter_path
                ],
                "autoApprove": [],
                "disabled": False,
                "env": {}
            }
        }
    }]
    
    try:
        # 创建助手实例
        bot = Assistant(
            llm=llm_cfg,
            name='TXT管理助手',
            description='桌面TXT文件统计与读取',
            system_message=system,
            function_list=tools,
        )
        print("助手初始化成功！")
        return bot
    except Exception as e:
        print(f"助手初始化失败: {str(e)}")
        raise

def app_gui():
    """图形界面模式
    """
    try:
        print("正在启动 Web 界面...")
        # 初始化助手
        bot = init_agent_service()
        # 配置聊天界面
        chatbot_config = {
            'prompt.suggestions': [
                '桌面上总共有多少个txt文件？',
                '请列出桌面上所有的txt文件',
                '请帮我读取桌面上的 test.txt 文件'
            ]
        }
        
        print("Web 界面准备就绪，正在启动服务...")
        # 启动 Web 界面
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run()
    except Exception as e:
        print(f"启动 Web 界面失败: {str(e)}")
        print("请检查网络连接和 API Key 配置")

if __name__ == '__main__':
    # 运行模式选择
    app_gui()          # 图形界面模式（默认）
