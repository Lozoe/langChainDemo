"""基于 Assistant 实现的开发规范检查智能助手

这个模块提供了一个智能助手，可以检查代码规范并执行以下操作：
1. 检查指定目录中的代码风格规范
2. 搜索代码中的TODO和FIXME注释
3. 验证项目结构是否符合最佳实践
4. 获取指定编程语言的命名规范
5. 检查Python文件中的导入语句组织是否符合PEP 8规范
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
    """初始化开发规范检查助手服务
    
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
    system = ('你扮演一个资深的开发代码审查专家与规范助手。你的职责是帮助我：\n'
             '1. 检查项目的代码风格规范\n'
             '2. 查找和管理项目中的TODO及FIXME等技术债\n'
             '3. 验证当前项目结构是否符合行业最佳实践\n'
             '4. 提供各种主流编程语言的命名规范指引\n'
             '5. 检查和优化Python代码里的模块导入(import)组织方式。\n'
             '你可以通过调用配置的开发规范检查器MCP工具来完成上述工作，请根据用户指令执行代码质量检查。')
    
    # 获取当前 dev_spec_mcp.py 的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dev_spec_mcp_path = os.path.join(current_dir, 'dev_spec_mcp.py')

    # MCP 工具配置
    tools = [{
        "mcpServers": {
            "dev-spec-mcp": {
                "command": "python",
                "args": [
                    dev_spec_mcp_path
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
            name='开发规范检查助手',
            description='代码风格审计、规范检查与项目结构分析',
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
                '检查一下当前目录下的代码风格',
                '在这个项目中搜索所有的 TODO 和 FIXME 注释',
                '验证当前 Python 项目目录结构是否规范',
                '提供一下 Python 的常见命名规范指南',
                '请检查一下当前目录下的 dev_spec_mcp.py 的 import 语句组织'
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
