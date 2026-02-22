"""
测试开发规范MCP工具
"""
import subprocess
import time
import requests
import sys
from pathlib import Path

def test_dev_spec_tools():
    """测试开发规范检查工具"""
    print("正在测试开发规范检查工具...")
    
    # 先导入我们的工具模块
    try:
        from dev_spec_mcp import (
            check_code_style,
            search_todo_comments,
            validate_project_structure,
            get_naming_conventions,
            check_import_organization
        )
        print("✓ 成功导入开发规范工具")
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    
    # 测试1: 检查代码风格
    try:
        result = check_code_style(".")
        print(f"✓ 代码风格检查完成，扫描了 {result.get('total_files', 0)} 个文件")
        if result.get('summary'):
            print(f"  发现问题类型: {result['summary']}")
    except Exception as e:
        print(f"✗ 代码风格检查失败: {e}")
    
    # 测试2: 搜索TODO评论
    try:
        todos = search_todo_comments(".")
        print(f"✓ TODO/FIXME搜索完成，找到 {len(todos)} 条记录")
    except Exception as e:
        print(f"✗ TODO/FIXME搜索失败: {e}")
    
    # 测试3: 验证项目结构
    try:
        validation_result = validate_project_structure("python")
        print(f"✓ 项目结构验证完成，有效: {validation_result.get('is_valid', False)}")
    except Exception as e:
        print(f"✗ 项目结构验证失败: {e}")
    
    # 测试4: 获取命名规范
    try:
        naming_conv = get_naming_conventions("python")
        print(f"✓ 命名规范获取完成，语言: Python")
        print(f"  变量命名: {naming_conv.get('variables', 'N/A')}")
    except Exception as e:
        print(f"✗ 命名规范获取失败: {e}")
    
    # 测试5: 检查导入组织
    try:
        import_check = check_import_organization("dev_spec_mcp.py")
        print(f"✓ 导入组织检查完成: {import_check.get('file', 'N/A')}")
    except Exception as e:
        print(f"✗ 导入组织检查失败: {e}")
    
    return True

if __name__ == "__main__":
    print("开始测试开发规范MCP工具...")
    test_dev_spec_tools()
    print("测试完成！")