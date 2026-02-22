import os
import re
from pathlib import Path
from typing import List, Dict, Any
from mcp.server.fastmcp import FastMCP

# 创建 MCP Server
mcp = FastMCP("开发规范检查器")

@mcp.tool()
def check_code_style(directory: str = ".") -> Dict[str, Any]:
    """检查指定目录中的代码风格规范
    
    Args:
        directory: 要检查的目录路径，默认为当前目录
        
    Returns:
        包含检查结果的字典
    """
    path = Path(directory).resolve()
    
    if not path.exists():
        return {"error": f"目录不存在: {directory}"}
    
    results = {
        "total_files": 0,
        "violations": [],
        "summary": {}
    }
    
    # 检查Python文件的常见规范问题
    py_files = list(path.rglob("*.py"))
    
    for py_file in py_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            violations = []
            
            # 检查行长度（超过88字符）
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if len(line) > 88:
                    violations.append({
                        "file": str(py_file),
                        "line": i,
                        "type": "line_length",
                        "message": f"Line too long ({len(line)} > 88 characters)"
                    })
            
            # 检查缺少的文档字符串
            if '"""' not in content and "'''" not in content:
                violations.append({
                    "file": str(py_file),
                    "type": "missing_docstring",
                    "message": "Missing module docstring"
                })
            
            # 检查不符合命名约定的变量/函数名
            # 简单检查：全大写的变量名（除了常量）
            pattern = r'\b([a-z][a-zA-Z0-9_]*)\s*=\s*'
            matches = re.findall(pattern, content)
            for match in matches:
                if match.isupper():
                    violations.append({
                        "file": str(py_file),
                        "type": "naming_convention",
                        "message": f"Variable name '{match}' appears to violate naming convention (should be lowercase with underscores)"
                    })
            
            results["total_files"] += 1
            results["violations"].extend(violations)
            
        except Exception as e:
            results["violations"].append({
                "file": str(py_file),
                "type": "error",
                "message": f"Error reading file {py_file}: {str(e)}"
            })
    
    # 生成摘要
    violation_types = [v["type"] for v in results["violations"]]
    summary = {vt: violation_types.count(vt) for vt in set(violation_types)}
    results["summary"] = summary
    
    return results


@mcp.tool()
def search_todo_comments(directory: str = ".") -> List[Dict[str, Any]]:
    """搜索代码中的TODO和FIXME注释
    
    Args:
        directory: 要搜索的目录路径，默认为当前目录
        
    Returns:
        包含TODO/FIXME注释信息的列表
    """
    path = Path(directory).resolve()
    
    if not path.exists():
        return [{"error": f"Directory does not exist: {directory}"}]
    
    todo_items = []
    patterns = [r'TODO:', r'FIXME:', r'XXX:', r'HACK:']
    
    # 搜索各种类型的源代码文件
    source_files = []
    for ext in ['*.py', '*.js', '*.ts', '*.jsx', '*.tsx', '*.java', '*.cpp', '*.c', '*.h']:
        source_files.extend(list(path.rglob(ext)))
    
    for source_file in source_files:
        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines, 1):
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        todo_items.append({
                            "file": str(source_file),
                            "line_number": i,
                            "type": pattern.strip(':'),
                            "content": line.strip(),
                            "context": line.strip()
                        })
                        
        except Exception as e:
            todo_items.append({
                "file": str(source_file),
                "error": f"Error reading file: {str(e)}"
            })
    
    return todo_items


@mcp.tool()
def validate_project_structure(project_type: str = "python") -> Dict[str, Any]:
    """验证项目结构是否符合最佳实践
    
    Args:
        project_type: 项目类型 ("python", "javascript", "java")
        
    Returns:
        验证结果和建议
    """
    result = {
        "project_type": project_type,
        "issues": [],
        "recommendations": [],
        "is_valid": True
    }
    
    current_dir = Path.cwd()
    
    if project_type.lower() == "python":
        # Python项目常见结构验证
        expected_files = ["requirements.txt", "setup.py", "pyproject.toml", "README.md"]
        expected_dirs = ["src", "tests", "docs", "config", "scripts"]
        
        found_expected_files = []
        missing_expected_files = []
        
        for ef in expected_files:
            ef_path = current_dir / ef
            if ef_path.exists():
                found_expected_files.append(ef)
            else:
                missing_expected_files.append(ef)
        
        found_expected_dirs = []
        missing_expected_dirs = []
        
        for ed in expected_dirs:
            ed_path = current_dir / ed
            if ed_path.exists():
                found_expected_dirs.append(ed)
            else:
                missing_expected_dirs.append(ed)
        
        if missing_expected_files:
            result["issues"].append({
                "type": "missing_files",
                "files": missing_expected_files,
                "message": f"Missing recommended files: {missing_expected_files}"
            })
        
        if missing_expected_dirs:
            result["recommendations"].append({
                "type": "missing_directories",
                "dirs": missing_expected_dirs,
                "message": f"Consider adding these directories: {missing_expected_dirs}"
            })
        
        # 检查src是否有合适的子结构
        src_path = current_dir / "src"
        if src_path.exists():
            src_contents = list(src_path.iterdir())
            if not src_contents:
                result["recommendations"].append({
                    "type": "empty_src",
                    "message": "The src directory is empty, consider adding your main package/modules here"
                })

    elif project_type.lower() == "javascript":
        expected_files = ["package.json", "README.md", "webpack.config.js", "rollup.config.js", "vite.config.js"]
        expected_dirs = ["src", "dist", "public", "tests", "docs", "__tests__", "__mocks__"]
        
        found_expected_files = []
        missing_expected_files = []
        
        for ef in expected_files:
            ef_path = current_dir / ef
            if ef_path.exists():
                found_expected_files.append(ef)
            else:
                missing_expected_files.append(ef)
        
        found_expected_dirs = []
        missing_expected_dirs = []
        
        for ed in expected_dirs:
            ed_path = current_dir / ed
            if ed_path.exists():
                found_expected_dirs.append(ed)
            else:
                missing_expected_dirs.append(ed)
        
        if missing_expected_files:
            result["issues"].append({
                "type": "missing_files",
                "files": missing_expected_files,
                "message": f"Missing recommended files: {missing_expected_files}"
            })
        
        if missing_expected_dirs:
            result["recommendations"].append({
                "type": "missing_directories",
                "dirs": missing_expected_dirs,
                "message": f"Consider adding these directories: {missing_expected_dirs}"
            })

    else:
        result["is_valid"] = False
        result["issues"].append({
            "type": "unsupported_type",
            "message": f"Project type '{project_type}' is not supported yet"
        })
    
    if not result["issues"]:
        result["is_valid"] = True
    else:
        result["is_valid"] = len(result["issues"]) == 0
    
    return result


@mcp.tool()
def get_naming_conventions(language: str = "python") -> Dict[str, str]:
    """获取指定编程语言的命名规范
    
    Args:
        language: 编程语言 ("python", "javascript", "java", "go")
        
    Returns:
        命名规范指南
    """
    conventions = {
        "python": {
            "variables": "snake_case (lower_with_underscores)",
            "functions": "snake_case (lower_with_underscores)",
            "classes": "PascalCase (UpperFirstWithNoSpaces)",
            "constants": "UPPER_CASE_WITH_UNDERSCORES",
            "modules": "snake_case (lower_with_underscores)",
            "packages": "snake_case (lower_with_underscores)"
        },
        "javascript": {
            "variables": "camelCase",
            "functions": "camelCase",
            "classes": "PascalCase",
            "constants": "UPPER_CASE or camelCase depending on scope",
            "modules": "camelCase or kebab-case",
            "packages": "kebab-case"
        },
        "java": {
            "variables": "camelCase",
            "functions": "camelCase",
            "classes": "PascalCase",
            "constants": "UPPER_CASE_WITH_UNDERSCORES",
            "packages": "lowercase_with_dots.for.subpackages"
        },
        "go": {
            "variables": "camelCase or PascalCase for exported",
            "functions": "camelCase or PascalCase for exported",
            "classes_structs": "PascalCase",
            "constants": "UPPER_CASE or CamelCase",
            "packages": "lowercase_snake_case_preferred"
        }
    }
    
    if language.lower() in conventions:
        return conventions[language.lower()]
    else:
        return {
            "info": f"Naming conventions for {language} are not defined in this system.",
            "available": list(conventions.keys())
        }


@mcp.tool()
def check_import_organization(file_path: str) -> Dict[str, Any]:
    """检查Python文件中的导入语句组织是否符合PEP 8规范
    
    Args:
        file_path: Python文件路径
        
    Returns:
        导入组织检查结果
    """
    path = Path(file_path)
    
    if not path.exists():
        return {"error": f"File does not exist: {file_path}"}
    
    if path.suffix != '.py':
        return {"error": f"Not a Python file: {file_path}"}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        return {"error": f"Error reading file: {str(e)}"}
    
    # 检查导入部分
    import_lines = []
    stdlib_imports = []
    third_party_imports = []
    local_imports = []
    
    in_import_section = True
    
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        
        # 检测是否结束导入部分
        if in_import_section:
            if stripped_line.startswith('#') or stripped_line == '':
                continue
            
            if (not stripped_line.startswith('import ') and 
                not stripped_line.startswith('from ') and 
                not stripped_line.startswith('try:') and
                not stripped_line.startswith('except')):
                
                in_import_section = False
                continue
        
        if in_import_section:
            import_lines.append((i+1, stripped_line))
            
            # 分类导入语句
            if stripped_line.startswith('import ') or stripped_line.startswith('from '):
                # 简单判断标准库、第三方库和本地库
                parts = stripped_line.replace('from ', '').replace('import ', '').split('.')
                first_part = parts[0].strip()
                
                stdlib_modules = [
                    'os', 'sys', 'json', 're', 'datetime', 'collections', 'itertools',
                    'functools', 'pathlib', 'argparse', 'logging', 'random', 'math',
                    'urllib', 'http', 'email', 'xml', 'csv', 'io', 'time', 'threading',
                    'multiprocessing', 'subprocess', 'shutil', 'glob', 'zipfile'
                ]
                
                if first_part in stdlib_modules:
                    stdlib_imports.append((i+1, stripped_line))
                else:
                    # 假设不是标准库的就是第三方库（这不完全准确，但作为示例）
                    third_party_imports.append((i+1, stripped_line))
    
    result = {
        "file": file_path,
        "total_import_lines": len(import_lines),
        "stdlib_imports": stdlib_imports,
        "third_party_imports": third_party_imports,
        "local_imports": local_imports,
        "recommendation": "Imports should be organized in 3 groups: standard library, third-party, local application/library specific"
    }
    
    return result


if __name__ == "__main__":
    # 如果检测到被 mcp dev 调用，就导出官方 Server 对象
    import sys
    if "mcp" in sys.modules and hasattr(mcp, "_server"):
        # fastmcp 内部已经包了一个官方 Server，直接暴露
        mcp._server.run()
    else:
        # 平时手动 python dev_spec_mcp.py 就走 fastmcp 自己的启动
        mcp.run()