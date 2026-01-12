#!/usr/bin/env python3
"""
自动更新贡献者列表
- 从 metadata.json 获取工具作者
- 从 tools.json 获取实际工具函数数量
- 生成简洁的 CONTRIBUTORS.md
- 更新前端展示
"""
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

def get_tool_authors_with_details() -> Dict[str, Dict]:
    """从 metadata.json 和 tools.json 获取详细的工具作者信息"""
    root_dir = Path(__file__).parent.parent
    servers_dir = root_dir / "servers"
    tools_json_path = root_dir / "data" / "tools.json"
    
    # 先加载 tools.json 获取完整信息
    if tools_json_path.exists():
        with open(tools_json_path, 'r', encoding='utf-8') as f:
            tools_data = json.load(f)
    else:
        # 如果 tools.json 不存在，从 metadata 文件构建
        tools_data = {"tools": []}
        for server_path in sorted(servers_dir.iterdir()):
            if server_path.is_dir() and not server_path.name.startswith('_'):
                metadata_path = server_path / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                            tools_data["tools"].append(metadata)
                    except:
                        continue
    
    # 构建作者信息字典
    author_info = defaultdict(lambda: {
        'collections': [],
        'tools': [],
        'categories': set()
    })
    
    for tool in tools_data.get('tools', []):
        author = tool.get('author', '@unknown')
        collection_name = tool.get('name', 'Unknown')
        category = tool.get('category', 'general')
        tool_functions = tool.get('tools', [])
        
        author_info[author]['collections'].append(collection_name)
        author_info[author]['tools'].extend(tool_functions)
        author_info[author]['categories'].add(category)
    
    # 转换为最终格式
    result = {}
    for author, info in author_info.items():
        result[author] = {
            'collections': sorted(info['collections']),  # Sort collections alphabetically
            'collections_count': len(info['collections']),
            'tools': sorted(info['tools']),  # Sort tools alphabetically
            'tools_count': len(info['tools']),
            'categories': sorted(list(info['categories']))  # Sort categories alphabetically
        }
    
    return result

def generate_contributors_md(authors_details: Dict[str, Dict]) -> str:
    """生成简洁的 CONTRIBUTORS.md 内容"""
    content = """# Contributors

Thank you to all our contributors! 🎉

## Tool Authors

| Author | Collections | Tools | Main Areas |
|--------|------------|-------|------------|
"""
    
    # 按工具数量排序
    sorted_authors = sorted(authors_details.items(), key=lambda x: x[1]['tools_count'], reverse=True)
    
    for author, details in sorted_authors:
        collections_str = ', '.join(details['collections'])
        categories_str = ', '.join(details['categories'])
        
        if author.startswith('@'):
            github_username = author[1:]
            author_link = f"[{author}](https://github.com/{github_username})"
        else:
            author_link = author
            
        content += f"| **{author_link}** | {details['collections_count']} ({collections_str}) | {details['tools_count']} | {categories_str} |\n"
    
    content += f"""

## Stats

- **Contributors**: {len(authors_details)}
- **Total Collections**: {sum(d['collections_count'] for d in authors_details.values())}
- **Total Tools**: {sum(d['tools_count'] for d in authors_details.values())}

## How to Contribute

Check out our [Contributing Guide](CONTRIBUTING.md) to get started!

---

*Auto-generated from metadata.json files*
"""
    
    return content

def merge_contributors_data(existing_data: List[Dict], new_data: List[Dict]) -> List[Dict]:
    """Merge new contributors data with existing data."""
    # Create mapping by author name
    existing_map = {c['author']: c for c in existing_data}
    merged = []
    
    # Process new contributors
    for new_contributor in new_data:
        author = new_contributor['author']
        if author in existing_map:
            # Update existing contributor
            existing = existing_map[author]
            # Update counts and lists from new data
            existing['collections'] = new_contributor['collections']
            existing['collections_count'] = new_contributor['collections_count']
            existing['tools_count'] = new_contributor['tools_count']
            existing['categories'] = new_contributor['categories']
            merged.append(existing)
        else:
            # Add new contributor
            merged.append(new_contributor)
    
    return merged

def save_contributors_json(authors_details: Dict[str, Dict]):
    """保存贡献者数据为 JSON 供前端使用"""
    root_dir = Path(__file__).parent.parent
    
    # 构造前端需要的贡献者列表
    contributors_list = []
    for author, details in authors_details.items():
        contributors_list.append({
            'author': author,
            'collections': details['collections'],
            'collections_count': details['collections_count'],
            'tools_count': details['tools_count'],
            'categories': details['categories']
        })
    
    # 按工具数量排序
    contributors_list.sort(key=lambda x: x['tools_count'], reverse=True)
    
    contributors_json = root_dir / "data" / "contributors.json"
    
    # Load existing data if exists
    existing_data = None
    if contributors_json.exists():
        try:
            with open(contributors_json, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except Exception as e:
            print(f"⚠️ 无法加载现有 contributors.json: {e}")
    
    # Merge if existing data found
    if existing_data and 'contributors' in existing_data:
        contributors_list = merge_contributors_data(existing_data['contributors'], contributors_list)
        # Re-sort after merge
        contributors_list.sort(key=lambda x: x['tools_count'], reverse=True)
    
    data = {
        'contributors': contributors_list,
        'total_contributors': len(authors_details),
        'total_collections': sum(d['collections_count'] for d in authors_details.values()),
        'total_tools': sum(d['tools_count'] for d in authors_details.values())
    }
    
    with open(contributors_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 贡献者数据已保存到: {contributors_json}")

def main():
    """主函数"""
    print("🔍 正在收集贡献者信息...")
    
    # 获取详细的工具作者信息
    authors_details = get_tool_authors_with_details()
    
    print(f"🔬 找到 {len(authors_details)} 位工具作者")
    
    # 生成 CONTRIBUTORS.md
    content = generate_contributors_md(authors_details)
    
    root_dir = Path(__file__).parent.parent
    contributors_md = root_dir / "data" / "CONTRIBUTORS.md"
    
    with open(contributors_md, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ CONTRIBUTORS.md 已更新: {contributors_md}")
    
    # 保存 JSON 数据供前端使用
    save_contributors_json(authors_details)
    
    print("\n🎉 贡献者列表更新完成！")
    print("💡 提示：运行 'python scripts/generate_simple_showcase.py' 更新前端展示")
    
    # 打印统计摘要
    print("\n📊 贡献者统计：")
    for author, details in sorted(authors_details.items(), key=lambda x: x[1]['tools_count'], reverse=True)[:5]:
        print(f"  {author}: {details['collections_count']} collections, {details['tools_count']} tools")

if __name__ == "__main__":
    main()