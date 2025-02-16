import openai
import os
import requests

# 从环境变量获取 API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORGANIZATION = os.getenv("OPENAI_ORGANIZATION")
BING_API_KEY = os.getenv("BING_API_KEY")
BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"

# 创建 OpenAI 客户端
client = openai.OpenAI(api_key=OPENAI_API_KEY, organization=OPENAI_ORGANIZATION)

def search_bing(color_code):
    """使用 Bing 搜索 color_code 相关的 RGB 颜色信息"""
    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    params = {"q": f"Valspar {color_code} color chip RGB", "count": 20, "textDecorations": True, "textFormat": "HTML"}
    
    response = requests.get(BING_SEARCH_URL, headers=headers, params=params)
    response.raise_for_status()
    results = response.json()
    
    # 提取前 20 条搜索结果的标题和摘要
    search_results = []
    for result in results.get("webPages", {}).get("value", [])[:20]:
        title = result.get("name", "")
        snippet = result.get("snippet", "")
        search_results.append(f"Title: {title}\nSnippet: {snippet}")
    
    return "\n\n".join(search_results)

def query_openai_for_color(color_code):
    """调用 OpenAI API，分析 Bing 搜索结果，提取 RGB 信息"""
    search_data = search_bing(color_code)
    prompt = f"Based on the following Bing search results, extract the RGB code for Valspar {color_code} color:\n\n{search_data}\n\nPlease return only the RGB code, such as (255, 0, 0)."

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content.strip()

# 测试查询
color_code = "8002-12A"
color_info = query_openai_for_color(color_code)
print(f"{color_code} 的颜色信息: {color_info}")
