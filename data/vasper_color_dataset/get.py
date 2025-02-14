import requests
from bs4 import BeautifulSoup
import json
import time

# Valspar 色卡主页 URL
MAIN_URL = "https://www.valspar.com/en/colors/browse-colors/lowes"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

# 获取网页内容
def get_page_content(url):
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to retrieve {url}, Status Code: {response.status_code}")
        return None

# 解析主页，提取所有颜色类别和链接
def get_color_categories():
    page_content = get_page_content(MAIN_URL)
    categories = []
    
    if page_content:
        soup = BeautifulSoup(page_content, 'html.parser')
        category_links = soup.find_all("a", class_="category-tile")
        
        for link in category_links:
            href = link.get("href")
            if href and href.startswith("/en/colors/browse-colors/lowes/"):
                category_url = "https://www.valspar.com" + href
                categories.append(category_url)
    
    return categories

# 解析颜色类别页面，提取所有颜色链接
def get_colors_from_category(category_url):
    page_content = get_page_content(category_url)
    color_links = []
    
    if page_content:
        soup = BeautifulSoup(page_content, 'html.parser')
        color_items = soup.find_all("a", class_="color-tile")
        
        for item in color_items:
            link = item.get("href")
            if link and link.startswith("/en/colors/browse-colors/lowes/"):
                color_links.append("https://www.valspar.com" + link)
    
    return color_links

# 解析某个颜色的详细信息
def parse_color_details(color_url):
    page_content = get_page_content(color_url)
    
    if page_content:
        soup = BeautifulSoup(page_content, 'html.parser')
        name_tag = soup.find("h1", class_="color-name")
        hex_tag = soup.find("span", class_="color-hex")
        description_tag = soup.find("div", class_="color-description")
        family_tag = soup.find(text="Family")
        undertone_tag = soup.find(text="Undertone")
        collection_tag = soup.find(text="Collection")
        lrv_tag = soup.find(text="LRV")
        rgb_tag = soup.find(text="RGB")
        
        name = name_tag.text.strip() if name_tag else "Unknown"
        hex_code = hex_tag.text.strip() if hex_tag else "Unknown"
        description = description_tag.text.strip() if description_tag else "No description available"
        family = family_tag.find_next("div").text.strip() if family_tag else "Unknown"
        undertone = undertone_tag.find_next("div").text.strip() if undertone_tag else "Unknown"
        collection = collection_tag.find_next("div").text.strip() if collection_tag else "Unknown"
        lrv = lrv_tag.find_next("div").text.strip() if lrv_tag else "Unknown"
        rgb = rgb_tag.find_next("div").text.strip() if rgb_tag else "Unknown"
        
        return {
            "name": name,
            "hex": hex_code,
            "description": description,
            "family": family,
            "undertone": undertone,
            "collection": collection,
            "LRV": lrv,
            "RGB": rgb,
            "url": color_url
        }
    
    return None

# 爬取所有色卡数据
def scrape_valspar_colors():
    print("Fetching Valspar color data...")
    all_colors = []
    categories = get_color_categories()
    
    for category_url in categories:
        print(f"Scraping category: {category_url}")
        color_links = get_colors_from_category(category_url)
        
        for color_url in color_links:
            print(f"Scraping color: {color_url}")
            color_data = parse_color_details(color_url)
            if color_data:
                all_colors.append(color_data)
            time.sleep(1)  # 避免请求过快
    
    # 保存到JSON文件
    with open("valspar_colors.json", "w", encoding="utf-8") as f:
        json.dump(all_colors, f, indent=4, ensure_ascii=False)
    
    print(f"Scraped {len(all_colors)} colors and saved to valspar_colors.json.")

if __name__ == "__main__":
    scrape_valspar_colors()
