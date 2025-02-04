import requests
from bs4 import BeautifulSoup
import re

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

def get_weather_data():
    # 模拟实际请求的URL（根据提供的HTML结构调整）
    url = 'https://weather.cma.cn/web/weather/59082.html'  # 韶关的正确页面链接
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print("无法获取天气数据，请检查网络连接或网站状态")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    weather_data = []

    # 查找7天天气预报容器
    forecast_div = soup.find('div', id='dayList')
    if not forecast_div:
        print("找不到天气预报容器")
        return None

    # 提取每一天的天气预报
    for day_div in forecast_div.find_all('div', class_='day'):
        try:
            # 提取日期
            date = day_div.find('div', class_='day-item').get_text(strip=True).replace('\n', ' ')
            
            # 提取天气状况（日间）
            day_weather = day_div.find_all('div', class_='day-item')[2].get_text(strip=True)
            night_weather = day_div.find_all('div', class_='day-item')[6].get_text(strip=True)
            weather = f"{day_weather}转{night_weather}"
            
            # 提取温度
            temp_high = day_div.find('div', class_='high').get_text(strip=True)
            temp_low = day_div.find('div', class_='low').get_text(strip=True)
            
            # 提取风力
            wind_day = day_div.find_all('div', class_='day-item')[3].get_text(strip=True)
            wind_speed_day = day_div.find_all('div', class_='day-item')[4].get_text(strip=True)
            wind = f"{wind_day}{wind_speed_day}"

            weather_data.append({
                '日期': date,
                '天气': weather,
                '最高温': int(temp_high.replace('℃', '')),
                '最低温': int(temp_low.replace('℃', '')),
                '风力': wind
            })
        except Exception as e:
            print(f"解析数据时出错: {str(e)}")
            continue

    return weather_data

def analyze_weather(data):
    if not data:
        return

    # 温度分析
    high_temps = [d['最高温'] for d in data]
    low_temps = [d['最低温'] for d in data]
    
    avg_high = sum(high_temps) / len(high_temps)
    avg_low = sum(low_temps) / len(low_temps)
    
    max_high = max(high_temps)
    min_low = min(low_temps)
    
    # 降雨天数分析
    rainy_days = len([d for d in data if '雨' in d['天气']])
    
    # 风力分析
    max_wind = 0
    for d in data:
        if match := re.search(r'(\d+)m/s', d['风力']):
            wind_speed = int(match.group(1))
            if wind_speed > max_wind:
                max_wind = wind_speed

    print("\n韶关7日天气预报分析结果：")
    print(f"1. 平均最高温度：{avg_high:.1f}℃")
    print(f"2. 平均最低温度：{avg_low:.1f}℃")
    print(f"3. 极端最高温度：{max_high}℃")
    print(f"4. 极端最低温度：{min_low}℃")
    print(f"5. 预计有雨天数：{rainy_days}天")
    print(f"6. 最大风速：{max_wind}m/s")

if __name__ == "__main__":
    weather_data = get_weather_data()
    
    if weather_data:
        print("成功获取韶关7日天气预报：")
        for day in weather_data:
            print(f"{day['日期']}: {day['天气']}，{day['最高温']}℃/{day['最低温']}℃，{day['风力']}")
        
        analyze_weather(weather_data)
    else:
        print("未能获取天气数据")