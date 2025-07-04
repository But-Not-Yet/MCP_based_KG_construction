import requests
import json
import re

class ChineseEntityRelationExtractor:
    """中文实体抽取与关系抽取（使用Silicon Flow API）"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.siliconflow.cn/v1/chat/completions"
        self.model = "deepseek-ai/DeepSeek-R1"  # 使用中文模型
        
    def call_api(self, prompt):
        """调用Silicon Flow API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"API调用失败: {e}")
            return None
    
    def extract_triplets(self, text):
        """从中文文本中抽取三元组（头实体-关系-尾实体）"""
        prompt = f"""
请从以下中文文本中抽取实体和它们之间的关系，以三元组的形式输出。

文本："{text}"

请按照以下格式输出三元组：
(头实体, 关系, 尾实体)

注意事项：
1. 只输出三元组，每行一个
2. 头实体和尾实体必须是文本中出现的具体实体
3. 关系要准确描述两个实体之间的关系
4. 如果没有明确的关系，不要强行创造
5. 常见的关系类型包括：工作于、位于、属于、创建、管理、投资、合作等

示例格式：
(张三, 工作于, 阿里巴巴)
(北京, 位于, 中国)
(苹果公司, 生产, iPhone)
"""
        
        response = self.call_api(prompt)
        if response:
            return self.parse_triplets(response)
        return []
    
    def parse_triplets(self, response):
        """解析API返回的三元组"""
        triplets = []
        lines = response.strip().split('\n')
        
        for line in lines:
            # 匹配 (头实体, 关系, 尾实体) 格式
            match = re.search(r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)', line)
            if match:
                head = match.group(1).strip()
                relation = match.group(2).strip()
                tail = match.group(3).strip()
                triplets.append((head, relation, tail))
        
        return triplets
    
    def print_results(self, text, triplets):
        """打印结果"""
        print("=" * 50)
        print("原文本：")
        print(text)
        print("\n抽取的三元组：")
        if triplets:
            for i, (head, relation, tail) in enumerate(triplets, 1):
                print(f"{i}. ({head}, {relation}, {tail})")
        else:
            print("未找到明确的实体关系")
        print("=" * 50)

def main():
    print("中文实体抽取与关系抽取系统（基于Silicon Flow API）")
    
    # API密钥
    api_key = "sk-igsxqudwjumptrovmyuuxemhjvhwqxnhegsuuswqpipnxfre"
    
    # 创建抽取器
    extractor = ChineseEntityRelationExtractor(api_key)
    
    print("\n开始使用！输入'quit'退出程序")
    
    while True:
        text = input("\n请输入要分析的中文文本: ").strip()
        
        if text.lower() == 'quit':
            print("程序退出，再见！")
            break
        
        if not text:
            print("文本不能为空，请重新输入")
            continue
        
        print("正在分析中...")
        triplets = extractor.extract_triplets(text)
        extractor.print_results(text, triplets)

if __name__ == "__main__":
    main()
