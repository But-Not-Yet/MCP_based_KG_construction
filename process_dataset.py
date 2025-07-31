import json
import os
from collections import defaultdict

def process_news_dataset():
    """
    读取 train.json 数据集，根据新闻类别将其拆分并格式化输出到不同的 txt 文件中。
    """
    # 定义输入文件和输出目录
    input_file_path = 'data/train.json'
    output_dir = 'data/processed_dataset'

    # 1. 确保输出目录存在
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"输出目录 '{output_dir}' 已准备就绪。")
    except OSError as e:
        print(f"错误：无法创建目录 {output_dir}: {e}")
        return

    # 2. 检查输入文件是否存在
    if not os.path.exists(input_file_path):
        print(f"错误：输入文件不存在于 '{input_file_path}'")
        return

    # 使用 defaultdict 方便地将语句按类别分组
    categorized_data = defaultdict(list)

    # 3. 读取和处理数据
    print(f"正在从 '{input_file_path}' 读取数据...")
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 忽略空行
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    label_desc = record.get('label_desc')
                    sentence = record.get('sentence')

                    if label_desc and sentence:
                        categorized_data[label_desc].append(sentence)
                    else:
                        print(f"警告：跳过缺少 'label_desc' 或 'sentence' 的行: {line.strip()}")
                except json.JSONDecodeError:
                    print(f"警告：跳过无效的 JSON 行: {line.strip()}")
    except Exception as e:
        print(f"读取或处理文件时出错: {e}")
        return

    print("数据读取和分类完成。")

    # 4. 将分类后的数据写入单独的文件
    if not categorized_data:
        print("没有可写的数据。")
        return

    print("正在将数据写入文件...")
    for category, sentences in categorized_data.items():
        # 清理类别名称以用作文件名
        safe_filename = "".join(c for c in category if c.isalnum() or c in ('_', '-')).rstrip()
        output_file_path = os.path.join(output_dir, f"{safe_filename}.txt")

        try:
            with open(output_file_path, 'w', encoding='utf-8') as f_out:
                for i, sentence in enumerate(sentences, 1):
                    f_out.write(f"{i}\t{sentence}\n")
            print(f"-> 已成功写入 {len(sentences)} 条语句到 '{output_file_path}'")
        except IOError as e:
            print(f"错误：无法写入文件 {output_file_path}: {e}")

    print(f"\n处理完成！所有文件都已保存到 '{output_dir}' 目录中。")

if __name__ == "__main__":
    process_news_dataset() 