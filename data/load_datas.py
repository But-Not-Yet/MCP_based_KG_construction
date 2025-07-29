# 通过政务文本语句构建低质量政务文本语句
import pandas as pd
import random


# 读取原始 CSV
df = pd.read_csv("D:\mcp_getting_started\data\政务文本语句.csv")
sentences = df["sentence"].tolist()

# 定义伪造规则
verbs = ["出台", "发布", "实施", "推动", "完成", "监督", "检查", "处理", "优化", "完善"]

def corrupt_sentence(sentence):
    corruption_types = ['remove_verb', 'word_shuffle', 'fragment', 'repeat_part']
    corruption = random.choice(corruption_types)

    if corruption == 'remove_verb':
        for verb in verbs:
            if verb in sentence:
                return sentence.replace(verb, '')


    elif corruption == 'word_shuffle':
        words = list(sentence)
        random.shuffle(words)
        return ''.join(words[:len(sentence)//2])

    elif corruption == 'fragment':
        if len(sentence) > 20:
            return sentence[:random.randint(5, len(sentence)//2)]
        else:
            return sentence

    elif corruption == 'repeat_part':
        if len(sentence) > 15:
            part = sentence[:random.randint(5, 10)]
            return part + part + sentence[random.randint(len(part), len(sentence)):]
        else:
            return sentence

    return sentence

# 生成低质量句子
low_quality_sentences = [corrupt_sentence(s) for s in sentences]

# 保存为 CSV
df["low_quality_sentence"] = low_quality_sentences
df.to_csv("政务主谓宾句子_低质量版本_1000条.csv", index=False, encoding="utf-8-sig")
