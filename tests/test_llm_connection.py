import os
import openai
from dotenv import load_dotenv

# --- 配置区 ---
# 加载项目根目录下的 .env 文件
# 如果您的 .env 文件在其他位置，请修改下面的路径
try:
    load_dotenv()
    print("✅ .env 文件加载成功。")
except Exception as e:
    print(f"⚠️  加载 .env 文件失败: {e}")

# 勾八就放两周够干嘛的，两周还一大堆事，无敌了，算下来睡觉的时间都不一定有
# 从环境变量中读取配置
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
# 您可以在 .env 中添加 OPENAI_MODEL，否则使用默认值
MODEL_NAME = os.getenv("MODEL")

# --- 测试逻辑 ---
def test_connection():
    """
    测试与 LLM 服务器的连接和认证。
    """
    print("-" * 50)
    print("🚀 开始测试 LLM 连接...")
    print(f"   - API Base URL: {BASE_URL or '未设置'}")
    print(f"   - 模型: {MODEL_NAME}")
    print(f"   - API Key: {'已设置' if API_KEY else '未设置'}")
    print("-" * 50)

    if not API_KEY:
        print("❌ 错误: 环境变量 OPENAI_API_KEY 未设置。")
        print("   请在 .env 文件中添加 OPENAI_API_KEY=your_api_key_here")
        return




    try:
        # --- API 调用（兼容新旧版本） ---
        reply = ""
        # 检查是否为新版 openai >= 1.0.0
        if hasattr(openai, "OpenAI"):
            print("🔧 检测到新版 (>=1.0.0) openai 库，使用客户端模式。")
            client = openai.OpenAI(
                api_key=API_KEY,
                base_url=BASE_URL,
                timeout=30.0,
            )
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": "Hello! Who are dire straits?"}],
            )
            reply = response.choices[0].message.content.strip()
        
        # 否则，假定为旧版 < 1.0.0
        else:
            print("🔧 检测到旧版 (<1.0.0) openai 库，使用旧版 API 模式。")
            openai.api_key = API_KEY
            if BASE_URL:
                openai.api_base = BASE_URL
            
            response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": "Hello! Who are dire straits?"}],
                timeout=30.0,
            )
            reply = response['choices'][0]['message']['content'].strip()

        print("\n✅ 连接成功！")
        print(f"🤖 模型回复:\n{reply}")

    # --- 异常处理（兼容新旧版本） ---
    except Exception as e:
        # 新版 openai 的特定异常
        if hasattr(openai, "AuthenticationError") and isinstance(e, openai.AuthenticationError):
            print("\n❌ 认证失败! API Key 无效或权限不足。")
            print(f"   - 错误详情: {e}")
        elif hasattr(openai, "APIConnectionError") and isinstance(e, openai.APIConnectionError):
            print(f"\n❌ 连接失败! 无法访问 API 地址: {BASE_URL}")
            print(f"   - 错误详情: {e}")
        elif hasattr(openai, "NotFoundError") and isinstance(e, openai.NotFoundError):
            print(f"\n❌ 找不到模型! 模型 '{MODEL_NAME}' 可能不存在。")
        elif hasattr(openai, "RateLimitError") and isinstance(e, openai.RateLimitError):
            print(f"\n❌ 请求速率超限!")
        
        # 旧版和通用的异常
        else:
            # 尝试从异常消息中识别旧版错误类型
            error_str = str(e).lower()
            if "authentication" in error_str or "api key" in error_str:
                print("\n❌ 认证失败! API Key 无效或权限不足。")
            elif "connection" in error_str or "timed out" in error_str:
                print(f"\n❌ 连接失败! 无法访问 API 地址: {BASE_URL}")
            elif "does not exist" in error_str:
                 print(f"\n❌ 找不到模型! 模型 '{MODEL_NAME}' 可能不存在。")
            else:
                print(f"\n❌ 发生未知错误:")
                print(f"   - 错误类型: {type(e).__name__}")
        
        print(f"   - 完整错误: {e}")


if __name__ == "__main__":
    test_connection() 