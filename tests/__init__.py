import sys
import os
 
# 确保项目根目录在 sys.path 中，便于直接执行测试脚本
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR) 