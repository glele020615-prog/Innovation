import json
import os

file_path = r"scan_history.json"

class R_W_JSON():
    def __init__(self,file_path = r"scan_history.json"):
        super().__init__()
        self.file_path = file_path

    def save_to_history(self, data_entry):

        # 1. 先读取现有数据
        history = self.load_all_history()

        # 2. 添加新条目
        history.append(data_entry)

        # 3. 写回文件
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=4)

    def load_all_history(self):
        if not os.path.exists(self.file_path):
            return []

        # 检查文件大小，如果为0字节，直接返回空列表
        if os.path.getsize(self.file_path) == 0:
            return []

        with open(self.file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return data
            except json.JSONDecodeError:
                # 如果文件内容乱码或格式错误，也返回空列表
                return []

    def delete_entry_by_path(self, target_path):
        """根据路径从 JSON 中删除记录"""
        try:
            # 1. 加载所有数据
            history = self.load_all_history()

            # 2. 过滤掉目标路径 (保留不匹配的)
            # 使用 os.path.normpath 确保路径格式一致，防止因斜杠方向不同导致匹配失败
            new_history = [
                item for item in history
                if os.path.normpath(item.get('file_path', '')) != os.path.normpath(target_path)
            ]

            # 3. 写回文件
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(new_history, f, ensure_ascii=False, indent=4)
            return True
        except Exception as e:
            print(f"删除 JSON 记录失败: {e}")
            return False