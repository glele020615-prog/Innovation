from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# 1. 收集 nilearn 包内的所有数据文件（含 glass_brain_files/*.json 等模板）
datas = collect_data_files('nilearn')

# 2. 收集 nilearn 全部子模块，避免延迟导入被漏掉
hiddenimports = collect_submodules('nilearn')
