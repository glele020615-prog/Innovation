# 智影识微：基于 MRI 影像的 MCI 进展风险智能预测平台 — 项目说明文档

> 本项目是一个基于 PySide6 的桌面应用程序，用于阿尔茨海默症（AD）/ 轻度认知障碍（MCI）的辅助诊断与进展风险预测。核心算法为 **SA-STGCN（Self-Attention Spatio-Temporal Graph Convolutional Network，自注意力时空图卷积网络）**，以静息态 fMRI 数据为输入，输出患病风险及可解释分析。

---

## 一、总体架构

程序由主窗口（`项目.py`）统一调度，内部通过 `QStackedWidget` 管理 6 个页面：

| 序号 | 页面 | 对应模块 / 类 |
|------|------|---------------|
| 0 | 首页概览 | `HomePageWidget` / `HomeImageCard`（位于 `项目.py`） |
| 1 | 数据加载 | `DataManagerPage`（`项目.py`） |
| 2 | 预处理 | `PreprocessingPage`（`Data_Pre.py`） |
| 3 | 特征提取 | `FeatureExtractionPage`（`feature_extraction_page.py`） |
| 4 | 疾病预测诊断 | `DiagnosisPage`（`predict.py`） |
| 5 | 个体报告分析 | `IndividualReportPage`（`individual_report_page.py`） |

页面间通过主窗口的 `shared_case_data` 传递病例数据，诊断完成后自动跳转到个体报告页。

---

## 二、文件清单与详细说明

### 1. 主程序与界面

#### `项目.py` — 主程序入口与界面骨架
- **作用**：整个应用的启动入口和主控窗口（`ADDiagnosisSystem`）。负责：
  - 构建顶部导航栏（`build_top_navigation`）、首页（`build_home_page`）。
  - 初始化并堆叠 6 个功能页面（`init_pages`）。
  - 定义深色科技风全局样式表（深蓝背景 + 亮蓝高亮）。
  - 提供跨页面数据共享与跳转逻辑（`open_individual_report`）。
  - 内置多个自定义控件：`CoverImageWidget`（封面图）、`HomePageWidget`（首页）、`AngledNavButton`（斜角导航按钮）、`NavBeamSeparator`（导航分隔条）、`HomeImageCard`（首页功能卡片）、`DataManagerPage`（数据加载页）。
- **关键辅助函数**：
  - `get_resource_path(*parts)`：按是否打包（PyInstaller）返回资源文件绝对路径。
  - `get_preferred_ui_font(point_size=10)`：加载系统中文字体（微软雅黑/苹方/思源黑体）。
- **使用**：直接运行 `python 项目.py` 启动（需先安装依赖）。

#### `ui/fmri_analyzer.ui` 与 `other/fmri_analyzer.ui` / `other/fmri_analyzer.py`
- **作用**：Qt Designer 生成的 UI 文件及其对应的 Python 代码（`Ui_MainWindow`）。这是一个 **较旧版本的 fMRI 分析界面**（含数据导入、预处理勾选、模型选择、可视化、报告生成等标签页），由 `pyuic` 编译生成。`other/` 下为早期原型代码，与主程序 `项目.py` 是两套独立界面，**前者是正式版，后者是参考原型**。
- **使用**：正式运行使用 `项目.py`；原型仅供历史参考或二次开发。

#### `other/main.py` / `other/main_window.py` / `other/open_window.py` / `other/test.py`
- **作用**：早期实验性窗口与测试脚本（如 `main_window.py` 组装旧界面、`test.py` 功能验证）。属于开发过程留存代码。
- **使用**：非正式入口，调试时可单独运行。

#### `other/test.ui` / `other/test.py`
- **作用**：临时测试界面与脚本。

---

### 2. 影像查看与可视化

#### `viewer.py` — fMRI 影像浏览器控件
- **作用**：提供 `fMRIViewWidget` 类，用于在界面中加载、切片浏览 NIfTI（`.nii`/`.nii.gz`）脑影像，并渲染指定切片的 2D 预览图。
- **使用**：被 `项目.py` 及其他页面 `import` 使用：`from viewer import fMRIViewWidget`。
- **核心能力**：读取影像维度/体素信息、按轴（x/y/z）渲染切片、配合鼠标滚轮或滑块切换层。

---

### 3. 数据预处理

#### `Data_Pre.py` — 预处理页面
- **作用**：定义 `PreprocessingPage` 类，提供 MRI/fMRI 数据预处理流程界面与逻辑，包括：
  - 头动校正（Motion Correction）
  - 时间层校正（Slice Timing）
  - 空间标准化（Spatial Normalization）
  - 平滑（Smoothing）、滤波（Filtering）、去线性漂移（Detrending）
  - 参数配置（平滑核大小、高/低通滤波、TR 等）
- **使用**：作为第 2 个标签页，由主程序 `from Data_Pre import PreprocessingPage` 加载。

---

### 4. 特征提取

#### `feature_extraction_page.py` — 特征提取页面
- **作用**：定义 `FeatureExtractionPage` 类，从预处理后的 fMRI 数据中提取用于模型输入的特征，例如：
  - 基于 AAL116 脑区模板提取时序信号
  - 构建脑功能连接矩阵（connectome）
  - 为 SA-STGCN 模型准备图结构输入
- **使用**：第 3 个标签页，依赖 `Node_AAL116.node` 脑区模板与 AAL116 重采样文件。

---

### 5. 疾病预测诊断（核心模型）

#### `predict.py` — 诊断预测页面
- **作用**：定义 `DiagnosisPage` 类，加载训练好的 SA-STGCN 模型，对提取的特征进行推理，输出：
  - 患病风险概率 / 分类结果（如 pMCI / sMCI / AD / NC）
  - 关键脑区与可解释分析
- **使用**：第 4 个标签页，`from predict import DiagnosisPage`。推理结果通过主窗口 `open_individual_report(case_data)` 传递并跳转。

#### `SelfAttentionBlock_ST_GCN.py` — 自注意力时空图卷积网络
- **作用**：定义 SA-STGCN 的核心神经网络模块（`SelfAttentionBlock` 等）。将脑区视为图节点、功能连接视为边，结合时空图卷积与自注意力机制，捕捉脑网络随时间的动态依赖关系。
- **使用**：被 `predict.py` 在加载/构建模型时 `import`，是预测能力的核心算法实现。
- **关键产出**：训练得到的权重文件 `best_acc_model_fold_1.pt`（5 折交叉验证中准确率最高的第 1 折模型），由 `predict.py` 加载用于推理。

---

### 6. 报告生成与可解释性

#### `individual_report_page.py` — 个体报告页面
- **作用**：定义 `IndividualReportPage` 类，接收诊断结果，生成面向个体/临床的报告视图，包含风险结论、关键脑区、可视化图表等。
- **使用**：第 5 个标签页，由主程序在诊断完成后自动跳转并 `load_case(case_data)`。

#### `report_explain_utils.py` — 报告解释工具
- **作用**：辅助生成报告中的可解释内容（如关键脑区说明、风险解读文本模板）。
- **使用**：被 `individual_report_page.py` 调用。

#### `explainability_utils.py` — 可解释性工具
- **作用**：提供模型可解释性相关计算（如注意力权重可视化、脑区贡献度排序、热力图生成）。
- **使用**：被 `predict.py` / `individual_report_page.py` 调用，用于产出可解释结果。

#### `Read_Write_JSON.py` — JSON 读写工具
- **作用**：定义 `R_W_JSON` 类，统一读写病例/扫描配置 JSON（如 `scan_history.json`、特征缓存）。
- **使用**：被主程序及多个页面 `from Read_Write_JSON import R_W_JSON` 调用，用于持久化病例数据与扫描历史。

---

### 7. 数据与资源文件

| 文件 / 目录 | 说明 |
|-------------|------|
| `Data/Dicom/` | 原始 DICOM 序列（89 个 `.dcm` 文件） |
| `Data/nii/` | 转换后的 NIfTI 影像目录 |
| `Data/output/` `Data/test_output/` | FreeSurfer / 预处理输出（`.mgz`、`.label`、`.annot`、日志等） |
| `Data/pMCI/` `Data/sMCI/` `Data/test/` | 不同类别（进展型/稳定型 MCI）样本数据 |
| `Data/tmp/` | 临时文件（`.pklz` 缓存、`.json`、`.rst` 报告等） |
| `Node_AAL116.node` | AAL116 脑区节点模板（图卷积节点定义） |
| `label_order_jian.node` | 脑区标签顺序定义文件 |
| `best_acc_model_fold_1.pt` | 训练好的 SA-STGCN 推理权重 |
| `assets/` | 界面图片：`blackground.png`（首页背景）、`fmri_preview.png`、`connectome_preview.png`、`ai_analysis_preview.png`（首页功能卡片图） |
| `scan_history.json` | 扫描/病例历史记录 |
| `individual_report.pdf` | 已生成的个体报告样例 |
| `report_cache/` | 报告缓存图片 |
| `FreeSurfer/` | FreeSurfer 相关说明/工具文件 |

---

### 8. 打包与构建

| 文件 | 说明 |
|------|------|
| `ADDiagnosisSystem.spec` | PyInstaller 打包配置（包含入口、隐藏导入、资源收集等） |
| `build_exe.bat` / `build_exe.ps1` | Windows 下打包成 exe 的脚本（批处理 / PowerShell） |
| `dist/` | 打包生成的可执行文件与依赖（含大量 `.dll` 等） |
| `build/` | 打包中间产物 |
| `__pycache__/` | Python 字节码缓存 |

---

## 三、运行与使用方式

### 1. 环境依赖
- Python 3.x（建议 3.9+）
- PySide6
- numpy、nibabel、matplotlib、torch（用于模型推理）
- 其他：FreeSurfer 相关工具（预处理）

### 2. 启动开发版
```bash
cd f:/Innovation
python 项目.py
```

### 3. 标准使用流程
1. **首页概览**：查看系统简介，点击「开始」进入数据加载。
2. **数据加载**：通过 `DataManagerPage` 选择/导入 NIfTI 或 DICOM 影像，预览脑切片，查看维度、体素、TR 等参数。
3. **预处理**：在 `PreprocessingPage` 勾选所需步骤并设置参数，运行预处理。
4. **特征提取**：`FeatureExtractionPage` 基于 AAL116 提取时序与功能连接特征。
5. **疾病预测诊断**：`DiagnosisPage` 加载 `best_acc_model_fold_1.pt`，推理输出风险与可解释结果。
6. **个体报告分析**：自动跳转 `IndividualReportPage`，生成并查看/导出个体诊断报告。

### 4. 打包为独立 exe
```bash
build_exe.bat     # 或 build_exe.ps1
```
产物位于 `dist/` 目录，可直接在 Windows 上运行，无需 Python 环境。

---

## 四、模块依赖关系图（简化）

```
项目.py (主控/导航/样式/数据共享)
 ├── viewer.py            (fMRI 切片浏览控件)
 ├── Read_Write_JSON.py   (JSON 持久化)
 ├── Data_Pre.py          (预处理页)
 ├── feature_extraction_page.py (特征提取页)
 ├── predict.py           (诊断预测页)
 │    └── SelfAttentionBlock_ST_GCN.py  (SA-STGCN 模型)
 │    └── explainability_utils.py        (可解释性)
 │    └── best_acc_model_fold_1.pt       (权重)
 └── individual_report_page.py (报告页)
      └── report_explain_utils.py
      └── explainability_utils.py
```

---

## 五、注意事项
- `other/` 目录下的文件为旧版/实验性代码，与主程序 `项目.py` 是 **两套独立界面**，修改正式功能请以根目录源码为准。
- `Data/` 内数据量较大，打包时需注意资源路径；程序已通过 `get_resource_path` 兼容 PyInstaller 打包后的资源定位。
- 本报告及系统输出 **仅供科研参考**，不应作为临床诊断的唯一依据。
