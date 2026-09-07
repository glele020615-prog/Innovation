# 灵视 - NeuroSight
# 智影识微：基于 MRI 影像的 MCI 进展风险智能预测平台 — 项目说明文档

> 本项目是一个基于 PySide6 的桌面应用程序，用于阿尔茨海默症（AD）/ 轻度认知障碍（MCI）的辅助诊断与进展风险预测。核心算法为 **SA-STGCN（Self-Attention Spatio-Temporal Graph Convolutional Network，自注意力时空图卷积网络）**，以静息态 fMRI 数据为输入，输出患病风险及可解释分析。

---

## 一、总体架构

程序启动后先进入**登录窗口**（`login_window.py`），账号密码校验通过后按角色分发到两个独立界面（退出登录均可回到登录界面）：

- **doctor（医生）** → 主窗口 `ADDiagnosisSystem`（`项目.py`），内部通过 `QStackedWidget` 管理 6 个页面：

| 序号 | 页面 | 对应模块 / 类 |
|------|------|---------------|
| 0 | 首页概览 | `HomePageWidget` / `HomeImageCard`（位于 `项目.py`） |
| 1 | 数据加载 | `DataManagerPage`（`项目.py`） |
| 2 | 预处理 | `PreprocessingPage`（`Data_Pre.py`） |
| 3 | 特征提取 | `FeatureExtractionPage`（`feature_extraction_page.py`） |
| 4 | 疾病预测诊断 | `DiagnosisPage`（`predict.py`） |
| 5 | 个体报告分析 | `IndividualReportPage`（`individual_report_page.py`） |

页面间通过主窗口的 `shared_case_data` 传递病例数据，诊断完成后自动跳转到个体报告页。

- **patient（患者）** → 患者端窗口 `PatientWindow`（`patient_window.py`）：我的报告（PDF 查看/导入）、认知训练（HTML 游戏）、训练记录（SQLite 汇总）。

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

#### `login_window.py` — 登录窗口
- **作用**：应用的第一屏。校验账号密码（调用 `auth.verify`），成功后回调 `on_login_success(role, username, display_name)`，由 `项目.py` 按角色分发到医生端或患者端。
- **界面**：以 `assets/login.png` 为整窗背景（等比放大、居中裁剪），左侧品牌区为半透明遮罩，右侧登录卡片；背景图加载失败时回退为纯深色底。
- **使用**：随 `项目.py` 自动打开；也可单独运行做界面自检（`python login_window.py`）。

#### `patient_window.py` — 患者端窗口
- **作用**：患者登录后的主界面（`PatientWindow`），侧边栏切换三个页面：
  - **我的报告**：基于 `PySide6.QtPdf` 查看/导入个人诊断 PDF；
  - **认知训练**：`QWebEngineView` 加载 5 个 HTML 认知游戏（项目内 `game\` 目录，开发时若项目内没有则回退 `F:\game`；打包后在 `_MEIPASS/game`）；
  - **训练记录**：从本地 SQLite（`training_logs` 表）汇总训练成绩。
  - 右下角带 **AI 健康问答悬浮球**（见「6.6 患者端 AI 悬浮球」）。
- **使用**：随 `项目.py` 登录患者账号进入；也可单独运行做界面自检（`python patient_window.py`，绕过登录直接打开）。

#### `patient_chat_page.py` — 患者端 AI 问答页
- **作用**：患者端悬浮球弹出的对话界面（`PatientChatPage`）。**没有任何工具调用**：只做「RAG 检索 → 检索片段拼进 system prompt → 无 tools 的流式对话补全」，只能回答知识库内的健康科普问题；未命中时如实告知并引导咨询主治医生。
- **会话独立存储**：`~/.智影agent/patient_sessions.json`，与医生端会话互相隔离。
- **使用**：由患者端悬浮球打开，也可单独运行自检（`python patient_chat_page.py`）。

#### `auth.py` — 账号体系
- **作用**：本地 SQLite 账号库（PBKDF2-SHA256 密码哈希，12 万次迭代 + 随机盐）。首次运行自动建表并写入两个种子账号（见「三、登录与账号」）。提供 `verify`（登录校验）、`add_user`（添加账号）、`change_password`（改密）、`get_account` 等接口。
- **账号库位置**：开发环境 `F:\Innovation\Data\accounts.db`；打包版 `%APPDATA%\ZhyingShiWei\Data\accounts.db`。可用环境变量 `ZYSW_DATA_DIR` 覆盖目录。

#### `agent_chat_page.py` / `patient_chat_page.py` / `floating_ball.py` — AI 助手对话页与悬浮球
- **作用**：`agent_chat_page.py` 实现医生端 AI 诊断助手对话界面（详见「六、Agent 子系统」）；`patient_chat_page.py` 实现患者端知识库问答页；`floating_ball.py` 提供可拖动的悬浮球入口，**双模式**：`FloatingBall(parent, mode="doctor")` 弹出医生端诊断助手（工具调用 + RAG），`mode="patient"` 弹出患者端健康问答助手（仅 RAG，无工具）。

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
| `assets/` | 界面图片：`login.png`（登录窗口背景）、`blackground.png`（首页背景）、`fmri_preview.png`、`connectome_preview.png`、`ai_analysis_preview.png`（首页功能卡片图） |
| `scan_history.json` | 扫描/病例历史记录 |
| `Data/accounts.db` | 本地 SQLite 账号库（账号/报告/训练记录三张表；开发环境位置，打包版在 `%APPDATA%\ZhyingShiWei\Data\`） |
| `individual_report.pdf` | 已生成的个体报告样例 |
| `report_cache/` | 报告缓存图片 |
| `FreeSurfer/` | FreeSurfer 相关说明/工具文件 |

---

### 8. 打包与构建

| 文件 | 说明 |
|------|------|
| `NeuroSight.spec` | PyInstaller 打包配置（包含入口、隐藏导入、资源收集、图标等） |
| `build_exe.bat` / `build_exe.ps1` | Windows 下打包成 exe 的脚本（批处理 / PowerShell），产物为 `dist\NeuroSight\NeuroSight.exe` |
| `dist/` | 打包生成的可执行文件与依赖（含大量 `.dll` 等） |
| `build/` | 打包中间产物 |
| `__pycache__/` | Python 字节码缓存 |

---

## 三、运行与使用方式

### 1. 环境依赖
- Python 3.x（建议 3.9+）
- PySide6（含 Addons：患者端用到 QtWebEngine / QtPdf）
- numpy、nibabel、matplotlib、torch（用于模型推理）
- 其他：FreeSurfer 相关工具（预处理）

### 2. 登录与账号（新用户如何加入）

系统**没有自助注册界面**：账号由管理员创建，写入本地 SQLite 账号库（PBKDF2 哈希存储，无法手写明文插入）。

**默认种子账号**（仅账号库为空时首次运行自动创建）：

| 角色 | 用户名 | 初始密码 | 进入界面 |
|------|--------|----------|----------|
| 医生 | `doctor` | `Doctor@2026` | 影像分析与诊断端 |
| 患者 | `patient` | `Patient@2026` | 报告与认知训练端 |

**添加新用户**：调用 `auth.add_user(用户名, 密码, 角色, 显示名)`，例：

```powershell
cd F:\Innovation
python -c "import auth; print(auth.add_user('zhangsan', 'Zs@2026abc', 'patient', '张三'))"
python -c "import auth; print(auth.add_user('lisi', 'Ls@2026abc', 'doctor', '李医生'))"
```

- 密码至少 6 位（建议含大小写字母与数字）；用户名重复会返回失败提示。
- 命令把账号写进**开发环境账号库** `F:\Innovation\Data\accounts.db`。
- **给打包版（exe）部署机添加账号**时，账号库在 `%APPDATA%\ZhyingShiWei\Data\accounts.db`，先让脚本指向该目录再执行：

```powershell
$env:ZYSW_DATA_DIR = "$env:APPDATA\ZhyingShiWei\Data"
python -c "import auth; print(auth.add_user('zhangsan', 'Zs@2026abc', 'patient', '张三'))"
```

患者首次登录后建议通过「修改密码」功能更换初始密码（`auth.change_password`）。

### 3. 启动开发版
```bash
cd f:/Innovation
python 项目.py
```

### 4. 标准使用流程

**医生端**：
1. **首页概览**：查看系统简介，点击「开始」进入数据加载。
2. **数据加载**：通过 `DataManagerPage` 选择/导入 NIfTI 或 DICOM 影像，预览脑切片，查看维度、体素、TR 等参数。
3. **预处理**：在 `PreprocessingPage` 勾选所需步骤并设置参数，运行预处理。
4. **特征提取**：`FeatureExtractionPage` 基于 AAL116 提取时序与功能连接特征。
5. **疾病预测诊断**：`DiagnosisPage` 加载 `best_acc_model_fold_1.pt`，推理输出风险与可解释结果。
6. **个体报告分析**：自动跳转 `IndividualReportPage`，生成并查看/导出个体诊断报告。

**患者端**：登录后在我的报告页查看/导入 PDF 报告；在认知训练页点击「训练入口」开始 5 个 HTML 认知游戏；训练成绩自动汇总到训练记录页。完成后点侧边栏「⏏ 退出登录」回到登录界面。

### 5. 打包为独立 exe
```bash
build_exe.bat     # 或 build_exe.ps1
```
产物位于 `dist/` 目录，可直接在 Windows 上运行，无需 Python 环境。

> 打包配置已包含患者端依赖：`login_window` / `patient_window` / `auth` 显式声明在 hiddenimports 中，`PySide6.QtWebEngineWidgets` 等 WebEngine/QtPdf 模块随包携带，登录背景图 `assets\login.png` 与认知训练游戏目录 `game\` 由构建后校验清单检查。游戏 HTML 源文件在项目 `game\` 目录（构建前有缺失检查），修改游戏后重新打包即可同步到 exe。

全部重新打包的方法
方式一：用脚本一键打包（推荐，和以前一样）
直接双击 build_exe.bat，或在 PowerShell 里运行：

```powershell
cd F:\Innovation
.\build_exe.ps1
```
脚本已加入 --additional-hooks-dir 和 --hidden-import，会加载 hook-nilearn.py 收集 nilearn 全部数据文件。脚本自带 --clean --noconfirm，会重新构建。

方式二：直接用 spec 文件打包
```powershell
cd F:\Innovation
pyinstaller NeuroSight.spec
```
想彻底清空旧产物再打（可选）
```powershell
cd F:\Innovation
Remove-Item -Recurse -Force dist, build
.\build_exe.ps1
```
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

---

## 六、Agent 子系统（AI 诊断助手）

AI 助手不只是聊天：它能**全程代跑**「导入 → 预处理 → 特征提取 → 预测 → 可解释 → 报告」这条链路。

### 6.1 目录结构

```
agent/
├── tools.py            纯注册表：TOOLS / TOOL_MAP / TOOL_SCHEMAS / describe_groups()
├── runtime.py          MessageBuilder · ToolExecutor(超时) · StopCondition(8步) · AgentLoop · Planner
├── llm_client.py       OpenAI 兼容客户端（Key 缺失给友好提示而非抛栈）
├── keyring.py          密钥解析：环境变量 > 用户配置 > 项目配置
├── config.py           统一配置入口（含 max_steps / tool_timeout / language）
├── security.py         路径白名单 safe_path()（拒绝系统目录 / UNC / 未授权目录）
├── redaction.py        脱敏：绝对路径 → <病例路径>，患者 ID → <ID>
├── logging_utils.py    结构化日志：session-*.jsonl + events.jsonl（均脱敏）
├── session_manager.py  多会话（schema_version=2，落盘脱敏）
├── user_memory.py      长期记忆：用户身份 / 讲解深度 / 报告格式偏好
├── rag.py              检索（embedding 优先，TF-IDF 回退，带相关度阈值）
├── prompts/            zh.py / en.py —— system prompt 按语言拆分
├── eval/               cases.jsonl（20 条金标）+ run.py（mock / 真 LLM）
├── knowledge/
│   ├── *.md            知识库语料
│   ├── brain_regions.json   脑区中文名 + AD 关联（唯一数据源）
│   ├── build_index.py  索引构建（chunks.json / vectors.npy / meta.json）
│   └── .index/         索引产物（已 gitignore）
└── skills/             工具按子系统切分（共 19 个工具）
    ├── base.py         BaseTool 协议（name/description/parameters/run/validate）
    ├── scan.py         ScanStore + list/get/import/delete/find_data_files
    ├── preprocess.py   fMRIPrep 命令生成 + nilearn 轻量预处理
    ├── feature_extract.py  纯函数 run_feature_extraction()（UI 线程与 Agent 共用）
    ├── predict.py      预测 + CaseStore + 梯度可解释性
    ├── report.py       render_case_report()（报告渲染，UI 与 Agent 共用）
    ├── knowledge.py    search_knowledge / get_region_info / list_regions
    └── pipeline.py     describe_pipeline / check_ready
```

新增工具 = 在对应 skill 里加一个 `BaseTool` 子类并放进该模块的 `*_tools` 列表；`tools.py` 无需改动。

### 6.2 关键实现约定

| 约定 | 说明 |
|------|------|
| 纯函数优先 | `run_feature_extraction()`、`render_case_report()` 不依赖 Qt；页面的 `QThread` 只做信号转发，Agent 在非 Qt 线程里也能调用 |
| 序列化 | 所有进 LLM / 落盘的 dict 先过 `jsonable()`（numpy → list、Path → str、nan → None） |
| 超时 | `ToolExecutor` 用线程池跑工具，超时返回 `{"success": false, "error_type": "timeout"}` 而不是炸线程 |
| 停止条件 | `max_steps=8` + 最后一轮是否还有 tool_calls + 连续重复调用；触顶时给 LLM 发「用尽预算」提示让其收尾 |
| 路径容错 | LLM 常给模糊路径（`Data/pMCI/.../matr_002_S_0729.mat`），`resolve_fuzzy_path()` 支持省略号、只给文件名、缺日期后缀三种补全 |
| FC/BOLD 推断 | 只给一个文件路径时，`infer_counterpart()` 会按 `FC ↔ BOLD` 兄弟目录自动补另一个 |
| 单一数据源 | 脑区中文名与 AD 知识只在 `agent/knowledge/brain_regions.json`，报告页与 Agent 共用 `brain_region_names.py` |
| 找文件 | `find_data_files` 按样本编号搜真实路径。用户只说"002_S_0729 这个样本"时，Agent 先搜再跑，不会自己拼路径去猜 |
| 防死循环 | 除"重复调用"外，还追踪**同一工具连续失败**（默认 3 次）——LLM 靠"换个后缀再试一次"猜路径会被打断并强制换路线 |
| 模型单例 | 诊断页 `predict.py` 与 Agent 共用 `agent.skills.predict.shared_model()`，打包后只有一份权重；`aboutToQuit` 自动 `release_model()` |

### 6.3 密钥与隐私

- 仓库里只有 `agent_config.example.json`（占位 key）；`agent_config.json` 已加入 `.gitignore`。
- 真实 key 存 `~/.智影agent/config.json`（首次启动自动从 example 拷贝，权限 600）。
- **填写入口**：对话页左下角「⚙ 设置」→ **API** 页，可填 Key、改接口地址/模型，并一键「测试连接」。
- 也可用环境变量：`AGENT_API_KEY` / `AGENT_BASE_URL` / `AGENT_MODEL` / `AGENT_LLM_PROVIDER`（优先级最高）。
- 生效的 provider / model / base_url 会显示在对话页右上角状态栏。

### 6.3.1 数据目录（可换盘 / 可便携）

配置、会话、日志、脱敏表、知识库索引都在**同一个数据目录**下，位置可改：

| 优先级 | 方式 | 适用场景 |
|---|---|---|
| 1 | 环境变量 `AGENT_HOME` | 临时指定、多实例、脚本 |
| 2 | 程序目录下的 `agent_home.txt` | **便携模式**：整目录拷到别的机器，配置跟着走 |
| 3 | `~/.智影agent` | 默认（C 盘用户目录，不是固定的） |

不想到 C 盘：打开「⚙ 设置」→ **数据目录** 页 →「更改位置」选 D 盘任意目录。
迁移是**先复制到新位置、确认成功后再删旧的**，中途失败源目录不动，不会丢数据。
换盘后需重启程序生效。

`agent_home.txt` 里写相对路径时按程序所在目录解析，因此写入 `AgentData` 即可做 U 盘绿色版。

> 打包注意：`build_exe.ps1` 会在打包前删除开发机的 `agent_home.txt`，
> 否则 exe 会去找开发者本机的路径。该文件已加入 `.gitignore`。
- 会话与日志落盘前过 `redact()`：绝对路径 → `<病例路径#xxxxxx>`，ADNI/患者 ID → `<ID#xxxxxx>`。占位符**带短哈希**，因此可按 token 精确回放：`unredact_text()` 还原全文、`reveal(token)` 查单条、`reveal_candidates(kind)` 列出候选供医生挑认。原始映射存 `~/.智影agent/redaction_map.json`（仅本地，600 权限，上限 5000 条）。
- 导出会话默认脱敏；如需保留真实路径与患者编号的本机留档，UI 会要求二次确认后才导出原始版本。
- UI 左侧栏「📂 授权数据目录」可把外部数据目录加入白名单，未授权目录一律返回「路径越界」。

> 注意：`agent/secrets.py` 会遮蔽 Python 标准库的 `secrets` 模块（numpy.random 依赖它），因此密钥模块命名为 `agent/keyring.py`。

### 6.4 评测与回归

```bash
python -m agent.eval.run            # mock LLM，零 token，验证链路 wiring
python -m agent.eval.run --real     # 真 LLM，评估真实工具选择表现
python agent/test_phase2.py         # 工具层单元测试
python agent/test_rag.py            # 索引构建 + 固定 query 回归 + 阈值
python -m agent.knowledge.build_index   # 手动重建知识库索引
```

改动 system prompt / 工具 schema / 知识库后，跑一遍 eval，报告输出到 `agent/eval/eval_report.json`。

### 6.5 日志

`<数据目录>/logs/<日期>/`（数据目录默认 `~/.智影agent`，可在「⚙ 设置 → 数据目录」换到别的盘）：
- `session-<id>.jsonl`：每轮 messages 快照（脱敏）
- `events.jsonl`：`tool_call_start` / `tool_call_done` / `error` / `token_usage`，可用于 replay

日志是数据目录里最容易长大的一项（实测几 MB/周），换盘时它跟着一起走。

### 6.6 患者端 AI 悬浮球（知识库问答）

患者端主窗口右下角有一个可拖拽的悬浮球，点开是「🌿 AI 健康问答助手」。它和医生端 Agent 的核心区别：**没有任何工具调用**，只能回答已建知识库内的健康科普问题。

| 维度 | 医生端（agent_chat_page） | 患者端（patient_chat_page） |
|------|---------------------------|------------------------------|
| 执行链路 | AgentLoop：LLM ⇄ 19 个工具，可代跑诊断全流程 | RAG 检索 → 检索片段拼入 system prompt → **无 tools** 的流式对话补全 |
| 能力边界 | 导数据、预处理、预测、出报告 | 只基于知识库回答；未命中如实说「无法回答」并引导咨询医生；不给诊断、不推荐药物 |
| 会话存储 | `~/.智影agent/sessions.json` | `~/.智影agent/patient_sessions.json`（互相隔离） |
| 界面入口 | 主窗口内嵌页 + 悬浮球弹窗 | 患者端悬浮球弹窗 |

实现要点：
- 患者版 system prompt（`patient_chat_page.PATIENT_SYSTEM_PROMPT`）强制：只依据检索片段回答、结论必须带 `[来源: 文件#章节]` 引用、通俗简短（约 200 字内）、不做治疗决策。
- 悬浮球按 `mode` 惰性导入对应对话页（`floating_ball.ChatDialog`），患者端不会加载医生端工具栈；医生端向弹窗推送病例数据的接口在患者模式下自动忽略。
- TF-IDF 回退检索的中文分词用**滑窗二元组**（`build_index._tokenize`），保证"什么是轻度认知障碍"这类患者口吻的自然句能命中知识库；embedding 模式不受影响。
- 患者问答与医生端共用同一份 API Key 配置（`~/.智影agent/config.json`），部署机需先配好 Key。
