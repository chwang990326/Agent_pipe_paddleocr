# PaddleOCR 工业装配质检 Agent 架构

## 目标

本项目从“被动 OCR/视觉识别接口”升级为“主动决策与行动的工业 Agent”。PaddleOCR 与形状识别能力不再直接面向业务流程，而是作为 Controller Agent 可调用的工具，由 Agent 自主完成感知、语义纠错、核验、决策、告警、日志持久化。

## 核心目录

```text
agent/
  brain.py                 # 认知中枢：任务拆解与可审计推理摘要
  controller.py            # Agent 对外门面
  state_machine.py         # ReAct 状态机工作流
  memory.py                # 短期去重记忆、轨迹日志、CSV 报表
  parsing.py               # 钢管字符解析与标准化
  config.py                # 环境变量配置
  tools/
    perception.py          # Tool_Read_Pipe_Text / Tool_Analyze_Shape
    reasoning.py           # Tool_Semantic_OCR_Correction
    action.py              # Tool_Query_ERP / Tool_Trigger_Alert / 仿真预备
    registry.py            # 工具注册表
  integrations/
    ocr_client.py          # Jetson TX2 OCR 服务适配器
    shape_client.py        # 形状识别服务适配器
    llm_client.py          # LLM 语义 OCR 纠错适配器
    erp_client.py          # ERP/MES 或本地 BOM 适配器
    alert_client.py        # 飞书/钉钉/通用 Webhook 适配器
    simulation_client.py   # 机械臂或装配仿真接口适配器
data/
  bom.sample.json          # 本地 BOM 样例
run_agent.py               # 单次轨迹 CLI
agent_api.py               # Flask Agent API
```

旧脚本 `shibie.py`、`demo.py` 暂时保留，作为现有 GUI、摄像头、OCR 与分类模型验证资产。后续可以逐步把其中的 OCR 和形状分类逻辑抽到服务层，再由 Agent 工具调用。

## 四大组件

1. 认知中枢：`ControllerBrain` 与 `PipeInspectionWorkflow`
   当前负责状态机编排、工具选择、语义纠错触发条件、BOM 核验和行动分支。LLM 不替代全部规则，而是在 OCR 边界缺陷处承担必须使用语言模型的语义纠错能力。

2. 感知工具箱：`Tool_Read_Pipe_Text`、`Tool_Analyze_Shape`
   OCR 工具可通过 `AGENT_OCR_ENDPOINT` 调用 Jetson TX2 上的 PaddleOCR Flask 服务。未配置 endpoint 时会返回 mock 结果，方便先跑通 Agent 闭环。

3. 推理工具箱：`Tool_Semantic_OCR_Correction`
   当 OCR 置信度低于阈值，或 ERP/BOM 查无精确物料时，Agent 会把原始 OCR、工位、任务、形状上下文和 BOM 候选项发给 LLM。LLM 必须返回结构化 JSON：是否应用纠错、纠正后的物料号、置信度、解释摘要和候选项。纠错置信度达标后，Agent 才会用纠正后的物料号重新查 BOM。

4. 行动工具箱：`Tool_Query_ERP`、`Tool_Trigger_Alert`、`Tool_Prepare_Assembly_Simulation`
   ERP/MES 未接入时读取 `data/bom.sample.json`。告警默认 dry-run，不会真正发送；配置飞书或钉钉 webhook 后可切换为实发。

5. 记忆与持久化：`AgentMemory`
   短期记忆记录当前批次/工位识别过的构件签名，避免同一构件被重复识别和重复报错。长期数据写入 JSONL 轨迹和 CSV 批次报表。

## ReAct 状态机轨迹

```text
triggered
  -> perception: Tool_Read_Pipe_Text, Tool_Analyze_Shape
  -> reasoning: ControllerBrain 生成计划与推理摘要
  -> validation: Tool_Query_ERP
  -> reasoning conditional:
       low_confidence or erp_not_found
       -> Tool_Semantic_OCR_Correction
       -> corrected_text accepted
       -> Tool_Query_ERP again
  -> agentic rag:
       Tool_Process_Change_RAG_Check reviews latest process-change documents
  -> decision:
       matched  -> Tool_Prepare_Assembly_Simulation -> finished
       blocked_by_process_change -> Tool_Trigger_Alert -> suspended_for_human_review
       mismatch -> Tool_Trigger_Alert -> suspended_for_human_review
       duplicate -> skip -> finished
       error/low_confidence_without_safe_correction -> Tool_Trigger_Alert -> suspended_for_human_review
```

## 语义级 OCR 纠错示例

现场 OCR 原始输出：

```text
0345B-DN5OO
```

Agent 首次查询 BOM 发现查无精确物料，同时 OCR 置信度低于阈值，于是调用 LLM。LLM 结合候选 BOM、冶金材料牌号、船舶装配上下文和视觉易混字符规则，返回：

```json
{
  "applied": true,
  "corrected_text": "Q345B-DN500",
  "confidence": 0.86,
  "reason_summary": "0345B 不符合常见钢材牌号；结合 Q/0 与 O/0 易混、BOM 候选和 DN 管径规范，修正为 Q345B-DN500。",
  "candidates_considered": ["Q345B-DN500"]
}
```

Agent 只有在 `confidence >= AGENT_SEMANTIC_CORRECTION_MIN_CONFIDENCE` 时才应用纠错，然后用 `Q345B-DN500` 重新查 BOM。

## CSV 工程规范

`memory.py` 导出的 CSV 使用 `utf-8-sig`、全字段引用，并对所有导出字段加文本保护前缀，默认是制表符 `\t`。这样批次时间戳、超长物料 ID、批次号在 Office 中打开时会被当作文本，不会被自动转换成科学计数法或日期格式。前缀可通过 `AGENT_CSV_TEXT_GUARD` 调整。

## 生产接入配置

```bash
AGENT_OCR_ENDPOINT=http://tx2-ip:8090/ocr
AGENT_SHAPE_ENDPOINT=http://shape-service/analyze
AGENT_ERP_ENDPOINT=http://erp-service/query-bom
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
AGENT_LLM_ENDPOINT=
AGENT_LLM_MODEL=
AGENT_LLM_API_KEY=
AGENT_SEMANTIC_CORRECTION_ENABLED=1
AGENT_SEMANTIC_CORRECTION_REQUIRED=1
AGENT_SEMANTIC_CORRECTION_MIN_CONFIDENCE=0.70
AGENT_PROCESS_RAG_ENABLED=1
AGENT_PROCESS_RAG_REQUIRED=0
AGENT_PROCESS_RAG_DOCS_DIR=data/process_docs
AGENT_PROCESS_RAG_TOP_K=4
AGENT_PROCESS_RAG_MIN_CONFIDENCE=0.70
AGENT_ALERT_WEBHOOK=https://...
AGENT_ALERT_CHANNEL=feishu
AGENT_ALERT_DRY_RUN=0
```

## 快速运行

普通一致路径：

```bash
python run_agent.py --workstation A-01 --component-id sensor-001
```

语义纠错演示：

```bash
$env:AGENT_LLM_ENDPOINT='mock://semantic-correction'
$env:AGENT_MOCK_OCR_TEXT='0345B-DN5OO'
$env:AGENT_MOCK_OCR_CONFIDENCE='0.62'
python run_agent.py --workstation A-03 --component-id semantic-demo-001
```

工艺变更 RAG 拦截演示：

```bash
$env:AGENT_LLM_ENDPOINT='mock://process-rag'
$env:AGENT_MOCK_OCR_TEXT='316L-DN500'
python run_agent.py --workstation A-04 --component-id rag-demo-001
```

启动 Agent API：

```bash
python agent_api.py
```

默认监听 `http://0.0.0.0:8091`，接口为 `POST /agent/pipe-inspection`。
