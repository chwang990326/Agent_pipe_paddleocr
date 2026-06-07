# PaddleOCR 工业装配质检 Agent 架构

## 目标

本项目从“被动 OCR/视觉识别接口”升级为“主动决策与行动的工业 Agent”。PaddleOCR 与形状识别能力不再直接面向业务流程，而是作为 Controller Agent 可调用的工具，由 Agent 自主完成感知、核验、决策、告警、日志持久化。

## 目录规划

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
    action.py              # Tool_Query_ERP / Tool_Trigger_Alert / 仿真预备
    registry.py            # 工具注册表
  integrations/
    ocr_client.py          # Jetson TX2 OCR 服务适配器
    shape_client.py        # 形状识别服务适配器
    erp_client.py          # ERP/MES 或本地 BOM 适配器
    alert_client.py        # 飞书/钉钉/通用 Webhook 适配器
    simulation_client.py   # 机械臂或装配仿真接口适配器
data/
  bom.sample.json          # 本地 BOM 样例
docs/
  AGENT_ARCHITECTURE.md    # 当前架构说明
run_agent.py               # 单次轨迹 CLI
agent_api.py               # Flask Agent API
```

旧脚本 `shibie.py`、`demo.py` 暂时保留，作为现有 GUI、摄像头、OCR 与分类模型验证资产。后续可以逐步把其中的 OCR 和形状分类逻辑抽到服务层，再由 Agent 工具调用。

## 四大组件

1. 认知中枢：`ControllerBrain` 与 `PipeInspectionWorkflow`
   当前提供确定性规则大脑，便于离线调试。后续接入本地 Qwen-14B 时，建议提供 OpenAI-compatible HTTP 服务，并让 `brain.py` 只输出结构化计划和推理摘要，不把底层工具耦合进 LLM 调用。

2. 感知工具箱：`Tool_Read_Pipe_Text`、`Tool_Analyze_Shape`
   OCR 工具可通过 `AGENT_OCR_ENDPOINT` 调用 Jetson TX2 上的 PaddleOCR Flask 服务。未配置 endpoint 时会返回 mock 结果，方便先跑通 Agent 闭环。

3. 行动工具箱：`Tool_Query_ERP`、`Tool_Trigger_Alert`、`Tool_Prepare_Assembly_Simulation`
   ERP/MES 未接入时读取 `data/bom.sample.json`。告警默认 dry-run，不会真正发送；配置飞书或钉钉 webhook 后可切换为实发。

4. 记忆与持久化：`AgentMemory`
   短期记忆记录当前批次/工位识别过的构件签名，避免同一构件被重复识别和重复报错。长期数据写入 JSONL 轨迹和 CSV 批次报表。

## ReAct 状态机轨迹

```text
triggered
  -> perception: Tool_Read_Pipe_Text, Tool_Analyze_Shape
  -> reasoning: ControllerBrain 生成计划与推理摘要
  -> validation: Tool_Query_ERP
  -> decision:
       matched  -> Tool_Prepare_Assembly_Simulation -> finished
       mismatch -> Tool_Trigger_Alert -> suspended_for_human_review
       duplicate -> skip -> finished
       error/low_confidence -> Tool_Trigger_Alert -> suspended_for_human_review
```

## CSV 工程规范

`memory.py` 导出的 CSV 使用 `utf-8-sig`、全字段引用，并对所有导出字段加文本保护前缀，默认是制表符 `\t`。这样批次时间戳、超长物料 ID、批次号在 Office 中打开时会被当作文本，不会被自动转换成科学计数法或日期格式。前缀可通过 `AGENT_CSV_TEXT_GUARD` 调整。

## 需要准备或确认的外部条件

- Jetson TX2 OCR 服务地址：例如 `http://tx2-ip:8090/ocr`，配置到 `AGENT_OCR_ENDPOINT`。
- 形状识别服务地址：如已有模型服务，配置到 `AGENT_SHAPE_ENDPOINT`；没有时先使用 mock 或从 `shibie.py` 抽服务。
- ERP/MES 查询接口：确认请求字段、鉴权方式、BOM 返回格式，配置到 `AGENT_ERP_ENDPOINT`。
- 告警 Webhook：飞书或钉钉机器人 webhook，配置 `AGENT_ALERT_WEBHOOK` 和 `AGENT_ALERT_CHANNEL=feishu|dingtalk`。
- 机械臂/装配仿真预备接口：配置 `AGENT_SIMULATION_ENDPOINT`。
- 构件唯一 ID 来源：建议由产线传感器、PLC 事件或视觉跟踪模块提供 `component_id`，去重会更可靠。
- 本地 LLM 服务：建议部署 Qwen 工业微调模型并提供结构化 JSON 输出接口，后续接入 `brain.py`。

## 快速运行

```bash
python run_agent.py --workstation A-01 --component-id sensor-001
```

返回 `matched` 表示 BOM 校验通过；把 `--workstation A-02` 与默认 mock OCR `304L-DN500` 搭配运行，会触发 mismatch 路径。

启动 Agent API：

```bash
python agent_api.py
```

默认监听 `http://0.0.0.0:8091`，接口为 `POST /agent/pipe-inspection`。
