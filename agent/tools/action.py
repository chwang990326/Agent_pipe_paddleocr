"""Action tools: the Agent's hands."""

from __future__ import annotations

from agent.integrations.alert_client import AlertClient
from agent.integrations.erp_client import ERPClient
from agent.integrations.simulation_client import AssemblySimulationClient
from agent.schemas import to_plain_data
from agent.tools.base import ToolContext, ToolResult, timed_call


class QueryERPTool:
    name = "Tool_Query_ERP"
    description = "Query ERP/MES BOM expectations for the recognized material."

    def run(self, payload: dict, context: ToolContext) -> ToolResult:
        def _call():
            record = ERPClient(context.config).query_expected_record(
                material_id=payload["material_id"],
                workstation=payload["workstation"],
            )
            return to_plain_data(record)

        return timed_call(_call)


class TriggerAlertTool:
    name = "Tool_Trigger_Alert"
    description = "Send an intervention alert to Feishu, DingTalk, or another receiver."

    def run(self, payload: dict, context: ToolContext) -> ToolResult:
        def _call():
            return AlertClient(context.config).send(
                title=payload.get("title", "Pipe inspection alert"),
                message=payload.get("message", ""),
                details=payload.get("details", {}),
            )

        return timed_call(_call)


class PrepareAssemblySimulationTool:
    name = "Tool_Prepare_Assembly_Simulation"
    description = "Prepare downstream robot or assembly simulation after BOM validation."

    def run(self, payload: dict, context: ToolContext) -> ToolResult:
        def _call():
            return AssemblySimulationClient(context.config).prepare(payload)

        return timed_call(_call)
