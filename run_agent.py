"""CLI entry point for the industrial pipe inspection Agent."""

from __future__ import annotations

import argparse
import json
import sys

from agent import build_default_agent
from agent.schemas import to_plain_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one pipe-inspection Agent trajectory.")
    parser.add_argument(
        "--task",
        default="Inspect current pipe material preparation at workshop area A.",
        help="Global task passed to the Controller Agent.",
    )
    parser.add_argument("--frame", default=None, help="Path to the current camera frame.")
    parser.add_argument("--workstation", default="A-01", help="Current workstation or area id.")
    parser.add_argument("--component-id", default=None, help="Sensor or tracker id for duplicate control.")
    parser.add_argument("--batch-id", default=None, help="Production batch id.")
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    agent = build_default_agent()
    state = agent.inspect_pipe(
        task=args.task,
        frame_path=args.frame,
        workstation=args.workstation,
        component_id=args.component_id,
        batch_id=args.batch_id,
    )
    print(json.dumps(to_plain_data(state), ensure_ascii=False, indent=2))
    return 0 if state.decision and state.decision.status in {"matched", "duplicate"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
