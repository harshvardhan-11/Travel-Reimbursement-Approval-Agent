"""Run smoke test on the refactored travel_reimbursement_agent.ipynb"""
from __future__ import annotations

import ast
import asyncio
import inspect
import json
import sys
import types
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "travel_reimbursement_agent.ipynb"


async def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    module = types.ModuleType("notebook_smoke_runtime")
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    for number, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        code = compile(source, f"{NOTEBOOK.name}:cell-{number}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        value = eval(code, namespace, namespace)
        if inspect.iscoroutine(value):
            await value

    results = namespace["results"]
    audits = namespace["audits"]
    assert len(results) == 5
    assert [result["claim_id"] for result in results] == [f"CLM-00{i}" for i in range(1, 6)]
    assert all(audit["execution_mode"] in {"llm_agentic", "deterministic_policy"} for audit in audits)
    assert sum(result["approved_amount"] or 0 for result in results) == 1950.0
    assert sum(result["deducted_amount"] or 0 for result in results) == 480.0
    pending = sum(
        namespace["claim_total"](namespace["CLAIMS"][result["claim_id"]])
        for result in results
        if result["decision"] == "MANUAL_REVIEW"
    )
    assert float(pending) == 3220.0
    print("SUCCESS: Notebook smoke test passed with all assertions verified!")
    print(f"Modes used: {[a['execution_mode'] for a in audits]}")
    for a in audits:
        print(f" - {a['claim_id']}: mode={a['execution_mode']}, tools_selected={a.get('tools_selected_by_agent')}")


if __name__ == "__main__":
    asyncio.run(main())
