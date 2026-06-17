import ast
import operator
from datetime import UTC, datetime


async def current_datetime() -> str:
    """Return the current date and time in ISO-8601 UTC format."""
    return datetime.now(UTC).isoformat()


_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Only numeric arithmetic expressions are supported.")


async def calculator(expression: str) -> str:
    """Evaluate a safe arithmetic expression using numbers and +, -, *, /, **."""
    parsed = ast.parse(expression, mode="eval")
    return str(_eval_node(parsed.body))


async def file_metadata_extractor(
    filename: str,
    content_type: str = "",
    size_bytes: int = 0,
) -> str:
    """Return simple file metadata for a PD-only prescription crop."""
    return f"filename={filename}; content_type={content_type}; size_bytes={size_bytes}"


async def prescription_validator_placeholder(extracted_json: str) -> str:
    """Validate PD-only prescription extraction shape without processing patient information."""
    if "patient" in extracted_json.lower() or "uhid" in extracted_json.lower():
        return "Rejected PI-like fields from validation context."
    return "PD extraction placeholder validation passed for review workflow."
