from __future__ import annotations

__lazy_modules__ = {"bashlex", "subprocess"}

import dataclasses
import subprocess

import bashlex

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
        Mapping,
        Sequence,
    )

    # a function that takes a command and the environment, and returns the result
    EnvironmentExecutor = Callable[[list[str], dict[str, str]], str]


def local_environment_executor(command: Sequence[str], env: Mapping[str, str]) -> str:
    return subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, check=True).stdout


@dataclasses.dataclass(frozen=True, kw_only=True)
class NodeExecutionContext:
    environment: dict[str, str]
    input: str
    executor: EnvironmentExecutor


def evaluate(
    value: str, environment: Mapping[str, str], executor: EnvironmentExecutor | None = None
) -> str:
    if not value:
        # empty string evaluates to empty string
        # (but trips up bashlex)
        return ""

    command_node = bashlex.parsesingle(value)

    if len(command_node.parts) != 1:
        msg = f"{value!r} has too many parts"
        raise ValueError(msg)

    value_word_node = command_node.parts[0]

    return evaluate_node(
        value_word_node,
        context=NodeExecutionContext(
            environment=dict(environment),
            input=value,
            executor=executor or local_environment_executor,
        ),
    )


def evaluate_node(node: bashlex.ast.node, context: NodeExecutionContext) -> str:
    if node.kind == "word":
        return evaluate_word_node(node, context=context)
    elif node.kind == "commandsubstitution":
        node_result = evaluate_command_node(node.command, context=context)
        # bash removes training newlines in command substitution
        return node_result.rstrip()
    elif node.kind == "parameter":
        return evaluate_parameter_node(node, context=context)
    else:
        msg = f"Unsupported bash construct: {node.kind!r}"
        raise ValueError(msg)


def _map_part_word_offsets(
    raw_span: str, part_offsets: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """
    Pairs each part's raw-span offsets with its offsets in the unescaped word
    text.

    bashlex reports expansion positions in raw-input coordinates, while
    node.word has quotes removed and backslash escapes resolved (a backslash
    before any character collapses to that character, and quote characters
    are dropped). Replaying that transformation keeps substitutions aligned
    even when escaped literals share the word with an expansion.
    """
    ends = dict(part_offsets)
    result: list[tuple[int, int]] = []
    raw_i = 0
    word_i = 0
    while raw_i < len(raw_span):
        if raw_i in ends:
            part_len = ends[raw_i] - raw_i
            result.append((word_i, word_i + part_len))
            word_i += part_len
            raw_i = ends[raw_i]
            continue
        char = raw_span[raw_i]
        if char == "\\" and raw_i + 1 < len(raw_span):
            raw_i += 2
            word_i += 1
        elif char in "\"'":
            raw_i += 1
        else:
            raw_i += 1
            word_i += 1
    return result


def evaluate_word_node(node: bashlex.ast.node, context: NodeExecutionContext) -> str:
    value: str = node.word

    if not node.parts:
        return value

    raw_span = context.input[node.pos[0] : node.pos[1]]
    part_offsets = sorted((p.pos[0] - node.pos[0], p.pos[1] - node.pos[0]) for p in node.parts)
    mapped = _map_part_word_offsets(raw_span, part_offsets)

    result: list[str] = []
    prev_end = 0
    for part, (start, end) in zip(node.parts, mapped, strict=True):
        result.append(value[prev_end:start])
        result.append(evaluate_node(part, context=context))
        prev_end = end

    result.append(value[prev_end:])
    return "".join(result)


def evaluate_command_node(node: bashlex.ast.node, context: NodeExecutionContext) -> str:
    if any(n.kind == "operator" for n in node.parts):
        return evaluate_nodes_as_compound_command(node.parts, context=context)
    else:
        return evaluate_nodes_as_simple_command(node.parts, context=context)


def evaluate_nodes_as_compound_command(
    nodes: Sequence[bashlex.ast.node], context: NodeExecutionContext
) -> str:
    # bashlex doesn't support any operators besides ';' inside command
    # substitutions, so we only need to handle that case. We do so assuming
    # that `set -o errexit` is on, because it's easier to code!

    result = ""
    for node in nodes:
        if node.kind == "command":
            result += evaluate_command_node(node, context=context)
        elif node.kind == "operator":
            if node.op != ";":
                msg = f"Unsupported bash operator: {node.op!r}"
                raise ValueError(msg)
        else:
            msg = f"Unsupported bash node in compound command: {node.kind!r}"
            raise ValueError(msg)

    return result


def evaluate_nodes_as_simple_command(
    nodes: Iterable[bashlex.ast.node], context: NodeExecutionContext
) -> str:
    command = [evaluate_node(part, context=context) for part in nodes]
    return context.executor(command, context.environment)


def evaluate_parameter_node(node: bashlex.ast.node, context: NodeExecutionContext) -> str:
    return context.environment.get(node.value, "")
