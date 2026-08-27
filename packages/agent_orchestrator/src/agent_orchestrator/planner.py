"""Query-routing planner (decisions #9, #33, #34, #36, #41, #42).

A forced-tool-use Claude call decides which evidence-gathering agents
(retriever, code-navigation) are needed for a question, given any prior
evidence already accumulated in the conversation. The planner never calls
those agents itself and never answers the question -- routing only. On a
malformed response it retries the same routing call exactly once, feeding
back what went wrong, before giving up.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import anthropic

from shared.evidence import EvidenceItem

DEFAULT_PLANNER_MODEL = "claude-haiku-4-5-20251001"

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the routing planner for a codebase question-answering system. Your \
only job is to decide which evidence-gathering agents are needed to answer \
the user's question -- you never answer the question yourself, and you \
never rely on your own general/pretrained knowledge to fill in an answer.

Two evidence sources are available:
- retriever: a semantic search over the codebase (embeddings + hybrid \
search). Give it a self-contained natural-language query.
- code_navigation: an exact-name lookup for symbol definitions and \
references (no reasoning, no fuzzy matching). Give it exact symbol names \
extracted from the question, never natural-language descriptions.

You may request one, both, or neither agent. Use both when the question \
needs a semantic search over relevant code as well as an exact \
definition/reference lookup.

If prior evidence from earlier turns in this conversation is provided and \
it already contains what's needed to answer the current question, request \
no agents and mark the existing evidence as sufficient.

If the question is unrelated to this codebase -- on any turn, regardless \
of what prior evidence already exists in the conversation -- request no \
agents and mark existing evidence as not sufficient. Do not attempt to be \
helpful by answering from general knowledge; an unrelated or unanswerable \
question is not your job to resolve, it belongs to downstream refusal \
handling.

Always call the route_query tool with your decision.\
"""

ROUTE_QUERY_TOOL: dict = {
    "name": "route_query",
    "description": (
        "Decide which evidence-gathering agents (if any) are needed to "
        "answer the user's question about this codebase, and what to ask "
        "each of them."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of the routing decision, for observability (decision #18).",
            },
            "agents_needed": {
                "type": "array",
                "items": {"type": "string", "enum": ["retriever", "code_navigation"]},
                "uniqueItems": True,
                "description": (
                    "Which agents to invoke. Can be empty: for a single-shot question, "
                    "empty means off-topic (critic will refuse). For a conversation "
                    "follow-up, empty means prior-turn evidence already suffices."
                ),
            },
            "retriever_query": {
                "type": ["string", "null"],
                "description": (
                    "Self-contained search query. Non-null iff 'retriever' is in "
                    "agents_needed. Must resolve any conversation-history "
                    "pronouns/references into a standalone query -- the retriever "
                    "never sees conversation history, only this string."
                ),
            },
            "code_navigation_symbols": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Exact symbol name(s) to look up. Non-empty iff 'code_navigation' "
                    "is in agents_needed."
                ),
            },
            "existing_context_sufficient": {
                "type": "boolean",
                "description": (
                    "Only meaningful when agents_needed is empty. True: prior-turn "
                    "evidence in this conversation already answers the question -- "
                    "reuse it, don't invoke new agents. False: no evidence applies -- "
                    "either this is the first turn, or the question is off-topic for "
                    "this codebase even though prior evidence exists; the critic sees "
                    "no evidence for this turn and refuses. Always false for "
                    "single-shot queries."
                ),
            },
        },
        "required": [
            "reasoning",
            "agents_needed",
            "retriever_query",
            "code_navigation_symbols",
            "existing_context_sufficient",
        ],
    },
}


@dataclass(frozen=True, slots=True)
class PlannerDecision:
    reasoning: str
    agents_needed: list[Literal["retriever", "code_navigation"]]
    retriever_query: str | None
    code_navigation_symbols: list[str]
    existing_context_sufficient: bool


class PlannerValidationError(RuntimeError):
    """Raised when the planner's structured output is internally inconsistent."""


class PlannerClient(Protocol):
    def plan(self, question: str, prior_evidence: list[EvidenceItem]) -> PlannerDecision: ...


def _format_prior_evidence(prior_evidence: list[EvidenceItem]) -> str:
    lines = []
    for item in prior_evidence:
        start, end = item.line_range
        location = f"{item.file_path}:{start}-{end}"
        if item.source == "retriever":
            detail = f"score={item.score}" if item.score is not None else ""
        else:
            detail = f"symbol={item.symbol_name} ({item.symbol_kind}, {item.reference_kind})"
        snippet = item.content.strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        lines.append(f"- [{item.source}] {location} {detail}: {snippet}")
    return "\n".join(lines)


def _build_user_message(question: str, prior_evidence: list[EvidenceItem]) -> str:
    if not prior_evidence:
        return f"Question: {question}"
    return (
        f"Question: {question}\n\n"
        "Prior evidence gathered in this conversation so far:\n"
        f"{_format_prior_evidence(prior_evidence)}"
    )


def _validate_decision(decision: PlannerDecision) -> None:
    retriever_requested = "retriever" in decision.agents_needed
    retriever_query_present = bool(decision.retriever_query)
    if retriever_requested != retriever_query_present:
        raise PlannerValidationError(
            "Inconsistent planner decision: 'retriever' in agents_needed is "
            f"{retriever_requested} but retriever_query is {decision.retriever_query!r} "
            f"(full decision: {decision!r})"
        )

    code_navigation_requested = "code_navigation" in decision.agents_needed
    symbols_present = len(decision.code_navigation_symbols) > 0
    if code_navigation_requested != symbols_present:
        raise PlannerValidationError(
            "Inconsistent planner decision: 'code_navigation' in agents_needed is "
            f"{code_navigation_requested} but code_navigation_symbols is "
            f"{decision.code_navigation_symbols!r} (full decision: {decision!r})"
        )


def _attempt_plan(
    client: anthropic.Anthropic,
    model: str,
    messages: list[dict[str, Any]],
) -> tuple[Any, PlannerDecision | None, PlannerValidationError | None]:
    """One forced-tool-use call plus validation. Returns (response, decision,
    error) -- exactly one of decision/error is non-None -- so the caller can
    retry once on failure using the raw `response` to build retry feedback.

    Robustness layer: a forced-tool-use response can still omit a required
    field or get cut off by max_tokens -- observed live (a missing
    `existing_context_sufficient`, reproduced even against the original,
    unmodified prompt), so this is checked defensively rather than trusting
    the schema's `required` list.
    """
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=[ROUTE_QUERY_TOOL],
        tool_choice={"type": "tool", "name": "route_query"},
        messages=messages,
    )

    tool_use = next(block for block in response.content if block.type == "tool_use")
    tool_input: dict[str, Any] = tool_use.input

    logger.info(
        "planner routing decision (stop_reason=%s): %s", response.stop_reason, tool_input
    )

    if response.stop_reason == "max_tokens":
        return response, None, PlannerValidationError(
            "Planner response hit max_tokens before its tool call finished -- "
            "tool_input is a truncated/malformed partial parse, not a normal "
            f"validation failure (full tool_input: {tool_input!r})"
        )

    required_fields = (
        "reasoning",
        "agents_needed",
        "retriever_query",
        "code_navigation_symbols",
        "existing_context_sufficient",
    )
    missing_fields = [field for field in required_fields if field not in tool_input]
    if missing_fields:
        return response, None, PlannerValidationError(
            f"Planner tool_use response is missing required field(s) {missing_fields} "
            f"(full tool_input: {tool_input!r})"
        )

    decision = PlannerDecision(
        reasoning=tool_input["reasoning"],
        agents_needed=tool_input["agents_needed"],
        retriever_query=tool_input["retriever_query"],
        code_navigation_symbols=tool_input["code_navigation_symbols"],
        existing_context_sufficient=tool_input["existing_context_sufficient"],
    )
    try:
        _validate_decision(decision)
    except PlannerValidationError as error:
        return response, None, error

    return response, decision, None


class AnthropicPlannerClient:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_PLANNER_MODEL) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def plan(self, question: str, prior_evidence: list[EvidenceItem]) -> PlannerDecision:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": _build_user_message(question, prior_evidence)}
        ]

        response, decision, error = _attempt_plan(self._client, self._model, messages)
        if decision is not None:
            return decision

        # Exactly one retry, mirroring the critic's retry: feed back the
        # model's own malformed attempt via a proper tool_result block (the
        # API requires this immediately after an assistant tool_use turn)
        # plus the specific validation error, before giving up.
        logger.warning("planner validation failed on first attempt, retrying once: %s", error)
        tool_use_block = next(block for block in response.content if block.type == "tool_use")
        retry_messages = messages + [
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_block.id,
                        "is_error": True,
                        "content": (
                            f"That tool call was invalid: {error} Call route_query "
                            "again, making sure every required field is populated "
                            "correctly this time."
                        ),
                    }
                ],
            },
        ]
        _, decision, error = _attempt_plan(self._client, self._model, retry_messages)
        if decision is not None:
            return decision
        raise error
