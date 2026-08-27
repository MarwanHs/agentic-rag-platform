"""Critic-synthesizer (decisions #11, #12, #16, #26, #35, #36, #49).

A forced-tool-use Claude call evaluates a closed evidence set -- already
combined from the retriever and code-navigation agents via the shared
EvidenceItem schema (decision #35) -- and either answers, with every claim
traceable to a cited evidence index, or refuses. On a malformed response it
retries the same synthesis call exactly once, feeding back what went wrong;
this never re-triggers evidence gathering (decision #12), so it's still a
gate over a fixed evidence set, not an orchestrator that goes back for more
input. It runs even when the evidence list is empty, since it is the only
component that produces final answer/refusal text (decision #11) and there
is no bypass that doesn't reintroduce a second place that text gets
generated.

Mapping cited_evidence_indices to API-layer Citation objects, and computing
sources_used from the planner's routing decision, belongs to the (not yet
built) orchestrating function -- out of scope here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import anthropic

from shared.evidence import EvidenceItem

# Decision #36: the one call in this system that gets the stronger model --
# highest-stakes call, worth it.
DEFAULT_CRITIC_MODEL = "claude-sonnet-5"

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the critic-synthesizer for a codebase question-answering system. \
You are given a question and a closed set of evidence gathered by other \
agents. Your only job is to answer strictly from that evidence, or refuse.

Refuse rather than fill gaps (decision #11): if the evidence doesn't \
sufficiently answer the question, refuse -- even if the answer "sounds \
obvious" or you're confident you know it from general/pretrained \
knowledge. You are not being asked what you know; you are being asked what \
this evidence supports.

Mandatory per-claim citation: every claim in your answer must be traceable \
to at least one cited evidence index. Never make an uncited assertion.

The evidence list mixes two different signal types -- do not treat them as \
equally certain just because they're in the same list. Retriever evidence \
carries a `score`, a probabilistic relevance ranking: a high score means \
likely relevant, not certainly correct. Code-navigation evidence carries a \
`symbol_kind`/`reference_kind`, an exact, deterministic structural fact: \
if it says a definition or reference exists at a location, it does.

If the evidence list is empty, always refuse. Never attempt to answer a \
question with no evidence using general knowledge, no matter how simple \
the question seems.

Always call the synthesize_answer tool with your decision.\
"""

SYNTHESIZE_ANSWER_TOOL: dict = {
    "name": "synthesize_answer",
    "description": (
        "Answer the question from the given evidence, citing which evidence "
        "items support it -- or refuse if the evidence doesn't sufficiently "
        "answer it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": ["string", "null"],
                "description": (
                    "The answer, citing claims via cited_evidence_indices. "
                    "Null iff refused is true."
                ),
            },
            "refused": {
                "type": "boolean",
                "description": "True if the evidence doesn't sufficiently answer the question.",
            },
            "reason": {
                "type": ["string", "null"],
                "description": "Explanation for refusal. Null iff refused is false.",
            },
            "cited_evidence_indices": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "Indices (into the numbered evidence list given) supporting the "
                    "answer. Non-empty iff refused is false -- every claim must be "
                    "backed by cited evidence, no zero-citation answers."
                ),
            },
        },
        "required": ["answer", "refused", "reason", "cited_evidence_indices"],
    },
}


@dataclass(frozen=True, slots=True)
class CriticDecision:
    answer: str | None
    refused: bool
    reason: str | None
    cited_evidence_indices: list[int]


class CriticValidationError(RuntimeError):
    """Raised when the critic's structured output is internally inconsistent."""


class CriticClient(Protocol):
    def synthesize(self, question: str, evidence: list[EvidenceItem]) -> CriticDecision: ...


def _format_evidence(evidence: list[EvidenceItem]) -> str:
    lines = []
    for i, item in enumerate(evidence):
        start, end = item.line_range
        location = f"{item.file_path}:{start}-{end}"
        if item.source == "retriever":
            detail = f"score={item.score}" if item.score is not None else ""
        else:
            detail = f"symbol={item.symbol_name} ({item.symbol_kind}, {item.reference_kind})"
        snippet = item.content.strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        lines.append(f"[{i}] source={item.source} {location} {detail}: {snippet}")
    return "\n".join(lines)


def _build_user_message(question: str, evidence: list[EvidenceItem]) -> str:
    if not evidence:
        return f"Question: {question}\n\nEvidence: none."
    return (
        f"Question: {question}\n\n"
        "Evidence:\n"
        f"{_format_evidence(evidence)}"
    )


def _validate_decision(decision: CriticDecision, evidence: list[EvidenceItem]) -> None:
    answer_present = decision.answer is not None
    if answer_present == decision.refused:
        raise CriticValidationError(
            f"Inconsistent critic decision: answer is {decision.answer!r} but "
            f"refused is {decision.refused} (full decision: {decision!r})"
        )

    reason_present = decision.reason is not None
    if reason_present != decision.refused:
        raise CriticValidationError(
            f"Inconsistent critic decision: reason is {decision.reason!r} but "
            f"refused is {decision.refused} (full decision: {decision!r})"
        )

    indices_present = bool(decision.cited_evidence_indices)
    if indices_present == decision.refused:
        raise CriticValidationError(
            "Inconsistent critic decision: cited_evidence_indices is "
            f"{decision.cited_evidence_indices!r} but refused is {decision.refused} "
            f"(full decision: {decision!r})"
        )

    for i in decision.cited_evidence_indices:
        if not (0 <= i < len(evidence)):
            raise CriticValidationError(
                f"Critic cited evidence index {i} out of range for evidence list of "
                f"length {len(evidence)} (full decision: {decision!r})"
            )


def _attempt_synthesis(
    client: anthropic.Anthropic,
    model: str,
    messages: list[dict[str, Any]],
    evidence: list[EvidenceItem],
) -> tuple[Any, CriticDecision | None, CriticValidationError | None]:
    """One forced-tool-use call plus validation. Returns (response, decision,
    error) -- exactly one of decision/error is non-None -- rather than always
    raising, so the caller can retry once on failure using the raw `response`
    to build retry feedback before deciding whether to give up.

    Robustness layer (decisions #53/#54/#58/#59): a forced-tool-use response
    can still omit a required field or get cut off by max_tokens -- observed
    live, independent of system-prompt wording (reproduced against multiple
    prompt variants, including the original unmodified one), so this is
    checked defensively rather than trusting the schema's `required` list.
    """
    response = client.messages.create(
        model=model,
        # Decision #54: free-text answer plus a potentially long
        # cited_evidence_indices array needs more headroom than the default.
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        tools=[SYNTHESIZE_ANSWER_TOOL],
        tool_choice={"type": "tool", "name": "synthesize_answer"},
        messages=messages,
    )

    tool_use = next(block for block in response.content if block.type == "tool_use")
    tool_input: dict[str, Any] = tool_use.input

    logger.info(
        "critic synthesis decision (stop_reason=%s): %s", response.stop_reason, tool_input
    )

    if response.stop_reason == "max_tokens":
        return response, None, CriticValidationError(
            "Critic response hit max_tokens before its tool call finished -- "
            "tool_input is a truncated/malformed partial parse, not a normal "
            f"validation failure (full tool_input: {tool_input!r})"
        )

    required_fields = ("answer", "refused", "reason", "cited_evidence_indices")
    missing_fields = [field for field in required_fields if field not in tool_input]
    if missing_fields:
        return response, None, CriticValidationError(
            f"Critic tool_use response is missing required field(s) {missing_fields} "
            f"(full tool_input: {tool_input!r})"
        )

    decision = CriticDecision(
        answer=tool_input["answer"],
        refused=tool_input["refused"],
        reason=tool_input["reason"],
        cited_evidence_indices=tool_input["cited_evidence_indices"],
    )
    try:
        _validate_decision(decision, evidence)
    except CriticValidationError as error:
        return response, None, error

    return response, decision, None


class AnthropicCriticClient:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_CRITIC_MODEL) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def synthesize(self, question: str, evidence: list[EvidenceItem]) -> CriticDecision:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": _build_user_message(question, evidence)}
        ]

        response, decision, error = _attempt_synthesis(self._client, self._model, messages, evidence)
        if decision is not None:
            return decision

        # Exactly one retry, feeding back the model's own malformed attempt
        # via a proper tool_result block (required immediately after an
        # assistant turn containing a tool_use block) plus the specific
        # validation error, before giving up.
        logger.warning("critic validation failed on first attempt, retrying once: %s", error)
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
                            f"That tool call was invalid: {error} Call "
                            "synthesize_answer again, making sure every required "
                            "field is populated correctly this time."
                        ),
                    }
                ],
            },
        ]
        _, decision, error = _attempt_synthesis(self._client, self._model, retry_messages, evidence)
        if decision is not None:
            return decision
        raise error
