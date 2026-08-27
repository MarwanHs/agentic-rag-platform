"""Critic-synthesizer (decisions #11, #12, #16, #26, #35, #36, #49).

A single forced-tool-use Claude call evaluates a closed evidence set --
already combined from the retriever and code-navigation agents via the
shared EvidenceItem schema (decision #35) -- and either answers, with every
claim traceable to a cited evidence index, or refuses. This call never
re-triggers evidence gathering (decision #12): it is a gate, not an
orchestrator. It runs even when the evidence list is empty, since it is the
only component that produces final answer/refusal text (decision #11) and
there is no bypass that doesn't reintroduce a second place that text gets
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


class AnthropicCriticClient:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_CRITIC_MODEL) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def synthesize(self, question: str, evidence: list[EvidenceItem]) -> CriticDecision:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[SYNTHESIZE_ANSWER_TOOL],
            tool_choice={"type": "tool", "name": "synthesize_answer"},
            messages=[{"role": "user", "content": _build_user_message(question, evidence)}],
        )

        tool_use = next(block for block in response.content if block.type == "tool_use")
        tool_input: dict[str, Any] = tool_use.input

        logger.info("critic synthesis decision: %s", tool_input)

        decision = CriticDecision(
            answer=tool_input["answer"],
            refused=tool_input["refused"],
            reason=tool_input["reason"],
            cited_evidence_indices=tool_input["cited_evidence_indices"],
        )
        _validate_decision(decision, evidence)
        return decision
