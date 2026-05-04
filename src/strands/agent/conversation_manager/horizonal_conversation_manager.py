"""Horizonal conversation management inspired by Merleau-Ponty's phenomenology of temporal experience.

Structures context as a temporal field with three zones:

- **Primal Impression Zone**: Recent messages in full fidelity (the "living present")
- **Retention Zone**: Medium-past with graduated compression (still "held in grasp")
- **Sedimented Horizon**: Deep past distilled to essential patterns and constraints

Compression is not uniform — it's graduated, like temporal experience itself.
"""

import copy
import logging
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from ..._async import run_async
from ...event_loop.streaming import process_stream
from ...hooks import BeforeModelCallEvent, HookRegistry
from ...types.content import ContentBlock, Message, Messages
from ...types.exceptions import ContextWindowOverflowException
from ...types.tools import ToolResultContent
from .conversation_manager import ConversationManager

if TYPE_CHECKING:
    from ..agent import Agent

logger = logging.getLogger(__name__)

_PRESERVE_CHARS = 200

SEDIMENTATION_PROMPT = """\
You are a conversation context distiller. Update the structured context summary
by integrating new conversation messages into the existing summary.

EXISTING CONTEXT SUMMARY:
{existing_sediment}

NEW MESSAGES TO INTEGRATE:
{new_messages}

CURRENT TRAJECTORY:
{trajectory}

Output an updated structured summary in this exact format:

## Context Horizon
### Identity & Preferences
- Key facts about the user and their preferences

### Decisions Made
- Decisions that have been made and should not be revisited

### Active Constraints
- Constraints, rules, or requirements currently in effect

### Key Results
- Important tool results or findings that may be referenced later

### Trajectory
- What the conversation is currently working toward

Rules:
- ACCUMULATE: layer new information on top of existing, do not discard prior context
- PRESERVE causal links: if a decision was made because of a result, keep that link
- Be concise but complete - this summary is the only record of the deep past
- Write in third person, bullet points only
- Do NOT add commentary or respond conversationally"""


class HorizonalConversationManager(ConversationManager):
    """Phenomenologically-inspired temporal context management.

    Instead of a flat buffer (sliding window) or single summary, this manager structures
    conversation as a temporal field with graduated compression, mirroring Husserl's analysis
    where each retention becomes "the retention of a retention, and the layer of time between
    it and myself becomes thicker."

    Example:
        ```python
        from strands import Agent
        from strands.agent.conversation_manager.horizonal_conversation_manager import (
            HorizonalConversationManager,
        )

        agent = Agent(
            conversation_manager=HorizonalConversationManager(
                primal_zone_tokens=40000,
                retention_zone_tokens=60000,
            )
        )
        ```
    """

    def __init__(
        self,
        primal_zone_tokens: int = 40000,
        retention_zone_tokens: int = 60000,
        enable_protention: bool = True,
        *,
        per_turn: bool | int = False,
    ):
        """Initialize the horizonal conversation manager.

        All zone boundaries are token-based, estimated at ~4 chars per token.

        Args:
            primal_zone_tokens: Token budget for the primal impression zone (the "living
                present"). Recent messages are preserved in full fidelity up to this limit.
                Default 40K tokens (~10-15 messages with tool results).
            retention_zone_tokens: Token budget for the retention zone (graduated compression).
                Messages between sedimented horizon and primal zone are progressively
                compressed within this budget. Default 60K tokens.
            enable_protention: Maintain a trajectory annotation for conversational direction.
            per_turn: Proactive management frequency. False=only on overflow, True=every call,
                int=every N calls.

        Raises:
            ValueError: If per_turn is 0 or negative.
        """
        if isinstance(per_turn, int) and not isinstance(per_turn, bool) and per_turn <= 0:
            raise ValueError(f"per_turn must be a positive integer, True, or False, got {per_turn}")

        super().__init__()
        self.primal_zone_tokens = primal_zone_tokens
        self.retention_zone_tokens = retention_zone_tokens
        self.enable_protention = enable_protention
        self.per_turn = per_turn

        self._sedimented_message: Message | None = None
        self._trajectory: str = ""
        self._model_call_count = 0
        self._sedimentation_call_count = 0
        self._sedimentation_input_tokens = 0
        self._sedimentation_output_tokens = 0

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    @override
    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register hook for proactive per-turn management.

        Args:
            registry: The hook registry to register callbacks with.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        super().register_hooks(registry, **kwargs)
        registry.add_callback(BeforeModelCallEvent, self._on_before_model_call)

    def _on_before_model_call(self, event: BeforeModelCallEvent) -> None:
        """Apply management before model calls based on per_turn config.

        Args:
            event: The before model call event.
        """
        if self.per_turn is False:
            return

        self._model_call_count += 1
        should_apply = self.per_turn is True or (
            isinstance(self.per_turn, int) and self._model_call_count % self.per_turn == 0
        )

        if should_apply:
            logger.debug(
                "model_call_count=<%d>, per_turn=<%s> | applying horizonal management",
                self._model_call_count,
                self.per_turn,
            )
            self.apply_management(event.agent)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @override
    def apply_management(self, agent: "Agent", **kwargs: Any) -> None:
        """Apply horizonal temporal structure to the agent's messages.

        Triggers when estimated tokens exceed primal_zone_tokens + retention_zone_tokens.

        Args:
            agent: The agent whose conversation history will be managed.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        estimated = self._estimate_tokens(agent.messages)
        budget = self.primal_zone_tokens + self.retention_zone_tokens
        if estimated <= budget:
            return

        logger.debug(
            "estimated_tokens=<%d>, budget=<%d> | triggering horizonal management",
            estimated,
            budget,
        )
        self._apply_horizonal_structure(agent)

    @override
    def reduce_context(self, agent: "Agent", e: Exception | None = None, **kwargs: Any) -> None:
        """Reduce context when the model's context window is exceeded.

        First tries compressing the retention zone, then forces sedimentation.

        Args:
            agent: The agent whose conversation history will be reduced.
            e: The exception that triggered the reduction, if any.
            **kwargs: Additional keyword arguments for future extensibility.

        Raises:
            ContextWindowOverflowException: If context cannot be reduced further.
        """
        if len(agent.messages) <= 2:
            if e is not None:
                raise ContextWindowOverflowException("Unable to reduce context further!") from e
            return

        if self._compress_retention_zone(agent.messages):
            return

        self._apply_horizonal_structure(agent, force=True)

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_message_tokens(message: Message) -> int:
        """Estimate token count for a single message (~4 chars per token).

        Args:
            message: Message to estimate.

        Returns:
            Estimated token count.
        """
        total = 0
        for block in message.get("content", []):
            if "text" in block:
                total += len(block["text"]) // 4
            elif "toolUse" in block:
                tu = block["toolUse"]
                total += len(str(tu.get("input", {}))) // 4 + 20
            elif "toolResult" in block:
                tr = block["toolResult"]
                for item in tr.get("content", []):
                    if "text" in item:
                        total += len(item["text"]) // 4
                    elif "image" in item:
                        total += 1000  # rough estimate for images
            elif "image" in block:
                total += 1000
        return max(total, 10)  # minimum 10 tokens per message

    def _estimate_tokens(self, messages: Messages) -> int:
        """Estimate total token count for a message list.

        Args:
            messages: Messages to estimate.

        Returns:
            Estimated total tokens.
        """
        return sum(self._estimate_message_tokens(m) for m in messages)

    # ------------------------------------------------------------------
    # Horizonal structure
    # ------------------------------------------------------------------

    def _apply_horizonal_structure(self, agent: "Agent", force: bool = False) -> None:
        """Restructure messages into the three temporal zones, all token-based.

        Walks backwards from the end to allocate primal zone (up to primal_zone_tokens),
        then retention zone (up to retention_zone_tokens). Everything else is sedimented.

        Args:
            agent: The agent whose messages will be restructured.
            force: If True, aggressively sediment even when under budget.
        """
        messages = agent.messages
        if len(messages) <= 2:
            return

        sediment_offset = 1 if (
            self._sedimented_message
            and len(messages) > 0
            and self._is_sediment_message(messages[0])
        ) else 0

        # Walk backwards to find primal zone boundary
        primal_tokens = 0
        primal_start = len(messages)
        for i in range(len(messages) - 1, sediment_offset - 1, -1):
            msg_tokens = self._estimate_message_tokens(messages[i])
            if primal_tokens + msg_tokens > self.primal_zone_tokens:
                break
            primal_tokens += msg_tokens
            primal_start = i

        # Walk backwards from primal_start to find retention zone boundary
        retention_tokens = 0
        retention_start = primal_start
        for i in range(primal_start - 1, sediment_offset - 1, -1):
            msg_tokens = self._estimate_message_tokens(messages[i])
            if retention_tokens + msg_tokens > self.retention_zone_tokens:
                break
            retention_tokens += msg_tokens
            retention_start = i

        # Everything before retention_start (after sediment_offset) gets sedimented
        to_sediment_count = retention_start - sediment_offset

        if not force and to_sediment_count <= 0:
            # Nothing to sediment — just apply graduated compression to retention
            self._apply_graduated_compression(messages, sediment_offset, primal_start)
            return

        if force:
            to_sediment_count = max(to_sediment_count, 2)

        available = messages[sediment_offset:primal_start]
        to_sediment_count = min(to_sediment_count, len(available))

        to_sediment_count = self._adjust_split_for_tool_pairs(available, to_sediment_count)
        if to_sediment_count <= 0:
            return

        to_sediment = available[:to_sediment_count]
        retention = available[to_sediment_count:]

        if self.enable_protention:
            self._update_trajectory(messages)

        self._sedimented_message = self._generate_sediment(to_sediment, agent)

        self.removed_message_count += to_sediment_count
        if sediment_offset:
            self.removed_message_count -= 1

        compressed_retention = self._graduate_compress_messages(retention)
        primal = messages[primal_start:]
        messages[:] = [self._sedimented_message] + compressed_retention + primal

    # ------------------------------------------------------------------
    # Graduated compression (retention zone)
    # ------------------------------------------------------------------

    def _apply_graduated_compression(self, messages: Messages, start: int, end: int) -> None:
        """Apply graduated compression to retention zone messages in-place.

        Args:
            messages: Full message list.
            start: Start index of retention zone.
            end: End index of retention zone.
        """
        zone_size = end - start
        if zone_size <= 0:
            return

        for i in range(start, end):
            temporal_distance = 1.0 - ((i - start) / max(zone_size - 1, 1))
            self._compress_message_in_place(messages[i], temporal_distance)

    def _graduate_compress_messages(self, messages: list[Message]) -> list[Message]:
        """Return deep copies of messages with graduated compression.

        Args:
            messages: Messages to compress.

        Returns:
            Compressed copies.
        """
        total = len(messages)
        result = []
        for i, msg in enumerate(messages):
            temporal_distance = 1.0 - (i / max(total - 1, 1))
            compressed = copy.deepcopy(msg)
            self._compress_message_in_place(compressed, temporal_distance)
            result.append(compressed)
        return result

    def _compress_message_in_place(self, message: Message, temporal_distance: float) -> None:
        """Compress a single message based on temporal distance from present.

        Args:
            message: Message to compress (modified in-place).
            temporal_distance: 0.0=close to present (light), 1.0=far (heavy).
        """
        content = message.get("content", [])
        if not content:
            return

        new_content: list[ContentBlock] = []
        for block in content:
            if "toolResult" in block and temporal_distance > 0.3:
                new_content.append(self._compress_tool_result(block, temporal_distance))
            elif "image" in block and temporal_distance > 0.2:
                img = block["image"]
                source: Any = img.get("source", {})
                data = source.get("bytes", b"")
                new_content.append({"text": f"[image: {img.get('format', '?')}, {len(data) if data else 0} bytes]"})
            elif "text" in block and temporal_distance > 0.6:
                new_content.append(self._compress_text_block(block, temporal_distance))
            else:
                new_content.append(block)

        message["content"] = new_content

    def _compress_tool_result(self, block: ContentBlock, temporal_distance: float) -> ContentBlock:
        """Compress a tool result block based on temporal distance.

        Args:
            block: Content block containing toolResult.
            temporal_distance: 0.0=light, 1.0=heavy compression.

        Returns:
            Compressed content block.
        """
        tool_result: Any = block["toolResult"]
        items = tool_result.get("content", [])
        new_items: list[ToolResultContent] = []

        for item in items:
            if "image" in item:
                img = item["image"]
                source: Any = img.get("source", {})
                data = source.get("bytes", b"")
                new_items.append({"text": f"[image: {img.get('format', '?')}, {len(data) if data else 0} bytes]"})
            elif "text" in item:
                text = item["text"]
                if temporal_distance > 0.7 and len(text) > 2 * _PRESERVE_CHARS:
                    prefix = text[:_PRESERVE_CHARS]
                    suffix = text[-_PRESERVE_CHARS:]
                    removed = len(text) - 2 * _PRESERVE_CHARS
                    new_items.append({"text": f"{prefix}...\n[truncated: {removed} chars]\n...{suffix}"})
                elif temporal_distance > 0.4 and len(text) > 4 * _PRESERVE_CHARS:
                    keep = max(int(len(text) * (1.0 - temporal_distance * 0.6)), 2 * _PRESERVE_CHARS)
                    new_items.append({"text": text[:keep] + f"\n[truncated: {len(text) - keep} chars]"})
                else:
                    new_items.append(item)
            else:
                new_items.append(item)

        updated: Any = {**{k: v for k, v in tool_result.items() if k != "content"}, "content": new_items}
        return {"toolResult": updated}

    def _compress_text_block(self, block: ContentBlock, temporal_distance: float) -> ContentBlock:
        """Compress a text block based on temporal distance.

        Args:
            block: Text content block.
            temporal_distance: 0.0=light, 1.0=heavy compression.

        Returns:
            Compressed content block.
        """
        text = block.get("text", "")
        if not text or len(text) < 200:
            return block

        if temporal_distance > 0.8 and len(text) > 500:
            return {"text": text[:200] + f"\n[condensed: {len(text) - 200} chars removed]"}
        elif temporal_distance > 0.6 and len(text) > 1000:
            return {"text": text[:500] + f"\n[condensed: {len(text) - 500} chars removed]"}

        return block

    # ------------------------------------------------------------------
    # Sedimentation
    # ------------------------------------------------------------------

    def _generate_sediment(self, messages: list[Message], agent: "Agent") -> Message:
        """Generate or update the sedimented horizon from messages being retired.

        Sedimentation is accumulative — new layers on top of old.

        Args:
            messages: Messages to distill into the sedimented horizon.
            agent: Agent whose model is used for distillation.

        Returns:
            A user-role message containing the structured context horizon.
        """
        existing = ""
        if self._sedimented_message:
            for block in self._sedimented_message.get("content", []):
                if "text" in block:
                    existing += block["text"]

        msg_text = self._format_messages_for_prompt(messages)
        prompt = SEDIMENTATION_PROMPT.format(
            existing_sediment=existing or "(No prior context)",
            new_messages=msg_text,
            trajectory=self._trajectory or "(No trajectory established)",
        )

        summarization_messages: Messages = [
            {"role": "user", "content": [{"text": prompt}]},
        ]

        async def _call_model() -> Message:
            chunks = agent.model.stream(
                summarization_messages,
                tool_specs=None,
                system_prompt="You are a precise context distiller. Output only the structured summary.",
            )
            result_message: Message | None = None
            async for event in process_stream(chunks):
                if "stop" in event:
                    _, result_message, usage, _ = event["stop"]
                    if usage:
                        self._sedimentation_input_tokens += usage.get("inputTokens", 0)
                        self._sedimentation_output_tokens += usage.get("outputTokens", 0)
            if result_message is None:
                raise RuntimeError("Failed to generate sedimented horizon")
            return result_message

        self._sedimentation_call_count += 1
        message = run_async(_call_model)
        return {
            "role": "user",
            "content": message.get("content", [{"text": "(context summary unavailable)"}]),
        }

    def _format_messages_for_prompt(self, messages: list[Message]) -> str:
        """Format messages into readable text for the sedimentation prompt.

        Args:
            messages: Messages to format.

        Returns:
            Formatted string representation.
        """
        lines: list[str] = []
        for msg in messages:
            role = msg.get("role", "?")
            for block in msg.get("content", []):
                if "text" in block:
                    lines.append(f"[{role}] {block['text'][:500]}")
                elif "toolUse" in block:
                    tu = block["toolUse"]
                    lines.append(f"[{role}] Called tool: {tu.get('name', '?')}({tu.get('input', {})})")
                elif "toolResult" in block:
                    tr = block["toolResult"]
                    result_text = ""
                    for item in tr.get("content", []):
                        if "text" in item:
                            result_text = item["text"][:200]
                            break
                    lines.append(f"[{role}] Tool result: {result_text}")
        return "\n".join(lines[-50:])

    # ------------------------------------------------------------------
    # Protention
    # ------------------------------------------------------------------

    def _update_trajectory(self, messages: Messages) -> None:
        """Extract current conversational trajectory from recent user messages.

        Protention captures intent/direction, not specific expected outcomes.

        Args:
            messages: Full message list.
        """
        for msg in reversed(messages):
            if msg.get("role") == "user":
                for block in msg.get("content", []):
                    if "text" in block and "## Context Horizon" not in block.get("text", ""):
                        self._trajectory = block["text"][:200]
                        return

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def sedimentation_overhead(self) -> dict[str, int]:
        """Token overhead from sedimentation calls (not tracked in agent metrics).

        Returns:
            Dict with call_count, input_tokens, output_tokens.
        """
        return {
            "call_count": self._sedimentation_call_count,
            "input_tokens": self._sedimentation_input_tokens,
            "output_tokens": self._sedimentation_output_tokens,
        }

    def _is_sediment_message(self, message: Message) -> bool:
        """Check if a message is a sedimented horizon message.

        Args:
            message: Message to check.

        Returns:
            True if the message contains the Context Horizon marker.
        """
        if message.get("role") != "user":
            return False
        for block in message.get("content", []):
            if "text" in block and "## Context Horizon" in block.get("text", ""):
                return True
        return False

    def _adjust_split_for_tool_pairs(self, messages: list[Message], split_point: int) -> int:
        """Adjust split point to avoid breaking toolUse/toolResult pairs.

        Args:
            messages: Messages being split.
            split_point: Initial split point.

        Returns:
            Adjusted split point.
        """
        if split_point >= len(messages):
            return len(messages)

        while split_point < len(messages):
            content = messages[split_point].get("content", [])
            has_tool_result = any("toolResult" in c for c in content)
            has_tool_use = any("toolUse" in c for c in content)

            if has_tool_result:
                split_point += 1
            elif has_tool_use and split_point + 1 < len(messages) and any(
                "toolResult" in c for c in messages[split_point + 1].get("content", [])
            ):
                break
            elif has_tool_use:
                split_point += 1
            else:
                break

        return min(split_point, len(messages))

    def _compress_retention_zone(self, messages: Messages) -> bool:
        """Try to compress tool results in retention zone.

        Args:
            messages: Full message list.

        Returns:
            True if any compression was applied.
        """
        sediment_offset = 1 if (
            self._sedimented_message and len(messages) > 0 and self._is_sediment_message(messages[0])
        ) else 0

        # Find primal boundary by walking backwards with token budget
        primal_tokens = 0
        primal_start = len(messages)
        for i in range(len(messages) - 1, sediment_offset - 1, -1):
            msg_tokens = self._estimate_message_tokens(messages[i])
            if primal_tokens + msg_tokens > self.primal_zone_tokens:
                break
            primal_tokens += msg_tokens
            primal_start = i

        changed = False
        for i in range(sediment_offset, primal_start):
            for block in messages[i].get("content", []):
                if "toolResult" in block:
                    tr: Any = block["toolResult"]
                    for item in tr.get("content", []):
                        if "text" in item:
                            text = item["text"]
                            if "[truncated:" not in text and len(text) > 2 * _PRESERVE_CHARS:
                                prefix = text[:_PRESERVE_CHARS]
                                suffix = text[-_PRESERVE_CHARS:]
                                removed = len(text) - 2 * _PRESERVE_CHARS
                                item["text"] = f"{prefix}...\n[truncated: {removed} chars]\n...{suffix}"
                                changed = True
        return changed

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    @override
    def get_state(self) -> dict[str, Any]:
        """Get the current state for session persistence.

        Returns:
            Dictionary containing all manager state.
        """
        state = super().get_state()
        state["sedimented_message"] = self._sedimented_message
        state["trajectory"] = self._trajectory
        state["model_call_count"] = self._model_call_count
        state["sedimentation_call_count"] = self._sedimentation_call_count
        state["sedimentation_input_tokens"] = self._sedimentation_input_tokens
        state["sedimentation_output_tokens"] = self._sedimentation_output_tokens
        return state

    @override
    def restore_from_session(self, state: dict[str, Any]) -> list[Message] | None:
        """Restore state from a session.

        Args:
            state: Previous state dictionary.

        Returns:
            Sedimented horizon message to prepend, if it exists.
        """
        super().restore_from_session(state)
        self._sedimented_message = state.get("sedimented_message")
        self._trajectory = state.get("trajectory", "")
        self._model_call_count = state.get("model_call_count", 0)
        self._sedimentation_call_count = state.get("sedimentation_call_count", 0)
        self._sedimentation_input_tokens = state.get("sedimentation_input_tokens", 0)
        self._sedimentation_output_tokens = state.get("sedimentation_output_tokens", 0)
        return [self._sedimented_message] if self._sedimented_message else None
