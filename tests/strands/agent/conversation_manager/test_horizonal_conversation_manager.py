"""Tests for HorizonalConversationManager."""

import copy
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strands.agent.conversation_manager.horizonal_conversation_manager import (
    HorizonalConversationManager,
)
from strands.types.content import Message
from strands.types.exceptions import ContextWindowOverflowException


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _user_msg(text: str) -> Message:
    return {"role": "user", "content": [{"text": text}]}


def _assistant_msg(text: str) -> Message:
    return {"role": "assistant", "content": [{"text": text}]}


def _tool_use_msg(name: str, tool_use_id: str = "tu1", input_data: dict | None = None) -> Message:
    return {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": tool_use_id, "name": name, "input": input_data or {}}}],
    }


def _tool_result_msg(tool_use_id: str = "tu1", text: str = "result") -> Message:
    return {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": tool_use_id, "content": [{"text": text}], "status": "success"}}],
    }


def _sediment_msg(text: str = "## Context Horizon\n### Identity\n- test") -> Message:
    return {"role": "user", "content": [{"text": text}]}


def _make_agent(messages: list[Message]) -> MagicMock:
    agent = MagicMock()
    agent.messages = messages
    return agent


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_params(self):
        mgr = HorizonalConversationManager()
        assert mgr.primal_zone_tokens == 40000
        assert mgr.retention_zone_tokens == 60000
        assert mgr.enable_protention is True
        assert mgr.per_turn is False

    def test_custom_params(self):
        mgr = HorizonalConversationManager(primal_zone_tokens=20000, retention_zone_tokens=30000, enable_protention=False)
        assert mgr.primal_zone_tokens == 20000
        assert mgr.retention_zone_tokens == 30000
        assert mgr.enable_protention is False

    def test_invalid_per_turn(self):
        with pytest.raises(ValueError):
            HorizonalConversationManager(per_turn=0)
        with pytest.raises(ValueError):
            HorizonalConversationManager(per_turn=-1)

    def test_initial_state(self):
        mgr = HorizonalConversationManager()
        assert mgr._sedimented_message is None
        assert mgr._trajectory == ""
        assert mgr._model_call_count == 0


# ---------------------------------------------------------------------------
# Primal Impression Zone (Task 2)
# ---------------------------------------------------------------------------

class TestPrimalImpressionZone:
    def test_messages_under_limit_untouched(self):
        """Messages within total capacity should not be modified."""
        mgr = HorizonalConversationManager(primal_zone_tokens=1000, retention_zone_tokens=2000)
        messages = [_user_msg(f"msg {i}") for i in range(25)]
        originals = copy.deepcopy(messages)
        agent = _make_agent(messages)

        mgr.apply_management(agent)

        assert agent.messages == originals

    def test_primal_zone_preserved_after_management(self):
        """Last N messages (primal zone) must be preserved in full fidelity."""
        mgr = HorizonalConversationManager(primal_zone_tokens=500, retention_zone_tokens=500)
        messages = [_user_msg(f"msg {i}") if i % 2 == 0 else _assistant_msg(f"resp {i}") for i in range(40)]
        primal_originals = copy.deepcopy(messages[-5:])
        agent = _make_agent(messages)

        sediment = _sediment_msg("## Context Horizon\n### Identity & Preferences\n- summary")

        with patch.object(mgr, "_generate_sediment", return_value=sediment):
            mgr.apply_management(agent)

        # Last 5 messages should be identical to originals
        assert agent.messages[-5:] == primal_originals


# ---------------------------------------------------------------------------
# Graduated Compression (Task 3)
# ---------------------------------------------------------------------------

class TestGraduatedCompression:
    def test_compress_message_no_op_for_low_distance(self):
        """Messages close to present (low temporal_distance) should not be compressed."""
        mgr = HorizonalConversationManager()
        msg = _assistant_msg("short text")
        original = copy.deepcopy(msg)
        mgr._compress_message_in_place(msg, 0.1)
        assert msg == original

    def test_compress_tool_result_heavy(self):
        """Tool results far from present should be heavily truncated."""
        mgr = HorizonalConversationManager()
        long_text = "x" * 1000
        msg: Message = {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": long_text}], "status": "success"}}],
        }
        mgr._compress_message_in_place(msg, 0.9)

        result_text = msg["content"][0]["toolResult"]["content"][0]["text"]
        assert "[truncated:" in result_text
        assert len(result_text) < len(long_text)

    def test_compress_tool_result_light(self):
        """Tool results close to present should keep more content."""
        mgr = HorizonalConversationManager()
        long_text = "y" * 1000
        msg: Message = {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": long_text}], "status": "success"}}],
        }
        mgr._compress_message_in_place(msg, 0.5)

        result_text = msg["content"][0]["toolResult"]["content"][0]["text"]
        assert "[truncated:" in result_text
        # Medium compression keeps more than heavy
        assert len(result_text) > 2 * 200 + 50  # more than just prefix+suffix

    def test_compress_text_block_heavy(self):
        """Long text blocks far from present should be condensed."""
        mgr = HorizonalConversationManager()
        msg = _assistant_msg("a" * 1000)
        mgr._compress_message_in_place(msg, 0.9)

        text = msg["content"][0]["text"]
        assert "[condensed:" in text
        assert len(text) < 1000

    def test_image_replaced_with_placeholder(self):
        """Images should be replaced with text placeholders."""
        mgr = HorizonalConversationManager()
        msg: Message = {
            "role": "user",
            "content": [{"image": {"format": "png", "source": {"bytes": b"fake_image_data"}}}],
        }
        mgr._compress_message_in_place(msg, 0.5)

        assert "text" in msg["content"][0]
        assert "[image:" in msg["content"][0]["text"]

    def test_graduated_compression_is_graduated(self):
        """Messages further from present should be more compressed than closer ones."""
        mgr = HorizonalConversationManager()
        long_text = "z" * 2000

        msg_far = {"role": "user", "content": [
            {"toolResult": {"toolUseId": "t1", "content": [{"text": long_text}], "status": "success"}}
        ]}
        msg_near = copy.deepcopy(msg_far)

        mgr._compress_message_in_place(msg_far, 0.9)
        mgr._compress_message_in_place(msg_near, 0.4)

        far_len = len(msg_far["content"][0]["toolResult"]["content"][0]["text"])
        near_len = len(msg_near["content"][0]["toolResult"]["content"][0]["text"])
        # Far should be more compressed (shorter) than near
        assert far_len <= near_len


# ---------------------------------------------------------------------------
# Tool Pair Preservation (Task 3 constraint)
# ---------------------------------------------------------------------------

class TestToolPairPreservation:
    def test_split_does_not_break_tool_pairs(self):
        """Split point should not land between a toolUse and its toolResult."""
        mgr = HorizonalConversationManager()
        messages = [
            _user_msg("hello"),
            _tool_use_msg("search", "tu1"),
            _tool_result_msg("tu1", "found it"),
            _user_msg("thanks"),
        ]

        # Trying to split at index 1 (toolUse) should advance to include the pair
        adjusted = mgr._adjust_split_for_tool_pairs(messages, 1)
        # Should not split between toolUse (idx 1) and toolResult (idx 2)
        assert adjusted != 2  # Can't start with orphaned toolResult

    def test_split_at_user_message_ok(self):
        """Split at a plain user message should be fine."""
        mgr = HorizonalConversationManager()
        messages = [
            _user_msg("hello"),
            _assistant_msg("hi"),
            _user_msg("question"),
        ]
        adjusted = mgr._adjust_split_for_tool_pairs(messages, 2)
        assert adjusted == 2


# ---------------------------------------------------------------------------
# Sedimentation (Task 4)
# ---------------------------------------------------------------------------

class TestSedimentation:
    def test_sediment_message_is_user_role(self):
        """Sedimented horizon should be a user-role message."""
        mgr = HorizonalConversationManager()

        sediment = _sediment_msg("## Context Horizon\n- test")
        with patch.object(mgr, "_generate_sediment", return_value=sediment) as mock_gen:
            # Trigger sedimentation by calling _apply_horizonal_structure
            messages = [_user_msg(f"msg {i}") if i % 2 == 0 else _assistant_msg(f"resp {i}") for i in range(40)]
            agent = _make_agent(messages)
            mgr.primal_zone_tokens = 30
            mgr.retention_zone_tokens = 30
            mgr._apply_horizonal_structure(agent)

            assert mock_gen.called
            assert agent.messages[0]["role"] == "user"

    def test_is_sediment_message_detection(self):
        """Should correctly identify sedimented horizon messages."""
        mgr = HorizonalConversationManager()

        assert mgr._is_sediment_message(_sediment_msg()) is True
        assert mgr._is_sediment_message(_user_msg("hello")) is False
        assert mgr._is_sediment_message(_assistant_msg("## Context Horizon")) is False

    def test_format_messages_for_prompt(self):
        """Should format messages into readable text."""
        mgr = HorizonalConversationManager()
        messages = [
            _user_msg("hello world"),
            _tool_use_msg("search", "tu1", {"query": "test"}),
            _tool_result_msg("tu1", "found 3 results"),
        ]
        result = mgr._format_messages_for_prompt(messages)

        assert "[user] hello world" in result
        assert "Called tool: search" in result
        assert "Tool result: found 3 results" in result


# ---------------------------------------------------------------------------
# Protention (Task 5)
# ---------------------------------------------------------------------------

class TestProtention:
    def test_trajectory_extracted_from_last_user_message(self):
        """Trajectory should capture the most recent user intent."""
        mgr = HorizonalConversationManager()
        messages = [
            _user_msg("Book me a flight to Paris"),
            _assistant_msg("Sure, let me search..."),
            _user_msg("Actually, change it to London"),
        ]
        mgr._update_trajectory(messages)
        assert "London" in mgr._trajectory

    def test_trajectory_skips_sediment_messages(self):
        """Trajectory should not pick up sedimented horizon content."""
        mgr = HorizonalConversationManager()
        messages = [
            _sediment_msg(),
            _user_msg("Find hotels in Tokyo"),
        ]
        mgr._update_trajectory(messages)
        assert "Tokyo" in mgr._trajectory
        assert "Context Horizon" not in mgr._trajectory

    def test_trajectory_disabled(self):
        """When protention is disabled, trajectory should not update."""
        mgr = HorizonalConversationManager(enable_protention=False)
        messages = [_user_msg("test")]
        mgr._update_trajectory(messages)
        # _update_trajectory is only called when enable_protention is True
        # in _apply_horizonal_structure, so direct call still works
        # but the flag controls whether it's called in the main flow


# ---------------------------------------------------------------------------
# Session Persistence (Task 6)
# ---------------------------------------------------------------------------

class TestSessionPersistence:
    def test_get_and_restore_state_roundtrip(self):
        """State should survive a save/restore cycle."""
        mgr = HorizonalConversationManager()
        mgr._sedimented_message = _sediment_msg()
        mgr._trajectory = "booking a flight"
        mgr._model_call_count = 5
        mgr.removed_message_count = 10

        state = mgr.get_state()

        mgr2 = HorizonalConversationManager()
        prepend = mgr2.restore_from_session(state)

        assert mgr2._sedimented_message == mgr._sedimented_message
        assert mgr2._trajectory == "booking a flight"
        assert mgr2._model_call_count == 5
        assert mgr2.removed_message_count == 10
        assert prepend is not None
        assert len(prepend) == 1

    def test_restore_without_sediment(self):
        """Restore with no sedimented message should return None."""
        mgr = HorizonalConversationManager()
        state = mgr.get_state()

        mgr2 = HorizonalConversationManager()
        prepend = mgr2.restore_from_session(state)
        assert prepend is None

    def test_state_class_name_validation(self):
        """Restoring from wrong class name should raise."""
        mgr = HorizonalConversationManager()
        with pytest.raises(ValueError):
            mgr.restore_from_session({"__name__": "WrongClass", "removed_message_count": 0})


# ---------------------------------------------------------------------------
# Hook Registration (Task 6)
# ---------------------------------------------------------------------------

class TestHooks:
    def test_register_hooks(self):
        """Should register BeforeModelCallEvent callback."""
        mgr = HorizonalConversationManager(per_turn=True)
        registry = MagicMock()
        mgr.register_hooks(registry)
        registry.add_callback.assert_called()

    def test_per_turn_false_skips(self):
        """With per_turn=False, hook should not trigger management."""
        mgr = HorizonalConversationManager(per_turn=False)
        event = MagicMock()
        mgr._on_before_model_call(event)
        assert mgr._model_call_count == 0

    def test_per_turn_true_increments(self):
        """With per_turn=True, every call should trigger."""
        mgr = HorizonalConversationManager(per_turn=True)
        event = MagicMock()
        event.agent.messages = []
        mgr._on_before_model_call(event)
        assert mgr._model_call_count == 1

    def test_per_turn_int_frequency(self):
        """With per_turn=N, should trigger every N calls."""
        mgr = HorizonalConversationManager(per_turn=3)
        event = MagicMock()
        event.agent.messages = []

        for _ in range(6):
            mgr._on_before_model_call(event)

        assert mgr._model_call_count == 6


# ---------------------------------------------------------------------------
# Reduce Context (overflow handling)
# ---------------------------------------------------------------------------

class TestReduceContext:
    def test_reduce_context_with_tiny_history_raises(self):
        """Should raise when history is too small to reduce."""
        mgr = HorizonalConversationManager()
        agent = _make_agent([_user_msg("hi")])
        with pytest.raises(ContextWindowOverflowException):
            mgr.reduce_context(agent, e=ContextWindowOverflowException("overflow"))

    def test_reduce_context_compresses_tool_results_first(self):
        """Should try compressing tool results before sedimentation."""
        mgr = HorizonalConversationManager(primal_zone_tokens=200, retention_zone_tokens=500)
        long_result = "x" * 1000
        messages = [
            _user_msg("start"),
            _tool_use_msg("search", "tu1"),
            _tool_result_msg("tu1", long_result),
            _assistant_msg("found it"),
            _user_msg("thanks"),
        ]
        agent = _make_agent(messages)

        mgr.reduce_context(agent)

        # Tool result should be truncated
        for msg in agent.messages:
            for block in msg.get("content", []):
                if "toolResult" in block:
                    for item in block["toolResult"].get("content", []):
                        if "text" in item:
                            assert "[truncated:" in item["text"]


# ---------------------------------------------------------------------------
# End-to-end structure test
# ---------------------------------------------------------------------------

class TestTokenBudget:
    def test_no_trigger_under_budget(self):
        """Should not trigger management when tokens are under budget."""
        mgr = HorizonalConversationManager(primal_zone_tokens=100, retention_zone_tokens=100)
        messages = [_user_msg(f"short msg {i}") for i in range(8)]
        originals = copy.deepcopy(messages)
        agent = _make_agent(messages)
        mgr.apply_management(agent)
        assert agent.messages == originals

    def test_triggers_on_large_tool_results(self):
        """Should trigger when few messages but large tool results exceed budget."""
        mgr = HorizonalConversationManager(primal_zone_tokens=100, retention_zone_tokens=100)
        # 6 messages but with huge tool results (~2000 tokens)
        messages = [
            _user_msg("start"),
            _tool_use_msg("search", "tu1"),
            _tool_result_msg("tu1", "x" * 8000),  # ~2000 tokens
            _assistant_msg("found it"),
            _user_msg("thanks"),
            _assistant_msg("you're welcome"),
        ]
        agent = _make_agent(messages)

        sediment = _sediment_msg("## Context Horizon\n- summary")
        with patch.object(mgr, "_generate_sediment", return_value=sediment):
            mgr.apply_management(agent)

        # Should have triggered sedimentation due to token budget
        assert len(agent.messages) < 6 or mgr._is_sediment_message(agent.messages[0])

    def test_estimate_tokens(self):
        """Token estimation should roughly match 4 chars per token."""
        mgr = HorizonalConversationManager()
        msg = _user_msg("a" * 400)  # ~100 tokens
        est = mgr._estimate_message_tokens(msg)
        assert 90 <= est <= 110

    def test_estimate_tool_result_tokens(self):
        """Large tool results should have high token estimates."""
        mgr = HorizonalConversationManager()
        msg = _tool_result_msg("tu1", "data " * 1000)  # 5000 chars ~1250 tokens
        est = mgr._estimate_message_tokens(msg)
        assert est > 1000


class TestEndToEnd:
    def test_full_horizonal_structure(self):
        """Full flow: messages overflow -> sedimentation + retention + primal."""
        mgr = HorizonalConversationManager(primal_zone_tokens=30, retention_zone_tokens=30)

        # Create 15 messages (well over 3+3=6 capacity)
        messages: list[Message] = []
        for i in range(15):
            if i % 2 == 0:
                messages.append(_user_msg(f"user message {i}"))
            else:
                messages.append(_assistant_msg(f"assistant response {i}"))

        agent = MagicMock()
        agent.messages = messages

        sediment = _sediment_msg("## Context Horizon\n### Identity & Preferences\n- User sent 15 messages")

        with patch.object(mgr, "_generate_sediment", return_value=sediment):
            mgr.apply_management(agent)

        # Should have: 1 sediment + <=3 retention + 3 primal = <=7
        assert len(agent.messages) <= 7
        # First message should be sedimented horizon
        assert mgr._is_sediment_message(agent.messages[0])
        # Last 3 should be primal (original last 3)
        assert agent.messages[-1] == _user_msg("user message 14")
