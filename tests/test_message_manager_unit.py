"""Unit tests for MessageManager that don't require a database connection.

Tests the _is_heartbeat_message and _combine_assistant_tool_messages methods
which are pure logic operating on PydanticMessage objects.
"""

import json

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall as OpenAIToolCall,
    Function as OpenAIFunction,
)

from letta.schemas.enums import MessageRole
from letta.schemas.letta_message_content import TextContent
from letta.schemas.message import Message as PydanticMessage
from letta.services.message_manager import MessageManager


def _make_msg(role, content=None, tool_calls=None, name=None, tool_call_id=None, agent_id="agent-test"):
    """Helper to build a PydanticMessage for unit tests."""
    return PydanticMessage(
        role=role,
        content=content or [],
        agent_id=agent_id,
        tool_calls=tool_calls,
        name=name,
        tool_call_id=tool_call_id,
    )


# ======================================================================================================================
# _is_heartbeat_message
# ======================================================================================================================


class TestIsHeartbeatMessage:
    """Tests for MessageManager._is_heartbeat_message."""

    def setup_method(self):
        self.mm = MessageManager()

    def test_heartbeat_system_message_detected(self):
        msg = _make_msg(
            MessageRole.system,
            content=[TextContent(text=json.dumps({"type": "heartbeat", "reason": "scheduled"}))],
        )
        assert self.mm._is_heartbeat_message(msg) is True

    def test_non_heartbeat_system_message(self):
        msg = _make_msg(MessageRole.system, content=[TextContent(text="You are a helpful assistant.")])
        assert self.mm._is_heartbeat_message(msg) is False

    def test_user_message_not_heartbeat(self):
        msg = _make_msg(MessageRole.user, content=[TextContent(text="Hello")])
        assert self.mm._is_heartbeat_message(msg) is False

    def test_assistant_empty_content_not_heartbeat(self):
        """Assistant messages with content=[] must NOT be treated as heartbeats.

        This is the key regression test: the old code used _extract_message_text() == ""
        which incorrectly flagged assistant messages with empty content as heartbeats.
        """
        msg = _make_msg(
            MessageRole.assistant,
            content=[],
            tool_calls=[
                OpenAIToolCall(
                    id="c1",
                    type="function",
                    function=OpenAIFunction(name="send_message", arguments='{"message": "Hi"}'),
                )
            ],
        )
        assert self.mm._is_heartbeat_message(msg) is False

    def test_tool_message_not_heartbeat(self):
        msg = _make_msg(MessageRole.tool, content=[TextContent(text="OK")], name="send_message", tool_call_id="c1")
        assert self.mm._is_heartbeat_message(msg) is False

    def test_system_message_empty_content_not_heartbeat(self):
        msg = _make_msg(MessageRole.system, content=[])
        assert self.mm._is_heartbeat_message(msg) is False

    def test_system_message_non_json_content(self):
        msg = _make_msg(MessageRole.system, content=[TextContent(text="plain text system message")])
        assert self.mm._is_heartbeat_message(msg) is False

    def test_system_message_json_not_heartbeat_type(self):
        msg = _make_msg(
            MessageRole.system,
            content=[TextContent(text=json.dumps({"type": "function_call_success", "message": "ok"}))],
        )
        assert self.mm._is_heartbeat_message(msg) is False


# ======================================================================================================================
# _combine_assistant_tool_messages
# ======================================================================================================================


class TestCombineAssistantToolMessages:
    """Tests for _combine_assistant_tool_messages.

    Verifies the fix: assistant messages with empty content (non-reasoning models)
    are no longer silently dropped by the heartbeat filter.
    """

    def setup_method(self):
        self.mm = MessageManager()

    def test_assistant_empty_content_with_tool_response_combined(self):
        """Assistant(content=[]) + matching Tool should be combined into 1 message."""
        assistant = _make_msg(
            MessageRole.assistant,
            content=[],
            tool_calls=[
                OpenAIToolCall(
                    id="c1",
                    type="function",
                    function=OpenAIFunction(name="send_message", arguments='{"message": "Hello!"}'),
                )
            ],
        )
        tool = _make_msg(
            MessageRole.tool,
            content=[TextContent(text='{"status": "OK", "message": "None", "time": "..."}')],
            name="send_message",
            tool_call_id="c1",
        )
        combined = self.mm._combine_assistant_tool_messages([assistant, tool])
        assert len(combined) == 1

    def test_assistant_empty_content_not_dropped(self):
        """Assistant with content=[] should survive the filter even without a matching tool msg."""
        assistant = _make_msg(
            MessageRole.assistant,
            content=[],
            tool_calls=[
                OpenAIToolCall(
                    id="c1",
                    type="function",
                    function=OpenAIFunction(name="send_message", arguments='{"message": "Hi!"}'),
                )
            ],
        )
        combined = self.mm._combine_assistant_tool_messages([assistant])
        assert len(combined) == 1
        assert combined[0].role == MessageRole.assistant

    def test_user_messages_pass_through(self):
        user = _make_msg(MessageRole.user, content=[TextContent(text="What is 2+2?")])
        combined = self.mm._combine_assistant_tool_messages([user])
        assert len(combined) == 1
        assert combined[0].role == MessageRole.user

    def test_heartbeat_system_messages_filtered(self):
        """Heartbeat system messages should still be filtered out."""
        heartbeat = _make_msg(
            MessageRole.system,
            content=[TextContent(text=json.dumps({"type": "heartbeat", "reason": "scheduled"}))],
        )
        user = _make_msg(MessageRole.user, content=[TextContent(text="Hello")])
        combined = self.mm._combine_assistant_tool_messages([heartbeat, user])
        assert len(combined) == 1
        assert combined[0].role == MessageRole.user

    def test_non_heartbeat_system_messages_preserved(self):
        """Non-heartbeat system messages (e.g. function_call_success) should NOT be filtered."""
        sys_msg = _make_msg(
            MessageRole.system,
            content=[TextContent(text=json.dumps({"type": "function_call_success", "message": "ok"}))],
        )
        combined = self.mm._combine_assistant_tool_messages([sys_msg])
        assert len(combined) == 1

    def test_mixed_conversation_preserved(self):
        """A realistic conversation should preserve user + assistant + tool triplets."""
        user = _make_msg(MessageRole.user, content=[TextContent(text="Tell me a joke")])
        assistant = _make_msg(
            MessageRole.assistant,
            content=[],
            tool_calls=[
                OpenAIToolCall(
                    id="c1",
                    type="function",
                    function=OpenAIFunction(
                        name="send_message",
                        arguments=json.dumps({"message": "Why did the chicken cross the road?"}),
                    ),
                )
            ],
        )
        tool = _make_msg(
            MessageRole.tool,
            content=[TextContent(text='{"status": "OK", "message": "None", "time": "..."}')],
            name="send_message",
            tool_call_id="c1",
        )
        combined = self.mm._combine_assistant_tool_messages([user, assistant, tool])
        # user + combined(assistant+tool) = 2
        assert len(combined) == 2
        assert combined[0].role == MessageRole.user

    def test_non_send_message_tool_combined_with_result(self):
        """Assistant calling a non-send_message tool should combine with tool result."""
        assistant = _make_msg(
            MessageRole.assistant,
            content=[TextContent(text="Let me search for that.")],
            tool_calls=[
                OpenAIToolCall(
                    id="c1",
                    type="function",
                    function=OpenAIFunction(
                        name="archival_memory_search",
                        arguments='{"query": "pizza"}',
                    ),
                )
            ],
        )
        tool = _make_msg(
            MessageRole.tool,
            content=[TextContent(text='{"results": ["user likes pizza"]}')],
            name="archival_memory_search",
            tool_call_id="c1",
        )
        combined = self.mm._combine_assistant_tool_messages([assistant, tool])
        assert len(combined) == 1
        # The combined message should contain tool call info
        text = combined[0].content[0].text
        assert "archival_memory_search" in text
