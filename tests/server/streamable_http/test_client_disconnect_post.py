"""Tests for ClientDisconnect handling in StreamableHTTPServerTransport._handle_post_request.

Regression test for pattern 1: ClientDisconnect raised during POST should log at WARNING
(not ERROR) and should not attempt to send a response to the closed socket.

Inspired by upstream PRs:
- https://github.com/modelcontextprotocol/python-sdk/pull/1647 (scope: POST only)
- https://github.com/modelcontextprotocol/python-sdk/pull/1947 (semantics: notify writer, skip response)
"""

from __future__ import annotations as _annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.requests import ClientDisconnect, Request

from mcp.server.streamable_http import StreamableHTTPServerTransport


class TestClientDisconnectDuringPOST:
    """ClientDisconnect during POST should be handled gracefully."""

    def _make_scope(self, headers: dict[str, bytes] | None = None) -> dict[str, Any]:
        """Build a minimal ASGI scope for a POST request."""
        return {
            "type": "http",
            "method": "POST",
            "path": "/mcp",
            "query_string": b"",
            "headers": list((headers or {}).items()) if headers else [
                (b"content-type", b"application/json"),
                (b"accept", b"application/json, text/event-stream"),
            ],
        }

    @pytest.mark.anyio
    async def test_client_disconnect_logs_warning_not_error(self, caplog):
        """ClientDisconnect should produce a WARNING, not an ERROR."""
        transport = StreamableHTTPServerTransport(mcp_session_id=None)
        scope = self._make_scope()

        # Set up a dummy writer so the transport passes the None check
        mock_writer = MagicMock()
        mock_writer.send = AsyncMock()
        transport._read_stream_writer = mock_writer

        # Mock request.body() to raise ClientDisconnect (simulates client going away
        # mid-request body upload).
        mock_request = MagicMock(spec=Request)
        mock_request.body = AsyncMock(side_effect=ClientDisconnect())
        mock_request.headers = {
            "content-type": "application/json",
            "accept": "application/json, text/event-stream",
        }
        mock_request.scope = scope

        send_calls: list[Any] = []

        async def dummy_receive():
            return {"type": "http.request", "body": b""}

        async def dummy_send(message):
            send_calls.append(message)

        with caplog.at_level(logging.DEBUG, logger="mcp.server.streamable_http"):
            await transport._handle_post_request(
                scope, mock_request, dummy_receive, dummy_send
            )

        # Should log a WARNING, not an ERROR
        warning_records = [
            r for r in caplog.records if r.levelno == logging.WARNING
            and "Client disconnected" in r.getMessage()
        ]
        error_records = [
            r for r in caplog.records if r.levelno == logging.ERROR
        ]
        assert len(warning_records) == 1, (
            f"Expected exactly 1 WARNING with 'Client disconnected', got {len(warning_records)}"
        )
        assert len(error_records) == 0, (
            f"Expected 0 ERROR logs, got {len(error_records)}: {[r.getMessage() for r in error_records]}"
        )

    @pytest.mark.anyio
    async def test_client_disconnect_sends_response(self):
        """After ClientDisconnect, a 202 response is sent so middleware chains don't
        raise 'No response returned' (ASGI server drops it if socket is closed)."""
        transport = StreamableHTTPServerTransport(mcp_session_id=None)
        scope = self._make_scope()

        # Set up a dummy writer so the transport passes the None check
        mock_writer = MagicMock()
        mock_writer.send = AsyncMock()
        transport._read_stream_writer = mock_writer

        mock_request = MagicMock(spec=Request)
        mock_request.body = AsyncMock(side_effect=ClientDisconnect())
        mock_request.headers = {
            "content-type": "application/json",
            "accept": "application/json, text/event-stream",
        }
        mock_request.scope = scope

        send_calls: list[Any] = []

        async def dummy_receive():
            return {"type": "http.request", "body": b""}

        async def dummy_send(message):
            send_calls.append(message)

        await transport._handle_post_request(
            scope, mock_request, dummy_receive, dummy_send
        )

        # A response IS sent (202 Accepted) so middleware chains don't blow up
        assert len(send_calls) >= 1, (
            f"Expected at least 1 ASGI send (response), got {len(send_calls)}"
        )
        # First send should be http.response.start with 499 (Client Closed Request)
        assert send_calls[0]["type"] == "http.response.start"
        assert send_calls[0]["status"] == 499

    @pytest.mark.anyio
    async def test_client_disconnect_notifies_writer(self):
        """Writer should receive ClientDisconnect so the inner session task can unblock."""
        transport = StreamableHTTPServerTransport(mcp_session_id=None)
        scope = self._make_scope()

        # Capture what the writer receives
        writer_messages: list[Any] = []

        async def capture_writer(msg):
            writer_messages.append(msg)

        # Patch the internal writer
        with patch.object(transport, "_read_stream_writer", MagicMock(send=capture_writer)):
            mock_request = MagicMock(spec=Request)
            mock_request.body = AsyncMock(side_effect=ClientDisconnect())
            mock_request.headers = {
                "content-type": "application/json",
                "accept": "application/json, text/event-stream",
            }
            mock_request.scope = scope

            send_calls: list[Any] = []

            async def dummy_receive():
                return {"type": "http.request", "body": b""}

            async def dummy_send(message):
                send_calls.append(message)

            await transport._handle_post_request(
                scope, mock_request, dummy_receive, dummy_send
            )

        # Writer should have been notified with ClientDisconnect
        assert len(writer_messages) == 1, (
            f"Expected writer to receive 1 message, got {len(writer_messages)}"
        )
        assert isinstance(writer_messages[0], ClientDisconnect), (
            f"Expected ClientDisconnect sent to writer, got {type(writer_messages[0])}"
        )

    @pytest.mark.anyio
    async def test_client_disconnect_writer_suppresses_errors(self):
        """If the writer itself is broken, we should not crash (suppress(Exception))."""
        transport = StreamableHTTPServerTransport(mcp_session_id=None)
        scope = self._make_scope()

        broken_send = AsyncMock(side_effect=RuntimeError("writer is broken"))

        with patch.object(transport, "_read_stream_writer", MagicMock(send=broken_send)):
            mock_request = MagicMock(spec=Request)
            mock_request.body = AsyncMock(side_effect=ClientDisconnect())
            mock_request.headers = {
                "content-type": "application/json",
                "accept": "application/json, text/event-stream",
            }
            mock_request.scope = scope

            async def dummy_receive():
                return {"type": "http.request", "body": b""}

            send_calls: list[Any] = []

            async def dummy_send(message):
                send_calls.append(message)

            # Should not raise even though writer.send() fails
            await transport._handle_post_request(
                scope, mock_request, dummy_receive, dummy_send
            )

        # The broken writer.send was called once (suppressed)
        broken_send.assert_called_once()
        # Response is still sent even though writer was broken
        assert len(send_calls) >= 1
        assert send_calls[0]["type"] == "http.response.start"
        assert send_calls[0]["status"] == 499
