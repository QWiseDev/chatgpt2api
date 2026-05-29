from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

from services.protocol import conversation
from services.protocol.conversation import ConversationRequest, ImageOutput


class FakeAccountService:
    def __init__(self) -> None:
        self._next = 0
        self._lock = threading.Lock()
        self.marked: list[tuple[str, bool]] = []

    def get_available_access_token(
        self,
        plan_type: str | None = None,
        source_type: str | None = None,
        plan_types: tuple[str, ...] | None = None,
    ) -> str:
        with self._lock:
            self._next += 1
            return f"token-{self._next}"

    def get_account(self, access_token: str) -> dict[str, str]:
        return {"email": f"{access_token}@example.com"}

    def mark_image_result(self, access_token: str, success: bool) -> None:
        with self._lock:
            self.marked.append((access_token, success))

    def remove_invalid_token(self, access_token: str, event: str) -> bool:
        return False


class ImageGenerationConcurrencyTests(unittest.TestCase):
    def test_multiple_requested_images_start_concurrently_and_return_in_index_order(self) -> None:
        active = 0
        max_active = 0
        lock = threading.Lock()

        def fake_stream_image_outputs(_backend, request: ConversationRequest, index: int, total: int):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            yield ImageOutput(
                kind="progress",
                model=request.model,
                index=index,
                total=total,
                text=f"progress-{index}",
            )
            time.sleep(0.2)
            with lock:
                active -= 1
            yield ImageOutput(
                kind="result",
                model=request.model,
                index=index,
                total=total,
                data=[{"b64_json": f"image-{index}"}],
            )

        fake_accounts = FakeAccountService()
        with (
            mock.patch.object(conversation, "account_service", fake_accounts),
            mock.patch.object(conversation, "OpenAIBackendAPI", lambda access_token: object()),
            mock.patch.object(conversation, "stream_image_outputs", fake_stream_image_outputs),
        ):
            outputs = list(conversation.stream_image_outputs_with_pool(ConversationRequest(
                prompt="cat",
                model="gpt-image-2",
                n=3,
            )))

        self.assertGreaterEqual(max_active, 2)
        self.assertEqual(len([output for output in outputs if output.kind == "progress"]), 3)
        self.assertEqual([output.index for output in outputs if output.kind == "result"], [1, 2, 3])
        self.assertEqual(len(fake_accounts.marked), 3)
        self.assertTrue(all(success for _, success in fake_accounts.marked))


if __name__ == "__main__":
    unittest.main()
