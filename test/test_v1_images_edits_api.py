from __future__ import annotations

import base64
import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.ai as ai_module
from services import log_service as log_module
from services.protocol.conversation import ImageGenerationError


AUTH_HEADERS = {"Authorization": "Bearer chatgpt2api"}
PNG_BYTES = b"\x89PNG\r\n\x1a\n"
DATA_IMAGE_URL = f"data:image/png;base64,{base64.b64encode(PNG_BYTES).decode('ascii')}"


class ImagesEditsApiTests(unittest.TestCase):
    def setUp(self):
        self.handle_calls = []

        def fake_handle(payload):
            self.handle_calls.append(payload)
            return {"created": 1, "data": [{"b64_json": base64.b64encode(b"out").decode("ascii")}]}

        self.handler_patcher = mock.patch.object(ai_module.openai_v1_image_edit, "handle", fake_handle)
        self.handler_patcher.start()
        self.addCleanup(self.handler_patcher.stop)
        app = FastAPI()
        app.include_router(ai_module.create_router())
        self.client = TestClient(app)

    def test_edit_accepts_json_image_url(self):
        """测试图片编辑接口支持官方 JSON image_url 引用。"""
        response = self.client.post(
            "/v1/images/edits",
            headers=AUTH_HEADERS,
            json={
                "model": "gpt-image-2",
                "prompt": "edit",
                "images": [{"image_url": DATA_IMAGE_URL}],
                "n": 1,
                "response_format": "b64_json",
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(self.handle_calls), 1)
        payload = self.handle_calls[0]
        self.assertEqual(payload["prompt"], "edit")
        self.assertEqual(payload["n"], 1)
        self.assertEqual(payload["images"], [(PNG_BYTES, "image_url.png", "image/png")])

    def test_edit_success_logs_but_hides_upstream_context(self):
        """测试会话调试上下文只写日志，不暴露给接口调用方。"""
        records = []

        def handle_with_context(_payload):
            return {
                "created": 1,
                "data": [{"b64_json": base64.b64encode(b"out").decode("ascii")}],
                "_upstream_context": {"conversation_id": "conv-1"},
            }

        with (
            mock.patch.object(ai_module.openai_v1_image_edit, "handle", handle_with_context),
            mock.patch.object(
                log_module.log_service,
                "add",
                lambda type, summary="", detail=None, **data: records.append((type, summary, detail or data)),
            ),
        ):
            response = self.client.post(
                "/v1/images/edits",
                headers=AUTH_HEADERS,
                json={"model": "gpt-image-2", "prompt": "edit", "images": [{"image_url": DATA_IMAGE_URL}]},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertNotIn("_upstream_context", response.json())
        self.assertEqual(records[0][2]["upstream_context"]["conversation_id"], "conv-1")

    def test_edit_rejects_file_id_reference(self):
        """测试图片编辑接口对暂不支持的 file_id 返回明确错误。"""
        response = self.client.post(
            "/v1/images/edits",
            headers=AUTH_HEADERS,
            json={
                "model": "gpt-image-2",
                "prompt": "edit",
                "images": [{"file_id": "file-abc123"}],
            },
        )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("file_id image references are not supported", response.text)
        self.assertEqual(self.handle_calls, [])

    def test_edit_failure_logs_upstream_context(self):
        """测试上游返回文本失败时，调用日志保留上游会话上下文。"""
        records = []

        def fail_handle(_payload):
            raise ImageGenerationError(
                '{"size":"1792x1024","n":1}',
                status_code=400,
                error_type="invalid_request_error",
                code="content_policy_violation",
                account_email="account@example.test",
                upstream_context={
                    "conversation_id": "conv-1",
                    "tool_invoked": False,
                    "turn_use_case": "image gen",
                    "message_preview": '{"size":"1792x1024","n":1}',
                },
            )

        with (
            mock.patch.object(ai_module.openai_v1_image_edit, "handle", fail_handle),
            mock.patch.object(
                log_module.log_service,
                "add",
                lambda type, summary="", detail=None, **data: records.append((type, summary, detail or data)),
            ),
        ):
            response = self.client.post(
                "/v1/images/edits",
                headers=AUTH_HEADERS,
                json={"model": "gpt-image-2", "prompt": "edit", "images": [{"image_url": DATA_IMAGE_URL}]},
            )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(len(records), 1)
        detail = records[0][2]
        self.assertEqual(detail["error"], '{"size":"1792x1024","n":1}')
        self.assertEqual(detail["account_email"], "account@example.test")
        self.assertEqual(detail["upstream_context"]["conversation_id"], "conv-1")
        self.assertEqual(detail["upstream_context"]["message_preview"], '{"size":"1792x1024","n":1}')


if __name__ == "__main__":
    unittest.main()
