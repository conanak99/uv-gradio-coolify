import base64
import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from clients import fal as fal_api
from clients import nano_gpt as nano_gpt_api
import main


class NanoGptTests(unittest.TestCase):
    def test_image_to_data_url_uses_detected_mime_type(self):
        with tempfile.NamedTemporaryFile(suffix=".bin") as image_file:
            Image.new("RGB", (10, 10), "red").save(image_file, format="JPEG")
            image_file.flush()

            data_url = nano_gpt_api.image_to_data_url(image_file.name)

        prefix, encoded = data_url.split(",", 1)
        self.assertEqual(prefix, "data:image/jpeg;base64")
        self.assertTrue(base64.b64decode(encoded).startswith(b"\xff\xd8"))

    def test_run_nano_gpt_model_uses_env_key_and_edit_model(self):
        response = io.BytesIO(json.dumps({"data": [{"url": "https://image.test/out.png"}]}).encode())

        with (
            patch.dict(os.environ, {"NANO_GPT_KEY": "test-key"}),
            patch.object(
                nano_gpt_api, "image_to_data_url", return_value="data:image/png;base64,aW1hZ2U="
            ),
            patch.object(nano_gpt_api, "urlopen", return_value=response) as urlopen,
        ):
            result = nano_gpt_api.edit_image(
                "bytedance/seedream-v5.0-pro/edit",
                "/tmp/input.png",
                "Improve the lighting",
            )

        self.assertEqual(result, "https://image.test/out.png")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.headers["Authorization"], "Bearer test-key")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "bytedance/seedream-v5.0-pro/edit")
        self.assertEqual(payload["n"], 1)
        self.assertEqual(
            payload["input_references"], ["data:image/png;base64,aW1hZ2U="]
        )

    def test_run_nano_gpt_model_requires_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "NANO_GPT_KEY"):
                nano_gpt_api.edit_image(
                    "bytedance/seedream-v5.0-pro/edit",
                    "/tmp/input.png",
                    "Improve the lighting",
                )


class FalClientTests(unittest.TestCase):
    def test_edit_image_builds_fal_request(self):
        with patch.object(
            fal_api.fal_client,
            "subscribe",
            return_value={"images": [{"url": "https://image.test/fal.png"}]},
        ) as subscribe:
            result = fal_api.edit_image(
                "fal-ai/qwen-image-edit-2511",
                "https://image.test/input.png",
                "Improve the lighting",
            )

        self.assertEqual(result, "https://image.test/fal.png")
        subscribe.assert_called_once_with(
            "fal-ai/qwen-image-edit-2511",
            arguments={
                "prompt": "Improve the lighting",
                "image_urls": ["https://image.test/input.png"],
                "sync_mode": False,
                "num_images": 1,
                "enable_safety_checker": False,
            },
        )


class ProviderRoutingTests(unittest.TestCase):
    def test_seedream_routes_to_nano_gpt_client(self):
        with patch.object(
            main.nano_gpt_client,
            "edit_image",
            return_value="https://image.test/nano.png",
        ) as edit_image:
            result = main.run_model(
                "Seedream 5.0 Pro Edit (NanoGPT)",
                "/tmp/input.png",
                "Improve the lighting",
            )

        self.assertEqual(
            result,
            (
                "https://image.test/nano.png",
                "Seedream 5.0 Pro Edit (NanoGPT)",
            ),
        )
        edit_image.assert_called_once_with(
            "bytedance/seedream-v5.0-pro/edit",
            "/tmp/input.png",
            "Improve the lighting",
        )


class HistoryTests(unittest.TestCase):
    def test_history_keeps_latest_ten_entries_without_mutating_input(self):
        history: main.History = []

        for index in range(11):
            previous_history = history
            history = main.add_history_entry(
                history,
                operation="Generate",
                prompt=f"Prompt {index}",
                input_image=None,
                outputs=[(f"https://image.test/{index}.png", "Model")],
                settings="Aspect ratio: 1:1 · Images: 1",
            )
            self.assertIsNot(history, previous_history)

        self.assertEqual(len(history), 10)
        self.assertEqual(history[0]["prompt"], "Prompt 10")
        self.assertEqual(history[-1]["prompt"], "Prompt 1")

    def test_history_entry_view_returns_selected_input_and_outputs(self):
        outputs = [("https://image.test/edited.png", "Edit Model")]
        history = main.add_history_entry(
            [],
            operation="Edit",
            prompt="Make it brighter",
            input_image="/tmp/input.png",
            outputs=outputs,
            settings="Models: Edit Model",
        )

        details, prompt, input_image, selected_outputs = main.history_entry_view(
            history, history[0]["id"]
        )

        self.assertIn("Edit", details)
        self.assertEqual(prompt, "Make it brighter")
        self.assertEqual(input_image, "/tmp/input.png")
        self.assertEqual(selected_outputs, outputs)

    def test_generate_flow_adds_successful_request_to_history(self):
        outputs = [("https://image.test/generated.png", "Grok Imagine (1:1)")]

        with patch.object(main, "generate_image", return_value=outputs):
            updates = list(main.generate_image_flow("A lighthouse", "1:1", "1", []))

        self.assertEqual(len(updates), 2)
        final_update = updates[-1]
        self.assertEqual(len(final_update), 8)
        self.assertEqual(final_update[0], outputs)
        self.assertEqual(final_update[2][0]["operation"], "Generate")
        self.assertEqual(final_update[2][0]["prompt"], "A lighthouse")
        self.assertEqual(final_update[5], "A lighthouse")
        self.assertIsNone(final_update[6])
        self.assertEqual(final_update[7], outputs)


if __name__ == "__main__":
    unittest.main()
