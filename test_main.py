import base64
import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import main


class NanoGptTests(unittest.TestCase):
    def test_image_to_data_url_uses_detected_mime_type(self):
        with tempfile.NamedTemporaryFile(suffix=".bin") as image_file:
            Image.new("RGB", (10, 10), "red").save(image_file, format="JPEG")
            image_file.flush()

            data_url = main.image_to_data_url(image_file.name)

        prefix, encoded = data_url.split(",", 1)
        self.assertEqual(prefix, "data:image/jpeg;base64")
        self.assertTrue(base64.b64decode(encoded).startswith(b"\xff\xd8"))

    def test_run_nano_gpt_model_uses_env_key_and_edit_model(self):
        response = io.BytesIO(json.dumps({"data": [{"url": "https://image.test/out.png"}]}).encode())

        with (
            patch.dict(os.environ, {"NANO_GPT_KEY": "test-key"}),
            patch.object(main, "urlopen", return_value=response) as urlopen,
        ):
            result = main.run_nano_gpt_model(
                "Seedream 5.0 Pro Edit (NanoGPT)",
                "data:image/png;base64,aW1hZ2U=",
                "Improve the lighting",
            )

        self.assertEqual(
            result,
            ("https://image.test/out.png", "Seedream 5.0 Pro Edit (NanoGPT)"),
        )
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
                main.run_nano_gpt_model(
                    "Seedream 5.0 Pro Edit (NanoGPT)",
                    "data:image/png;base64,aW1hZ2U=",
                    "Improve the lighting",
                )


if __name__ == "__main__":
    unittest.main()
