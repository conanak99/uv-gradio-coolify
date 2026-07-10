import base64
import io
import json
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import ANY, patch

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

    def test_generate_images_sends_resolution_and_count(self):
        response = io.BytesIO(
            json.dumps(
                {
                    "data": [
                        {"url": "https://image.test/one.png"},
                        {"b64_json": "dHdv"},
                    ]
                }
            ).encode()
        )

        with (
            patch.dict(os.environ, {"NANO_GPT_KEY": "test-key"}),
            patch.object(nano_gpt_api, "urlopen", return_value=response) as urlopen,
        ):
            results = nano_gpt_api.generate_images(
                "seedream-v5.0-lite",
                "A lighthouse",
                "2560x1440",
                2,
            )

        self.assertEqual(
            results,
            [
                "https://image.test/one.png",
                "data:image/png;base64,dHdv",
            ],
        )
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(
            payload,
            {
                "model": "seedream-v5.0-lite",
                "prompt": "A lighthouse",
                "resolution": "2560x1440",
                "n": 2,
            },
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

    def test_seedream_generate_models_route_to_nano_gpt(self):
        cases = [
            (
                main.SEEDREAM_LITE_GENERATE_MODEL,
                "16:9",
                "seedream-v5.0-lite",
                "2560x1440",
            ),
            (
                main.SEEDREAM_PRO_GENERATE_MODEL,
                "3:4",
                "bytedance/seedream-v5.0-pro",
                "3:4",
            ),
            (
                main.SEEDREAM_LITE_GENERATE_MODEL,
                "4:3",
                "seedream-v5.0-lite",
                "3072x2048",
            ),
            (
                main.SEEDREAM_LITE_GENERATE_MODEL,
                "3:4",
                "seedream-v5.0-lite",
                "2048x3072",
            ),
        ]

        for model_name, ratio, model_id, resolution in cases:
            with (
                self.subTest(model=model_name),
                patch.object(
                    main.nano_gpt_client,
                    "generate_images",
                    return_value=["https://image.test/generated.png"],
                ) as generate_images,
            ):
                result = main.run_generate_model(
                    "A lighthouse",
                    model_name,
                    ratio,
                    2,
                )

                expected_ratio = {
                    "4:3": "4:3 → 3:2",
                    "3:4": (
                        "3:4 → 2:3"
                        if model_name == main.SEEDREAM_LITE_GENERATE_MODEL
                        else "3:4"
                    ),
                }.get(ratio, ratio)
                self.assertEqual(
                    result,
                    [
                        (
                            "https://image.test/generated.png",
                            f"{model_name} ({expected_ratio})",
                        )
                    ],
                )
                generate_images.assert_called_once_with(
                    model_id,
                    "A lighthouse",
                    resolution,
                    2,
                )

    def test_grok_generation_still_routes_to_fal(self):
        with patch.object(
            main.fal_client,
            "generate_images",
            return_value=["https://image.test/grok.png"],
        ) as generate_images:
            result = main.run_generate_model(
                "A lighthouse",
                main.GROK_GENERATE_MODEL,
                "4:3",
                1,
            )

        self.assertEqual(
            result,
            [
                (
                    "https://image.test/grok.png",
                    f"{main.GROK_GENERATE_MODEL} (4:3)",
                )
            ],
        )
        generate_images.assert_called_once_with("A lighthouse", "4:3", 1)

    def test_generate_image_runs_all_selected_models(self):
        models = [
            main.GROK_GENERATE_MODEL,
            main.SEEDREAM_LITE_GENERATE_MODEL,
            main.SEEDREAM_PRO_GENERATE_MODEL,
        ]

        def result_for_model(_prompt, model_name, ratio, _count):
            return [(f"https://image.test/{model_name}.png", f"{model_name} ({ratio})")]

        with patch.object(
            main,
            "run_generate_model",
            side_effect=result_for_model,
        ) as run_generate_model:
            progress_updates = []
            results = main.generate_image(
                "A lighthouse",
                models,
                "1:1",
                2,
                progress_updates.append,
            )

        self.assertEqual(len(results), 3)
        self.assertEqual(
            [len(update) for update in progress_updates],
            [1, 2, 3],
        )
        self.assertEqual(
            {call.args for call in run_generate_model.call_args_list},
            {
                ("A lighthouse", model_name, "1:1", 2)
                for model_name in models
            },
        )

    def test_edit_image_reports_each_completed_model(self):
        models = ["Qwen Image Edit", "FLUX.2 Klein 9B Edit"]

        def edit_result(model_name, _image_reference, _prompt):
            return (f"https://image.test/{model_name}.png", model_name)

        with (
            patch.object(
                main.fal_client,
                "upload_image",
                return_value="https://image.test/input.png",
            ),
            patch.object(main, "run_model", side_effect=edit_result),
        ):
            progress_updates = []
            results = main.edit_image(
                "/tmp/input.png",
                "Improve the lighting",
                models,
                progress_updates.append,
            )

        self.assertEqual(len(results), 2)
        self.assertEqual(
            [len(update) for update in progress_updates],
            [1, 2],
        )


class HistoryTests(unittest.TestCase):
    def setUp(self):
        with main.HISTORY_LOCK:
            main.HISTORY_STORE.clear()

    def test_history_keeps_latest_ten_entries(self):
        for index in range(11):
            main.add_history_entry(
                operation="Generate",
                prompt=f"Prompt {index}",
                input_image=None,
                outputs=[(f"https://image.test/{index}.png", "Model")],
                settings="Aspect ratio: 1:1 · Images: 1",
            )

        history = main.get_history()
        self.assertEqual(len(history), 10)
        self.assertEqual(history[0]["prompt"], "Prompt 10")
        self.assertEqual(history[-1]["prompt"], "Prompt 1")

    def test_history_entry_view_returns_selected_input_and_outputs(self):
        outputs = [("https://image.test/edited.png", "Edit Model")]
        entry_id = main.add_history_entry(
            operation="Edit",
            prompt="Make it brighter",
            input_image="/tmp/input.png",
            outputs=outputs,
            settings="Models: Edit Model",
        )
        history = main.get_history()

        details, prompt, input_html, outputs_html = main.history_entry_view(
            history, entry_id
        )

        self.assertIn("Edit", details)
        self.assertEqual(prompt, "Make it brighter")
        self.assertIn("/gradio_api/file=/tmp/input.png", input_html)
        self.assertIn("https://image.test/edited.png", outputs_html)
        self.assertIn("Edit Model", outputs_html)

    def test_refresh_exposes_shared_history(self):
        entry_id = main.add_history_entry(
            operation="Generate",
            prompt="Visible on every device",
            input_image=None,
            outputs=[("https://image.test/shared.png", "Shared")],
            settings="Model: Shared",
        )

        selector, details, prompt, input_html, outputs_html = main.refresh_history()

        self.assertEqual(selector["value"], entry_id)
        self.assertEqual(prompt, "Visible on every device")
        self.assertIn("No input image", input_html)
        self.assertIn("https://image.test/shared.png", outputs_html)
        self.assertIn("Shared", outputs_html)

    def test_history_html_handles_data_urls_without_gradio_file_processing(self):
        outputs_html = main.history_outputs_html(
            [("data:image/png;base64,dHdv", "Inline image")]
        )

        self.assertIn('src="data:image/png;base64,dHdv"', outputs_html)
        self.assertIn("Inline image", outputs_html)

    def test_failed_job_is_not_added_to_history(self):
        with patch.object(
            main,
            "generate_image",
            side_effect=RuntimeError("provider failed"),
        ):
            flow = main.generate_image_flow(
                "This will fail",
                [main.SEEDREAM_PRO_GENERATE_MODEL],
                "1:1",
                "1",
            )
            next(flow)
            with self.assertRaisesRegex(RuntimeError, "provider failed"):
                list(flow)

        self.assertEqual(main.get_history(), [])

    def test_generate_flow_adds_successful_request_to_history(self):
        outputs = [
            (
                "https://image.test/generated.png",
                f"{main.SEEDREAM_LITE_GENERATE_MODEL} (1:1)",
            )
        ]

        with patch.object(
            main, "generate_image", return_value=outputs
        ) as generate_image:
            updates = list(
                main.generate_image_flow(
                    "A lighthouse",
                    [
                        main.SEEDREAM_LITE_GENERATE_MODEL,
                        main.SEEDREAM_PRO_GENERATE_MODEL,
                    ],
                    "1:1",
                    "1",
                )
            )

        self.assertEqual(len(updates), 2)
        final_update = updates[-1]
        self.assertEqual(len(final_update), 7)
        self.assertEqual(final_update[0], outputs)
        history = main.get_history()
        self.assertEqual(history[0]["operation"], "Generate")
        self.assertEqual(history[0]["prompt"], "A lighthouse")
        self.assertIn(
            main.SEEDREAM_LITE_GENERATE_MODEL,
            history[0]["settings"],
        )
        self.assertIn(
            main.SEEDREAM_PRO_GENERATE_MODEL,
            history[0]["settings"],
        )
        self.assertEqual(final_update[4], "A lighthouse")
        self.assertIn("No input image", final_update[5])
        self.assertIn("https://image.test/generated.png", final_update[6])
        generate_image.assert_called_once_with(
            "A lighthouse",
            [
                main.SEEDREAM_LITE_GENERATE_MODEL,
                main.SEEDREAM_PRO_GENERATE_MODEL,
            ],
            "1:1",
            1,
            ANY,
        )

    def test_generate_flow_streams_before_storing_complete_history(self):
        first_result = [("https://image.test/first.png", "First Model")]
        all_results = [
            *first_result,
            ("https://image.test/second.png", "Second Model"),
        ]
        release_second_result = threading.Event()

        def staged_generation(
            _prompt, _models, _ratio, _count, progress_callback
        ):
            progress_callback(first_result)
            release_second_result.wait(timeout=2)
            progress_callback(all_results)
            return all_results

        with patch.object(
            main,
            "generate_image",
            side_effect=staged_generation,
        ):
            flow = main.generate_image_flow(
                "Stream a lighthouse",
                [
                    main.GROK_GENERATE_MODEL,
                    main.SEEDREAM_PRO_GENERATE_MODEL,
                ],
                "1:1",
                "1",
            )
            next(flow)
            partial_update = next(flow)

            self.assertEqual(partial_update[0], first_result)
            self.assertEqual(main.get_history(), [])

            release_second_result.set()
            remaining_updates = list(flow)

        self.assertEqual(remaining_updates[0][0], all_results)
        self.assertEqual(remaining_updates[-1][0], all_results)
        completed_history = main.get_history()[0]
        self.assertEqual(completed_history["outputs"], all_results)

    def test_edit_flow_streams_before_storing_complete_history(self):
        first_result = [("https://image.test/edit-first.png", "First Edit")]
        all_results = [
            *first_result,
            ("https://image.test/edit-second.png", "Second Edit"),
        ]
        release_second_result = threading.Event()

        def staged_edit(_path, _prompt, _models, progress_callback):
            progress_callback(first_result)
            release_second_result.wait(timeout=2)
            progress_callback(all_results)
            return all_results

        with patch.object(main, "edit_image", side_effect=staged_edit):
            flow = main.edit_image_flow(
                "/tmp/input.png",
                "Stream an edit",
                ["First Edit", "Second Edit"],
            )
            next(flow)
            partial_update = next(flow)

            self.assertEqual(partial_update[0], first_result)
            self.assertEqual(main.get_history(), [])

            release_second_result.set()
            remaining_updates = list(flow)

        self.assertEqual(remaining_updates[0][0], all_results)
        self.assertEqual(remaining_updates[-1][0], all_results)
        completed_history = main.get_history()[0]
        self.assertEqual(completed_history["outputs"], all_results)

    def test_generation_completes_in_memory_after_client_disconnect(self):
        release_generation = threading.Event()
        outputs = [
            (
                "https://image.test/disconnected.png",
                f"{main.SEEDREAM_PRO_GENERATE_MODEL} (1:1)",
            )
        ]

        def delayed_generation(*_args):
            release_generation.wait(timeout=2)
            return outputs

        with patch.object(main, "generate_image", side_effect=delayed_generation):
            flow = main.generate_image_flow(
                "A lighthouse after disconnect",
                [main.SEEDREAM_PRO_GENERATE_MODEL],
                "1:1",
                "1",
            )
            next(flow)
            self.assertEqual(main.get_history(), [])

            flow.close()
            release_generation.set()

            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                history = main.get_history()
                if history:
                    break
                time.sleep(0.01)
            else:
                self.fail("Background generation did not complete after disconnect")

        self.assertEqual(history[0]["outputs"], outputs)


if __name__ == "__main__":
    unittest.main()
