import base64
import concurrent.futures
import io
import json
import os
import queue
import tempfile
import threading
import time
import unittest
from unittest.mock import ANY, patch

from PIL import Image

from clients import fal as fal_api
from clients import nano_gpt as nano_gpt_api
import history
import image_utils
import main
import workflows


class ImageUtilsTests(unittest.TestCase):
    def test_closest_aspect_ratio_uses_relative_ratio_distance(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as image_file:
            Image.new("RGB", (1600, 1000), "red").save(image_file)
            image_file.flush()

            result = image_utils.closest_aspect_ratio(
                image_file.name,
                nano_gpt_api.SEEDREAM_PRO_EDIT_ASPECT_RATIOS,
            )

        self.assertEqual(result, "3:2")

    def test_prepare_for_aspect_ratio_center_crops_copy(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as image_file:
            Image.new("RGB", (1000, 800), "red").save(image_file)
            image_file.flush()

            prepared_path, aspect_ratio = image_utils.prepare_for_aspect_ratio(
                image_file.name,
                nano_gpt_api.SEEDREAM_PRO_EDIT_ASPECT_RATIOS,
            )
            try:
                with Image.open(image_file.name) as original:
                    self.assertEqual(original.size, (1000, 800))
                with Image.open(prepared_path) as prepared:
                    self.assertEqual(prepared.size, (1000, 750))
            finally:
                os.remove(prepared_path)

        self.assertEqual(aspect_ratio, "4:3")


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

        with self.assertLogs("clients.nano_gpt", level="INFO") as captured_logs:
            with (
                patch.dict(os.environ, {"NANO_GPT_KEY": "test-key"}),
                patch.object(
                    nano_gpt_api,
                    "image_to_data_url",
                    return_value="data:image/png;base64,aW1hZ2U=",
                ) as image_to_data_url,
                patch.object(
                    nano_gpt_api,
                    "prepare_for_aspect_ratio",
                    return_value=("/tmp/prepared.png", "4:3"),
                ) as prepare_for_aspect_ratio,
                patch.object(
                    nano_gpt_api, "urlopen", return_value=response
                ) as urlopen,
            ):
                result = nano_gpt_api.edit_image(
                    "bytedance/seedream-v5.0-pro/edit",
                    "/tmp/input.png",
                    "Improve the lighting",
                )

        self.assertEqual(result, "https://image.test/out.png")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, nano_gpt_api.IMAGE_EDITS_URL)
        self.assertEqual(request.headers["Authorization"], "Bearer test-key")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "bytedance/seedream-v5.0-pro/edit")
        self.assertEqual(payload["n"], 1)
        self.assertEqual(payload["size"], "4:3")
        self.assertEqual(
            payload["imageDataUrl"], "data:image/png;base64,aW1hZ2U="
        )
        self.assertNotIn("input_references", payload)
        prepare_for_aspect_ratio.assert_called_once_with(
            "/tmp/input.png",
            nano_gpt_api.SEEDREAM_PRO_EDIT_ASPECT_RATIOS,
        )
        image_to_data_url.assert_called_once_with("/tmp/prepared.png")
        logs = "\n".join(captured_logs.output)
        self.assertIn("request endpoint=", logs)
        self.assertIn("response endpoint=", logs)
        self.assertNotIn("test-key", logs)
        self.assertNotIn("aW1hZ2U=", logs)
        self.assertNotIn("Improve the lighting", logs)

    def test_run_nano_gpt_model_requires_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "NANO_GPT_KEY"):
                nano_gpt_api.edit_image(
                    "bytedance/seedream-v5.0-pro/edit",
                    "/tmp/input.png",
                    "Improve the lighting",
                )

    def test_invalid_base64_image_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "invalid base64"):
            nano_gpt_api._cache_base64_image("not-valid-base64")

    def test_generate_images_sends_resolution_and_count(self):
        generated_image = io.BytesIO()
        Image.new("RGB", (8, 8), "blue").save(generated_image, format="PNG")
        generated_bytes = generated_image.getvalue()
        response = io.BytesIO(
            json.dumps(
                {
                    "data": [
                        {"url": "https://image.test/one.png"},
                        {
                            "b64_json": base64.b64encode(
                                generated_bytes
                            ).decode()
                        },
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
            results[0],
            "https://image.test/one.png",
        )
        self.assertTrue(results[1].endswith(".png"))
        with open(results[1], "rb") as generated_file:
            self.assertEqual(generated_file.read(), generated_bytes)
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(
            urlopen.call_args.args[0].full_url,
            nano_gpt_api.IMAGES_URL,
        )
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
        with self.assertLogs("clients.fal", level="INFO") as captured_logs:
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
        logs = "\n".join(captured_logs.output)
        self.assertIn("request operation=edit", logs)
        self.assertIn("response operation=edit", logs)
        self.assertNotIn("Improve the lighting", logs)
        self.assertNotIn("https://image.test/input.png", logs)
        self.assertNotIn("https://image.test/fal.png", logs)


class ProviderRoutingTests(unittest.TestCase):
    def test_elapsed_button_label_formats_seconds_and_minutes(self):
        with patch.object(workflows.time, "monotonic", return_value=112.9):
            self.assertEqual(
                workflows.elapsed_button_label("Editing", 100.0),
                "Editing... 12s",
            )

        with patch.object(workflows.time, "monotonic", return_value=165.0):
            self.assertEqual(
                workflows.elapsed_button_label("Generating", 100.0),
                "Generating... 1m 05s",
            )

    def test_seedream_routes_to_nano_gpt_client(self):
        with patch.object(
            workflows.nano_gpt_client,
            "edit_image",
            return_value="https://image.test/nano.png",
        ) as edit_image:
            result = workflows.run_model(
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
                workflows.SEEDREAM_LITE_GENERATE_MODEL,
                "16:9",
                "seedream-v5.0-lite",
                "2560x1440",
            ),
            (
                workflows.SEEDREAM_PRO_GENERATE_MODEL,
                "3:4",
                "bytedance/seedream-v5.0-pro",
                "3:4",
            ),
            (
                workflows.SEEDREAM_LITE_GENERATE_MODEL,
                "4:3",
                "seedream-v5.0-lite",
                "3072x2048",
            ),
            (
                workflows.SEEDREAM_LITE_GENERATE_MODEL,
                "3:4",
                "seedream-v5.0-lite",
                "2048x3072",
            ),
        ]

        for model_name, ratio, model_id, resolution in cases:
            with (
                self.subTest(model=model_name),
                patch.object(
                    workflows.nano_gpt_client,
                    "generate_images",
                    return_value=["https://image.test/generated.png"],
                ) as generate_images,
            ):
                result = workflows.run_generate_model(
                    "A lighthouse",
                    model_name,
                    ratio,
                    2,
                )

                expected_ratio = {
                    "4:3": "4:3 → 3:2",
                    "3:4": (
                        "3:4 → 2:3"
                        if model_name == workflows.SEEDREAM_LITE_GENERATE_MODEL
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
            workflows.fal_client,
            "generate_images",
            return_value=["https://image.test/grok.png"],
        ) as generate_images:
            result = workflows.run_generate_model(
                "A lighthouse",
                workflows.GROK_GENERATE_MODEL,
                "4:3",
                1,
            )

        self.assertEqual(
            result,
            [
                (
                    "https://image.test/grok.png",
                    f"{workflows.GROK_GENERATE_MODEL} (4:3)",
                )
            ],
        )
        generate_images.assert_called_once_with("A lighthouse", "4:3", 1)

    def test_generate_image_runs_all_selected_models(self):
        models = [
            workflows.GROK_GENERATE_MODEL,
            workflows.SEEDREAM_LITE_GENERATE_MODEL,
            workflows.SEEDREAM_PRO_GENERATE_MODEL,
        ]

        def result_for_model(_prompt, model_name, ratio, _count):
            return [(f"https://image.test/{model_name}.png", f"{model_name} ({ratio})")]

        with patch.object(
            workflows,
            "run_generate_model",
            side_effect=result_for_model,
        ) as run_generate_model:
            progress_updates = []
            results = workflows.generate_image(
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
                workflows.fal_client,
                "upload_image",
                return_value="https://image.test/input.png",
            ),
            patch.object(workflows, "run_model", side_effect=edit_result),
        ):
            progress_updates = []
            results = workflows.edit_image(
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
        with history.HISTORY_LOCK:
            history.HISTORY_STORE.clear()

    def test_history_keeps_latest_ten_entries(self):
        for index in range(11):
            history.add_history_entry(
                operation="Generate",
                prompt=f"Prompt {index}",
                input_image=None,
                outputs=[(f"https://image.test/{index}.png", "Model")],
                settings="Aspect ratio: 1:1 · Images: 1",
            )

        entries = history.get_history()
        self.assertEqual(len(entries), 10)
        self.assertEqual(entries[0]["prompt"], "Prompt 10")
        self.assertEqual(entries[-1]["prompt"], "Prompt 1")

    def test_history_addition_logs_safe_metadata(self):
        with self.assertLogs("history", level="INFO") as captured_logs:
            entry_id = history.add_history_entry(
                operation="Generate",
                prompt="Private prompt text",
                input_image=None,
                outputs=[("https://image.test/private.png", "Model")],
                settings="Model: Test",
            )

        logs = "\n".join(captured_logs.output)
        self.assertIn(f"history added id={entry_id}", logs)
        self.assertIn("operation=Generate", logs)
        self.assertIn("outputs=1", logs)
        self.assertNotIn("Private prompt text", logs)
        self.assertNotIn("https://image.test/private.png", logs)

    def test_history_entry_view_returns_selected_input_and_outputs(self):
        outputs = [("https://image.test/edited.png", "Edit Model")]
        entry_id = history.add_history_entry(
            operation="Edit",
            prompt="Make it brighter",
            input_image="/tmp/input.png",
            outputs=outputs,
            settings="Models: Edit Model",
        )
        entries = history.get_history()

        details, prompt, input_html, outputs_html = history.history_entry_view(
            entries, entry_id
        )

        self.assertIn("Edit", details)
        self.assertEqual(prompt, "Make it brighter")
        self.assertIn("/gradio_api/file=/tmp/input.png", input_html)
        self.assertIn("https://image.test/edited.png", outputs_html)
        self.assertIn("Edit Model", outputs_html)

    def test_refresh_exposes_shared_history(self):
        entry_id = history.add_history_entry(
            operation="Generate",
            prompt="Visible on every device",
            input_image=None,
            outputs=[("https://image.test/shared.png", "Shared")],
            settings="Model: Shared",
        )

        selector, details, prompt, input_html, outputs_html = history.refresh_history()

        self.assertEqual(selector["value"], entry_id)
        self.assertEqual(prompt, "Visible on every device")
        self.assertIn("No input image", input_html)
        self.assertIn("https://image.test/shared.png", outputs_html)
        self.assertIn("Shared", outputs_html)

    def test_history_html_handles_data_urls_without_gradio_file_processing(self):
        outputs_html = history.history_outputs_html(
            [("data:image/png;base64,dHdv", "Inline image")]
        )

        self.assertIn('src="data:image/png;base64,dHdv"', outputs_html)
        self.assertIn("Inline image", outputs_html)

    def test_failed_job_is_not_added_to_history(self):
        with patch.object(
            workflows,
            "generate_image",
            side_effect=RuntimeError("provider failed"),
        ):
            flow = workflows.generate_image_flow(
                "This will fail",
                [workflows.SEEDREAM_PRO_GENERATE_MODEL],
                "1:1",
                "1",
            )
            next(flow)
            with self.assertRaisesRegex(RuntimeError, "provider failed"):
                list(flow)

        self.assertEqual(history.get_history(), [])

    def test_job_progress_emits_heartbeat_while_waiting(self):
        future: concurrent.futures.Future[list[history.GalleryItem]] = (
            concurrent.futures.Future()
        )
        progress_queue: queue.Queue[list[history.GalleryItem]] = queue.Queue()
        progress = workflows.iter_job_progress(
            future,
            progress_queue,
            heartbeat_seconds=0.01,
        )

        self.assertIsNone(next(progress))
        future.set_result([])
        self.assertEqual(list(progress), [])

    def test_generate_flow_adds_successful_request_to_history(self):
        outputs = [
            (
                "https://image.test/generated.png",
                f"{workflows.SEEDREAM_LITE_GENERATE_MODEL} (1:1)",
            )
        ]

        with patch.object(
            workflows, "generate_image", return_value=outputs
        ) as generate_image:
            updates = list(
                workflows.generate_image_flow(
                    "A lighthouse",
                    [
                        workflows.SEEDREAM_LITE_GENERATE_MODEL,
                        workflows.SEEDREAM_PRO_GENERATE_MODEL,
                    ],
                    "1:1",
                    "1",
                )
            )

        self.assertEqual(updates[0][1]["value"], "Generating... 0s")
        self.assertEqual(len(updates), 2)
        final_update = updates[-1]
        self.assertEqual(len(final_update), 7)
        self.assertEqual(final_update[0], outputs)
        entries = history.get_history()
        self.assertEqual(entries[0]["operation"], "Generate")
        self.assertEqual(entries[0]["prompt"], "A lighthouse")
        self.assertIn(
            workflows.SEEDREAM_LITE_GENERATE_MODEL,
            entries[0]["settings"],
        )
        self.assertIn(
            workflows.SEEDREAM_PRO_GENERATE_MODEL,
            entries[0]["settings"],
        )
        self.assertEqual(final_update[4], "A lighthouse")
        self.assertIn("No input image", final_update[5])
        self.assertIn("https://image.test/generated.png", final_update[6])
        generate_image.assert_called_once_with(
            "A lighthouse",
            [
                workflows.SEEDREAM_LITE_GENERATE_MODEL,
                workflows.SEEDREAM_PRO_GENERATE_MODEL,
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
            workflows,
            "generate_image",
            side_effect=staged_generation,
        ):
            flow = workflows.generate_image_flow(
                "Stream a lighthouse",
                [
                    workflows.GROK_GENERATE_MODEL,
                    workflows.SEEDREAM_PRO_GENERATE_MODEL,
                ],
                "1:1",
                "1",
            )
            next(flow)
            partial_update = next(flow)

            self.assertEqual(partial_update[0], first_result)
            self.assertEqual(history.get_history(), [])

            release_second_result.set()
            remaining_updates = list(flow)

        self.assertEqual(remaining_updates[0][0], all_results)
        self.assertEqual(remaining_updates[-1][0], all_results)
        completed_history = history.get_history()[0]
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

        with patch.object(workflows, "edit_image", side_effect=staged_edit):
            flow = workflows.edit_image_flow(
                "/tmp/input.png",
                "Stream an edit",
                ["First Edit", "Second Edit"],
            )
            initial_update = next(flow)
            partial_update = next(flow)

            self.assertEqual(initial_update[1]["value"], "Editing... 0s")
            self.assertEqual(partial_update[0], first_result)
            self.assertEqual(history.get_history(), [])

            release_second_result.set()
            remaining_updates = list(flow)

        self.assertEqual(remaining_updates[0][0], all_results)
        self.assertEqual(remaining_updates[-1][0], all_results)
        completed_history = history.get_history()[0]
        self.assertEqual(completed_history["outputs"], all_results)

    def test_generation_completes_in_memory_after_client_disconnect(self):
        release_generation = threading.Event()
        outputs = [
            (
                "https://image.test/disconnected.png",
                f"{workflows.SEEDREAM_PRO_GENERATE_MODEL} (1:1)",
            )
        ]

        def delayed_generation(*_args):
            release_generation.wait(timeout=2)
            return outputs

        with patch.object(workflows, "generate_image", side_effect=delayed_generation):
            flow = workflows.generate_image_flow(
                "A lighthouse after disconnect",
                [workflows.SEEDREAM_PRO_GENERATE_MODEL],
                "1:1",
                "1",
            )
            next(flow)
            self.assertEqual(history.get_history(), [])

            flow.close()
            release_generation.set()

            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                entries = history.get_history()
                if entries:
                    break
                time.sleep(0.01)
            else:
                self.fail("Background generation did not complete after disconnect")

        self.assertEqual(entries[0]["outputs"], outputs)


class UiConfigTests(unittest.TestCase):
    def test_model_preferences_restore_only_available_models(self):
        edit_model = main.DEFAULT_EDIT_MODELS[0]
        generate_model = main.DEFAULT_GENERATE_MODELS[-1]

        edit_models, generate_models = main.load_model_preferences(
            [edit_model, "Removed Edit Model"],
            ["Removed Generate Model", generate_model],
        )

        self.assertEqual(edit_models, [edit_model])
        self.assertEqual(generate_models, [generate_model])

    def test_model_preferences_fall_back_to_defaults_for_invalid_state(self):
        edit_models, generate_models = main.load_model_preferences(
            None,
            [{"bad": "state"}],
        )

        self.assertEqual(edit_models, main.DEFAULT_EDIT_MODELS)
        self.assertEqual(generate_models, main.DEFAULT_GENERATE_MODELS)

    def test_model_preference_save_ignores_removed_models(self):
        edit_model = main.DEFAULT_EDIT_MODELS[-1]
        generate_model = main.DEFAULT_GENERATE_MODELS[0]

        self.assertEqual(
            main.save_edit_model_preference([edit_model, "Removed Edit Model"]),
            [edit_model],
        )
        self.assertEqual(
            main.save_generate_model_preference(
                [generate_model, "Removed Generate Model"]
            ),
            [generate_model],
        )

    def test_edit_prompt_preference_round_trips_text(self):
        self.assertEqual(
            main.edit_prompt_preference("Make the background warmer"),
            "Make the background warmer",
        )

    def test_edit_prompt_preference_ignores_invalid_state(self):
        self.assertEqual(main.edit_prompt_preference(["not", "text"]), "")

    def test_form_controls_use_ios_safe_font_size(self):
        config = main.demo.get_config_file()
        prompt_inputs = [
            component
            for component in config["components"]
            if "prompt-input" in component.get("props", {}).get("elem_classes", [])
        ]
        history_selects = [
            component
            for component in config["components"]
            if "history-select"
            in component.get("props", {}).get("elem_classes", [])
        ]

        self.assertEqual(len(prompt_inputs), 2)
        self.assertEqual(len(history_selects), 1)
        self.assertTrue(history_selects[0]["props"]["allow_custom_value"])
        self.assertIn("font-size: 16px", main.PROMPT_CSS)
        self.assertIn(".history-select input", main.PROMPT_CSS)

    def test_launch_allows_nano_output_cache(self):
        with patch.object(main.demo, "launch") as launch:
            main.launch_app(7860)

        launch.assert_called_once_with(
            server_name="0.0.0.0",
            server_port=7860,
            css=main.PROMPT_CSS,
            allowed_paths=[str(nano_gpt_api.OUTPUT_CACHE_PATH)],
        )


if __name__ == "__main__":
    unittest.main()
