from unittest.mock import Mock

import torch
from PIL import Image

import app
from app import Item, Instance, _pack_labeled_instances, _parse_segment_payload_json


def test_caller_labels_preserve_subject_object_categories_and_context() -> None:
    labels, score, mask = _parse_segment_payload_json(
        '{"subjects":[{"name":" students ","context":" group portrait "}],'
        '"objects":[{"name":" graduation cap","context":"lower photo"}]}'
    )
    assert labels == {
        "subjects": [Item(name="students", context="group portrait")],
        "objects": [Item(name="graduation cap", context="lower photo")],
    }
    assert 0 <= score <= 1
    assert 0 <= mask <= 1


def test_caller_labeled_path_uses_photo_analysis_instance_packing() -> None:
    items = [Item(name="students", context="group portrait")]
    instances = [[Instance(score=0.9, bbox=(1.2, 2.4, 10.6, 20.5), mask=None)]]
    assert _pack_labeled_instances(items, instances, include_context=True) == [
        {
            "name": "students",
            "context": "group portrait",
            "instances": [
                {
                    "score": 0.9,
                    "bbox": [1, 2, 11, 20],
                    "mask_png_base64": None,
                    "mask_area": None,
                }
            ],
        }
    ]


def test_sam_uses_context_for_caller_labeled_item_without_losing_name(
    monkeypatch,
) -> None:
    processor = Mock()
    processor.return_value = {
        "original_sizes": torch.tensor([[20, 20]]),
    }
    processor.post_process_instance_segmentation.return_value = [
        {"masks": None, "boxes": None, "scores": None}
    ]
    model = Mock(return_value=object())
    monkeypatch.setattr(app, "_load_sam3_if_needed", lambda: None)
    app.app.state.sam3_processor = processor
    app.app.state.sam3_model = model

    item = Item(
        name="Tashfeen Malik",
        context="woman in a light hijab in the small portrait at lower right",
    )
    instances = app._sam3_segment(
        Image.new("RGB", (20, 20)), [item], use_context_prompt=True
    )

    assert instances == [[]]
    assert processor.call_args.kwargs["text"] == item.context
    packed = _pack_labeled_instances([item], instances, include_context=True)
    assert packed[0]["name"] == "Tashfeen Malik"


def test_legacy_sam_prompt_still_uses_name(monkeypatch) -> None:
    processor = Mock()
    processor.return_value = {
        "original_sizes": torch.tensor([[20, 20]]),
    }
    processor.post_process_instance_segmentation.return_value = [
        {"masks": None, "boxes": None, "scores": None}
    ]
    monkeypatch.setattr(app, "_load_sam3_if_needed", lambda: None)
    app.app.state.sam3_processor = processor
    app.app.state.sam3_model = Mock(return_value=object())

    item = Item(name="students", context="front row at left")
    app._sam3_segment(Image.new("RGB", (20, 20)), [item])

    assert processor.call_args.kwargs["text"] == "students"


def test_startup_loads_sam3_before_service_is_ready(monkeypatch) -> None:
    sam3_model = object()
    sam3_processor = object()

    monkeypatch.setattr(app.genai, "Client", Mock(return_value=object()))

    def load_sam3() -> None:
        app.app.state.sam3_model = sam3_model
        app.app.state.sam3_processor = sam3_processor

    monkeypatch.setattr(app, "_load_sam3_if_needed", load_sam3)

    app._load_models()

    assert app.app.state.sam3_model is sam3_model
    assert app.app.state.sam3_processor is sam3_processor
    assert app.healthz().status_code == 200


def test_healthz_rejects_an_unloaded_sam3(monkeypatch) -> None:
    monkeypatch.setattr(app.app.state, "sam3_model", None, raising=False)
    monkeypatch.setattr(app.app.state, "sam3_processor", None, raising=False)

    response = app.healthz()

    assert response.status_code == 503
    assert response.body == b"SAM3 not ready"
