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
