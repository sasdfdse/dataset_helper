import json
from contextlib import contextmanager

import cv2
import numpy as np
import pytest

from background_swap.common.settings import (BackgroundSwapSettings, effective_settings, generation_diff,
                                             settings_from_dict, settings_to_dict, validate_settings)
from background_swap.stages.references import (CACHE_NAME, MAX_PER_AXIS, ReferencePhotoError,
                                               clean_description, describe_prompt, describe_references,
                                               pending_references)
from tests.bg_fakes import FakeDescriber


class _Progress:
    def __init__(self):
        self.stages, self.advanced = [], 0

    def stage(self, name, total):
        self.stages.append((name, total))

    def advance(self, n=1):
        self.advanced += n

    def event(self, kind, **fields):
        pass


def _opener(describer, opened):
    @contextmanager
    def open_describer():
        opened.append(1)
        yield describer
    return open_describer


def _photo(path, size=(40, 30), value=80):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((size[1], size[0], 3), value, np.uint8))
    return path


def _settings(tmp_path, **kw):
    base = dict(scene_axes={"venue": ["gym"], "lighting": [], "camera": ["blurry photo"]})
    base.update(kw)
    return BackgroundSwapSettings(**base)


def test_prompt_focuses_on_each_axis():
    assert "lighting" in describe_prompt("lighting") and "floor" in describe_prompt("surroundings_floor")
    assert "photo style" in describe_prompt("camera") and "venue" in describe_prompt("venue")
    assert describe_prompt("my_axis")      # unknown axes work too


def test_clean_description():
    assert clean_description('  "Warm yellow ceiling lights."\nmore') == "Warm yellow ceiling lights"
    assert clean_description("Answer: grey carpet") == "grey carpet"
    assert len(clean_description(" ".join(["w"] * 40)).split()) == 20
    assert clean_description("체육관") == "" and clean_description("") == ""


def test_no_reference_images_changes_nothing(tmp_path):
    s = _settings(tmp_path)
    opened = []
    assert describe_references(s, _opener(FakeDescriber(), opened), _Progress(),
                               tmp_path / CACHE_NAME) == (s, [])
    assert opened == []


def test_references_are_optional_per_axis_and_add_candidates(tmp_path):
    light = _photo(tmp_path / "refs" / "light.jpg")
    cam1, cam2 = _photo(tmp_path / "refs" / "c1.jpg"), _photo(tmp_path / "refs" / "c2.png", value=20)
    s = _settings(tmp_path, reference_images={"lighting": [light], "camera": [cam1, cam2]})
    cache = tmp_path / CACHE_NAME
    d = FakeDescriber(["warm stage spotlights", "grainy phone photo", "sharp DSLR photo"])
    opened, p = [], _Progress()
    s2, used = describe_references(s, _opener(d, opened), p, cache)
    assert opened == [1] and p.stages == [("references", 3)] and p.advanced == 3
    assert [(u["axis"], u["file"]) for u in used] == [("lighting", "light.jpg"), ("camera", "c1.jpg"),
                                                      ("camera", "c2.png")]
    # empty axis (lighting): captions only; filled axis (camera): values + captions; no photos (venue): unchanged
    assert s2.scene_axes == {"venue": ["gym"], "lighting": ["warm stage spotlights"],
                             "camera": ["blurry photo", "grainy phone photo", "sharp DSLR photo"]}
    assert s.scene_axes["lighting"] == []                  # the original is unchanged
    assert pending_references(s, cache) == []
    # second run: uses saved captions without opening the VLM
    opened2 = []
    s3, used2 = describe_references(s, _opener(FakeDescriber([]), opened2), _Progress(), cache)
    assert opened2 == [] and s3 == s2 and used2 == used
    # the same photo on another axis is described again for that axis
    s4 = _settings(tmp_path, reference_images={"venue": [light]})
    assert [(a, p.name) for a, p in pending_references(s4, cache)] == [("venue", "light.jpg")]


def test_unchecked_axis_references_are_ignored(tmp_path):
    photo = _photo(tmp_path / "r.jpg")
    s = _settings(tmp_path, reference_images={"camera": [photo]}, disabled_scene_axes=["camera"])
    e = effective_settings(s)
    assert e.reference_images == {} and "camera" not in e.scene_axes
    opened = []
    assert describe_references(e, _opener(FakeDescriber(), opened), _Progress(), tmp_path / CACHE_NAME)[1] == []
    assert opened == []


def test_per_axis_limit(tmp_path):
    photos = [_photo(tmp_path / f"p{i}.jpg", value=i) for i in range(MAX_PER_AXIS + 3)]
    s = _settings(tmp_path, reference_images={"venue": photos})
    assert len(pending_references(s, tmp_path / CACHE_NAME)) == MAX_PER_AXIS


def test_errors_for_missing_file_or_no_usable_description(tmp_path):
    s = _settings(tmp_path, reference_images={"venue": [tmp_path / "gone.jpg"]})
    with pytest.raises(ReferencePhotoError, match="찾을 수 없습니다"):
        describe_references(s, _opener(FakeDescriber(), []), _Progress(), tmp_path / CACHE_NAME)
    s = _settings(tmp_path, reference_images={"venue": [_photo(tmp_path / "a.jpg")]})
    with pytest.raises(ReferencePhotoError, match="장소|venue"):
        describe_references(s, _opener(FakeDescriber(["체육관"]), []), _Progress(), tmp_path / CACHE_NAME)


def test_settings_round_trip_validation_and_diff(tmp_path):
    photo = _photo(tmp_path / "r.jpg")
    base = dict(images_dir=tmp_path, labels_dir=tmp_path, output_dir=tmp_path / "out")
    s = BackgroundSwapSettings(scene_axes={"venue": ["gym"], "lighting": []},
                               reference_images={"lighting": [photo]}, **base)
    assert settings_from_dict(json.loads(json.dumps(settings_to_dict(s)))) == s
    assert validate_settings(s) == []                      # an empty axis is valid when it has reference photos
    no_ref = BackgroundSwapSettings(scene_axes={"venue": ["gym"], "lighting": []}, **base)
    assert any("참고 사진" in e for e in validate_settings(no_ref))
    gone = BackgroundSwapSettings(scene_axes={"venue": ["gym"]},
                                  reference_images={"venue": [tmp_path / "x.jpg"]}, **base)
    assert any("x.jpg" in e for e in validate_settings(gone))
    assert generation_diff(s, no_ref) == ["reference_images"]
