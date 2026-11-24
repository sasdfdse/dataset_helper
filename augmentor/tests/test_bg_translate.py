import json
from contextlib import contextmanager

import pytest

from background_swap.common.settings import BackgroundSwapSettings, KeepItem
from background_swap.stages.translate import (CACHE_NAME, TranslationError, apply_translations,
                                              build_translation_messages, clean_translation,
                                              collect_sources, has_hangul, pending_sources,
                                              translate_inputs)
from tests.bg_fakes import FakeTranslator


class _Progress:
    def __init__(self):
        self.stages, self.advanced = [], 0

    def stage(self, name, total):
        self.stages.append((name, total))

    def advance(self, n=1):
        self.advanced += n

    def event(self, kind, **fields):
        pass


def _settings(**kw):
    base = dict(
        keep_items=[KeepItem(name="골대", method="text", prompt="하얀 골대"),
                    KeepItem(name="field", method="color", hsv_lower=[0, 0, 0], hsv_upper=[1, 1, 1])],
        scene_axes={"venue": ["체육관", "convention hall"], "lighting": ["체육관", "warm light"]},
    )
    base.update(kw)
    return BackgroundSwapSettings(**base)


def _opener(translator, opened):
    @contextmanager
    def open_translator():
        opened.append(1)
        yield translator
    return open_translator


def test_has_hangul():
    assert has_hangul("골대") and has_hangul("white 골대")
    assert not has_hangul("goal post") and not has_hangul("") and not has_hangul("café")


def test_collect_sources_unique_in_order_with_kinds():
    s = _settings()
    assert collect_sources(s, ["로봇", "ball"]) == [
        ("keep", "골대"), ("keep", "하얀 골대"), ("scene", "체육관"), ("class", "로봇")]


def test_clean_translation():
    assert clean_translation('  "White goal post."  \nextra line') == "White goal post"
    assert clean_translation("Translation: gym") == "gym"
    assert clean_translation("school   gymnasium") == "school gymnasium"
    assert clean_translation("") == ""


def test_build_translation_messages_contains_text_and_english_instruction():
    msgs = build_translation_messages("골대")
    assert msgs[0]["role"] == "system" and "English" in msgs[0]["content"]
    assert msgs[-1] == {"role": "user", "content": "골대"}
    roles = [m["role"] for m in msgs[1:-1]]
    assert roles and roles == ["user", "assistant"] * (len(roles) // 2)   # example translation pairs


def test_apply_translations_replaces_only_korean_and_keeps_other_fields():
    s = _settings()
    mapping = {"골대": "goal", "하얀 골대": "white goal post", "체육관": "gymnasium", "로봇": "robot"}
    s2, names = apply_translations(s, ["로봇", "ball"], mapping)
    assert [(k.name, k.prompt, k.method) for k in s2.keep_items] == [
        ("goal", "white goal post", "text"), ("field", "", "color")]
    assert s2.keep_items[1].hsv_lower == [0, 0, 0]
    assert s2.scene_axes == {"venue": ["gymnasium", "convention hall"],
                             "lighting": ["gymnasium", "warm light"]}
    assert names == ["robot", "ball"]
    assert s.keep_items[0].name == "골대"     # the original is unchanged


def test_translate_inputs_translates_once_and_caches(tmp_path):
    s = _settings()
    cache = tmp_path / CACHE_NAME
    tr = FakeTranslator({"골대": "goal", "하얀 골대": "white goal post", "체육관": "gymnasium",
                         "로봇": "robot"})
    opened, p = [], _Progress()
    s2, names, used = translate_inputs(s, ["로봇"], _opener(tr, opened), p, cache)
    assert opened == [1] and tr.calls == [["골대", "하얀 골대", "체육관", "로봇"]]
    assert p.stages == [("translate", 4)] and p.advanced == 4
    assert s2.keep_items[0].prompt == "white goal post" and names == ["robot"]
    assert [(t.kind, t.source, t.text) for t in used] == [
        ("keep", "골대", "goal"), ("keep", "하얀 골대", "white goal post"),
        ("scene", "체육관", "gymnasium"), ("class", "로봇", "robot")]
    assert json.loads(cache.read_text(encoding="utf-8"))["체육관"] == "gymnasium"
    assert pending_sources(s, ["로봇"], cache) == []

    # second run: uses saved translations without opening the model
    opened2 = []
    s3, _, used2 = translate_inputs(s, ["로봇"], _opener(FakeTranslator({}), opened2), _Progress(), cache)
    assert opened2 == [] and s3 == s2 and used2 == used


def test_translate_inputs_without_korean_opens_nothing(tmp_path):
    s = BackgroundSwapSettings(scene_axes={"venue": ["gym"]})
    opened = []
    s2, names, used = translate_inputs(s, ["ball"], _opener(FakeTranslator({}), opened), _Progress(),
                                       tmp_path / CACHE_NAME)
    assert (s2, names, used, opened) == (s, ["ball"], [], [])
    assert not (tmp_path / CACHE_NAME).exists()


def test_translate_inputs_only_translates_new_texts(tmp_path):
    cache = tmp_path / CACHE_NAME
    cache.write_text(json.dumps({"골대": "goal", "하얀 골대": "white goal post"}), encoding="utf-8")
    tr = FakeTranslator({"체육관": "gymnasium"})
    _, _, used = translate_inputs(_settings(), [], _opener(tr, []), _Progress(), cache)
    assert tr.calls == [["체육관"]]
    assert {t.source: t.text for t in used} == {"골대": "goal", "하얀 골대": "white goal post",
                                                "체육관": "gymnasium"}


@pytest.mark.parametrize("bad", ["", "골대", "  ", "\"\"", "体育馆", "123"])
def test_translate_inputs_rejects_unusable_translation(tmp_path, bad):
    s = BackgroundSwapSettings(scene_axes={"venue": ["체육관", "강당"]})
    cache = tmp_path / CACHE_NAME
    tr = FakeTranslator({"체육관": "gymnasium", "강당": bad})
    with pytest.raises(TranslationError, match="강당"):
        translate_inputs(s, [], _opener(tr, []), _Progress(), cache)
    # successful translations are saved and not redone next run
    assert json.loads(cache.read_text(encoding="utf-8")) == {"체육관": "gymnasium"}


def test_translate_inputs_ignores_broken_cache(tmp_path):
    cache = tmp_path / CACHE_NAME
    cache.write_text("{not json", encoding="utf-8")
    s = BackgroundSwapSettings(scene_axes={"venue": ["체육관"]})
    _, _, used = translate_inputs(s, [], _opener(FakeTranslator({"체육관": "gym"}), []), _Progress(), cache)
    assert used[0].text == "gym"
    cache.write_text(json.dumps({"체육관": "여전히 한국어", "x": 3}), encoding="utf-8")
    assert pending_sources(s, [], cache) == ["체육관"]


def test_translate_inputs_rejects_wrong_result_count(tmp_path):
    class Short:
        def translate_batch(self, texts):
            return []
    s = BackgroundSwapSettings(scene_axes={"venue": ["체육관"]})
    with pytest.raises(ValueError, match="translate"):
        translate_inputs(s, [], _opener(Short(), []), _Progress(), tmp_path / CACHE_NAME)
