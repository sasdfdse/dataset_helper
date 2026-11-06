from background_swap.prompts import (
    LLM_SYSTEM_PROMPT,
    PromptRequest,
    TemplatePromptGenerator,
    build_llm_messages,
    derive_seed,
    negative_prompt,
    sample_scene,
    sanitize_prompt,
)


def test_derive_seed_stable_and_distinct():
    assert derive_seed(0, "a", 1, 1) == derive_seed(0, "a", 1, 1)
    assert len({derive_seed(0, "a", v, k) for v in (1, 2) for k in (1, 2)}) == 4
    assert 0 <= derive_seed(0, "a", 1, 1) < 2**31


def test_sample_scene_deterministic_one_value_per_axis():
    axes = {"venue": ["a", "b", "c"], "lighting": ["x", "y"]}
    s = sample_scene(axes, 42)
    assert s == sample_scene(axes, 42) and list(s) == ["venue", "lighting"]
    assert s["venue"] in axes["venue"]


def test_template_prompt():
    req = PromptRequest({"venue": "sports hall", "lighting": "warm LED"}, ["green field mat", "fence"], 0)
    assert TemplatePromptGenerator().generate_batch([req]) == [
        "sports hall, warm LED, surrounding green field mat and fence, photorealistic"]


def test_template_prompt_without_keep_names():
    req = PromptRequest({"venue": "gym"}, [], 0)
    assert TemplatePromptGenerator().generate_batch([req]) == ["gym, photorealistic"]


def test_sanitize_removes_class_names_whole_word():
    out = sanitize_prompt('"A hall with a Robot and robotics banners, ball pit"', ["robot", "ball"])
    assert "Robot" not in out and "robotics" in out and " ball " not in f" {out} "
    assert not out.startswith('"')


def test_sanitize_cleans_punctuation_and_first_line():
    out = sanitize_prompt("“A hall , robot , with banners .”\nsecond line", ["robot"])
    assert out == "A hall, with banners."


def test_sanitize_multi_word_class_and_special_chars():
    out = sanitize_prompt("a gym with a ball boy and c++ gear", ["ball boy", "c++"])
    assert "ball boy" not in out and "c++" not in out


def test_sanitize_skips_leading_blank_lines():
    assert sanitize_prompt("\n\n  hello world  \nignored", []) == "hello world"


def test_sanitize_truncates_to_60_words():
    assert len(sanitize_prompt(" ".join(["w"] * 80), []).split()) == 60


def test_negative_prompt():
    assert negative_prompt(["robot"]) == "robot, blurry, distorted, text, watermark, extra objects"


def test_llm_messages_include_forbidden_words():
    msgs = build_llm_messages(PromptRequest({"venue": "gym"}, ["mat"], 0), ["robot"])
    assert msgs[0]["role"] == "system" and "robot" in msgs[1]["content"] and "gym" in msgs[1]["content"]
    assert msgs[1]["role"] == "user"
    assert "venue: gym" in msgs[1]["content"]
    assert "Keep elements: mat" in msgs[1]["content"]
    assert "Forbidden words: robot" in msgs[1]["content"]
    assert msgs[0]["content"] == LLM_SYSTEM_PROMPT


def test_llm_messages_none_placeholders():
    msgs = build_llm_messages(PromptRequest({"venue": "gym"}, [], 0), [])
    assert "Keep elements: none" in msgs[1]["content"]
    assert "Forbidden words: none" in msgs[1]["content"]
