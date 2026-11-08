from background_swap.stages.prompts import (
    LLM_SYSTEM_PROMPT,
    PromptRequest,
    TemplatePromptGenerator,
    build_llm_messages,
    derive_seed,
    finalize_prompt,
    looks_like_keyword_list,
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
    assert "Scene elements: gym" in msgs[1]["content"]
    assert "Keep elements: mat" in msgs[1]["content"]
    assert "Forbidden words: robot" in msgs[1]["content"]
    assert msgs[0]["content"] == LLM_SYSTEM_PROMPT


def test_llm_messages_none_placeholders():
    msgs = build_llm_messages(PromptRequest({"venue": "gym"}, [], 0), [])
    assert "Keep elements: none" in msgs[1]["content"]
    assert "Forbidden words: none" in msgs[1]["content"]


AXES = ["venue", "lighting", "surroundings_floor"]


def test_derive_seed_pinned_value():
    assert derive_seed(0, "a", 1, 1) == 1427502002


def test_llm_messages_do_not_expose_axis_keys():
    scene = {"venue": "school gymnasium", "lighting": "warm LED lighting",
             "surroundings_floor": "grey carpet", "style": "clean DSLR photo"}
    user = build_llm_messages(PromptRequest(scene, ["mat"], 0), ["robot"])[1]["content"]
    for key in scene:
        assert f"{key}:" not in user and key not in user.replace(scene["lighting"], "")
    assert "_" not in user.replace("\n", " ")
    assert ("Scene elements: school gymnasium; warm LED lighting; grey carpet; "
            "clean DSLR photo") in user


def test_llm_messages_skip_empty_scene_values():
    user = build_llm_messages(PromptRequest({"a": "gym", "b": " ", "c": ""}, [], 0), [])[1]["content"]
    assert "Scene elements: gym\n" in user


def test_system_prompt_demands_natural_sentence_with_example():
    low = LLM_SYSTEM_PROMPT.lower()
    assert "single natural english sentence" in low and "example" in low
    assert "no labels" in low and "underscores" in low and "colons" in low


def test_looks_like_keyword_list_positives():
    assert looks_like_keyword_list("Surroundings_floor: grey carpet", AXES)
    assert looks_like_keyword_list("venue, lighting", AXES)
    assert looks_like_keyword_list("Surroundings floor grey carpet", AXES)
    assert looks_like_keyword_list("gym_hall_clean_photo", [])
    assert looks_like_keyword_list("gym: grey carpet", [])


def test_looks_like_keyword_list_negative_for_natural_sentence():
    text = "A spacious school gymnasium with grey carpet and sponsor banners on the walls."
    assert not looks_like_keyword_list(text, AXES)
    assert not looks_like_keyword_list(text, [""])
    # 한 단어 축 이름이 자연스러운 문장 속에 쓰인 것만으로는 키워드 나열이 아니다.
    assert not looks_like_keyword_list(
        "A school gymnasium venue lit by warm LED lighting, with grey carpet.", AXES)


def test_finalize_prompt_passes_good_sentence_through():
    req = PromptRequest({"venue": "gym"}, ["mat"], 0)
    good = "A spacious school gymnasium with grey carpet and sponsor banners on the walls."
    assert finalize_prompt(good, req, ["robot"], AXES) == good


def test_finalize_prompt_falls_back_on_keyword_output():
    req = PromptRequest({"venue": "sports hall", "lighting": "warm LED"}, ["fence"], 0)
    out = finalize_prompt("Surroundings_floor_grey_carpet_clean_DSLR_photo", req, ["robot"], AXES)
    assert out == "sports hall, warm LED, surrounding fence, photorealistic"


def test_finalize_prompt_falls_back_on_empty_output():
    req = PromptRequest({"venue": "sports hall"}, [], 0)
    assert finalize_prompt("", req, ["robot"], AXES) == "sports hall, photorealistic"
    assert finalize_prompt('"robot"', req, ["robot"], AXES) == "sports hall, photorealistic"


def test_finalize_prompt_fallback_is_sanitized():
    req = PromptRequest({"venue": "robot arena"}, [], 0)
    assert finalize_prompt("venue: x", req, ["robot"], AXES) == "arena, photorealistic"


def test_sanitize_removes_longest_class_name_first():
    out = sanitize_prompt("a ball boy near the ball", ["ball", "ball boy"])
    assert out == "a near the"


def test_sanitize_multi_word_class_across_whitespace():
    out = sanitize_prompt("a gym with a ball  \t boy here", ["ball boy"])
    assert out == "a gym with a here"


def test_sanitize_keeps_mismatched_quotes():
    assert sanitize_prompt('"hello world', []) == '"hello world'
    assert sanitize_prompt("hello world'", []) == "hello world'"
    assert sanitize_prompt("“hello” and “world”", []) == "hello” and “world"


def test_sanitize_strips_nested_matching_quotes():
    assert sanitize_prompt("\"'hello world'\"", []) == "hello world"
    assert sanitize_prompt("‘hello world’", []) == "hello world"


def test_sanitize_strips_trailing_comma_after_truncation():
    text = " ".join(["w"] * 59) + ", tail words"
    out = sanitize_prompt(text, [])
    assert len(out.split()) == 60 and not out.endswith(",")


def test_negative_prompt_skips_empty_names():
    assert negative_prompt(["robot", "", "  "]) == "robot, blurry, distorted, text, watermark, extra objects"


def test_template_prompt_skips_empty_values():
    req = PromptRequest({"a": "", "venue": "gym", "b": "  "}, [], 0)
    assert TemplatePromptGenerator().generate_batch([req]) == ["gym, photorealistic"]
    assert TemplatePromptGenerator().generate_batch([PromptRequest({"a": ""}, [], 0)]) == ["photorealistic"]


def test_sanitize_removal_repeats_until_stable():
    out = sanitize_prompt("a y x z b", ["x", "y z"])
    assert "y z" not in out and "x" not in out.split()
    assert out == "a b"


def test_system_prompt_treats_camera_style_as_style_phrase():
    low = LLM_SYSTEM_PROMPT.lower()
    assert "camera or photo style" in low and "dslr" in low and "end of the sentence" in low
