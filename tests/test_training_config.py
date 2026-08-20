"""P4 contract tests: the pinned hyperparameters, the anti-drift guard that
configs/train.json and configs/models.json name the SAME base revision, and the
BEHAVIOR of training/modal_app.py's pure functions.

training/modal_app.py IS imported here, by path. The earlier claim that importing
it "constructs Modal objects" is not a reason to skip it: modal.App /
Image.from_registry / Volume.from_name are lazy handles that touch no network at
import time, which is why the import below succeeds under plain pytest. Scanning
the file with ast.parse alone is not a test of it -- a do-nothing stub with the
same shape passes every string check -- so the label masking, the smoke-override
merge and the collator are exercised for real, against a deterministic fake
tokenizer.

Network-free: pure JSON, ast.parse, and in-process calls with no model, no GPU
and no Hub access.
"""
import ast
import copy
import importlib.util
import json

import pytest

import eval as E
from conftest import REPO_ROOT

TRAIN_JSON = REPO_ROOT / "configs" / "train.json"
MODELS_JSON = REPO_ROOT / "configs" / "models.json"
MODAL_APP = REPO_ROOT / "training" / "modal_app.py"

EXPECTED_TARGET_MODULES = {
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
}


@pytest.fixture(scope="module")
def train_cfg():
    assert TRAIN_JSON.exists(), f"{TRAIN_JSON} missing -- P4 contract file"
    with open(TRAIN_JSON) as f:
        return json.load(f)


def test_train_json_parses_and_has_every_section(train_cfg):
    for section in ("base_model", "lora", "training", "smoke_overrides",
                    "data", "output"):
        assert section in train_cfg, f"configs/train.json is missing {section!r}"


def test_base_model_is_the_instruct_tuned_qwen3_repo_pinned_by_sha(train_cfg):
    base = train_cfg["base_model"]
    assert base["repo"] == "Qwen/Qwen3-1.7B"
    # Qwen3 dropped the -Instruct suffix; the unsuffixed repo is the post-trained
    # chat variant. The -Base repo would silently "work" and train the wrong model.
    assert not base["repo"].endswith("-Base")
    rev = base["revision"]
    assert isinstance(rev, str) and len(rev) == 40 and all(
        c in "0123456789abcdef" for c in rev), f"revision must be a full SHA: {rev!r}"
    assert base["spec"] == f"hf:{base['repo']}@{rev}"


def test_lora_hyperparameters_are_the_pinned_ones(train_cfg):
    lora = train_cfg["lora"]
    assert lora["r"] == 16
    assert lora["lora_alpha"] == 32
    assert lora["lora_dropout"] == 0.05
    assert set(lora["target_modules"]) == EXPECTED_TARGET_MODULES
    assert len(lora["target_modules"]) == len(EXPECTED_TARGET_MODULES)


def test_training_hyperparameters_are_the_pinned_ones(train_cfg):
    t = train_cfg["training"]
    assert t["learning_rate"] == pytest.approx(2e-4, rel=0, abs=1e-12)
    assert t["num_train_epochs"] == 3
    assert t["per_device_train_batch_size"] == 8
    assert t["gradient_accumulation_steps"] == 2
    assert t["max_seq_length"] == 2048
    assert t["seed"] == 42
    assert t["warmup_ratio"] == pytest.approx(0.03, rel=0, abs=1e-12)


def test_effective_batch_size_is_sixteen(train_cfg):
    t = train_cfg["training"]
    assert t["per_device_train_batch_size"] * t["gradient_accumulation_steps"] == 16


def test_smoke_overrides_cap_the_run_at_twenty_steps(train_cfg):
    assert train_cfg["smoke_overrides"] == {"max_steps": 20}
    # max_steps must NOT leak into the pinned training block: an unqualified run
    # has to be the full 3 epochs.
    assert "max_steps" not in train_cfg["training"]


def test_training_condition_equals_the_zero_shot_eval_condition(train_cfg):
    """eval.py's HFGenerator renders prompts with enable_thinking=False; training
    must render them identically or we are measuring prompt transfer, not the
    behavior."""
    assert train_cfg["data"]["enable_thinking"] is False


def test_output_contract_names_adapter_merged_and_train_log(train_cfg):
    out = train_cfg["output"]
    assert out["adapter_dirname"]
    assert out["merged_dirname"]
    assert out["train_log_filename"] == "train_log.json"
    assert "16bit" in out["merged_save_method"], "merged model must be fp16"


def test_base_revision_matches_configs_models_json(train_cfg):
    """THE anti-drift guard: the model the ablation/eval pins and the model we
    fine-tune must be the same weights."""
    if not MODELS_JSON.exists():
        pytest.skip("configs/models.json not written yet (P1)")
    with open(MODELS_JSON) as f:
        models = json.load(f)
    base_spec = models.get("base")
    assert base_spec, "configs/models.json has no 'base' entry"
    repo, sep, revision = base_spec.partition("@")
    assert sep, f"configs/models.json:base is not pinned by revision: {base_spec!r}"
    assert repo == f"hf:{train_cfg['base_model']['repo']}", (
        f"base repo drift: models.json={repo!r} "
        f"train.json=hf:{train_cfg['base_model']['repo']!r}")
    assert revision == train_cfg["base_model"]["revision"], (
        f"base revision drift: models.json={revision!r} "
        f"train.json={train_cfg['base_model']['revision']!r}")


def test_modal_app_parses_as_python():
    assert MODAL_APP.exists(), f"{MODAL_APP} missing -- P4 deliverable"
    tree = ast.parse(MODAL_APP.read_text(), filename=str(MODAL_APP))
    names = {n.name for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "train" in names, "modal_app.py defines no train function"
    assert "main" in names, "modal_app.py defines no local entrypoint 'main'"


def test_modal_app_uses_the_current_modal_api_not_modal_stub():
    """modal.Stub was removed in modal 1.x; code written against it raises
    AttributeError. Checked over the AST so prose in comments doesn't count."""
    tree = ast.parse(MODAL_APP.read_text(), filename=str(MODAL_APP))
    attrs = {f"{n.value.id}.{n.attr}" for n in ast.walk(tree)
             if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    assert "modal.Stub" not in attrs, "modal.Stub does not exist in modal 1.x"
    assert "modal.App" in attrs
    assert "local_entrypoint" in MODAL_APP.read_text()


def test_modal_app_sets_an_explicit_timeout():
    """Modal's default @app.function timeout is 300s and would kill a real run."""
    src = MODAL_APP.read_text()
    assert "timeout=" in src


def test_modal_app_does_not_hardcode_pinned_hyperparameters(train_cfg):
    """configs/train.json is the single source of truth: the app must read the
    numbers, never duplicate them."""
    src = MODAL_APP.read_text()
    for literal in ("lora_alpha=32", "r=16,", "learning_rate=2e-4",
                    "learning_rate=0.0002", "num_train_epochs=3",
                    "max_seq_length=2048", "warmup_ratio=0.03"):
        assert literal not in src, (
            f"training/modal_app.py hardcodes {literal!r}; it must come from "
            f"configs/train.json")
    assert train_cfg["base_model"]["revision"] not in src, (
        "training/modal_app.py hardcodes the base revision; read it from "
        "configs/train.json")


# =====================================================================
# The module itself, imported and executed.
# =====================================================================
@pytest.fixture(scope="module")
def app_mod():
    """training/modal_app.py, imported by path.

    No monkeypatching of modal is needed: the module-level modal objects are
    lazy handles. If this import ever starts touching the network, that is a
    defect in modal_app.py, not a reason to go back to string-scanning it.
    """
    spec = importlib.util.spec_from_file_location("slmforge_modal_app", MODAL_APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------- fake tokenizer
class FakeTokenizer:
    """Deterministic character-level stand-in for a HF tokenizer.

    It implements exactly the three surfaces build_training_features touches --
    ``apply_chat_template``, ``__call__(...)['input_ids']`` and ``eos_token`` --
    and it records the kwargs of every apply_chat_template call so the test can
    assert the prompt is rendered with the SAME call eval.py's HFGenerator makes.

    Character-level means the tokenization of ``prompt + target`` always has the
    tokenization of ``prompt`` as an exact prefix, which is the normal case.
    MERGES simulates a BPE pair that spans the join, which is the pathological
    case build_training_features has a fallback for.
    """
    eos_token = "<|im_end|>"
    MERGES = ()

    def __init__(self):
        self.vocab = {}
        self.calls = []

    def _id(self, tok):
        return self.vocab.setdefault(tok, 100 + len(self.vocab))

    def decode(self, ids):
        inverse = {v: k for k, v in self.vocab.items()}
        return "".join(inverse[i] for i in ids)

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, enable_thinking=None):
        self.calls.append({"messages": messages, "tokenize": tokenize,
                           "add_generation_prompt": add_generation_prompt,
                           "enable_thinking": enable_thinking})
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                       for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def __call__(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            if text[i:i + 2] in self.MERGES:
                ids.append(self._id(text[i:i + 2]))
                i += 2
            else:
                ids.append(self._id(text[i]))
                i += 1
        return {"input_ids": ids}


class JoinMergingTokenizer(FakeTokenizer):
    """A tokenizer whose BPE merges "\\nA" -- the last prompt character with the
    first target character -- so retokenizing prompt+target changes the prompt's
    own tokens. build_training_features must notice and fall back."""
    MERGES = ("\nA",)


class NoThinkingKwargTokenizer(FakeTokenizer):
    """An older tokenizer whose chat template does not accept enable_thinking."""

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False):
        return super().apply_chat_template(
            messages, tokenize=tokenize,
            add_generation_prompt=add_generation_prompt)


PASSAGE = ("Marie Curie was awarded the Nobel Prize in Physics in 1903 and the "
           "Nobel Prize in Chemistry in 1911.")
QUESTION = "In what year did Marie Curie win the Nobel Prize in Chemistry?"
ANSWER = ('Answer: 1911\nCitation: "the Nobel Prize in Chemistry in 1911"')


def training_row(assistant=ANSWER, passage=PASSAGE, question=QUESTION):
    """One row in the P3 training schema -- system is eval.SYSTEM_CORE verbatim
    and user is built by eval.user_msg, exactly as datagen emits it."""
    return {"messages": [
        {"role": "system", "content": E.SYSTEM_CORE},
        {"role": "user", "content": E.user_msg(passage, question)},
        {"role": "assistant", "content": assistant},
    ], "meta": {"gold_label": "answerable"}}


def refusal_row():
    return training_row(assistant=E.REFUSAL_STRING,
                        passage="The building opened in 1974.",
                        question="Who designed the building?")


def prompt_boundary(tok, row, enable_thinking=False):
    """The token index where the completion starts, computed independently of
    build_training_features so the test cannot inherit its off-by-one."""
    text = tok.apply_chat_template(row["messages"][:-1], tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=enable_thinking)
    return len(tok(text, add_special_tokens=False)["input_ids"]), text


# ------------------------------------------------------- build_training_features
def test_build_training_features_masks_the_prompt_and_trains_on_the_completion(
        app_mod):
    """THE label-masking contract. A silent bug here trains the model on its own
    prompt (or on nothing) and every MVP number becomes meaningless."""
    tok = FakeTokenizer()
    rows = [training_row(), refusal_row()]
    feats, stats = app_mod.build_training_features(tok, rows, 4096, False)

    assert len(feats) == 2 and stats["kept"] == 2 and stats["skipped"] == 0
    for row, f in zip(rows, feats):
        n, _ = prompt_boundary(tok, row)
        ids, labels, mask = f["input_ids"], f["labels"], f["attention_mask"]
        assert len(ids) == len(labels) == len(mask)
        assert mask == [1] * len(ids)
        assert labels[:n] == [-100] * n, "prompt tokens must not contribute loss"
        assert labels[n:] == ids[n:], "completion tokens must be trained on verbatim"
        assert all(t != -100 for t in labels[n:]), "completion must not be masked"
        # boundary, both sides -- catches an off-by-one in either direction
        assert labels[n - 1] == -100, "last prompt token leaked into the loss"
        assert labels[n] != -100, "first completion token was masked away"
        assert len(labels) > n, "there is nothing to learn from"


def test_the_unmasked_region_is_exactly_the_assistant_turn_plus_eos(app_mod):
    """Decoding the supervised tokens must give back the target text and nothing
    else -- no leading template fragment, no missing first character."""
    tok = FakeTokenizer()
    row = training_row()
    feats, _ = app_mod.build_training_features(tok, [row], 4096, False)
    n, _ = prompt_boundary(tok, row)
    supervised = [t for t in feats[0]["labels"] if t != -100]
    assert tok.decode(supervised) == ANSWER + tok.eos_token
    assert tok.decode(feats[0]["input_ids"][:n]).endswith("<|im_start|>assistant\n")


def test_the_prompt_half_is_rendered_with_the_call_eval_py_makes(app_mod):
    """add_generation_prompt=True and enable_thinking passed through: this is
    what makes the trained prefix byte-identical to the zero_shot eval prefix."""
    tok = FakeTokenizer()
    row = training_row()
    app_mod.build_training_features(tok, [row], 4096, False)
    render = tok.calls[0]
    assert render["add_generation_prompt"] is True
    assert render["enable_thinking"] is False
    assert render["tokenize"] is False
    assert [m["role"] for m in render["messages"]] == ["system", "user"], \
        "the assistant turn must NOT be inside the rendered prompt"
    assert render["messages"][0]["content"] == E.SYSTEM_CORE

    tok2 = FakeTokenizer()
    app_mod.build_training_features(tok2, [row], 4096, True)
    assert tok2.calls[0]["enable_thinking"] is True


def test_a_tokenizer_without_enable_thinking_still_builds_features(app_mod):
    tok = NoThinkingKwargTokenizer()
    feats, stats = app_mod.build_training_features(tok, [training_row()], 4096,
                                                   False)
    assert stats["kept"] == 1 and len(feats) == 1
    assert any(t != -100 for t in feats[0]["labels"])


def test_retokenization_across_the_join_falls_back_to_exact_concatenation(app_mod):
    """When the joint tokenization changes the prompt's own tokens, the mask
    would silently point at the wrong boundary. The fallback must rebuild the
    sequence from the two token lists so the boundary stays exact."""
    tok = JoinMergingTokenizer()
    row = training_row()
    feats, stats = app_mod.build_training_features(tok, [row], 4096, False)

    assert stats["prefix_mismatch"] == 1, "the mismatch was not even detected"
    n, prompt_text = prompt_boundary(tok, row)
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tok(ANSWER + tok.eos_token, add_special_tokens=False)["input_ids"]
    assert feats[0]["input_ids"] == prompt_ids + target_ids
    assert feats[0]["labels"] == [-100] * n + target_ids
    assert tok.decode([t for t in feats[0]["labels"] if t != -100]) == \
        ANSWER + tok.eos_token


def test_truncation_keeps_input_ids_and_labels_in_lockstep(app_mod):
    tok = FakeTokenizer()
    row = training_row()
    n, _ = prompt_boundary(tok, row)
    limit = n + 5
    feats, stats = app_mod.build_training_features(tok, [row], limit, False)
    assert stats["truncated"] == 1
    f = feats[0]
    assert len(f["input_ids"]) == len(f["labels"]) == len(f["attention_mask"]) == limit
    assert f["labels"][:n] == [-100] * n
    assert f["labels"][n:] == f["input_ids"][n:]


def test_a_row_truncated_down_to_prompt_only_is_dropped_not_trained_on(app_mod):
    """An all -100 row contributes no gradient and (with some trainers) produces
    NaN loss. It must be counted as skipped, not silently kept."""
    tok = FakeTokenizer()
    row = training_row()
    n, _ = prompt_boundary(tok, row)
    feats, stats = app_mod.build_training_features(tok, [row], n - 10, False)
    assert feats == []
    assert stats["skipped"] == 1 and stats["kept"] == 0


def test_rows_whose_last_turn_is_not_the_assistant_are_skipped(app_mod):
    tok = FakeTokenizer()
    bad = {"messages": [{"role": "system", "content": E.SYSTEM_CORE},
                        {"role": "user", "content": E.user_msg(PASSAGE, QUESTION)}]}
    feats, stats = app_mod.build_training_features(tok, [bad, training_row()],
                                                   4096, False)
    assert stats["skipped"] == 1
    assert stats["kept"] == 1 and len(feats) == 1


def test_completion_only_loss_false_supervises_every_token(app_mod):
    """The flag has to actually do something -- otherwise the completion-only
    masking above could be an accident of the default path."""
    tok = FakeTokenizer()
    feats, _ = app_mod.build_training_features(tok, [training_row()], 4096, False,
                                               completion_only_loss=False)
    f = feats[0]
    assert -100 not in f["labels"]
    assert f["labels"] == f["input_ids"]


def test_feature_stats_account_for_every_row(app_mod):
    tok = FakeTokenizer()
    rows = [training_row(), refusal_row(),
            {"messages": [{"role": "user", "content": "x"}]}]
    feats, stats = app_mod.build_training_features(tok, rows, 4096, False)
    assert stats["rows"] == 3
    assert stats["kept"] == len(feats)
    assert stats["kept"] + stats["skipped"] == stats["rows"]


# ------------------------------------------------------------------ PadCollator
def _batch(app_mod, lengths, pad_id=7):
    """Feature dicts of the given token lengths, first half of each masked."""
    batch = []
    for n in lengths:
        ids = list(range(1000, 1000 + n))
        labels = [-100] * (n // 2) + ids[n // 2:]
        batch.append({"input_ids": ids, "attention_mask": [1] * n,
                      "labels": labels})
    return batch, app_mod.PadCollator(pad_id)


def test_pad_collator_right_pads_to_the_longest_sequence(app_mod):
    import torch
    lengths = [3, 7, 5]
    batch, collate = _batch(app_mod, lengths)
    out = collate([dict(b) for b in batch])
    assert set(out) == {"input_ids", "attention_mask", "labels"}
    for key in out:
        assert out[key].shape == (3, 7), f"{key} has shape {tuple(out[key].shape)}"
        assert out[key].dtype == torch.long
    for i, n in enumerate(lengths):
        assert out["input_ids"][i][:n].tolist() == batch[i]["input_ids"]
        assert out["input_ids"][i][n:].tolist() == [7] * (7 - n)


def test_pad_collator_pads_labels_with_minus_100_not_the_pad_id(app_mod):
    """Padding labels with the pad token id would train the model to emit padding
    -- the classic silent seq2seq bug."""
    lengths = [3, 7]
    batch, collate = _batch(app_mod, lengths, pad_id=7)
    out = collate([dict(b) for b in batch])
    padded = out["labels"][0].tolist()
    assert padded[3:] == [-100] * 4
    assert 7 not in padded[3:], "labels were padded with the pad token id"
    assert padded[:3] == batch[0]["labels"]


def test_pad_collator_zeroes_the_attention_mask_over_padding(app_mod):
    lengths = [2, 6]
    batch, collate = _batch(app_mod, lengths)
    out = collate([dict(b) for b in batch])
    for i, n in enumerate(lengths):
        mask = out["attention_mask"][i].tolist()
        assert mask == [1] * n + [0] * (6 - n)
        assert sum(mask) == n


def test_pad_collator_is_a_no_op_when_lengths_already_match(app_mod):
    batch, collate = _batch(app_mod, [4, 4])
    out = collate([dict(b) for b in batch])
    assert out["input_ids"].shape == (2, 4)
    for i in range(2):
        assert out["attention_mask"][i].tolist() == [1] * 4
        assert out["input_ids"][i].tolist() == batch[i]["input_ids"]
        assert out["labels"][i].tolist() == batch[i]["labels"]


def test_pad_collator_does_not_mutate_the_features_it_is_given(app_mod):
    batch, collate = _batch(app_mod, [3, 6])
    before = copy.deepcopy(batch)
    collate(batch)
    assert batch == before, "the collator mutated the cached dataset rows"


# --------------------------------------------------------------- resolve_config
def test_resolve_config_applies_the_smoke_overrides_for_real(app_mod, train_cfg):
    resolved = app_mod.resolve_config(app_mod.load_config(), True)
    assert resolved["training"]["max_steps"] == train_cfg["smoke_overrides"]["max_steps"] == 20
    assert resolved["_smoke"] is True
    # the pinned numbers must survive the merge untouched
    assert resolved["training"]["learning_rate"] == train_cfg["training"]["learning_rate"]
    assert resolved["lora"]["r"] == train_cfg["lora"]["r"]


def test_resolve_config_without_smoke_leaves_the_full_run_uncapped(app_mod,
                                                                  train_cfg):
    resolved = app_mod.resolve_config(app_mod.load_config(), False)
    assert "max_steps" not in resolved["training"], \
        "a non-smoke run must be the full 3 epochs, not 20 steps"
    assert resolved["_smoke"] is False
    assert resolved["training"]["num_train_epochs"] == 3


def test_resolve_config_does_not_mutate_the_caller_config(app_mod):
    cfg = app_mod.load_config()
    before = copy.deepcopy(cfg)
    app_mod.resolve_config(cfg, True)
    assert cfg == before, "resolve_config mutated the config it was handed"


def test_resolve_config_deep_merges_a_section_valued_override(app_mod):
    """A key naming a section merges INTO that section; anything else lands in
    'training' where TrainingArguments expects it."""
    cfg = {"training": {"learning_rate": 2e-4, "seed": 42},
           "lora": {"r": 16, "lora_alpha": 32},
           "smoke_overrides": {"lora": {"r": 4}, "max_steps": 20}}
    resolved = app_mod.resolve_config(cfg, True)
    assert resolved["lora"] == {"r": 4, "lora_alpha": 32}
    assert resolved["training"]["max_steps"] == 20
    assert resolved["training"]["learning_rate"] == 2e-4
    assert resolved["training"]["seed"] == 42
    assert "lora" not in resolved["training"], \
        "a section override leaked into the training block"


def test_load_config_reads_the_pinned_train_json(app_mod, train_cfg):
    assert app_mod.load_config() == train_cfg
