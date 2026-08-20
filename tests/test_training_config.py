"""Training contract tests: the pinned hyperparameters, the anti-drift guard
that configs/train.json and configs/models.json name the SAME base revision,
and the BEHAVIOR of training/modal_app.py's pure functions under the pivot's
two-message example schema.

training/modal_app.py IS imported here, by path, and its functions are CALLED.
An earlier version of this file only ast.parse'd and substring-scanned the
module, which meant a ten-line do-nothing stub passed it byte-identically. So:
label masking, the smoke-override merge, the collator, the prompt-contract
verification and the train_log builder are all exercised for real, against
deterministic fakes whose every token index is known by hand.

What is deliberately NOT asserted here: that the shared prompt renderer named by
configs/train.json:data.prompt_renderer imports today. It is owned by another
lane and was not on disk when this was written. Asserting it exists would fail
this suite for another lane's unfinished work; asserting it "exists or is
skipped" would hide the gap. Instead
:func:`test_resolve_render_fn_never_silently_substitutes_a_stub` pins the only
property that matters in BOTH worlds -- the loader returns a real callable or
raises loudly, and never invents a renderer.

Network-free: pure JSON, ast.parse, and in-process calls with no model, no GPU
and no Hub access.
"""
import ast
import copy
import importlib.util
import json
import re

import pytest

from conftest import REPO_ROOT

TRAIN_JSON = REPO_ROOT / "configs" / "train.json"
MODELS_JSON = REPO_ROOT / "configs" / "models.json"
MODAL_APP = REPO_ROOT / "training" / "modal_app.py"

EXPECTED_TARGET_MODULES = {
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
}


@pytest.fixture(scope="module")
def train_cfg():
    assert TRAIN_JSON.exists(), f"{TRAIN_JSON} missing -- training contract file"
    with open(TRAIN_JSON) as f:
        return json.load(f)


# =====================================================================
# configs/train.json -- the pinned numbers
# =====================================================================
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
    """The assignment rule: a disappointing model is a DATA problem. These
    numbers must not move to suit the new target."""
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


def test_output_contract_names_adapter_merged_and_train_log(train_cfg):
    out = train_cfg["output"]
    assert out["adapter_dirname"]
    assert out["merged_dirname"]
    assert out["train_log_filename"] == "train_log.json"
    assert "16bit" in out["merged_save_method"], "merged model must be fp16"


def test_base_revision_matches_configs_models_json(train_cfg):
    """THE anti-drift guard: the model the ablation/eval pins and the model we
    fine-tune must be the same weights. No skip -- if models.json is missing,
    that IS the failure."""
    assert MODELS_JSON.exists(), f"{MODELS_JSON} missing -- nothing pins the base"
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


# =====================================================================
# configs/train.json -- what the schema change forced
# =====================================================================
def test_data_block_names_the_shared_prompt_renderer(train_cfg):
    """The pivot removed the system message, so 'system == eval.SYSTEM_CORE' no
    longer binds training to eval. The binding is now the shared renderer, and
    its dotted path has to be configuration -- otherwise modal_app.py hardcodes
    another lane's module name."""
    dotted = train_cfg["data"]["prompt_renderer"]
    assert isinstance(dotted, str) and dotted.strip()
    assert "." in dotted, f"prompt_renderer must be a dotted path, got {dotted!r}"
    module, _, attr = dotted.rpartition(".")
    assert module and attr
    assert module.split(".")[0] in ("scenarios", "harness"), (
        f"prompt_renderer {dotted!r} does not name the SHARED renderer -- a "
        f"training run may not nominate its own prompt renderer")


def test_data_block_no_longer_claims_the_abstainer_schema(train_cfg):
    """The old note pinned {system,user,assistant} + eval.SYSTEM_CORE. A stale
    contract note is how the next reader trains the wrong thing."""
    blob = json.dumps(train_cfg["data"])
    assert "SYSTEM_CORE" not in blob
    assert "[system, user, assistant]" not in blob


def test_completion_only_loss_and_thinking_flags_survived_the_pivot(train_cfg):
    assert train_cfg["data"]["completion_only_loss"] is True
    # the chat template call must match the eval generator's, or we measure
    # prompt transfer instead of the behavior.
    assert train_cfg["data"]["enable_thinking"] is False


# =====================================================================
# training/modal_app.py -- structural facts checked over the AST
# =====================================================================
def test_modal_app_parses_as_python():
    assert MODAL_APP.exists(), f"{MODAL_APP} missing -- training deliverable"
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
    assert "timeout=" in MODAL_APP.read_text()


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


def test_modal_app_carries_no_self_attestation_flag():
    """The previous train_log wrote system_prompt_equals_eval_SYSTEM_CORE=True as
    a hardcoded literal and the loop checker trusted it. A checker must never
    trust a boolean written by the thing it is checking."""
    src = MODAL_APP.read_text()
    assert "system_prompt_equals_eval_SYSTEM_CORE" not in src
    assert "SYSTEM_CORE" not in src


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
TOKEN_RE = re.compile(r"<[^>]+>|[^\s<]+")


class ExactTokenizer:
    """A tokenizer whose every token index is known by hand.

    Tokens are either an angle-bracket special or a run of non-space,
    non-'<' characters, so ``"delta<eos>"`` is two tokens and the count of a
    rendered prompt can be written down literally in a test.

    The chat template is deliberately explicit:

        "<bos> <user> {content} <end>"          per message
        + " <assistant>"                        when add_generation_prompt

    It records the kwargs of every apply_chat_template call so a test can assert
    the prompt half is rendered with the same call the eval generator makes.
    """
    eos_token = "<eos>"
    MERGES = ()

    def __init__(self):
        self.vocab = {}
        self.calls = []

    def _id(self, tok):
        return self.vocab.setdefault(tok, 100 + len(self.vocab))

    def ids(self, text):
        return self(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids):
        inverse = {v: k for k, v in self.vocab.items()}
        return " ".join(inverse[i] for i in ids)

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, enable_thinking=None):
        self.calls.append({"messages": copy.deepcopy(messages),
                           "tokenize": tokenize,
                           "add_generation_prompt": add_generation_prompt,
                           "enable_thinking": enable_thinking})
        text = "<bos>"
        for m in messages:
            text += f" <{m['role']}> {m['content']} <end>"
        if add_generation_prompt:
            text += " <assistant>"
        return text

    def __call__(self, text, add_special_tokens=False):
        toks, i = [], 0
        raw = TOKEN_RE.findall(text)
        while i < len(raw):
            if i + 1 < len(raw) and (raw[i], raw[i + 1]) in self.MERGES:
                toks.append(raw[i] + raw[i + 1])
                i += 2
            else:
                toks.append(raw[i])
                i += 1
        return {"input_ids": [self._id(t) for t in toks]}


class JoinMergingTokenizer(ExactTokenizer):
    """A tokenizer that merges the last prompt token with the first target token,
    so retokenizing prompt+target changes the PROMPT's own tokens. Without the
    fallback the mask boundary would silently point one token too far."""
    MERGES = (("<assistant>", "gamma"),)


class NoThinkingKwargTokenizer(ExactTokenizer):
    """An older tokenizer whose chat template does not accept enable_thinking."""

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False):
        return super().apply_chat_template(
            messages, tokenize=tokenize,
            add_generation_prompt=add_generation_prompt)


# ------------------------------------------------------------- example fixtures
USER_MSG = "alpha beta"
ASSISTANT_MSG = "gamma delta"

# The rendered prompt for USER_MSG under ExactTokenizer is, character for
# character:  "<bos> <user> alpha beta <end> <assistant>"
# which is exactly these six tokens, in this order:
PROMPT_TOKENS = ["<bos>", "<user>", "alpha", "beta", "<end>", "<assistant>"]
# and the supervised half is exactly these three:
TARGET_TOKENS = ["gamma", "delta", "<eos>"]
BOUNDARY = len(PROMPT_TOKENS)          # == 6, hand-counted, not derived


CONSTRAINT = {"id": "c-lex-014", "type": "banned_words",
              "params": {"words": ["coffee"]},
              "family": "lexical", "split": "train"}
SCENARIO = {"scenario_id": "s-0042",
            "task": "write about cafes without saying coffee",
            "constraints": ["c-lex-014"], "k": 1,
            "pressure": "adversarial", "split": "train"}


def example(user=USER_MSG, assistant=ASSISTANT_MSG, **extra):
    """One row in the pinned pivot schema: exactly [user, assistant], plus the
    constraints/provenance blocks and the source scenario the shared renderer
    is re-run on."""
    row = {
        "messages": [{"role": "user", "content": user},
                     {"role": "assistant", "content": assistant}],
        "constraints": [copy.deepcopy(CONSTRAINT)],
        "provenance": {"teacher": "openai:gpt-5.6-sol", "attempt": 1},
        "scenario": copy.deepcopy(SCENARIO),
    }
    row.update(extra)
    return row


# ================================================================= label masking
def test_the_label_mask_boundary_is_exact_on_a_hand_counted_example(app_mod):
    """THE load-bearing test. Every token index below was counted by hand from
    ExactTokenizer's template, not read back out of the implementation, so a
    one-token shift in EITHER direction fails here."""
    tok = ExactTokenizer()
    feats, stats = app_mod.build_training_features(tok, [example()], 4096, False)
    assert stats["kept"] == 1
    f = feats[0]

    expected_ids = [tok._id(t) for t in PROMPT_TOKENS + TARGET_TOKENS]
    assert f["input_ids"] == expected_ids
    assert f["attention_mask"] == [1] * 9

    # the whole label vector, written out literally
    expected_labels = [-100, -100, -100, -100, -100, -100,
                       tok._id("gamma"), tok._id("delta"), tok._id("<eos>")]
    assert f["labels"] == expected_labels

    # and the boundary named explicitly, both sides
    assert f["labels"][BOUNDARY - 1] == -100, \
        "the last prompt token (<assistant>) leaked into the loss"
    assert f["labels"][BOUNDARY] == tok._id("gamma"), \
        "the first completion token was masked away"

    # a shift by one in either direction must NOT match what we got
    shift_left = [-100] * (BOUNDARY - 1) + expected_ids[BOUNDARY - 1:]
    shift_right = [-100] * (BOUNDARY + 1) + expected_ids[BOUNDARY + 1:]
    assert f["labels"] != shift_left, "labels are shifted one token left"
    assert f["labels"] != shift_right, "labels are shifted one token right"


def test_the_supervised_region_is_exactly_the_assistant_turn_plus_eos(app_mod):
    tok = ExactTokenizer()
    feats, _ = app_mod.build_training_features(tok, [example()], 4096, False)
    supervised = [t for t in feats[0]["labels"] if t != -100]
    assert supervised == [tok._id(t) for t in TARGET_TOKENS]
    assert tok.decode(supervised) == "gamma delta <eos>"
    # nothing from the template leaked in
    assert tok._id("<assistant>") not in supervised
    assert tok._id("<end>") not in supervised


def test_masked_and_unmasked_regions_partition_the_sequence(app_mod):
    tok = ExactTokenizer()
    feats, _ = app_mod.build_training_features(tok, [example()], 4096, False)
    f = feats[0]
    assert len(f["input_ids"]) == len(f["labels"]) == len(f["attention_mask"])
    n = sum(1 for t in f["labels"] if t == -100)
    assert n == BOUNDARY
    assert f["labels"][n:] == f["input_ids"][n:], \
        "completion tokens must be trained on verbatim"


def test_the_prompt_half_is_rendered_with_the_generation_prompt_call(app_mod):
    """add_generation_prompt=True and enable_thinking passed through: this is
    what makes the trained prefix byte-identical to the eval prefix. And the
    assistant turn must not be inside the rendered prompt."""
    tok = ExactTokenizer()
    app_mod.build_training_features(tok, [example()], 4096, False)
    render = tok.calls[0]
    assert render["add_generation_prompt"] is True
    assert render["enable_thinking"] is False
    assert render["tokenize"] is False
    assert [m["role"] for m in render["messages"]] == ["user"], \
        "the prompt half must be the user turn alone"

    tok2 = ExactTokenizer()
    app_mod.build_training_features(tok2, [example()], 4096, True)
    assert tok2.calls[0]["enable_thinking"] is True


def test_no_system_message_is_ever_synthesised(app_mod):
    """The eval harness sends no system turn. Inventing one here is exactly the
    drift the prompt contract exists to prevent."""
    tok = ExactTokenizer()
    app_mod.build_training_features(tok, [example()], 4096, False)
    for call in tok.calls:
        roles = [m["role"] for m in call["messages"]]
        assert "system" not in roles, f"a system message was synthesised: {roles}"


def test_a_row_carrying_a_system_message_is_a_hard_error(app_mod):
    """Not stripped, not skipped. A dataset built to the old three-message schema
    must stop the run, loudly."""
    row = example()
    row["messages"].insert(0, {"role": "system", "content": "you are helpful"})
    # the diagnostic has to name the actual problem -- falling through to the
    # generic "messages must be [user, assistant]" branch leaves the next reader
    # guessing whether stripping the turn would have been fine.
    with pytest.raises(ValueError, match="synthesise"):
        app_mod.build_training_features(ExactTokenizer(), [row], 4096, False)


def test_rows_that_are_not_exactly_user_then_assistant_are_hard_errors(app_mod):
    tok = ExactTokenizer()
    only_user = {"messages": [{"role": "user", "content": USER_MSG}]}
    with pytest.raises(ValueError):
        app_mod.build_training_features(tok, [only_user], 4096, False)

    swapped = {"messages": [{"role": "assistant", "content": ASSISTANT_MSG},
                            {"role": "user", "content": USER_MSG}]}
    with pytest.raises(ValueError):
        app_mod.build_training_features(tok, [swapped], 4096, False)

    empty_completion = example(assistant="   ")
    with pytest.raises(ValueError, match="empty"):
        app_mod.build_training_features(tok, [empty_completion], 4096, False)


def test_a_tokenizer_without_enable_thinking_still_builds_features(app_mod):
    tok = NoThinkingKwargTokenizer()
    feats, stats = app_mod.build_training_features(tok, [example()], 4096, False)
    assert stats["kept"] == 1
    assert feats[0]["labels"] == [-100] * BOUNDARY + [
        tok._id(t) for t in TARGET_TOKENS]


def test_retokenization_across_the_join_falls_back_to_exact_concatenation(app_mod):
    """When the joint tokenization merges the last prompt token with the first
    target token, a naive len(prompt_ids) boundary is off by one. The fallback
    must rebuild the sequence from the two token lists."""
    tok = JoinMergingTokenizer()
    feats, stats = app_mod.build_training_features(tok, [example()], 4096, False)
    assert stats["prefix_mismatch"] == 1, "the mismatch was not even detected"

    prompt_ids = tok.ids("<bos> <user> alpha beta <end> <assistant>")
    target_ids = tok.ids("gamma delta<eos>")
    assert len(prompt_ids) == BOUNDARY
    assert feats[0]["input_ids"] == prompt_ids + target_ids
    assert feats[0]["labels"] == [-100] * BOUNDARY + target_ids
    assert feats[0]["labels"][BOUNDARY] == tok._id("gamma")


def test_truncation_keeps_input_ids_and_labels_in_lockstep(app_mod):
    tok = ExactTokenizer()
    limit = BOUNDARY + 2
    feats, stats = app_mod.build_training_features(tok, [example()], limit, False)
    assert stats["truncated"] == 1
    f = feats[0]
    assert len(f["input_ids"]) == len(f["labels"]) == len(f["attention_mask"]) == limit
    assert f["labels"][:BOUNDARY] == [-100] * BOUNDARY
    assert f["labels"][BOUNDARY:] == f["input_ids"][BOUNDARY:]


def test_a_row_truncated_down_to_prompt_only_is_dropped_not_trained_on(app_mod):
    """An all -100 row contributes no gradient and (with some trainers) produces
    NaN loss. It must be counted as skipped, not silently kept."""
    tok = ExactTokenizer()
    feats, stats = app_mod.build_training_features(tok, [example()], BOUNDARY - 1,
                                                   False)
    assert feats == []
    assert stats["skipped"] == 1 and stats["kept"] == 0


def test_completion_only_loss_false_supervises_every_token(app_mod):
    """The flag has to actually do something -- otherwise the masking above could
    be an accident of the default path."""
    tok = ExactTokenizer()
    feats, _ = app_mod.build_training_features(tok, [example()], 4096, False,
                                               completion_only_loss=False)
    f = feats[0]
    assert -100 not in f["labels"]
    assert f["labels"] == f["input_ids"]


def test_feature_stats_name_the_rows_that_were_kept(app_mod):
    """train_log's prompt_contract evidence points at a row index, so that index
    has to be a row that was actually trained on."""
    tok = ExactTokenizer()
    rows = [example(assistant="gamma delta"), example(assistant="gamma delta")]
    feats, stats = app_mod.build_training_features(tok, rows, 4096, False)
    assert stats["rows"] == 2
    assert stats["kept"] == len(feats) == 2
    assert stats["kept_row_indices"] == [0, 1]

    short = app_mod.build_training_features(
        tok, [example(assistant="gamma delta"), example()], BOUNDARY + 1, False)
    assert short[1]["kept_row_indices"] == [0, 1]


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


def test_the_collator_survives_a_real_feature_batch_end_to_end(app_mod):
    """The two halves wired together: features straight out of the builder go
    through the collator with the mask still intact."""
    tok = ExactTokenizer()
    rows = [example(), example(assistant="gamma")]
    feats, _ = app_mod.build_training_features(tok, rows, 4096, False)
    out = app_mod.PadCollator(0)(feats)
    assert out["labels"].shape == (2, 9)
    assert out["labels"][0].tolist()[:BOUNDARY] == [-100] * BOUNDARY
    assert out["labels"][1].tolist()[BOUNDARY:] == [
        tok._id("gamma"), tok._id("<eos>"), -100]


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


# ======================================================= the shared renderer
def test_resolve_render_fn_imports_a_real_dotted_path(app_mod, monkeypatch):
    """A dotted path inside the allowed shared-renderer packages resolves to the
    callable that module actually holds."""
    import types

    mod = types.ModuleType("scenarios.fake_render_for_test")

    def render_user_message(scenario, constraint_lookup):
        return USER_MSG

    mod.render_user_message = render_user_message
    mod.NOT_A_FUNCTION = "text"
    monkeypatch.setitem(__import__("sys").modules,
                        "scenarios.fake_render_for_test", mod)

    dotted = "scenarios.fake_render_for_test.render_user_message"
    assert app_mod.resolve_render_fn(dotted) is render_user_message
    assert app_mod.resolve_render_fn(
        "scenarios.fake_render_for_test:render_user_message") is render_user_message
    with pytest.raises(AttributeError, match="prompt_renderer"):
        app_mod.resolve_render_fn("scenarios.fake_render_for_test.missing")
    with pytest.raises(TypeError):
        app_mod.resolve_render_fn("scenarios.fake_render_for_test.NOT_A_FUNCTION")


def test_resolve_render_fn_raises_loudly_instead_of_stubbing(app_mod):
    with pytest.raises(ImportError, match="prompt_renderer"):
        app_mod.resolve_render_fn("scenarios.no_such_module.render")
    with pytest.raises(ValueError):
        app_mod.resolve_render_fn("")
    with pytest.raises(ValueError):
        app_mod.resolve_render_fn("render_user_message")  # no module part


def test_a_run_may_not_nominate_its_own_prompt_renderer(app_mod):
    """The renderer has to be the shared one the eval side uses. Pointing at any
    other module would let a run certify its own prompt -- the same allowlist
    scripts/check_loop.py enforces from the validating side."""
    assert app_mod.ALLOWED_RENDERER_ROOTS == ("scenarios", "harness")
    for dotted in ("json.dumps", "training.modal_app.render_prompt_text",
                   "datagen.whatever.render"):
        with pytest.raises(ValueError, match="shared renderer modules"):
            app_mod.resolve_render_fn(dotted)


def test_resolve_render_fn_never_silently_substitutes_a_stub(app_mod, train_cfg):
    """The configured renderer belongs to another lane and may not be on disk
    yet. Whichever world we are in, the loader must either hand back a real
    callable or fail loudly -- it must never invent one, because a local
    reimplementation of the prompt IS the drift this contract prevents."""
    dotted = train_cfg["data"]["prompt_renderer"]
    try:
        fn = app_mod.resolve_render_fn(dotted)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        assert "prompt_renderer" in str(exc) or "not callable" in str(exc), (
            "the failure must name the config key an integrator has to fix, "
            f"got: {exc}")
    else:
        assert callable(fn)
        assert fn.__module__ != app_mod.__name__, (
            "the renderer resolved to something defined inside modal_app.py -- "
            "training must IMPORT the shared renderer, never reimplement it")


def test_render_arity_matches_the_rule_the_loop_checker_uses(app_mod):
    """Writer and validator must call the shared renderer identically, or one of
    them is verifying a prompt the other never built."""
    assert app_mod.render_arity(lambda scenario, lookup: "") == 2
    assert app_mod.render_arity(lambda scenario: "") == 1
    assert app_mod.render_arity(lambda *args: "") == 2
    assert app_mod.render_arity(lambda scenario, *, lookup=None: "") == 1
    assert app_mod.render_arity(lambda scenario, lookup=None: "") == 2


class RecordingRenderer:
    """Stands in for the shared renderer: records what it was asked to render."""

    def __init__(self, mapping=None):
        self.calls = []
        self.mapping = mapping or {}

    def __call__(self, scenario, constraint_lookup):
        self.calls.append((scenario, constraint_lookup))
        return self.mapping.get(scenario.get("scenario_id"), USER_MSG)


def test_verify_prompt_contract_delegates_to_the_injected_renderer(app_mod):
    """Proof that training does not compute the prompt itself: every row goes
    through the renderer it was handed, with that row's scenario and its
    {id: constraint} lookup."""
    rows = [example(), example()]
    renderer = RecordingRenderer()
    assert app_mod.verify_prompt_contract(rows, renderer) == 2
    assert len(renderer.calls) == 2
    scenario, lookup = renderer.calls[0]
    assert scenario == SCENARIO
    assert lookup == {"c-lex-014": CONSTRAINT}


def test_verify_prompt_contract_honours_a_one_argument_renderer(app_mod):
    seen = []

    def render(scenario):
        seen.append(scenario)
        return USER_MSG

    assert app_mod.verify_prompt_contract([example()], render) == 1
    assert seen == [SCENARIO]


def test_verify_prompt_contract_rejects_a_row_the_renderer_disagrees_with(app_mod):
    """THE contract that matters most. If the stored prompt is not what the
    shared renderer produces, the tuned model would be measured under a prompt
    it never saw."""
    rows = [example(), example(user="a different prompt entirely")]
    with pytest.raises(ValueError, match="row 1"):
        app_mod.verify_prompt_contract(rows, RecordingRenderer())


def test_verify_prompt_contract_catches_a_one_character_drift(app_mod):
    rows = [example(user=USER_MSG + " ")]
    with pytest.raises(ValueError):
        app_mod.verify_prompt_contract(rows, RecordingRenderer())


def test_verify_prompt_contract_refuses_a_row_with_no_scenario_to_re_render(app_mod):
    """Without the source scenario the claim cannot be re-derived, so it would
    become an unverifiable assertion -- which is the thing we are not doing."""
    row = example()
    row.pop("scenario")
    with pytest.raises(ValueError, match="self-attested"):
        app_mod.verify_prompt_contract([row], RecordingRenderer())

    no_task = example()
    no_task["scenario"] = {"scenario_id": "s-0042", "constraints": ["c-lex-014"]}
    with pytest.raises(ValueError, match="task"):
        app_mod.verify_prompt_contract([no_task], RecordingRenderer())


def test_verify_prompt_contract_resolves_ids_against_the_declared_corpora(app_mod):
    """Rows may carry ids instead of objects -- but only when the corpora that
    resolve them are declared, otherwise the prompt is unverifiable."""
    row = example()
    row.pop("scenario")
    row["scenario_id"] = "s-0042"
    row["constraints"] = ["c-lex-014"]

    with pytest.raises(ValueError, match="scenario corpus|scenarios_path"):
        app_mod.verify_prompt_contract([row], RecordingRenderer())

    renderer = RecordingRenderer()
    assert app_mod.verify_prompt_contract(
        [row], renderer,
        scenario_corpus={"s-0042": SCENARIO},
        constraint_corpus={"c-lex-014": CONSTRAINT}) == 1
    assert renderer.calls[0] == (SCENARIO, {"c-lex-014": CONSTRAINT})

    with pytest.raises(ValueError, match="not in the declared scenario corpus"):
        app_mod.verify_prompt_contract([row], renderer,
                                       scenario_corpus={"s-9999": SCENARIO})


def test_verify_prompt_contract_rejects_rows_with_no_constraints(app_mod):
    row = example()
    row["constraints"] = []
    with pytest.raises(ValueError, match="constraints"):
        app_mod.verify_prompt_contract([row], RecordingRenderer())


def test_verify_prompt_contract_rejects_a_renderer_that_returns_non_text(app_mod):
    def bad_renderer(scenario, constraint_lookup):
        return {"content": USER_MSG}

    with pytest.raises(TypeError):
        app_mod.verify_prompt_contract([example()], bad_renderer)


def test_load_side_corpus_is_absent_or_real_never_silently_empty(app_mod, tmp_path):
    assert app_mod.load_side_corpus(tmp_path, None, "id") is None
    assert app_mod.load_side_corpus(tmp_path, "", "id") is None

    # and the diagnostic has to say what the missing corpus cost -- falling
    # through to a bare OS error leaves the reader to work that out
    with pytest.raises(FileNotFoundError, match="re-rendered"):
        app_mod.load_side_corpus(tmp_path, "nope.jsonl", "id")

    (tmp_path / "empty.jsonl").write_text("\n\n")
    with pytest.raises(ValueError, match="empty"):
        app_mod.load_side_corpus(tmp_path, "empty.jsonl", "id")

    (tmp_path / "c.jsonl").write_text(json.dumps(CONSTRAINT) + "\n")
    assert app_mod.load_side_corpus(tmp_path, "c.jsonl", "id") == {
        "c-lex-014": CONSTRAINT}


def test_declared_side_corpora_exist_on_disk(train_cfg):
    """configs/train.json names them, so a run will try to load them; a path
    that resolves to nothing is a broken run, not a warning."""
    for key in ("scenarios_path", "constraints_path"):
        rel = train_cfg["data"].get(key)
        if rel is None:
            continue
        assert (REPO_ROOT / rel).exists(), \
            f"configs/train.json:data.{key} points at a missing file: {rel}"


# ===================================================== train_log.json, pinned
SHA256 = "a" * 64
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def train_log_kwargs(**overrides):
    kwargs = dict(
        base_repo="Qwen/Qwen3-1.7B",
        base_revision=REVISION,
        config_resolved={"training": {"seed": 42}, "_smoke": True},
        dataset_path="data/train/smoke_v0.jsonl",
        dataset_rows=12,
        dataset_sha256=SHA256,
        render_fn_dotted="scenarios.render.render_user_message",
        sample_row_index=3,
        sample_rendered_prompt="<bos> <user> alpha beta <end> <assistant>",
        steps=20,
        loss_history=[(1, 2.5), (2, 2.1)],
        wall_clock_seconds=412.5,
        adapter_path="smoke/adapter",
        adapter_bytes=1234,
        merged_path="smoke/merged",
        merged_bytes=5678,
        started_utc="2026-08-20T06:00:00+00:00",
        finished_utc="2026-08-20T06:07:00+00:00",
    )
    kwargs.update(overrides)
    return kwargs


def test_train_log_has_exactly_the_pinned_shape(app_mod):
    """The pinned shape, written out here as a literal so the module cannot move
    the goalposts by exporting a different constant."""
    log = app_mod.build_train_log(**train_log_kwargs())

    assert list(log) == ["base_model", "config_resolved", "dataset",
                         "prompt_contract", "steps", "loss_history",
                         "wall_clock_seconds", "artifacts", "started_utc",
                         "finished_utc"]
    assert list(log["base_model"]) == ["repo", "revision"]
    assert list(log["dataset"]) == ["path", "rows", "sha256"]
    # the side corpora appear ONLY when the run actually declared them, and
    # never displace the three pinned keys
    with_corpora = app_mod.build_train_log(**train_log_kwargs(
        scenarios_path="data/scenarios/corpus.jsonl",
        constraints_path="data/scenarios/constraint_instances.jsonl"))
    assert list(with_corpora["dataset"]) == [
        "path", "rows", "sha256", "scenarios_path", "constraints_path"]
    assert list(log["prompt_contract"]) == ["render_fn", "sample_row_index",
                                            "sample_rendered_prompt"]
    assert list(log["artifacts"]) == ["adapter", "adapter_bytes", "merged_fp16",
                                      "merged_bytes"]

    assert log["base_model"]["revision"] == REVISION
    assert log["dataset"]["rows"] == 12 and log["dataset"]["sha256"] == SHA256
    assert isinstance(log["steps"], int) and log["steps"] == 20
    assert isinstance(log["wall_clock_seconds"], float)
    assert log["artifacts"]["adapter_bytes"] == 1234
    assert log["config_resolved"] == {"training": {"seed": 42}, "_smoke": True}
    assert json.loads(json.dumps(log)) == log, "train_log must be JSON round-trippable"


def test_train_log_loss_history_is_step_loss_pairs(app_mod):
    log = app_mod.build_train_log(**train_log_kwargs(
        loss_history=[(1, 2.5), (2, "2.25")]))
    assert log["loss_history"] == [[1, 2.5], [2, 2.25]]
    for step, loss in log["loss_history"]:
        assert isinstance(step, int) and isinstance(loss, float)


def test_train_log_carries_no_self_attestation_boolean(app_mod):
    """The evidence rule: prompt_contract holds the actual rendered text so a
    checker can re-derive the claim. A boolean the writer sets about itself is
    not evidence, and there must not be one."""
    log = app_mod.build_train_log(**train_log_kwargs())
    for key, value in log.items():
        assert not isinstance(value, bool), f"top-level boolean claim: {key}"
    for key, value in log["prompt_contract"].items():
        assert not isinstance(value, bool), f"prompt_contract boolean: {key}"
    assert "equals" not in json.dumps(log).lower()


def test_train_log_prompt_contract_is_evidence_not_an_empty_claim(app_mod):
    with pytest.raises(ValueError, match="EVIDENCE"):
        app_mod.build_train_log(**train_log_kwargs(sample_rendered_prompt=""))
    with pytest.raises(ValueError):
        app_mod.build_train_log(**train_log_kwargs(sample_rendered_prompt=None))
    with pytest.raises(ValueError):
        app_mod.build_train_log(**train_log_kwargs(sample_row_index=-1))
    with pytest.raises(ValueError):
        app_mod.build_train_log(**train_log_kwargs(render_fn_dotted="render"))


def test_train_log_rejects_fabricated_identifiers_and_counts(app_mod):
    with pytest.raises(ValueError, match="revision"):
        app_mod.build_train_log(**train_log_kwargs(base_revision="main"))
    with pytest.raises(ValueError, match="sha256"):
        app_mod.build_train_log(**train_log_kwargs(dataset_sha256="deadbeef"))
    with pytest.raises(ValueError, match="steps"):
        app_mod.build_train_log(**train_log_kwargs(steps=0))
    with pytest.raises(ValueError, match="rows"):
        app_mod.build_train_log(**train_log_kwargs(dataset_rows=0))


def test_train_log_evidence_matches_what_the_feature_builder_actually_fed(app_mod):
    """End to end on the pure path: the prompt text stored as evidence is the
    same text whose tokens are the masked prefix of the trained row."""
    tok = ExactTokenizer()
    rows = [example()]
    feats, stats = app_mod.build_training_features(tok, rows, 4096, False)
    index = stats["kept_row_indices"][0]
    rendered = app_mod.render_prompt_text(tok, rows[index]["messages"][:1], False)

    log = app_mod.build_train_log(**train_log_kwargs(
        sample_row_index=index, sample_rendered_prompt=rendered))
    evidence = log["prompt_contract"]["sample_rendered_prompt"]

    assert evidence == "<bos> <user> alpha beta <end> <assistant>"
    masked_prefix = feats[0]["input_ids"][:BOUNDARY]
    assert tok.ids(evidence) == masked_prefix
    assert feats[0]["labels"][:BOUNDARY] == [-100] * len(masked_prefix)
