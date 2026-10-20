#!/usr/bin/env python
"""
The pinned configuration for the clean RQ1/RQ2 grid.

One configuration, five models, three replicates. This file exists so that the
grid cannot drift between runs the way the previous one did: of the 16 runs it
replaces, 4 carried a machine-written config record, 3 carried a code version,
and the dependency-hint arm for 11 of them was recoverable only from the file
name. Everything that changes behaviour is declared here, stamped into every
output record by MultiAgentAnnotator._build_run_meta, and checked before launch.

    python scripts/xrun/grid_config.py --check     verify the environment
    python scripts/xrun/grid_config.py --print     emit the 15 commands

Design notes on the two settings that are easy to get wrong:

  annotator_top_logprobs  Without it there is no type_max_entropy, so the
      entropy instrument silently disappears — that is why the legacy qwen35_v2
      run has a 34-test BH family where every other run has 36.

  include_relation_schema  No longer an arm on this grid. The MoBiKo prompts
      cannot paste a relation schema at all now — the vocabulary reaches the
      agents as a name list plus relation_search — so the setting is inert here
      and is not declared. It survives for CoNLL, whose prompts still render a
      schema when one is passed. Kept in this note because runs made before the
      change carry the key in run_meta and differ on it: False on the DeepSeek
      runs, and one run carried it in its FILE NAME (_noRS).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Repo root too: the pipeline module imports src.resources_updated.*
for _p in (ROOT, ROOT / "src" / "multi_agent_annotation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── the grid ────────────────────────────────────────────────────────────────

# Six models: two within-family SCALE pairs plus two further families.
#
# The scale pairs are the point. A spread of unrelated models can only show
# that uncertainty-readability differs between models; a pair that varies scale
# inside one family can show what it varies with. The probe of 2026-10-19 made
# this better than designed — reasoning status splits exactly along the pairs,
# so scale is contrasted once inside a reasoning family (Qwen) and once inside
# a non-reasoning one (Apertus).
#
# Apertus is load-bearing beyond being a family. Reasoning models emit
# chain-of-thought into the same stream the logprobs are captured from, so
# annotator_mean_logprob pools reasoning with answer tokens. Apertus is the only
# non-reasoning pair available and therefore the control that separates "this
# model's uncertainty is unreadable" from "this model's logprobs are mostly
# reasoning".
#
# PRECISION IS PINNED TO THE UNSUFFIXED ID. The RCP allow-list carries -fp8 and
# -bfloat16 variants of nearly all of these. Quantisation changes the logit
# distribution, hence logprobs and predictive entropy — the RQ2 instruments
# themselves — so the variant is not a deployment detail here. check() asserts
# no resolved id carries a precision suffix.
#
# gpt-oss is deliberately absent. It is EOL upstream, and it is the one model
# null on every deliberation signal measured so far: challenge->false-positive
# (OR 0.82 and 1.18, both n.s.), detection disagreement (0.84, 1.08), and
# error-type discrimination (0.78, 0.94). Its three archived runs stay in the
# paper as that contrast; it is not carried into the rebuilt grid.
#
# No mixed-model arms. The DeepSeek-annotator/Kimi-critic run was how the
# annotator-seat vs critic-seat question got asked, but Kimi is withdrawn and
# the Critic adds +0.004 AUC over the Annotator's own confidence in either
# seat, so the question no longer discriminates anything worth a cell.
MODELS = [
    "rcp-qwen3.6-35B",      # Qwen, small, reasoning
    "rcp-qwen3.5-397B",     # Qwen, large, reasoning
    "rcp-apertus-8B",       # Swiss-AI, small, NOT reasoning
    "rcp-apertus-70B",      # Swiss-AI, large, NOT reasoning
    "rcp-deepseek-4.1",     # DeepSeek, reasoning
    "rcp-glm-5.3",          # Zhipu, reasoning
]

# Both arms, every cell. The previous grid had solo arms for only some models,
# which is why "does the pipeline beat its own Annotator alone" could be asked
# of four models and answered for two. The solo arm is the control for every
# deliberation claim in the paper, so it is not optional per model.
#
# single_pass stops before the Critic is ever consulted, so the solo arm costs
# one Annotator call per sentence rather than a discounted pipeline.
ARMS = {
    "multi": {"single_pass": False, "deliberation_mode": "review"},
    "solo":  {"single_pass": True,  "deliberation_mode": "review"},
    # The blind-pair design: two Annotators and a Critic annotate independently,
    # the Critic then revises having seen the other two (anonymised, shuffled),
    # and contested spans are decided by BOTH a majority vote and the
    # Adjudicator. A third arm rather than a replacement, because the review
    # design is the baseline it has to be compared against — and all 17 archived
    # runs used that design on models the provider has since withdrawn, so
    # re-running it here is the only way to compare the two on equal footing.
    #
    # guideline_search is OFF in this arm: the guideline is pasted in full to
    # every agent, so the agents differ by sampling alone.
    "blind": {"single_pass": False, "deliberation_mode": "blind_pair",
              "guideline_search": "off"},
}

# Which arms --print and --check cover. Not every arm needs running every time:
# the blind arm is new and wants verifying on its own before 54 runs are
# committed, and the review arms are the expensive baseline.
DEFAULT_ARMS = ["blind"]

# Three, not one. The paper's noise floor is se = 0.012-0.014 and its weakest
# rows are the three that had R=1; on the RQ2 side the replicate spread on rho
# reaches 0.098, which is wider than the differences between models. A single
# run per cell reproduces both problems on clean data.
REPLICATES = 3

INPUT = ROOT / "data" / "manually_labeled_last"

# Guideline v4, asserted by check() below against the file the run will load.
# A guideline that silently moved is the confound the old DeepSeek pair could not
# rule out, which is the whole reason this constant exists — it was declared here
# and never actually compared to anything until the check was added.
GUIDELINE_SECTIONS_HASH = "09db658b91fcc806f5594f0661b22911"

# Every flag that changes behaviour. Absent = the pipeline default, which is
# exactly the ambiguity this file removes, so nothing is left implicit.
FLAGS = {
    "use_dependency_relation_hints": False,   # not the point of the paper
    "strict_critic": False,
    "cold_start": False,
    "use_precedent_memory": False,
    "single_pass": False,                     # overridden per arm, see ARMS
    "tool_choice": "auto",
    # Three rounds, not two. Post-fix, round 2 now actually runs and adds
    # nothing (-0.003 to -0.020 F1, all n.s.) — but that was ONE revision round,
    # and "you only let it revise once" is the obvious objection to a null. The
    # loop already breaks as soon as the Critic raises no disagreements and no
    # missing annotations, so a third round costs tokens only on the sentences
    # that have not converged, which are the ones the question is about.
    "max_rounds": 3,
    "annotator_top_logprobs": 20,             # required for the entropy instrument
    # Capture the Critic's and Adjudicator's token probabilities too. The
    # Annotator's are a single-model signal available before deliberation; these
    # are the only ones that can show the deliberation itself carrying
    # uncertainty. Capturing later means rerunning the whole grid.
    "critic_top_logprobs": 20,
    "adjudicator_top_logprobs": 20,
    "prompt_set": "mobiko",
    # Overridden per arm. Declared here so build_command never reads a key that
    # only some arms define — the kind of gap that turns into a KeyError three
    # models into a launch.
    "deliberation_mode": "review",
    "context_radius": 1,
    # The relation NAMES are in the prompt and their definitions are behind
    # relation_search; "optional" lets an agent look one up when unsure rather
    # than forcing a call per relation. Declared here because it changes what the
    # agents can produce, which is exactly the kind of setting that used to live
    # only in a file name.
    "relation_search": "optional",
    "guideline_search": "optional",
    "guideline_search_backend": "embedding",
}

OUT_DIR = ROOT / "data" / "auto_annotated"


def out_path(model: str, arm: str, rep: int) -> Path:
    return OUT_DIR / f"grid_{model}_{arm}_r{rep}.jsonl"


def build_command(model: str, arm: str, rep: int) -> list:
    """argv for one run. Temperature and seeds stay at pipeline defaults; the
    replicate index is carried by the output path, not by a config change, so
    the three runs of a cell are genuine replicates of one configuration."""
    flags = {**FLAGS, **ARMS[arm]}
    cmd = [
        sys.executable, str(ROOT / "src" / "multi_agent_annotation" / "demo_ag2.py"),
        "--annotator-model", model,
        "--critic-model", model,
        "--adjudicator-model", model,
        "--input", str(INPUT),
        "--output", str(out_path(model, arm, rep)),
        "--max-rounds", str(flags["max_rounds"]),
        "--tool-choice", flags["tool_choice"],
        "--guideline-search", flags["guideline_search"],
        "--deliberation-mode", flags["deliberation_mode"],
        "--relation-search", flags["relation_search"],
        "--guideline-search-backend", flags["guideline_search_backend"],
    ]
    # Store-true flags: pass only when enabled, and never rely on the default.
    if flags["use_dependency_relation_hints"]:
        cmd.append("--use-dependency-relation-hints")
    if flags["strict_critic"]:
        cmd.append("--strict-critic")
    if flags["cold_start"]:
        cmd.append("--cold-start")
    if flags["use_precedent_memory"]:
        cmd.append("--precedent-memory")
    if flags["single_pass"]:
        cmd.append("--single-pass")
    return cmd


# ── preflight ───────────────────────────────────────────────────────────────

def check(arms=None) -> int:
    """Fail loudly BEFORE burning GPU hours on runs that cannot be cited."""
    arms = list(arms or DEFAULT_ARMS)
    import multi_agent_annotation_ag2 as M
    problems, warnings = [], []

    version = os.getenv(M.CODE_VERSION_ENV) or M.resolve_code_version()
    if not version:
        problems.append(
            f"no code version: {M.CODE_VERSION_ENV} is unset and this is not a git "
            f"checkout. Export it on the machine that has the repo:\n"
            f"      export {M.CODE_VERSION_ENV}=$(git rev-parse --short HEAD)")
    elif version.endswith("-dirty"):
        warnings.append(
            f"code version is {version}: tracked files are modified, so the sha "
            f"does not name the code that will run. Commit before launching, or "
            f"accept that every record is stamped '-dirty'.")
    else:
        print(f"  ok    code version {version}")

    # Model keys must resolve, or the run silently falls back and the cell is
    # not the model the table will claim it is.
    try:
        from src.resources_updated.model_endpoints import MODEL_ENDPOINTS
        unknown = [m for m in MODELS if m not in MODEL_ENDPOINTS]
        if unknown:
            problems.append(f"model keys not in MODEL_ENDPOINTS: {unknown}\n"
                            f"      known: {sorted(MODEL_ENDPOINTS)}")
        else:
            # A withdrawn model surfaces as a gateway 403 partway into a batch,
            # after the earlier cells have already been written.
            gone = [m for m in MODELS if MODEL_ENDPOINTS[m].get("withdrawn")]
            if gone:
                problems.append(
                    "withdrawn models in MODELS: "
                    + ", ".join(f"{m} ({MODEL_ENDPOINTS[m]['withdrawn']})"
                                for m in gone))
            # Precision suffixes change the logit distribution, so they change
            # logprobs and entropy — the RQ2 instruments. A grid that mixes
            # -fp8 with unsuffixed ids is not measuring one thing.
            suffixed = [(m, MODEL_ENDPOINTS[m]["model"]) for m in MODELS
                        if any(MODEL_ENDPOINTS[m]["model"].endswith(sfx)
                               for sfx in ("-fp8", "-bfloat16", "-int4",
                                           "-float32", "-float16"))]
            if suffixed:
                problems.append(
                    "precision-suffixed model ids (pin the unsuffixed form; "
                    "quantisation moves the logprobs this paper measures): "
                    + ", ".join(f"{k} -> {v}" for k, v in suffixed))
            for m in MODELS:
                ep = MODEL_ENDPOINTS[m]
                r = ep.get("reasoning")
                if r is None:
                    warnings.append(f"{m} does not declare `reasoning`; the "
                                    f"enable_thinking toggle will be guessed")
                print(f"  ok    {m:22s} -> {ep.get('model'):46s} "
                      f"reasoning={r}")
    except Exception as exc:
        warnings.append(f"could not check model keys: {exc}")

    if not INPUT.exists():
        problems.append(f"input missing: {INPUT}")
    else:
        print(f"  ok    input {INPUT}")

    # The guideline material the runs will actually load, hashed the way the
    # pipeline hashes it.
    #
    # NOTE what goes into this hash: the pipeline calls _init_tool_state with
    # `all_sections = decision_support_sections + guidance_sections`, so
    # run_meta.guideline_sections_hash covers the DECISION TABLE AND THE
    # NARRATIVE GUIDELINE TOGETHER, in that order — not the .md alone. Hashing
    # only the guideline file gives a different digest and a false alarm.
    try:
        ds, gl = M._DEFAULT_DECISION_SUPPORT, M._DEFAULT_GUIDELINE
        missing = [str(x) for x in (ds, gl) if not x.exists()]
        if missing:
            problems.append(f"guideline material missing: {missing}")
        else:
            sections = M.load_decision_support(ds) + (
                M.load_guideline_from_docx(gl) if gl.suffix.lower() == ".docx"
                else M.load_guideline_from_md(gl))
            got = M._guideline_sections_hash(sections)
            if got != GUIDELINE_SECTIONS_HASH:
                problems.append(
                    f"guideline hash moved: {ds.name} + {gl.name} now hash to {got}, "
                    f"pinned {GUIDELINE_SECTIONS_HASH}. Either the guideline "
                    f"changed (the grid would not be comparable to the archived "
                    f"runs) or the pin is stale — decide which, do not just "
                    f"update the constant.")
            else:
                print(f"  ok    guideline {ds.name} + {gl.name} "
                      f"({len(sections)} sections) hash {got[:8]}")
    except Exception as exc:
        warnings.append(f"could not verify the guideline hash: {exc}")

    if "blind" in arms:
        # Without abstracts the context window falls back to sentence-only with
        # a log warning, the run completes, the output looks normal — and every
        # agent had LESS context than Mark and Davnah did, which biases the
        # agreement-vs-0.497 comparison in the humans' favour. A warning in a
        # log nobody reads after a 90-minute run is not a safeguard.
        try:
            sys.path.insert(0, str(ROOT / "src" / "multi_agent_annotation"))
            from abstract_context import ContextWindow
            cw = ContextWindow(radius=FLAGS["context_radius"])
            if not cw.index.abstracts:
                problems.append(
                    "blind arm needs data/abstracts/abstracts.txt: without it "
                    "every agent sees the sentence alone, which is NOT the "
                    "condition the human annotators worked under")
            else:
                sents = [l.strip() for l in INPUT.joinpath(
                    "overlap_sentences.txt").read_text().splitlines()
                    if l.strip()] if INPUT.is_dir() else []
                res = [cw.for_sentence(x) for x in sents]
                none = sum(1 for r in res if not r["n_before"] and not r["n_after"])
                both = sum(1 for r in res if r["n_before"] and r["n_after"])
                if none:
                    problems.append(
                        f"{none} of {len(sents)} sentences resolve to no context "
                        f"at all — the abstracts file and the input are out of sync")
                else:
                    print(f"  ok    context window: {both}/{len(sents)} with both "
                          f"neighbours, 0 with none ({len(cw.index.abstracts)} abstracts)")
        except Exception as exc:
            problems.append(f"blind arm context check failed: {exc}")

    for role in ("annotator", "critic", "adjudicator"):
        n = FLAGS[f"{role}_top_logprobs"]
        if n:
            print(f"  ok    {role:12s} top_logprobs = {n}")
        elif role == "annotator":
            problems.append("annotator_top_logprobs is falsy — no entropy instrument")
        else:
            warnings.append(f"{role}_top_logprobs is off: no {role} uncertainty, so "
                            f"the deliberation-vs-single-model question stays open")

    existing = [out_path(m, a, r) for m in MODELS for a in arms
                for r in range(1, REPLICATES + 1)
                if out_path(m, a, r).exists()]
    if existing:
        warnings.append(f"{len(existing)} output file(s) already exist and would be "
                        f"resumed or overwritten, e.g. {existing[0].name}")

    for w in warnings:
        print(f"  WARN  {w}")
    for p in problems:
        print(f"  FAIL  {p}")
    print(f"\n  arms: {', '.join(arms)}")
    print(f"  {len(MODELS)} models x {len(arms)} arm(s) x {REPLICATES} "
          f"replicates = {len(MODELS) * len(arms) * REPLICATES} runs")
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="preflight the environment")
    ap.add_argument("--print", action="store_true", dest="show",
                    help="emit the run commands")
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS,
                    choices=sorted(ARMS) + ["all"],
                    help=f"which arms to cover (default: {DEFAULT_ARMS})")
    args = ap.parse_args()
    arms = sorted(ARMS) if "all" in args.arms else args.arms
    if args.show:
        import multi_agent_annotation_ag2 as M
        v = os.getenv(M.CODE_VERSION_ENV) or M.resolve_code_version() or ""
        print(f"export {M.CODE_VERSION_ENV}={v}")
        for role in ("annotator", "critic", "adjudicator"):
            print(f"export {role.upper()}_TOP_LOGPROBS={FLAGS[f'{role}_top_logprobs']}")
        for model in MODELS:
            for arm in arms:
                for rep in range(1, REPLICATES + 1):
                    print(" ".join(build_command(model, arm, rep)))
        return 0
    return check(arms)


if __name__ == "__main__":
    raise SystemExit(main())
