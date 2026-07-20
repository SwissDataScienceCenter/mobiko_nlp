"""
reconstruct_loop.py — closed-loop cold-start guideline reconstruction (RQ-D, spec 11).

Drives the iteration loop of spec 11.2:

    for i = 0, 1, 2, …:
      (1) annotate the WORKING set with the multi-agent system under guideline G_i
      (2) mine confusion patterns from the deliberation logs (Layer 4), ≥ min-count
      (3) draft amendments for the top-K patterns (guideline_amender)
      (4) apply the programmatically-approved amendments → G_{i+1}  (append-only)
      (5) stop on the hard cap i_max, or when the loop can no longer progress

This is the FULLY AUTOMATIC loop: "approval" in step (4) is the amender's own
programmatic validity gate (well-formed JSON + operational decision_test naming a
real label pair); there is NO human in the loop. The expert guideline is never
shown to the loop — it is only used later, at evaluation time (spec 11.5).

SCOPE (this pass = loop orchestrator only): the per-iteration friction / held-out
F1 metrics and the dual stopping rule (11.3), plus the controls/coverage (11.5b,
11.8), are intentionally deferred. Friction counts ARE logged each iteration for
monitoring, but the loop stops only on: i_max reached · no confusion ≥ min-count ·
zero amendments accepted · a corpus-leak in the drafted guideline.

Nothing in the existing pipeline is modified. Annotation is run as a fresh
subprocess per iteration (``cold_start_annotate.py``) so the pipeline's
module-global tool state is reset between guidelines; mining and amendment reuse
the library functions in-process.

Usage:
  python reconstruct_loop.py \
      --g0 ./G0_cold_start.md \
      --working-jsonl ./working_set.jsonl \
      --out-dir ./output/reconstruction_run1 \
      --i-max 6 --top-k 5 --min-count 5 \
      --annotator-model qwen3-35B-vllm --critic-model qwen3-35B-vllm \
      --adjudicator-model qwen3-35B-vllm --amender-model qwen3-35B-vllm
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_THIS_DIR = Path(__file__).resolve().parent          # …/multi_agent_annotation/loop
_PKG_ROOT = _THIS_DIR.parent                         # …/multi_agent_annotation (shared core)
_SRC = _PKG_ROOT.parent                              # …/src (for resources_updated)
_REPO_ROOT = _SRC.parent                             # repo root
# Flat sibling imports span the package's subdirs (loop/ + evaluation/) and the
# shared core at the package root; make them all resolvable regardless of cwd.
for _p in (_SRC, _PKG_ROOT, _PKG_ROOT / "loop", _PKG_ROOT / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Reuse the real pipeline's building blocks (flat imports → resolved via the
# bootstrap above: deliberation_history is at the package root; the amender,
# cold_start_*, decision_table and stopping_rule are loop/ siblings).
from multi_agent_annotation_ag2 import MODEL_ENDPOINTS
from deliberation_history import load_records, analyze
from guideline_amender import (
    load_confusion_patterns,
    classify_pattern,
    collect_examples,
    amend_pattern,
    write_outputs,
    verify_no_corpus_leak,
    _make_client,
    generate_amendment,
)
import cold_start_annotate
import cold_start_init
import decision_table
import stopping_rule

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("reconstruct_loop")

_ANNOTATE_SCRIPT = _THIS_DIR / "cold_start_annotate.py"
_DEFAULT_SCHEMA = _SRC / "resources_updated" / "relation_schema_new.py"
_DEFAULT_SEEDS = _SRC / "resources_updated" / "manual_seeds_filled.py"


# ─────────────────────────────────────────────────────────────
# Stage 1 — annotation (subprocess, isolated per iteration)
# ─────────────────────────────────────────────────────────────

def run_annotation(args, guideline: Path, decision_tbl: Path,
                   input_path: Path, out_jsonl: Path,
                   num_sentences: Optional[int] = None,
                   input_format: str = "jsonl") -> None:
    """Annotate ``input_path`` under (``guideline``, ``decision_tbl``) → ``out_jsonl``.

    ``guideline`` drives the Critic/Adjudicator; ``decision_tbl`` drives the
    Annotator — both are cold-started and amended each iteration. ``input_path``
    is the working set for friction mining, or the held-out split for the F1
    guard (same (G_i, D_i), separate output file). ``input_format`` selects
    whether ``input_path`` is passed to the annotate subprocess as
    ``--input-jsonl`` (JSONL with a 'sentence'/'text' field) or ``--input-txt``
    (plain text, one sentence per line).

    On a fresh (non-resume) run we clear any stale output first, because the
    underlying ``annotate_batch`` appends. With --resume we leave it and let the
    pipeline skip already-annotated sentences.

    If the subprocess itself dies (crash, OOM, a flaky endpoint) we retry it up
    to ``args.subprocess_retries`` times — every retry forces ``--resume`` on the
    subprocess regardless of the top-level ``--resume`` flag, since any sentence
    a prior attempt already flushed to ``out_jsonl`` must not be redone. Only
    after exhausting all attempts do we raise.
    """
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume and out_jsonl.exists():
        out_jsonl.unlink()

    input_flag = "--input-jsonl" if input_format == "jsonl" else "--input-txt"
    base_cmd = [
        args.python, str(_ANNOTATE_SCRIPT),
        input_flag, str(Path(input_path).resolve()),
        "--guideline", str(guideline.resolve()),
        "--decision-support", str(decision_tbl.resolve()),
        "--schema", str(args.schema.resolve()),
        "--output", str(out_jsonl.resolve()),
        "--annotator-model", args.annotator_model,
        "--critic-model", args.critic_model,
        "--adjudicator-model", args.adjudicator_model,
        "--annotator-temp", str(args.annotator_temp),
        "--max-rounds", str(args.max_rounds),
        "--guideline-search-backend", args.guideline_search_backend,
        "--guideline-search", args.guideline_search,
        "--timeout", str(args.timeout),
    ]
    if args.critic_temp is not None:
        base_cmd += ["--critic-temp", str(args.critic_temp)]
    if args.adjudicator_temp is not None:
        base_cmd += ["--adjudicator-temp", str(args.adjudicator_temp)]
    if args.seeds:
        base_cmd += ["--seeds", str(args.seeds.resolve())]
    if num_sentences is not None:
        base_cmd += ["--num-sentences", str(num_sentences)]
    if args.strict_critic:
        base_cmd += ["--strict-critic"]
    if args.cold_start:
        base_cmd += ["--cold-start"]
    if args.tool_choice:
        base_cmd += ["--tool-choice", args.tool_choice]
    if args.precedent_memory:
        base_cmd += ["--precedent-memory"]

    # Expose the loop dir, the shared package root, src/ (resources_updated) and
    # the evaluation siblings on PYTHONPATH so cold_start_annotate's flat imports
    # resolve no matter the cwd (it also self-bootstraps the same set).
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_THIS_DIR), str(_PKG_ROOT), str(_SRC), str(_PKG_ROOT / "evaluation"),
         env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)

    max_attempts = max(1, args.subprocess_retries)
    last_returncode: Optional[int] = None
    for attempt in range(1, max_attempts + 1):
        resume_this_attempt = args.resume or attempt > 1
        cmd = list(base_cmd)
        if resume_this_attempt:
            cmd += ["--resume"]

        logger.info("  [annotate attempt %d/%d] %s", attempt, max_attempts, " ".join(cmd))
        proc = subprocess.run(cmd, cwd=str(_THIS_DIR), env=env)
        if proc.returncode == 0:
            return
        last_returncode = proc.returncode
        if attempt < max_attempts:
            backoff = min(60, 10 * attempt)
            logger.warning(
                "  annotation subprocess failed (exit %d) on attempt %d/%d — already-"
                "annotated sentences in %s are preserved; retrying in %ds with --resume…",
                proc.returncode, attempt, max_attempts, out_jsonl, backoff,
            )
            time.sleep(backoff)

    raise RuntimeError(
        f"annotation subprocess failed (exit {last_returncode}) after {max_attempts} attempt(s)"
    )


def _load_expected_sentences(
    input_path: Path, input_format: str, cap: Optional[int],
) -> List[str]:
    """The exact (deduplicated, order-preserved) sentence list ``run_annotation``
    would send to the annotate subprocess for this input — same loaders
    ``cold_start_annotate.main`` uses, so the comparison in
    ``_deliberations_complete`` is apples-to-apples."""
    if input_format == "jsonl":
        sentences = cold_start_annotate.load_sentences_from_jsonl(Path(input_path).resolve())
    else:
        sentences = cold_start_annotate.load_sentences_from_txt(Path(input_path).resolve())
    if cap is not None:
        sentences = sentences[:cap]
    return sentences


def _deliberations_complete(
    deliberations: Path, input_path: Path, input_format: str, cap: Optional[int],
) -> bool:
    """Whether ``deliberations`` already covers every expected sentence.

    A crash mid-batch leaves a non-empty ``deliberations.jsonl`` with only some
    sentences done — a plain "file exists and is non-empty" check would treat
    that as finished and never annotate the rest. This checks the actual
    sentence coverage instead, so --resume only skips re-annotation once the
    iteration is genuinely complete.
    """
    if not (deliberations.exists() and deliberations.stat().st_size > 0):
        return False
    expected = set(_load_expected_sentences(input_path, input_format, cap))
    if not expected:
        return False
    done: set = set()
    for line in deliberations.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(json.loads(line).get("sentence", ""))
        except Exception:
            continue
    return expected.issubset(done)


# ─────────────────────────────────────────────────────────────
# Stage 2 — mine confusions (Layer 4), in-process
# ─────────────────────────────────────────────────────────────

def mine_confusions(deliberations: Path, layer34_out: Path) -> Dict[str, Any]:
    """Run the Layer-4 analysis and persist it; return the report dict."""
    records = load_records(deliberations)
    report = analyze(records)
    report["n_sentences"] = len(records)
    layer34_out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def select_patterns(
    layer34_json: Path, top_k: int, min_count: int, entity_only: bool = True,
) -> List[Dict[str, Any]]:
    """Top-K confusions with count ≥ min_count (spec 11.2: ≥ 5 occurrences).

    With ``entity_only`` (the RQ-D default, spec 11.1: "reconstruct entity
    disambiguation only") the relation-validity and annotation-scope confusions
    are dropped, leaving entity-type confusions — the relation schema is fixed.
    """
    # Pull all (normalised) confusions, then apply the occurrence threshold and cap.
    everything = load_confusion_patterns(layer34_json, top_k=10_000)
    kept = [p for p in everything if p["count"] >= min_count]
    if entity_only:
        kept = [p for p in kept
                if classify_pattern(p["annotator"], p["critic"]) == "entity_type"]
    return kept[:top_k]


# ─────────────────────────────────────────────────────────────
# Stage 3+4 — draft amendments and apply (append accepted only)
# ─────────────────────────────────────────────────────────────

def draft_amendments(
    args,
    patterns: List[Dict[str, Any]],
    records: List[dict],
    guideline_path: Path,
    amend_dir: Path,
    today: str,
) -> Dict[str, Any]:
    """Draft one amendment per pattern, write outputs, run the leak check.

    ``guideline_path`` is the current G_i (used by write_outputs for the base
    text and the next-version filename). Returns a summary dict: amendments,
    status counts, output paths, the drafted next-guideline path (G_i + accepted
    amendments appended), and any detected corpus leaks.
    """
    guideline_text = guideline_path.read_text(encoding="utf-8")
    client, model_name = _make_client(args.amender_model)
    gen = lambda messages: generate_amendment(client, model_name, messages)

    amendments: List[Dict[str, Any]] = []
    for p in patterns:
        ex = collect_examples(records, p["annotator"], p["critic"], args.examples_per_pattern)
        logger.info("    amend %s → %s (%d×, %d ex.)",
                    p["annotator"], p["critic"], p["count"], len(ex))
        a = amend_pattern(p, ex, guideline_text,
                          generate_fn=gen, max_redrafts=args.max_redrafts)
        logger.info("      → %s (%d attempt(s))", a["status"], a.get("attempts", 0))
        amendments.append(a)

    paths = write_outputs(amendments, guideline_text, guideline_path, amend_dir, today)

    corpus_sentences = [(r.get("sentence") or "") for r in records]
    leaks = verify_no_corpus_leak(amendments, corpus_sentences, n=args.leak_ngram)

    counts = {"accepted": 0, "rejected": 0, "malformed": 0}
    for a in amendments:
        st = a.get("status", "malformed")
        counts[st] = counts.get(st, 0) + 1

    return {
        "amendments": amendments,
        "counts": counts,
        "paths": {k: str(v) for k, v in paths.items()},
        "drafted_guideline": paths["guideline"],
        "leaks": leaks,
    }


# ─────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────

def _accepted_triples_from_amendments(amendments_json: Path) -> List[tuple]:
    """Recover accepted (annotator, critic, decision_test) triples from a written
    ``amendments.json`` (its ``pattern`` field is ``"ANNOTATOR → CRITIC"``).

    Used to rebuild the cumulative rule-coverage set when --resume skips an
    already-completed iteration's drafting stage.
    """
    out: List[tuple] = []
    try:
        data = json.loads(Path(amendments_json).read_text(encoding="utf-8"))
    except Exception:
        return out
    for a in data:
        if a.get("status") != "accepted":
            continue
        left, sep, right = a.get("pattern", "").partition("→")
        if sep:
            out.append((left.strip(), right.strip(), a.get("decision_test", "")))
    return out


def run_loop(args) -> Dict[str, Any]:
    out_dir: Path = args.out_dir
    guidelines_dir = out_dir / "guidelines"
    tables_dir = out_dir / "tables"
    guidelines_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()

    # Cold start: both the guideline (G0) and the decision table (D0). Any not
    # supplied on the CLI is generated from the schema (names + one-line defs
    # only, spec 11.1) so the loop is runnable turnkey and the cold start is
    # recorded under out_dir/coldstart/.
    g0_src: Path
    d0_src: Path
    if args.g0 and args.d0:
        g0_src, d0_src = args.g0.resolve(), args.d0.resolve()
    else:
        gen = cold_start_init.generate(out_dir / "coldstart", today=today)
        g0_src = args.g0.resolve() if args.g0 else gen["g0"]
        d0_src = args.d0.resolve() if args.d0 else gen["d0"]
        logger.info("Generated cold-start artifact(s) in %s", out_dir / "coldstart")

    g0_path = guidelines_dir / "G0.md"
    d0_path = tables_dir / "D0.csv"
    if not (args.resume and g0_path.exists()):
        shutil.copyfile(g0_src, g0_path)
    if not (args.resume and d0_path.exists()):
        shutil.copyfile(d0_src, d0_path)
    logger.info("G0 = %s (from %s)", g0_path, g0_src)
    logger.info("D0 = %s (from %s)", d0_path, d0_src)

    # Resolve the working-set / held-out-set input path + format once — each is
    # supplied as either JSONL (--*-jsonl) or plain text (--*-txt), mutually
    # exclusive at the CLI level (argparse enforces exactly one, or neither for
    # held-out).
    working_path, working_format = (
        (args.working_jsonl, "jsonl") if args.working_jsonl else (args.working_txt, "txt")
    )
    if args.held_out_jsonl or args.held_out_txt:
        held_out_path, held_out_format = (
            (args.held_out_jsonl, "jsonl") if args.held_out_jsonl else (args.held_out_txt, "txt")
        )
    else:
        held_out_path, held_out_format = None, None

    # Stopping rule (spec §11.3): friction (PRIMARY) + held-out F1 (GUARD), with
    # guideline text delta and rule coverage logged for monitoring only.
    stopping_enabled = not args.no_stopping_rule
    held_out_ready = bool(held_out_path and args.held_out_human)
    if stopping_enabled and not held_out_ready:
        logger.warning(
            "Stopping rule: no held-out split configured (--held-out-jsonl/"
            "--held-out-txt + --held-out-human) — the echo-chamber GUARD is "
            "DISABLED; running on friction convergence only (cannot detect F1 "
            "regressions)."
        )
    expert_rules = (stopping_rule.load_expert_rules(args.expert_rules)
                    if args.expert_rules else None)
    if expert_rules is not None:
        logger.info("Loaded %d enumerated expert disambiguation rule(s) for coverage (§11.5b).",
                    len(expert_rules))

    manifest: Dict[str, Any] = {
        "config": {
            "g0": str(g0_src), "d0": str(d0_src),
            "working_set": str(working_path), "working_format": working_format,
            "i_max": args.i_max, "top_k": args.top_k, "min_count": args.min_count,
            "patterns": args.patterns,
            "annotator_model": args.annotator_model, "critic_model": args.critic_model,
            "adjudicator_model": args.adjudicator_model, "amender_model": args.amender_model,
            "annotator_temp": args.annotator_temp, "critic_temp": args.critic_temp,
            "adjudicator_temp": args.adjudicator_temp, "cold_start": args.cold_start,
            "max_rounds": args.max_rounds, "date": today,
            "stopping_rule": {
                "enabled": stopping_enabled,
                "friction_eps": args.stop_friction_eps,
                "friction_patience": args.stop_friction_patience,
                "f1_guard_patience": args.stop_f1_guard_patience,
                "held_out_set": str(held_out_path) if held_out_path else None,
                "held_out_format": held_out_format,
                "held_out_human": [str(p) for p in (args.held_out_human or [])],
                "held_out_f1_mode": args.held_out_f1_mode,
                "expert_rules": str(args.expert_rules) if args.expert_rules else None,
            },
        },
        "iterations": [],
        "final_guideline": str(g0_path),
        "final_decision_table": str(d0_path),
        "stopped": None,
        "stopping": None,  # filled with the friction ∥ held-out-F1 trajectory at the end
    }
    manifest_path = out_dir / "manifest.json"

    def flush_manifest() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                 encoding="utf-8")

    flush_manifest()

    # Per-iteration metric rows + the two driving series, plus the cumulative
    # accepted decision-tests (for the §11.5b coverage curve) and the previous
    # iteration's amendment counts (text-delta is reported against G_{i-1}).
    metric_rows: List[Dict[str, Any]] = []
    friction_series: List[float] = []
    f1_series: List[Optional[float]] = []
    cumulative_triples: List[tuple] = []
    last_accepted = 0
    last_injections = 0
    final_decision: Dict[str, Any] = {}

    def finalize_report() -> None:
        """Write the friction ∥ held-out-F1 trajectory (spec §11.3 REPORT)."""
        report = stopping_rule.build_report(metric_rows, final_decision)
        manifest["stopping"] = report
        (out_dir / "stopping_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        traj = stopping_rule.format_trajectory(report)
        (out_dir / "stopping_report.md").write_text(traj + "\n", encoding="utf-8")
        for line in traj.splitlines():
            logger.info(line)
        flush_manifest()

    for i in range(args.i_max):
        g_i = guidelines_dir / f"G{i}.md"
        g_next = guidelines_dir / f"G{i+1}.md"
        d_i = tables_dir / f"D{i}.csv"
        d_next = tables_dir / f"D{i+1}.csv"
        iter_dir = out_dir / f"iter_{i:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        deliberations = iter_dir / "deliberations.jsonl"
        layer34 = iter_dir / "layer34.json"
        amend_dir = iter_dir / "amend"

        logger.info("=" * 64)
        logger.info("ITERATION %d  (guideline: %s, table: %s)", i, g_i.name, d_i.name)
        logger.info("=" * 64)

        rec: Dict[str, Any] = {"iteration": i, "guideline_in": str(g_i),
                               "decision_table_in": str(d_i)}

        # Was this iteration fully drafted on a previous run? (resume). Metrics
        # below are still recomputed from its logs so the trajectory + stop
        # decision are correct; only the amend/apply work is skipped.
        iteration_complete = (
            args.resume and g_next.exists() and d_next.exists()
            and (amend_dir / "amendments.json").exists()
        )

        # (1) Annotate the working set under (G_i, D_i) ------------------------
        if args.resume and _deliberations_complete(
            deliberations, working_path, working_format, args.num_sentences
        ):
            logger.info("  [resume] working-set deliberations already complete: %s", deliberations)
        else:
            run_annotation(args, g_i, d_i, working_path, deliberations, args.num_sentences,
                           input_format=working_format)
        rec["deliberations"] = str(deliberations)

        # (2) Mine confusions + friction (PRIMARY stop signal, §11.3) ----------
        report = mine_confusions(deliberations, layer34)
        records = load_records(deliberations)
        friction = stopping_rule.friction_from_report(report, records)
        rec["layer34"] = str(layer34)
        rec["friction"] = friction
        logger.info("  friction: R1 disagreements=%d (PRIMARY) · all-steps=%d · rounds-used rate=%s",
                    friction["r1_disagreements"], friction["total_disagreements_all_steps"],
                    friction["rounds_used_rate"])

        # (2b) Held-out agent-vs-expert F1 (GUARD signal, §11.7) ---------------
        held_out = None
        if held_out_ready:
            ho_delib = iter_dir / "heldout_deliberations.jsonl"
            if args.resume and _deliberations_complete(
                ho_delib, held_out_path, held_out_format, args.held_out_num_sentences
            ):
                logger.info("  [resume] held-out deliberations already complete: %s", ho_delib)
            else:
                run_annotation(args, g_i, d_i, held_out_path, ho_delib,
                               args.held_out_num_sentences, input_format=held_out_format)
            held_out = stopping_rule.compute_held_out_f1(
                ho_delib, args.held_out_human, mode=args.held_out_f1_mode,
                names=args.held_out_names)
            rec["held_out_deliberations"] = str(ho_delib)
            rec["held_out_f1"] = held_out
            logger.info("  held-out F1 (%s) = %s  (n=%d eval sentences)", held_out["mode"],
                        f"{held_out['f1']:.4f}" if held_out["f1"] is not None else "—",
                        held_out["n_eval_sentences"])

        # (2c) Guideline text delta + rule coverage (MONITORING ONLY) ----------
        prev_g = guidelines_dir / f"G{i-1}.md" if i > 0 else None
        td = stopping_rule.text_delta(prev_g, g_i, rules_added=last_accepted,
                                      table_injections=last_injections,
                                      drift=not args.no_embedding_drift)
        rec["text_delta"] = td
        rcov = None
        if expert_rules is not None:
            rcov = stopping_rule.rule_coverage(
                expert_rules, cumulative_triples,
                semantic=args.rule_coverage_semantic,
                semantic_threshold=args.rule_coverage_threshold)
            rec["rule_coverage"] = rcov
            logger.info("  rule coverage (pairs) = %s  (%d/%d expert disambiguations)",
                        rcov["pair_coverage"], rcov["n_pair_matched"], rcov["n_expert_rules"])

        # Record this guideline's row + the two driving series.
        friction_series.append(friction["r1_disagreements"])
        f1_series.append(held_out["f1"] if held_out else None)
        metric_rows.append({
            "iteration": i, "guideline": g_i.name, "decision_table": d_i.name,
            "friction": friction, "held_out_f1": held_out,
            "text_delta": td, "rule_coverage": rcov,
        })

        # (3) DUAL STOPPING RULE (§11.3) — evaluated BEFORE drafting so we don't
        #     pay the amender once the loop has converged. The GUARD overrides. ─
        decision = stopping_rule.decide_stop(
            friction_series, f1_series,
            eps=args.stop_friction_eps,
            friction_patience=args.stop_friction_patience,
            f1_guard_patience=args.stop_f1_guard_patience)
        rec["stop_decision"] = decision
        final_decision = decision
        if stopping_enabled and decision["stop"]:
            logger.info("  STOP — %s", decision["reason"])
            if decision["echo_chamber"]:
                best = stopping_rule.best_f1_iteration(f1_series)
                logger.warning("  ⚠ ECHO-CHAMBER: held-out F1 regressed while friction fell; "
                               "highest held-out F1 was at iteration %s.", best)
            rec["stopped"] = decision["reason"]
            manifest["iterations"].append(rec)
            manifest["stopped"] = decision["reason"]
            manifest["final_guideline"] = str(g_i)
            manifest["final_decision_table"] = str(d_i)
            flush_manifest()
            break

        # Resume: skip drafting an already-completed iteration, but rebuild the
        # cumulative coverage set + text-delta counts from its amendments.json.
        if iteration_complete:
            logger.info("  [resume] iteration %d already drafted — skipping amend/apply.", i)
            triples = _accepted_triples_from_amendments(amend_dir / "amendments.json")
            cumulative_triples.extend(triples)
            last_accepted = len(triples)
            last_injections = len(triples)  # proxy; exact count was logged at draft time
            rec["resumed"] = True
            rec["guideline_out"] = str(g_next)
            rec["decision_table_out"] = str(d_next)
            manifest["iterations"].append(rec)
            manifest["final_guideline"] = str(g_next)
            manifest["final_decision_table"] = str(d_next)
            flush_manifest()
            continue

        # (4) Select confusions to amend ---------------------------------------
        patterns = select_patterns(layer34, args.top_k, args.min_count,
                                    entity_only=(args.patterns == "entity"))
        rec["patterns"] = patterns
        if not patterns:
            logger.info("  No confusion ≥ %d× — loop cannot progress. Stopping.", args.min_count)
            rec["stopped"] = f"no confusion with count ≥ {args.min_count}"
            manifest["iterations"].append(rec)
            manifest["stopped"] = rec["stopped"]
            flush_manifest()
            break

        # (5) Draft amendments --------------------------------------------------
        draft = draft_amendments(args, patterns, records, g_i, amend_dir, today)
        rec["amendments_counts"] = draft["counts"]
        rec["amend_paths"] = draft["paths"]
        logger.info("  amendments: %s", draft["counts"])

        if draft["leaks"]:
            logger.error("  CORPUS LEAK in drafted guideline (%d) — NOT promoting G%d. Stopping.",
                         len(draft["leaks"]), i + 1)
            rec["stopped"] = f"corpus leak detected ({len(draft['leaks'])})"
            rec["leaks"] = draft["leaks"]
            manifest["iterations"].append(rec)
            manifest["stopped"] = rec["stopped"]
            flush_manifest()
            break

        # (6) Apply (append accepted only) → G_{i+1} and D_{i+1} ----------------
        if draft["counts"]["accepted"] == 0:
            logger.info("  0 amendments accepted — guideline/table cannot change. Stopping.")
            rec["stopped"] = "no amendments accepted"
            manifest["iterations"].append(rec)
            manifest["stopped"] = rec["stopped"]
            flush_manifest()
            break

        # Guideline: the amender already drafted G_i + accepted sections appended.
        shutil.copyfile(draft["drafted_guideline"], g_next)
        rec["guideline_out"] = str(g_next)

        # Decision table: inject each accepted decision_test into the Question
        # column of the rows for the two labels it disambiguates. patterns and
        # draft["amendments"] are 1:1 in order, so zip to recover the label pair.
        accepted_triples = [
            (p["annotator"], p["critic"], a.get("decision_test", ""))
            for p, a in zip(patterns, draft["amendments"])
            if a.get("status") == "accepted"
        ]
        n_inject = decision_table.write_amended_table(d_i, d_next, accepted_triples)
        rec["decision_table_out"] = str(d_next)
        rec["decision_test_injections"] = n_inject
        rec["accepted_decision_tests"] = [
            {"annotator": a, "critic": b, "decision_test": t} for a, b, t in accepted_triples
        ]

        # Fold this iteration's accepted tests into the cumulative coverage set
        # and remember the counts so the NEXT iteration's text-delta is correct.
        cumulative_triples.extend(accepted_triples)
        last_accepted = draft["counts"]["accepted"]
        last_injections = n_inject

        logger.info("  → G%d = %s (%d accepted amendment(s) appended)",
                    i + 1, g_next, draft["counts"]["accepted"])
        logger.info("  → D%d = %s (%d decision_test injection(s))", i + 1, d_next, n_inject)

        manifest["iterations"].append(rec)
        manifest["final_guideline"] = str(g_next)
        manifest["final_decision_table"] = str(d_next)
        flush_manifest()
    else:
        manifest["stopped"] = f"reached i_max ({args.i_max})"
        logger.info("Reached i_max (%d).", args.i_max)
        flush_manifest()

    # Spec §11.3 REPORT: friction trajectory ALONGSIDE the held-out F1 trajectory.
    if not final_decision.get("reason"):
        final_decision["reason"] = manifest.get("stopped")
    finalize_report()

    logger.info("Done. Final guideline:      %s", manifest["final_guideline"])
    logger.info("      Final decision table: %s", manifest["final_decision_table"])
    logger.info("Manifest: %s", manifest_path)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Cold-start guideline reconstruction loop (RQ-D).")
    p.add_argument("--g0", type=Path, default=None,
                   help="Cold-start guideline G0 (.md): entity type names + one-line defs only. "
                        "Omit to auto-generate from the schema (spec 11.1).")
    p.add_argument("--d0", type=Path, default=None,
                   help="Cold-start decision table D0 (.csv): LABEL + one-line Definition only, "
                        "blank Question/Examples. Omit to auto-generate from the schema.")
    working_group = p.add_mutually_exclusive_group(required=True)
    working_group.add_argument("--working-jsonl", type=Path, default=None,
                   help="Working-set sentences (JSONL with a 'sentence'/'text' field).")
    working_group.add_argument("--working-txt", type=Path, default=None,
                   help="Working-set sentences (plain text, one sentence per line).")
    p.add_argument("--out-dir", type=Path, required=True, help="Run output directory.")

    p.add_argument("--schema", type=Path, default=_DEFAULT_SCHEMA, help="Relation schema (.py/.json).")
    p.add_argument("--seeds", type=Path, default=_DEFAULT_SEEDS, help="Seed examples (.py/.json).")

    p.add_argument("--i-max", type=int, default=6, help="Hard iteration cap (spec 11.2).")
    p.add_argument("--top-k", type=int, default=5, help="Top confusions amended per iteration.")
    p.add_argument("--min-count", type=int, default=2,
                   help="Minimum confusion occurrences to amend (spec 11.2: ≥ 5).")
    p.add_argument("--patterns", choices=["entity", "all"], default="entity",
                   help="'entity' (default, spec 11.1) reconstructs entity-type "
                        "disambiguation only; 'all' also amends relation/scope confusions.")
    p.add_argument("--max-redrafts", type=int, default=2)
    p.add_argument("--examples-per-pattern", type=int, default=5)
    p.add_argument("--leak-ngram", type=int, default=7)

    p.add_argument("--annotator-model", type=str, default="qwen3-35B-vllm")
    p.add_argument("--critic-model", type=str, default="qwen3-35B-vllm")
    p.add_argument("--adjudicator-model", type=str, default="qwen3-35B-vllm")
    p.add_argument("--amender-model", choices=list(MODEL_ENDPOINTS), default="qwen3-35B-vllm")
    p.add_argument("--annotator-temp", type=float, default=0.7,
                   help="Annotator sampling temperature forwarded to the annotate "
                        "subprocess (default 0.7 — higher = more diverse annotations).")
    p.add_argument("--critic-temp", type=float, default=None,
                   help="Critic temperature (default: 0.3, or 0.5 with --strict-critic).")
    p.add_argument("--adjudicator-temp", type=float, default=None,
                   help="Adjudicator temperature (default 0.1).")
    p.add_argument("--max-rounds", type=int, default=2)
    p.add_argument("--num-sentences", type=int, default=None,
                   help="Limit working-set size (smoke tests).")

    p.add_argument("--guideline-search-backend", type=str, default="embedding",
                   choices=["lexical", "embedding"])
    p.add_argument("--guideline-search", choices=["mandatory", "optional"], default="optional")
    p.add_argument("--strict-critic", action="store_true")
    p.add_argument("--cold-start", action="store_true",
                   help="Use cold-start prompts in the annotation subprocess: agents "
                        "disambiguate from domain expertise + explicit reasoning instead of "
                        "citing the (still-scaffold) guideline verbatim. Recommended for early "
                        "iterations; overrides --strict-critic for the Critic's prompt.")
    p.add_argument("--tool-choice", type=str, default=None, choices=["auto", "required", "none"])
    p.add_argument("--precedent-memory", action="store_true",
                   help="Enable the lookup_precedent tool / precedent store for the annotation "
                        "subprocess (default: disabled — not currently used).")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--subprocess-retries", type=int, default=3,
                   help="Retries for a crashed annotation subprocess within a single "
                        "iteration (flaky endpoint, OOM, …). Each retry forces --resume "
                        "on the subprocess so already-annotated sentences aren't redone "
                        "(default 3; set 1 to disable retrying).")

    # ── Stopping rule (spec §11.3): dual friction + held-out F1 ──────────────
    stop = p.add_argument_group("stopping rule (spec §11.3)")
    held_out_group = stop.add_mutually_exclusive_group()
    held_out_group.add_argument("--held-out-jsonl", type=Path, default=None,
                      help="Held-out split sentences (JSONL) for the agent-vs-expert F1 "
                           "GUARD — MUST be distinct from --working-jsonl/--working-txt (§11.7).")
    held_out_group.add_argument("--held-out-txt", type=Path, default=None,
                      help="Held-out split sentences (plain text, one sentence per line) — "
                           "MUST be distinct from --working-jsonl/--working-txt (§11.7).")
    stop.add_argument("--held-out-human", type=Path, nargs="+", default=None,
                      help="Expert gold annotation file(s) for the held-out split "
                           "(native project JSON or per-annotator JSONL). Required for the guard.")
    stop.add_argument("--held-out-names", nargs="+", default=None,
                      help="Display names for --held-out-human files, in the same order.")
    stop.add_argument("--held-out-f1-mode", default="strict",
                      choices=["strict", "boundary", "text_type", "text_only"],
                      help="Matching mode for the held-out F1 (default: strict = offset+type).")
    stop.add_argument("--held-out-num-sentences", type=int, default=None,
                      help="Cap held-out size (smoke tests).")
    stop.add_argument("--stop-friction-eps", type=float, default=0.05,
                      help="PRIMARY: max relative reduction in R1 disagreements to count as "
                           "converged (spec §11.3: < 5%%).")
    stop.add_argument("--stop-friction-patience", type=int, default=2,
                      help="PRIMARY: consecutive sub-eps iterations required to stop (spec: 2).")
    stop.add_argument("--stop-f1-guard-patience", type=int, default=2,
                      help="GUARD: consecutive held-out-F1 decreases that trip the "
                           "echo-chamber guard (spec: 2).")
    stop.add_argument("--no-stopping-rule", action="store_true",
                      help="Log all §11.3 metrics but do NOT stop on them (run to i_max / "
                           "the existing hard stops). Useful for ablations.")
    stop.add_argument("--no-embedding-drift", action="store_true",
                      help="Skip the (monitoring-only) embedding-similarity drift in the text delta.")
    stop.add_argument("--expert-rules", type=Path, default=None,
                      help="Enumerated expert disambiguations (JSON/CSV) for the rule-coverage "
                           "curve (§11.5b, monitoring only). Each rule has a competing label pair.")
    stop.add_argument("--rule-coverage-semantic", action="store_true",
                      help="Also compute semantic coverage (embedding match of expert rule text "
                           "vs reconstructed decision_test), not just label-pair coverage.")
    stop.add_argument("--rule-coverage-threshold", type=float, default=0.5,
                      help="Cosine threshold for --rule-coverage-semantic (default 0.5).")

    p.add_argument("--python", type=str, default=sys.executable,
                   help="Python interpreter for the annotation subprocess.")
    p.add_argument("--resume", action="store_true",
                   help="Skip stages whose outputs already exist.")
    args = p.parse_args()

    if args.held_out_names and args.held_out_human and \
            len(args.held_out_names) != len(args.held_out_human):
        p.error("--held-out-names must match --held-out-human count")
    if (args.held_out_jsonl or args.held_out_txt) and not args.held_out_human:
        p.error("--held-out-jsonl/--held-out-txt requires --held-out-human "
                "(the expert gold for the guard)")

    run_loop(args)


if __name__ == "__main__":
    main()