#!/usr/bin/env python
"""align_per_molecule.py — cross-method-aligned per-molecule results, for runs made BEFORE or AFTER the ids fix
(or any mix), for the default and the HP-tuned (hp_final) phase.

For one dataset + protocol + phase, for each method and fold it: recovers which cleaned-CSV row every saved
prediction row is (the "id"), VERIFIES that mapping, averages the ensemble members BY MOLECULE, and writes
<out>/<method>_seed<fold>.csv with [id, error]. With --wilcoxon-dir it also writes per-method (per-target)
inputs for wilcoxon_paired.py: fold,id,smiles,y_true,y_pred. Only ids + errors / test statistics need leave
your side.

Which config a fold used (hp_final): read from results/<m>/<ds>/<proto>/best_hp.json — per-fold winners for
time-sliding splits, the single winner otherwise — exactly what the pipeline aggregated. Stale config dirs are
ignored.

How a member's row ids are recovered — candidates are accepted only if they VERIFY (the cleaned labels at the
candidate ids reproduce the member's own labels_test.npy, NaN-aware, atol 1e-6):
  1. ids_test.npy                       -> any method (written by the fixed pipeline)
  2. test_idx order                     -> rows already in split order
  3. chemprop replay                    -> chemprop2*/chemeleon*: the pandas intersection order train_one
                                           produced; several plausible pandas orderings are tried, and one is
                                           accepted only if every verifying ordering implies the same
                                           prediction for every molecule (run with the chemprop2 env python)
  4. molformer replay                   -> old token-budget order = argsort(token lengths), replayed with the
                                           real tokenizer in the molformer env (--molformer-python)
  5. label-value match                  -> last resort; usually refused on real data (labels repeat)

molformer needs one more decision. One code version (c7179db..26ade66) wrote labels in split order while its
predictions came out in loader order, and nothing in the files records which version ran. When molformer rows
could be either split or loader order, the script decides from the PREDICTIONS: it pairs them (test + saved
validation rows pooled) with the cleaned labels under both orders and keeps the one the model fits, using a
threshold that grows as rows get fewer. Constant predictions make the order irrelevant and are accepted. A
fresh ids_test.npy (written in the same run as the predictions) is trusted when the predictions cannot decide.
If it still cannot be decided, BOTH orders are exported (<wilcoxon-dir>/alt_order/) and
wilcoxon_from_results.py reports the more conservative result, flagged — never a guess.

Cross-checks printed per method (and used by wilcoxon_from_results.py to stop):
  * data check     : the pipeline's own computation redone from the raw files (members averaged by row position,
                     labels from the first member) must equal results/.../<phase>/_summary.json.
  * recovered check: the metric from the recovered rows must equal that too, UNLESS a correction was applied
                     (bug-2 molformer, members from different code versions); such cells are flagged because the
                     pipeline's reported number for them was computed on mispaired rows.

Usage (from the repo root):
  <chemprop2-env>/bin/python scripts/align_per_molecule.py --dataset caco2_time_sliding --protocol custom \
      --phase hp_final --out aligned_caco2 --wilcoxon-dir wilcoxon_inputs/caco2 \
      --molformer-python <molformer-env>/bin/python
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SMILES_JOIN = {"chemprop2", "chemprop2_nofp", "chemeleon", "chemeleon_nofp"}
ALL_METHODS = ["chemprop2", "chemprop2_nofp", "chemeleon", "chemeleon_nofp",
               "molclr", "molfcl", "motil", "molformer"]
REPLAY_MOLFORMER = Path(__file__).resolve().parent / "replay_molformer_ids.py"


def split_sub(protocol, seed):
    return "v1_det_seed0" if protocol == "v1_det" else f"{protocol}_seed{seed}"


def as2d(a):
    return a[:, None] if a.ndim == 1 else a


def verify(id_arr, labels, full, tcols):
    """Cleaned labels at these ids must reproduce the saved labels row-for-row."""
    if id_arr is None or len(id_arr) == 0 or (np.asarray(id_arr) < 0).any():
        return False
    ref = full.iloc[id_arr][tcols].to_numpy(float)
    lab = as2d(labels).astype(float)
    if ref.shape != lab.shape:
        return False
    both = ~(np.isnan(ref) | np.isnan(lab))
    return bool((np.isnan(ref) == np.isnan(lab)).all() and np.allclose(ref[both], lab[both], atol=1e-6))


def expand_by_smiles(smiles_sequence, test_idx, full):
    """train_one takes rows with .loc[common] on an index built from the split: each SMILES in `common` expands to
    every split row carrying it, in split order. Returns the global id of each written row."""
    positions = {}
    for g in test_idx:
        positions.setdefault(full["smiles"].iloc[int(g)], []).append(int(g))
    return np.array([g for s in smiles_sequence for g in positions.get(s, [])], dtype=np.int64)


def chemprop_candidate_orders(full, test_idx):
    """Row orders chemprop2/chemeleon could have written, depending on how the installed pandas orders
    Index.intersection(test rows in split order, prediction rows in cleaned order)."""
    test_index = full.iloc[test_idx].set_index("smiles").index
    pred_index = full.iloc[np.sort(test_idx)].set_index("smiles").index
    unique_in = lambda seq: list(dict.fromkeys(seq))
    return {
        "chemprop replay (installed pandas)": expand_by_smiles(test_index.intersection(pred_index), test_idx, full),
        "chemprop order: prediction rows (cleaned order)": expand_by_smiles(unique_in(pred_index), test_idx, full),
        "chemprop order: split rows": expand_by_smiles(unique_in(test_index), test_idx, full),
        "chemprop order: sorted SMILES": expand_by_smiles(sorted(set(test_index)), test_idx, full),
    }


_REPLAY_CACHE = {}


def replay_molformer(molformer_python, root, dataset, protocol, seed):
    """(test_ids, val_ids) in old token-budget loader order, or (None, None, error)."""
    key = (str(molformer_python), str(root), dataset, protocol, int(seed))
    if key not in _REPLAY_CACHE:
        with tempfile.TemporaryDirectory() as tmp:
            out, out_val = Path(tmp) / "ids.npy", Path(tmp) / "val_ids.npy"
            cmd = [molformer_python, str(REPLAY_MOLFORMER), "--root", str(root), "--dataset", dataset,
                   "--protocol", protocol, "--seed", str(seed), "--out", str(out), "--out-val", str(out_val)]
            res = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
            if res.returncode != 0 or not out.exists():
                tail = (res.stderr or res.stdout).strip().splitlines()[-1:] or ["replay failed"]
                _REPLAY_CACHE[key] = (None, None, tail[0])
            else:
                _REPLAY_CACHE[key] = (np.load(out), np.load(out_val) if out_val.exists() else None, None)
    return _REPLAY_CACHE[key]


def label_match_ids(labels, full, tcols, test_idx):
    ref = full.iloc[test_idx][tcols].to_numpy(float)
    key_to_pos = {}
    for j, row in enumerate(ref):
        key_to_pos.setdefault(tuple(np.round(row, 6).tolist()), []).append(j)
    out, ambiguous = [], 0
    for row in as2d(labels).astype(float):
        cand = key_to_pos.get(tuple(np.round(row, 6).tolist()), [])
        if len(cand) == 1:
            out.append(int(test_idx[cand[0]]))
        else:
            ambiguous += 1
            out.append(-1)
    return np.asarray(out, np.int64), ambiguous


def pairing_score(pred, truth):
    """|Spearman| between predictions and truth, averaged over targets (rank-based: regression values and
    class probabilities alike). A model scores high only under the correct row pairing."""
    pred2, truth2 = as2d(pred).astype(float), as2d(truth).astype(float)
    scores = []
    for t in range(truth2.shape[1]):
        frame = pd.DataFrame({"p": pred2[:, t], "y": truth2[:, t]}).dropna()
        if len(frame) > 2 and frame["y"].nunique() > 1 and frame["p"].nunique() > 1:
            scores.append(abs(frame["p"].corr(frame["y"], method="spearman")))
    return float(np.nanmean(scores)) if scores else float("nan")


class Member:
    """One ensemble member's saved outputs."""

    def __init__(self, path, tcols):
        self.path = path
        self.pred = as2d(np.load(path / "pred_test.npy")).astype(float)
        if self.pred.shape[1] != len(tcols) and len(tcols) == 1 and self.pred.shape[1] == 2:
            self.pred = self.pred[:, 1:2]                  # 2-column class scores -> positive-class column
        self.labels = np.load(path / "labels_test.npy")
        self.ids = np.load(path / "ids_test.npy") if (path / "ids_test.npy").exists() else None
        self.ids_fresh = self.ids is not None and \
            os.path.getmtime(path / "ids_test.npy") >= os.path.getmtime(path / "pred_test.npy") - 2
        has_val = (path / "pred_val.npy").exists() and (path / "labels_val.npy").exists()
        self.pred_val = as2d(np.load(path / "pred_val.npy")).astype(float) if has_val else None
        if self.pred_val is not None and self.pred_val.shape[1] == 2 and len(tcols) == 1:
            self.pred_val = self.pred_val[:, 1:2]
        self.labels_val = np.load(path / "labels_val.npy") if has_val else None


def molformer_decision(member, candidate, candidate_val, how, ctx):
    """Is molformer's prediction order `candidate` (split order) or the replayed loader order? Returns a list of
    (ids, how, use_cleaned) options: one if decided, two if it cannot be decided (both exported)."""
    full, tcols = ctx["full"], ctx["tcols"]
    if not ctx["molformer_python"]:
        return None, ("molformer rows look like split order, but without --molformer-python the old token-budget "
                      "order cannot be ruled out — pass --molformer-python")
    replay_test, replay_val, err = replay_molformer(ctx["molformer_python"], ctx["root"], ctx["dataset"],
                                                    ctx["protocol"], ctx["seed"])
    if replay_test is None:
        return None, f"molformer replay failed ({err})"
    if np.array_equal(replay_test, candidate):
        return [(candidate, how, False)], None             # the two orders coincide
    pred_rows, cand_truth, loader_truth = [member.pred], [full.iloc[candidate][tcols]], [full.iloc[replay_test][tcols]]
    if member.pred_val is not None and candidate_val is not None and replay_val is not None \
            and len(member.pred_val) == len(candidate_val) == len(replay_val):
        pred_rows.append(member.pred_val)                  # validation rows share the test rows' code version
        cand_truth.append(full.iloc[candidate_val][tcols])
        loader_truth.append(full.iloc[replay_val][tcols])
    pred = np.vstack(pred_rows)
    cand_score = pairing_score(pred, pd.concat(cand_truth).to_numpy(float))
    loader_score = pairing_score(pred, pd.concat(loader_truth).to_numpy(float))
    n = len(pred)
    margin = max(0.1, 3.5 * np.sqrt(2.0 / n))              # grows as rows get fewer (noise ~ 1/sqrt(n))
    fit = f"[fit split={cand_score:.2f} vs loader={loader_score:.2f}, margin {margin:.2f}, n={n}]"
    if np.isnan(cand_score) and np.isnan(loader_score):
        return [(candidate, f"{how} (constant predictions: order irrelevant)", True)], None
    if cand_score - loader_score > margin:
        return [(candidate, f"{how} {fit}", False)], None
    if loader_score - cand_score > margin:
        return [(replay_test, f"molformer replay — predictions in loader order, labels/ids not (pre-26ade66 or "
                              f"stale ids); errors use cleaned labels {fit}", True)], None
    if how.startswith("ids_test.npy") and member.ids_fresh:
        return [(candidate, f"{how} (written with the predictions; predictions cannot decide {fit})", False)], None
    return [(candidate, f"AMBIGUOUS split order {fit}", True),
            (replay_test, f"AMBIGUOUS loader order {fit}", True)], None


def recover_member(method, member, ctx):
    """Returns (options, reason): options = [(ids, how, use_cleaned)], two entries if molformer is ambiguous."""
    full, tcols, test_idx, val_idx = ctx["full"], ctx["tcols"], ctx["test_idx"], ctx["val_idx"]
    labels = member.labels
    n = as2d(labels).shape[0]
    tried = []
    if member.ids is not None and len(member.ids) == n:
        ids = np.asarray(member.ids, np.int64)
        if verify(ids, labels, full, tcols):
            if method == "molformer":
                return molformer_decision(member, ids, val_idx, "ids_test.npy", ctx)
            return [(ids, "ids_test.npy", False)], None
        tried.append("ids_test.npy failed verify")
    if test_idx is None:
        return None, "missing test_idx.npy"
    if len(test_idx) == n and verify(test_idx, labels, full, tcols):
        if method == "molformer":
            return molformer_decision(member, test_idx, val_idx, "test_idx (in split order)", ctx)
        return [(test_idx, "test_idx (in split order)", False)], None
    tried.append("not in split order")

    if method in SMILES_JOIN:
        verified = {name: ids for name, ids in chemprop_candidate_orders(full, test_idx).items()
                    if len(ids) == n and verify(ids, labels, full, tcols)}
        if verified:
            implied = [dict(zip(ids.tolist(), map(tuple, np.round(member.pred, 12)))) for ids in verified.values()]
            if all(d == implied[0] for d in implied[1:]):
                name, ids = next(iter(verified.items()))
                return [(ids, name, False)], None
            tried.append(f"chemprop orderings disagree ({', '.join(verified)}) — cannot tell which pandas ordering ran")
        else:
            tried.append("no chemprop ordering reproduces the labels (rows genuinely out of order, or data changed)")

    if method == "molformer":
        if ctx["molformer_python"]:
            replay_test, _, err = replay_molformer(ctx["molformer_python"], ctx["root"], ctx["dataset"],
                                                   ctx["protocol"], ctx["seed"])
            if replay_test is not None and len(replay_test) == n and verify(replay_test, labels, full, tcols):
                return [(replay_test, "molformer replay", False)], None
            tried.append(f"molformer replay did not reproduce the labels ({err or 'verify'})")
        else:
            tried.append("molformer replay skipped: pass --molformer-python <molformer-env>/bin/python")

    matched, ambiguous = label_match_ids(labels, full, tcols, test_idx)
    if ambiguous == 0 and verify(matched, labels, full, tcols):
        return [(matched, "label-value match", False)], None
    tried.append(f"label match ambiguous on {ambiguous}/{n} rows")
    return None, "; ".join(tried)


def headline_metric(meta):
    if meta.get("source") == "tdc" and meta.get("metric"):
        name = meta["metric"].strip().lower().replace("_", "-")
        return {"auc": "roc-auc", "auroc": "roc-auc", "auprc": "pr-auc", "average-precision": "pr-auc"}.get(name, name)
    return "roc-auc" if meta["task_type"] == "cls" else "rmse"


def score(metric, y, p):
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score, roc_auc_score
    keep = ~(np.isnan(y) | np.isnan(p))
    y, p = y[keep], p[keep]
    if metric == "rmse":
        return float(np.sqrt(np.mean((y - p) ** 2)))
    if metric == "mae":
        return float(np.mean(np.abs(y - p)))
    if metric == "spearman":
        return float(spearmanr(y, p).statistic)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p) if metric == "roc-auc" else average_precision_score(y, p))


def cell_metric(metric, truth, pred):
    return float(np.nanmean([score(metric, truth[:, t], pred[:, t]) for t in range(truth.shape[1])]))


def align_cell(method, member_dirs, ctx):
    """Recover every member separately (a cell can mix code versions), align members BY MOLECULE, average.
    Returns (status, detail, result) with result = {"main": (ids, mean_pred), "alt": ... or None,
    "corrected": bool, "raw_metric": pipeline-style metric from the raw files}."""
    members = [Member(d, ctx["tcols"]) for d in member_dirs if (d / "pred_test.npy").exists()]
    if not members:
        return "NOT ALIGNED", "no pred_test.npy", None
    for m in members:
        if m.pred.shape[1] != len(ctx["tcols"]):
            return "NOT ALIGNED", f"{m.path.name}: pred has {m.pred.shape[1]} columns for {len(ctx['tcols'])} target(s)", None
    choices, hows = [], []
    for m in members:
        options, reason = recover_member(method, m, ctx)
        if options is None:
            return "NOT ALIGNED", f"{m.path.name}: {reason}", None
        for ids, _, _ in options:
            if len(np.unique(ids)) != len(ids):
                return "NOT ALIGNED", f"{m.path.name}: recovered ids are not unique", None
        choices.append(options)
        hows.append(options[0][1].split(" — ")[0].split(" [")[0].split(" (")[0])

    def build(pick):
        reference = np.sort(choices[0][pick if len(choices[0]) > 1 else 0][0])
        stacked = []
        for m, options in zip(members, choices):
            ids = options[pick if len(options) > 1 else 0][0]
            if not np.array_equal(np.sort(ids), reference):
                return None
            row_of = {int(g): r for r, g in enumerate(ids)}
            stacked.append(m.pred[[row_of[int(g)] for g in reference]])
        return reference, np.mean(np.stack(stacked, 0), 0)

    main = build(0)
    if main is None:
        return "NOT ALIGNED", "members cover different molecules", None
    ambiguous = any(len(o) > 1 for o in choices)
    alt = build(1) if ambiguous else None
    corrected = not ambiguous and (any(o[0][2] for o in choices) or
                                   len({tuple(o[0][0]) for o in choices}) > 1)   # labels fixed / members in different orders
    # what the pipeline itself computed: members averaged by ROW POSITION, labels from the first member
    raw_pred = np.mean(np.stack([m.pred for m in members], 0), 0)
    raw_metric = cell_metric(ctx["metric"], as2d(members[0].labels).astype(float), raw_pred)
    reference, mean_pred = main
    truth = ctx["full"].iloc[reference][ctx["tcols"]].to_numpy(float)
    error = np.nanmean(np.abs(mean_pred - truth), axis=1)
    pd.DataFrame({"id": reference, "error": error}).to_csv(ctx["out_dir"] / f"{method}_seed{ctx['seed']}.csv", index=False)
    counts = {h: hows.count(h) for h in dict.fromkeys(hows)}
    status = "AMBIGUOUS" if ambiguous else "ALIGNED"
    detail = f"{len(members)} members [" + ", ".join(f"{c}x {h}" for h, c in counts.items()) + f"] ({len(reference)} rows)"
    if ambiguous:
        detail += " — molformer order undecidable from predictions: both orders exported, conservative result reported"
    return status, detail, {"main": main, "alt": alt, "corrected": corrected, "ambiguous": ambiguous,
                            "raw_metric": raw_metric}


def write_wilcoxon_inputs(cells_by_method, ctx_base, wdir):
    """Per (method, target): fold,id,smiles,y_true,y_pred sorted by (fold, id); alternative-order files for
    undecidable molformer cells go to wdir/alt_order/. Also the two cross-checks against _summary.json."""
    full, tcols, metric = ctx_base["full"], ctx_base["tcols"], ctx_base["metric"]
    wdir.mkdir(parents=True, exist_ok=True)
    lines = []

    def table(by_seed, key, t):
        rows = []
        for seed in sorted(by_seed):
            result = by_seed[seed]
            ids, mean_pred = result[key] if result[key] is not None else result["main"]
            rows.append(pd.DataFrame({"fold": seed, "id": ids, "smiles": full["smiles"].iloc[ids].to_numpy(),
                                      "y_true": full.iloc[ids][tcols[t]].to_numpy(float),
                                      "y_pred": mean_pred[:, t]}).dropna(subset=["y_true", "y_pred"]))
        return pd.concat(rows).sort_values(["fold", "id"])

    for method, by_seed in cells_by_method.items():
        has_alt = any(r["alt"] is not None for r in by_seed.values())
        recovered = {}
        for t, target in enumerate(tcols):
            suffix = "" if len(tcols) == 1 else f"__{target}"
            main_table = table(by_seed, "main", t)
            main_table.to_csv(wdir / f"{method}{suffix}.csv", index=False)
            if has_alt:
                (wdir / "alt_order").mkdir(exist_ok=True)
                table(by_seed, "alt", t).to_csv(wdir / "alt_order" / f"{method}{suffix}.csv", index=False)
            for seed, g in main_table.groupby("fold"):
                recovered.setdefault(seed, []).append(score(metric, g.y_true.to_numpy(float), g.y_pred.to_numpy(float)))
        seeds = sorted(by_seed)
        recovered = [float(np.nanmean(recovered[s])) for s in seeds]
        raw = [by_seed[s]["raw_metric"] for s in seeds]
        corrected = [s for s in seeds if by_seed[s]["corrected"]]
        undecided = [s for s in seeds if by_seed[s]["ambiguous"]]
        summary = ctx_base["root"] / "results" / method / ctx_base["dataset"] / ctx_base["protocol"] / \
            ctx_base["phase"] / "_summary.json"
        verdict = "no _summary.json to check against"
        if summary.exists():
            reported = json.loads(summary.read_text()).get("per_seed_am") or []
            if len(reported) == len(seeds):
                data_ok = all(abs(a - b) < 1e-6 for a, b in zip(raw, reported) if b is not None and a == a)
                unexplained = [s for s, a, b in zip(seeds, recovered, reported)
                               if s not in corrected and s not in undecided and b is not None and a == a
                               and abs(a - b) >= 1e-6]
                if not data_ok:
                    verdict = f"DATA MISMATCH: files do not reproduce _summary.json {reported} — stop"
                elif unexplained:
                    verdict = f"UNEXPLAINED difference on folds {unexplained} vs _summary.json — stop"
                elif corrected:
                    verdict = (f"matches _summary.json except CORRECTED folds {corrected}: the pipeline's number for "
                               f"those was computed on mispaired rows (reported {[round(reported[seeds.index(s)], 4) for s in corrected]}"
                               f" -> recovered {[round(recovered[seeds.index(s)], 4) for s in corrected]})")
                else:
                    verdict = "matches _summary.json exactly"
                if undecided:
                    verdict += (f"; order-UNDECIDABLE folds {undecided} (molformer order not decidable from its "
                                f"predictions: both orders exported, not checked against _summary.json)")
            else:
                verdict = f"_summary.json has {len(reported)} folds, recovered {len(seeds)} — not compared"
        lines.append(f"  {method:16s} {metric} per fold " + ", ".join(f"s{s}={v:.4f}" for s, v in zip(seeds, recovered))
                     + f"  [{verdict}]" + ("  [has alt_order]" if has_alt else ""))
    return lines


def seed_config_dirs(base, phase_root, phase, config):
    """{seed: config dir} — from best_hp.json for hp_final (what the pipeline aggregated), else the unique dir
    holding each seed. Returns (mapping, problem)."""
    best = phase_root / "best_hp.json"
    if phase == "hp_final" and best.exists() and not config:
        rec = json.loads(best.read_text())
        if rec.get("per_fold"):
            mapping = {int(s): base / w["id"] for s, w in rec["folds"].items()}
        else:
            chosen = base / rec["id"]
            mapping = {int(p.name.split("seed")[1].split("_em")[0]): chosen for p in chosen.glob("seed*_em*")}
        missing = [s for s, d in mapping.items() if not any(d.glob(f"seed{s}_em*"))]
        return ({s: d for s, d in mapping.items() if s not in missing},
                f"best_hp.json names configs with no results for folds {missing}" if missing else None)
    dirs = [p for p in base.iterdir() if p.is_dir() and not p.name.startswith("_") and (not config or p.name == config)]
    holders = {}
    for d in dirs:
        for p in d.glob("seed*_em*"):
            holders.setdefault(int(p.name.split("seed")[1].split("_em")[0]), set()).add(d)
    clash = sorted(s for s, h in holders.items() if len(h) > 1)
    if clash:
        return {}, f"folds {clash} appear under several config dirs and no best_hp.json says which — pass --config"
    return {s: next(iter(h)) for s, h in holders.items()}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--protocol", default="custom")
    ap.add_argument("--phase", default="default", help="default or hp_final")
    ap.add_argument("--config", default=None, help="force one config dir name")
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS)
    ap.add_argument("--seed", type=int, default=None, help="one fold/seed; default = every fold found")
    ap.add_argument("--root", default=".", help="repo root (has cleaned/ splits/ results/); default cwd")
    ap.add_argument("--molformer-python", default=None, help="molformer env python, to replay old molformer order")
    ap.add_argument("--out", default="aligned_out")
    ap.add_argument("--wilcoxon-dir", default=None,
                    help="also write per-(method, target) inputs for wilcoxon_paired.py here (stays on your side)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    meta = json.loads((root / "cleaned" / f"{args.dataset}.meta.json").read_text())
    full = pd.read_csv(root / "cleaned" / f"{args.dataset}.csv")
    tcols = meta.get("target_columns") or [c for c in full.columns if c != "smiles"]
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx_base = {"root": root, "dataset": args.dataset, "protocol": args.protocol, "phase": args.phase,
                "full": full, "tcols": tcols, "metric": headline_metric(meta), "out_dir": out_dir,
                "molformer_python": args.molformer_python}

    report, cells_by_method = [], {}
    for method in args.methods:
        phase_root = root / "results" / method / args.dataset / args.protocol
        base = phase_root / args.phase
        if not base.exists():
            report.append((method, "-", "SKIP", "no results dir")); continue
        mapping, problem = seed_config_dirs(base, phase_root, args.phase, args.config)
        if problem:
            report.append((method, "-", "NOT ALIGNED", problem))
        for seed in sorted(mapping) if args.seed is None else [args.seed]:
            if seed not in mapping:
                report.append((method, seed, "NOT ALIGNED", "no config dir for this fold")); continue
            sub = root / "splits" / args.dataset / split_sub(args.protocol, seed)
            ctx = {**ctx_base, "seed": seed,
                   "test_idx": np.asarray(np.load(sub / "test_idx.npy"), np.int64) if (sub / "test_idx.npy").exists() else None,
                   "val_idx": np.asarray(np.load(sub / "val_idx.npy"), np.int64) if (sub / "val_idx.npy").exists() else None}
            status, detail, result = align_cell(method, sorted(mapping[seed].glob(f"seed{seed}_em*")), ctx)
            report.append((method, seed, status, f"{mapping[seed].name}: {detail}"))
            if result is not None:
                cells_by_method.setdefault(method, {})[seed] = result

    print(f"\nAlignment report — {args.dataset} / {args.protocol} / {args.phase}\n" + "-" * 78)
    print(f"{'method':<16}{'seed':>5}  {'status':<12} detail")
    for method, seed, status, detail in report:
        print(f"{method:<16}{str(seed):>5}  {status:<12} {detail}")
    print("-" * 78)
    print(f"{sum(r[2] in ('ALIGNED', 'AMBIGUOUS') for r in report)} usable cell(s) -> {out_dir}/  (columns: id, error)")
    if args.wilcoxon_dir:
        lines = write_wilcoxon_inputs(cells_by_method, ctx_base, root / args.wilcoxon_dir)
        print(f"\nWilcoxon inputs -> {root / args.wilcoxon_dir}/  (headline metric: {ctx_base['metric']}); checks:")
        print("\n".join(lines))
    bad = [f"{m}/seed{s}" for m, s, st, _ in report if st == "NOT ALIGNED"]
    if bad:
        print("NOT aligned (excluded, re-run with the current code): " + ", ".join(bad))


if __name__ == "__main__":
    main()
