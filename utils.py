
import os, json, gc, time, tempfile, warnings, textwrap, re
import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
warnings.filterwarnings("ignore")

# ── Frame sampler ─────────────────────────────────────────────────────────────
try:
    from decord import VideoReader, cpu as dcpu
    _DECORD_OK = True
except ImportError:
    _DECORD_OK = False

def sample_frames(video_path: str, n_frames: int = 8) -> list:
    """Uniformly sample n_frames PIL Images (RGB 224x224) from a video file."""
    SIZE = (224, 224)
    if _DECORD_OK:
        try:
            vr = VideoReader(video_path, ctx=dcpu(0))
            total = len(vr)
            if total == 0: raise ValueError()
            idx = np.linspace(0, total-1, n_frames, dtype=int)
            return [Image.fromarray(f).convert("RGB").resize(SIZE)
                    for f in vr.get_batch(idx).asnumpy()]
        except Exception:
            pass
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return [Image.fromarray(np.zeros((224,224,3),dtype=np.uint8))] * n_frames
    want  = set(np.linspace(0, total-1, n_frames, dtype=int))
    frames, i = [], 0
    while cap.isOpened() and len(frames) < n_frames:
        ret, fr = cap.read()
        if not ret: break
        if i in want:
            frames.append(
                Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).resize(SIZE))
        i += 1
    cap.release()
    while len(frames) < n_frames:
        frames.append(Image.fromarray(np.zeros((224,224,3),dtype=np.uint8)))
    return frames

# ── Dataset ───────────────────────────────────────────────────────────────────
class VideoCaptionDataset(Dataset):
    """Generic video captioning dataset for MSVD and MSR-VTT."""
    def __init__(self, video_ids, video_dir, captions, n_frames=8):
        self.ids = [v for v in video_ids
                    if os.path.exists(os.path.join(video_dir,f"{v}.mp4"))
                    and v in captions and len(captions[v])>0]
        self.video_dir = video_dir
        self.captions  = captions
        self.n_frames  = n_frames
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        v = self.ids[i]
        return {"video_id": v,
                "frames":   sample_frames(
                    os.path.join(self.video_dir,f"{v}.mp4"), self.n_frames),
                "references": self.captions[v]}

# ── Evaluator ─────────────────────────────────────────────────────────────────
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

class CaptionEvaluator:
    """Wraps pycocoevalcap: BLEU-1/4, METEOR, CIDEr, ROUGE-L."""
    def __init__(self):  self._h={}; self._r={}
    def add(self,v,h):   self._h[v]=h.strip()
    def add_refs(self,v,rs): self._r[v]=[r.strip() for r in rs if r.strip()]
    def add_refs(self,v,rs):
        self._r[v] = [
            " ".join(r) if isinstance(r, list) else str(r).strip()
            for r in rs
        ]
    def evaluate(self):
        common = set(self._h) & set(self._r)
        if not common:
            return {m:0 for m in ["BLEU_1","BLEU_4","METEOR","CIDEr","ROUGE_L"]}
        aid, imgs, anns, res = 0, [], [], []
        for v in sorted(common):
            imgs.append({"id":v})
            for r in self._r[v]:
                anns.append({"image_id":v,"caption":r,"id":aid}); aid+=1
            res.append({"image_id":v,"caption":self._h[v]})
        with tempfile.TemporaryDirectory() as tmp:
            ap=os.path.join(tmp,"a.json"); rp=os.path.join(tmp,"r.json")
            with open(ap,"w") as f: json.dump({"images":imgs,"annotations":anns},f)
            with open(rp,"w") as f: json.dump(res,f)
            coco=COCO(ap); cr=coco.loadRes(rp)
            ev=COCOEvalCap(coco,cr)
            ev.params["image_id"]=cr.getImgIds(); ev.evaluate()
        s={m:round(v,4) for m,v in ev.eval.items()}
        return {"BLEU_1":s.get("Bleu_1",0),"BLEU_4":s.get("Bleu_4",0),
                "METEOR":s.get("METEOR",0),"CIDEr":s.get("CIDEr",0),
                "ROUGE_L":s.get("ROUGE_L",0)}

# ── Per-video scores ──────────────────────────────────────────────────────────
def compute_per_video_scores(hypotheses: dict, references: dict) -> dict:
    """
    Compute per-video CIDEr, BLEU-4, METEOR, ROUGE-L.
    Returns { video_id: { metric: score } }
    Uses a singleton evaluator per video — expensive but complete.
    """
    ev  = CaptionEvaluator()
    out = {}
    common = set(hypotheses) & set(references)
    for vid in tqdm(sorted(common), desc="Per-video scores", ncols=80, leave=False):
        ev.reset()
        ev.add(vid, hypotheses[vid])
        ev.add_refs(vid, references[vid])
        out[vid] = ev.evaluate()
    return out

# ── Full evaluation (returns ALL artifacts) ───────────────────────────────────
def run_evaluation_full(model_fn, dataset, label="Model",
                        max_samples=None, compute_per_video=True):
    """
    Complete evaluation loop.  Returns a dict with ALL artifacts needed
    for the analysis notebook — no reruns required downstream.

    Returns:
      {
        "scores"          : { BLEU_1, BLEU_4, METEOR, CIDEr, ROUGE_L },
        "hypotheses"      : { video_id: hypothesis_string },
        "references"      : { video_id: [ref1, ref2, ...] },
        "examples"        : [ (video_id, hypothesis, ref_0), ... ],
        "per_video_scores": { video_id: { metric: score } },  # if requested
        "timing"          : { "total_s": float, "mean_s": float,
                               "std_s": float, "per_video_s": [float,...] }
      }
    """
    ev = CaptionEvaluator()
    n  = min(len(dataset), max_samples or len(dataset))
    hypotheses, references, examples, times = {}, {}, [], []

    for i in tqdm(range(n), desc=label, ncols=90):
        item = dataset[i]
        vid  = item["video_id"]
        t0   = time.perf_counter()
        with torch.no_grad():
            hyp = model_fn(item["frames"])
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        hypotheses[vid] = hyp
        references[vid] = item["references"]
        ev.add(vid, hyp)
        ev.add_refs(vid, item["references"])
        examples.append((vid, hyp,
                         item["references"][0] if item["references"] else ""))

    scores = ev.evaluate()
    per_video = (compute_per_video_scores(hypotheses, references)
                 if compute_per_video else {})
    timing = {
        "total_s" : round(sum(times), 3),
        "mean_s"  : round(float(np.mean(times)), 4),
        "std_s"   : round(float(np.std(times)),  4),
        "per_video_s": [round(t, 4) for t in times],
    }
    return {"scores": scores, "hypotheses": hypotheses, "references": references,
            "examples": examples, "per_video_scores": per_video, "timing": timing}

# ── Save all artifacts ────────────────────────────────────────────────────────
def save_all_artifacts(result: dict, model_name: str,
                       dataset_name: str, results_dir: str):
    """
    Persist every artifact from run_evaluation_full to disk.

    Files written (all under results_dir):
      <tag>_scores.json          – aggregate metric scores
      <tag>_hypotheses.json      – { video_id: hypothesis }
      <tag>_references.json      – { video_id: [refs] }
      <tag>_examples.csv         – video_id, hypothesis, reference_0
      <tag>_per_video_scores.csv – video_id, BLEU_1, BLEU_4, METEOR, CIDEr, ROUGE_L
      <tag>_timing.json          – inference timing statistics
    """
    import pandas as pd
    safe = (model_name.replace("+","_plus_").replace("-","_")
                      .replace(" ","_").replace("/","_"))
    tag  = f"{safe}_{dataset_name}"
    saved = []

    def _save(obj, fname):
        p = os.path.join(results_dir, fname)
        if fname.endswith(".json"):
            with open(p,"w") as f:
                json.dump(obj, f, indent=2)
        else:
            obj.to_csv(p, index=False)
        saved.append(p)

    # Scores
    _save({"model":model_name,"dataset":dataset_name,
           "scores":result["scores"]}, f"{tag}_scores.json")

    # Hypotheses & references
    _save(result["hypotheses"], f"{tag}_hypotheses.json")
    _save(result["references"], f"{tag}_references.json")

    # Examples CSV
    df_ex = pd.DataFrame(result["examples"],
                         columns=["video_id","hypothesis","reference_0"])
    _save(df_ex, f"{tag}_examples.csv")

    # Per-video scores CSV
    if result.get("per_video_scores"):
        rows = [{"video_id":vid, **s}
                for vid,s in result["per_video_scores"].items()]
        _save(pd.DataFrame(rows), f"{tag}_per_video_scores.csv")

    # Timing
    _save({"model":model_name,"dataset":dataset_name,
           **result["timing"]}, f"{tag}_timing.json")

    for p in saved:
        print(f"    saved {os.path.basename(p)}")
    return saved

# ── Ablation runner ───────────────────────────────────────────────────────────
def run_frame_ablation(model_factory, model_name, dataset_name,
                       video_dir, captions, ok_ids, results_dir,
                       frame_counts=(4,8,16), max_samples=150):
    """
    Run frame-count ablation and save to disk.
    model_factory: callable() -> model instance
    Returns dict { "N frames": { scores + timing } }
    """
    ev      = CaptionEvaluator()
    results = {}
    safe    = (model_name.replace("+","_plus_").replace("-","_")
                         .replace(" ","_").replace("/","_"))

    for nf in frame_counts:
        ds    = VideoCaptionDataset(ok_ids[:max_samples], video_dir,
                                    captions, n_frames=nf)
        model = model_factory()
        if hasattr(model, "eval"): model.eval()
        result = run_evaluation_full(model, ds,
                                     label=f"{model_name} ({nf}f)",
                                     max_samples=max_samples,
                                     compute_per_video=False)
        results[f"{nf} frames"] = {
            "scores": result["scores"],
            "timing": result["timing"],
        }
        print(f"  {nf}f  CIDEr={result['scores']['CIDEr']:.4f}"
              f"  BLEU-4={result['scores']['BLEU_4']:.4f}"
              f"  mean={result['timing']['mean_s']:.3f}s/video")
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Save
    out = {"model":model_name,"dataset":dataset_name,"results":results}
    p   = os.path.join(results_dir,
                       f"{safe}_{dataset_name}_ablation_frames.json")
    with open(p,"w") as f: json.dump(out, f, indent=2)
    print(f"  saved {os.path.basename(p)}")
    return results

# ── Display helpers ───────────────────────────────────────────────────────────
def show_scores(scores, model_name, dataset_name):
    print(f"\n── {model_name} on {dataset_name} " + "─"*30)
    for k,v in scores.items(): print(f"  {k:<10}: {v:.4f}")

def show_examples(examples, n=5):
    wrap = lambda s: "\n               ".join(textwrap.wrap(str(s),52))
    print("\nQualitative examples:")
    print("-"*68)
    for vid,hyp,ref in examples[:n]:
        print(f"  Video : {vid}\n  Ref   : {wrap(ref)}\n  Gen   : {wrap(hyp)}\n")
