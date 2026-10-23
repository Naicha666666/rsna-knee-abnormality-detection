"""32-concurrency Knee MRI Report Annotation using local vLLM Gemma 4 12B.
Extracts 12 abnormalities with (p, conf) and writes to labels_gemma4_soft.parquet.
Resumable via JSONL checkpointing.
"""
import os
import sys
import time
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

RUBRIC = """You grade knee MRI radiology reports for twelve findings.
Grade the report in whatever language it arrives in. Do not translate.

For EACH of the twelve findings, return two numbers:
  "p"    probability the finding is present in THIS knee, 0.0 to 1.0
  "conf" how much the report actually tells you about that finding, 0.0 to 1.0

The twelve findings:
  ACL              anterior cruciate ligament tear/rupture
  MCL              medial collateral ligament injury
  Medial Meniscus  medial meniscus tear
  Lateral Meniscus lateral meniscus tear
  Medial OA        medial compartment osteoarthritis / cartilage loss
  Lateral OA       lateral compartment osteoarthritis / cartilage loss
  PF OA            patellofemoral osteoarthritis / chondromalacia patellae
  Effusion         joint effusion / fluid in the joint
  Synovitis        synovitis, synovial thickening or proliferation
  Baker's          Baker's (popliteal) cyst
  Contusion        bone contusion / bone bruise / marrow oedema from trauma
  Fracture         fracture of any bone in the study

Return ONLY a valid JSON object matching this exact key structure:
{"ACL": {"p": ..., "conf": ...}, "MCL": {"p": ..., "conf": ...}, ...}
No prose, no explanations, no markdown formatting."""

KEY_MAP = {
    "baker s": "Baker's",
    "bakers": "Baker's",
    "baker's": "Baker's",
    "pf_oa": "PF OA",
    "pf oa": "PF OA",
    "medial_oa": "Medial OA",
    "medial oa": "Medial OA",
    "lateral_oa": "Lateral OA",
    "lateral oa": "Lateral OA",
    "medial_meniscus": "Medial Meniscus",
    "medial meniscus": "Medial Meniscus",
    "lateral_meniscus": "Lateral Meniscus",
    "lateral meniscus": "Lateral Meniscus",
}

def normalize_keys(d):
    out = {}
    for k, v in d.items():
        k_clean = k.strip().lower()
        target = KEY_MAP.get(k_clean, k.strip())
        out[target] = v
    return out

def label_one_report(uid, report_text, endpoint="http://127.0.0.1:8000/v1/chat/completions", max_retries=3):
    payload = {
        "model": "google/gemma-4-12B-it",
        "messages": [
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": f"<report>\n{report_text}\n</report>"}
        ],
        "temperature": 0.0,
        "max_tokens": 500
    }
    data_bytes = json.dumps(payload).encode("utf-8")
    
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                endpoint,
                data=data_bytes,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=40) as resp:
                res = json.loads(resp.read().decode("utf-8"))
            content = res["choices"][0]["message"]["content"].strip()
            
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON found in response: {content[:100]}")
            
            raw_dict = json.loads(match.group(0))
            norm_dict = normalize_keys(raw_dict)
            
            rec = {"StudyInstanceUID": uid}
            for lab in LABELS:
                item = norm_dict.get(lab, {})
                rec[lab] = float(item.get("p", 0.05))
                rec[lab + "__conf"] = float(item.get("conf", 0.05))
            return rec
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[ERROR] Failed {uid} after {max_retries} attempts: {e}", flush=True)
                rec = {"StudyInstanceUID": uid}
                for lab in LABELS:
                    rec[lab] = 0.05
                    rec[lab + "__conf"] = 0.05
                return rec
            time.sleep(0.5 * (attempt + 1))

def save_parquet(records, out_parquet):
    df = pd.DataFrame(records)
    df.to_parquet(out_parquet, index=False)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--train-csv", type=str, default="/home/eason/rsna-knee/data/raw/train.csv")
    parser.add_argument("--out-dir", type=str, default="/home/eason/rsna-knee/data/labels")
    parser.add_argument("--endpoint", type=str, default="http://127.0.0.1:8000/v1/chat/completions")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, "labels_gemma4_soft.jsonl")
    out_parquet = os.path.join(args.out_dir, "labels_gemma4_soft.parquet")

    finished = {}
    if os.path.exists(out_jsonl):
        with open(out_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        row = json.loads(line)
                        finished[row["StudyInstanceUID"]] = row
                    except:
                        pass
        print(f"Resuming: found {len(finished)} already completed records.", flush=True)

    tr = pd.read_csv(args.train_csv)
    all_reports = [(str(u), str(r) if pd.notna(r) else "") for u, r in zip(tr["StudyInstanceUID"], tr["Report"])]
    if args.limit:
        all_reports = all_reports[:args.limit]

    pending = [item for item in all_reports if item[0] not in finished]
    print(f"Total reports: {len(all_reports)} | Already finished: {len(finished)} | Pending: {len(pending)}", flush=True)

    if not pending:
        print("All reports already finished! Writing parquet...", flush=True)
        save_parquet(list(finished.values()), out_parquet)
        print(f"Done. Parquet saved to {out_parquet}", flush=True)
        return

    records = list(finished.values())
    t0 = time.time()
    n_completed = 0
    total_pending = len(pending)

    with open(out_jsonl, "a", encoding="utf-8") as f_jsonl:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(label_one_report, uid, rep, args.endpoint): uid for uid, rep in pending}
            for fut in as_completed(futures):
                rec = fut.result()
                records.append(rec)
                f_jsonl.write(json.dumps(rec) + "\n")
                f_jsonl.flush()
                n_completed += 1

                if n_completed % 25 == 0 or n_completed == total_pending:
                    elapsed = time.time() - t0
                    speed = n_completed / elapsed
                    eta = (total_pending - n_completed) / speed if speed > 0 else 0
                    pct = n_completed / total_pending * 100
                    print(f"[{n_completed}/{total_pending} ({pct:.1f}%)] Speed: {speed:.1f} rep/s | Elapsed: {elapsed:.0f}s | ETA: {eta/60:.1f}m", flush=True)
                    if n_completed % 100 == 0 or n_completed == total_pending:
                        save_parquet(records, out_parquet)

    save_parquet(records, out_parquet)
    total_time = time.time() - t0
    print(f"\nALL COMPLETED in {total_time:.1f}s ({total_time/60:.2f}m)!", flush=True)
    print(f"Final output: {out_parquet} with shape ({len(records)}, {len(records[0])})", flush=True)

if __name__ == "__main__":
    main()
