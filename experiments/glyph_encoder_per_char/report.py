from __future__ import annotations
import csv
from collections import defaultdict
from .metrics import mean_std, position_scores, text_metrics

def evaluate(jobs,predictions,case_sensitive=False):
    rows=[]
    for job in jobs:
        prediction=str(predictions[int(job["index"])]["ocr_predicted_text"]); scores=position_scores(job["target_text"],prediction,case_sensitive)
        masked=int(job["masked_character_index"]); others=[x for i,x in enumerate(scores) if i!=masked]
        rows.append({**job,"ocr_predicted_text":prediction,**text_metrics(job["target_text"],prediction,case_sensitive),
            "masked_character_correct":scores[masked],**{f"other_character_{i+1}_correct":x for i,x in enumerate(others)}})
    return rows

def cell(rows,key):
    mean,std=mean_std([float(x[key]) for x in rows]); return f"{mean:.6f}/{std:.6f}"

def write_reports(rows,output_dir):
    fields=["index","sample_index","filename","source_text","target_text","input_noise_standard_deviation","input_noise_seed","masking_proportion","masked_character_index","masked_character","glyph_mask_seed","generation_seed","ocr_predicted_text","ACC","NED","CER","masked_character_correct",*[f"other_character_{i}_correct" for i in range(1,5)],"input_path","output_path"]
    with (output_dir/"detailed_results.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows({k:x.get(k) for k in fields} for x in rows)
    groups=defaultdict(list)
    for row in rows: groups[(float(row["masking_proportion"]),int(row["masked_character_index"]))].append(row)
    with (output_dir/"summary.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f); writer.writerow(["masking_proportion","ACC_mean/std","NED_mean/std","CER_mean/std","masked_character_accuracy_mean/std",*[f"other_character_{i}_accuracy_mean/std" for i in range(1,5)],"masked_character_index"])
        for (proportion,position),members in sorted(groups.items()):
            writer.writerow([proportion,*[cell(members,x) for x in ("ACC","NED","CER","masked_character_correct")],*[cell(members,f"other_character_{i}_correct") for i in range(1,5)],position])
