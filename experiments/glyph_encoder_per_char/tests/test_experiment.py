from pathlib import Path
from experiments.glyph_encoder_per_char.data import expand_jobs
from experiments.glyph_encoder_per_char.metrics import position_scores,text_metrics
from experiments.glyph_encoder_per_char.report import evaluate

def test_paired_design(tmp_path):
    samples=[{"sample_index":i,"target_text":"abcde"} for i in range(50)]
    jobs=expand_jobs(samples,{"seed":42,"masking_proportions":[0,.1,.3,.5,.7]},tmp_path)
    assert len(jobs)==1250
    assert all(len([x for x in jobs if x["sample_index"]==i])==25 for i in range(50))
    zero=[x for x in jobs if x["masking_proportion"]==0]
    assert len(zero)==250 and {x["masked_character_index"] for x in zero}==set(range(5))

def test_metrics():
    assert text_metrics("abcde","abcde")=={"ACC":1.0,"NED":1.0,"CER":0.0}
    assert position_scores("abcde","abXde")==[1,1,0,1,1]

def test_evaluation_character_columns(tmp_path):
    jobs=expand_jobs([{"sample_index":0,"target_text":"abcde"}],{"seed":1,"masking_proportions":[0]},tmp_path)
    rows=evaluate(jobs,{x["index"]:{"ocr_predicted_text":"abXde"} for x in jobs})
    row=rows[2]
    assert row["masked_character_correct"]==0
    assert [row[f"other_character_{i}_correct"] for i in range(1,5)]==[1,1,1,1]
