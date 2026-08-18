from __future__ import annotations
import json,os,subprocess
from collections.abc import Mapping
from pathlib import Path
from omegaconf import OmegaConf
from .collage import render_collages
from .data import expand_jobs,prepare_samples,read_jsonl,write_jsonl
from .report import evaluate,write_reports
ROOT=Path(__file__).resolve().parents[2]
def resolve(value):
    path=Path(str(value)).expanduser(); return path.resolve() if path.is_absolute() else (ROOT/path).resolve()
def main(argv=None):
    import sys
    cfg=OmegaConf.load(Path(__file__).with_name("configs")/"experiment.yaml"); cfg=OmegaConf.merge(cfg,OmegaConf.from_dotlist(list(sys.argv[1:] if argv is None else argv))); config=OmegaConf.to_container(cfg,resolve=True)
    if not isinstance(config,Mapping):raise TypeError("configuration must be a mapping")
    out=resolve(config["output_dir"]); out.mkdir(parents=True,exist_ok=True); samples_file=out/"samples.jsonl"
    samples=prepare_samples(config,ROOT,out) if config["overwrite"] or not samples_file.is_file() else read_jsonl(samples_file)
    jobs=expand_jobs(samples,config,out); write_jsonl(out/"jobs.jsonl",jobs); expected=int(config["sample_count"])*25
    if len(jobs)!=expected:raise RuntimeError(f"Expected {expected} jobs, got {len(jobs)}")
    if config["stage"]=="prepare":print(json.dumps({"samples":len(samples),"jobs":len(jobs),"output_dir":str(out)},indent=2));return
    repository=resolve(config["textctrl"]["repository_dir"]); predictions=out/"predictions.jsonl"
    command=[str(config["textctrl"]["python_executable"]),str(Path(__file__).with_name("worker.py").resolve()),"--repository",str(repository),"--checkpoint",str(resolve(config["textctrl"]["checkpoint_path"])),"--ocr-checkpoint",str(resolve(config["ocr"]["checkpoint_path"])),"--manifest",str(out/"jobs.jsonl"),"--predictions",str(predictions),"--starting-layer",str(config["textctrl"]["starting_layer"]),"--num-inference-steps",str(config["textctrl"]["num_inference_steps"]),"--guidance-scale",str(config["textctrl"]["guidance_scale"])]
    if config["overwrite"]:command.append("--overwrite")
    env=os.environ.copy();env["PYTHONPATH"]=os.pathsep.join(filter(None,(str(ROOT),str(repository),env.get("PYTHONPATH",""))));subprocess.run(command,cwd=repository,env=env,check=True)
    predicted={int(x["index"]):x for x in read_jsonl(predictions)}
    if len(predicted)!=len(jobs):raise RuntimeError(f"Incomplete predictions: {len(predicted)}/{len(jobs)}")
    rows=evaluate(jobs,predicted,bool(config["metrics"]["case_sensitive"]));write_reports(rows,out);render_collages(samples,rows,out/"collages",config["collage"]);(out/"config.yaml").write_text(OmegaConf.to_yaml(cfg,resolve=True),encoding="utf-8")
if __name__=="__main__":main()
