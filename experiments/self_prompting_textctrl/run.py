"""Orchestrate preparation, both generators, OCR, reporting, and collages."""
from __future__ import annotations
import json, os, subprocess, sys
from collections.abc import Mapping
from pathlib import Path
from omegaconf import OmegaConf
from .collage import render_case_collage, render_special_collage
from .data import expand_jobs, prepare_samples, read_jsonl, write_jsonl
from .report import evaluate, write_reports

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def resolve(value):
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def command_environment(repository=None):
    environment = os.environ.copy()
    paths = [str(ROOT)]
    if repository is not None: paths.append(str(repository))
    if environment.get("PYTHONPATH"): paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    return environment


def run_subprocess(command, cwd=None, repository=None):
    subprocess.run([str(value) for value in command], cwd=cwd or ROOT,
                   env=command_environment(repository), check=True)


def selected_models(mode):
    if mode == "both": return ("regular", "self_prompting")
    if mode in ("regular", "self_prompting"): return (mode,)
    raise ValueError("mode must be one of: both, regular, self_prompting")


def main(argv=None):
    config = OmegaConf.load(HERE/"config.yaml")
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(sys.argv[1:] if argv is None else argv)))
    cfg = OmegaConf.to_container(config, resolve=True)
    if not isinstance(cfg, Mapping): raise TypeError("Configuration must be a mapping")
    output = resolve(cfg["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    models = selected_models(str(cfg["mode"]))
    output_samples_path = output/"samples.jsonl"
    reuse_path = cfg.get("samples_path")
    if reuse_path not in (None, ""):
        source_samples_path = resolve(reuse_path)
        if not source_samples_path.is_file(): raise FileNotFoundError(source_samples_path)
        samples = read_jsonl(source_samples_path)
        write_jsonl(output_samples_path, samples)
    else:
        samples = (prepare_samples(cfg,ROOT,output) if bool(cfg["overwrite"]) or not output_samples_path.is_file()
                   else read_jsonl(output_samples_path))
    jobs = expand_jobs(samples,cfg,output,models); manifest=output/"jobs.jsonl"; write_jsonl(manifest,jobs)
    expected = int(cfg["sample_count"])*8*len(models)
    if len(samples)!=int(cfg["sample_count"]) or len(jobs)!=expected:
        raise RuntimeError(f"Expected {cfg['sample_count']} samples/{expected} jobs; got {len(samples)}/{len(jobs)}")
    stage=str(cfg["stage"])
    if stage=="prepare":
        print(json.dumps({"mode":cfg["mode"],"samples":len(samples),"jobs":len(jobs),"output_dir":str(output)},indent=2)); return
    regular=cfg["regular"]; repository=resolve(regular["repository_dir"])
    if "regular" in models and stage in ("regular","all"):
        command=[regular["python_executable"],HERE/"regular_worker.py","--repository",repository,
                 "--checkpoint",resolve(regular["checkpoint_path"]),"--manifest",manifest,
                 "--starting-layer",regular["starting_layer"],"--num-inference-steps",regular["num_inference_steps"],
                 "--guidance-scale",regular["guidance_scale"]]
        if cfg["overwrite"]: command.append("--overwrite")
        run_subprocess(command,cwd=repository,repository=repository)
    self_cfg=cfg["self_prompting"]
    if "self_prompting" in models and stage in ("self_prompting","all"):
        checkpoint=resolve(self_cfg["checkpoint_path"]); unet_config=checkpoint/"unet"/"config.json"
        if not unet_config.is_file():
            raise FileNotFoundError(f"Self-prompting checkpoint must contain unet/config.json: {checkpoint}")
        command=[self_cfg["python_executable"],HERE/"self_prompting_worker.py","--manifest",manifest,
                 "--checkpoint",checkpoint,"--vae-path",resolve(self_cfg["vae_path"]),
                 "--unet-path",resolve(self_cfg["unet_path"]),"--scheduler-path",resolve(self_cfg["scheduler_path"]),
                 "--text-model-path",self_cfg["text_model_path"],"--font-path",resolve(self_cfg["font_path"]),
                 "--image-size",self_cfg["image_size"],"--max-text-length",self_cfg["max_text_length"],
                 "--num-inference-steps",self_cfg["num_inference_steps"],"--guidance-scale",self_cfg["guidance_scale"]]
        if self_cfg.get("revision") is not None: command.extend(["--revision",self_cfg["revision"]])
        if cfg["overwrite"]: command.append("--overwrite")
        run_subprocess(command)
    predictions_path=output/"ocr_predictions.jsonl"; ocr=cfg["ocr"]
    if stage in ("ocr","all"):
        command=[ocr["python_executable"],HERE/"ocr_worker.py","--repository",resolve(ocr["repository_dir"]),
                 "--checkpoint",resolve(ocr["checkpoint_path"]),"--manifest",manifest,
                 "--predictions",predictions_path]
        if cfg["overwrite"]: command.append("--overwrite")
        run_subprocess(command,cwd=resolve(ocr["repository_dir"]),repository=resolve(ocr["repository_dir"]))
    if stage in ("report","all"):
        if not predictions_path.is_file(): raise FileNotFoundError(predictions_path)
        predictions={int(row["index"]):row for row in read_jsonl(predictions_path)}
        if len(predictions)!=len(jobs): raise RuntimeError(f"Incomplete OCR: {len(predictions)}/{len(jobs)}")
        rows=evaluate(jobs,predictions,case_sensitive=bool(cfg["metrics"]["case_sensitive"]))
        write_reports(rows,output)
        render_case_collage(samples,rows,output/"capital_lowercase_collage.png",cfg["collage"])
        render_special_collage(samples,rows,output/"special_character_collage.png",cfg["collage"])
        OmegaConf.save(config,output/"config.yaml")
    print(json.dumps({"mode":cfg["mode"],"stage":stage,"samples":len(samples),"jobs":len(jobs),"output_dir":str(output)},indent=2))


if __name__=="__main__": main()
