"""Run TextCtrl's released ABINet over every generated image."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    for name in ("repository", "checkpoint", "manifest", "predictions"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--overwrite", action="store_true"); args = parser.parse_args()
    repository = Path(args.repository).resolve(); os.chdir(str(repository)); sys.path.insert(0,str(repository))
    import torch, torchvision.transforms as transforms
    from omegaconf import OmegaConf
    from PIL import Image
    from tqdm import tqdm
    from src.module.abinet import ABINetIterModel, CharsetMapper, postprocess
    if not torch.cuda.is_available(): raise RuntimeError("ABINet OCR requires CUDA")
    jobs = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    output_path = Path(args.predictions); output_path.parent.mkdir(parents=True,exist_ok=True)
    completed = {}
    if output_path.is_file() and not args.overwrite:
        completed = {int(row["index"]):row for row in
                     (json.loads(line) for line in output_path.read_text().splitlines() if line.strip())}
    if args.overwrite: output_path.write_text("",encoding="utf-8")
    config = OmegaConf.load("configs/inference.yaml").model.params.base_config.ocr_model
    model = ABINetIterModel(config).cuda(); model.load_state_dict(torch.load(args.checkpoint,map_location="cpu")); model.eval()
    charset = CharsetMapper(filename=str(config.charset_path),max_length=int(config.max_length)+1)
    resize, to_tensor = transforms.Resize([int(config.height),int(config.width)]),transforms.ToTensor()
    mode = "a" if output_path.is_file() and not args.overwrite else "w"
    with output_path.open(mode,encoding="utf-8",buffering=1) as output:
        for job in tqdm(jobs,desc="ABINet OCR"):
            index=int(job["index"])
            if index in completed: continue
            image_path=Path(job["output_path"])
            if not image_path.is_file(): raise FileNotFoundError(image_path)
            with Image.open(image_path) as image:
                value=resize(to_tensor(image.convert("RGB"))).unsqueeze(0).cuda()
            with torch.no_grad(): prediction=postprocess(model(value,mode="test"),charset,"alignment")[0][0]
            row={"index":index,"ocr_predicted_text":prediction,"output_path":str(image_path)}
            output.write(json.dumps(row,sort_keys=True)+"\n"); completed[index]=row


if __name__ == "__main__": main()
