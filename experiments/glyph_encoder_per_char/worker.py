"""GPU worker; runs in TextCtrl's pinned Python environment."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

def args(argv=None):
    p=argparse.ArgumentParser()
    for name in ("repository","checkpoint","ocr-checkpoint","manifest","predictions"): p.add_argument("--"+name, required=True)
    p.add_argument("--starting-layer",type=int,default=10); p.add_argument("--num-inference-steps",type=int,default=50)
    p.add_argument("--guidance-scale",type=float,default=2); p.add_argument("--overwrite",action="store_true")
    return p.parse_args(argv)

def read(path):
    with Path(path).open(encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]

def mask_token(embedding, position, proportion, seed):
    import torch
    if not 0 <= proportion <= 1: raise ValueError("masking proportion must be in [0,1]")
    result=embedding.clone(); count=round(proportion*result.shape[-1])
    if count:
        generator=torch.Generator(device=result.device).manual_seed(seed)
        indices=torch.randperm(result.shape[-1],generator=generator,device=result.device)[:count]
        result[:,position,indices]=0
    return result

def generate(pipe, image, source, target, job, steps, layer, guidance):
    import torch, torchvision.transforms as transforms
    from src.MuSA.GaMuSA import glyph_cosine_similarity, prepare_label
    from src.MuSA.utils import MuSA_TextCtrl, regiter_attention_editor_diffusers_Edit
    with torch.no_grad():
        torch.manual_seed(int(job["generation_seed"])); torch.cuda.manual_seed_all(int(job["generation_seed"]))
        start,_=pipe.inversion(image,image,[source],guidance_scale=guidance,num_inference_steps=steps,return_intermediates=True)
        prompts=[source,target]; clean=pipe.model.get_text_conditioning(prompts)
        clean[1:2]=mask_token(clean[1:2],int(job["masked_character_index"]),float(job["masking_proportion"]),int(job["glyph_mask_seed"]))
        embeddings=torch.cat([pipe.model.get_text_conditioning(["",""]),clean],0)
        latents=start.expand(2,-1,-1,-1).clone(); pipe.scheduler.set_timesteps(steps)
        ids,_=prepare_label(prompts,pipe.charset,pipe.max_length,pipe.device)
        controller=MuSA_TextCtrl(24,layer); regiter_attention_editor_diffusers_Edit(pipe.unet,controller); controller.start_ctrl()
        try:
            for index,timestep in enumerate(pipe.scheduler.timesteps):
                model_input=torch.cat([latents]*2); hint=torch.cat([image.expand(2,-1,-1,-1)]*2)
                control=pipe.control_model(hint,model_input,timestep,embeddings)
                prediction=pipe.unet(x=model_input,timestep=timestep,encoder_hidden_states=embeddings,control=control).sample
                uncond,cond=prediction.chunk(2); latents,_=pipe.step(uncond+guidance*(cond-uncond),timestep,latents)
                if (index+1)%5==0:
                    monitored=transforms.Resize([32,128])(pipe.latent2image_grad(latents))
                    controller.reset_alpha(glyph_cosine_similarity(pipe.monitor(monitored),ids))
        finally: controller.reset_ctrl(); controller.reset()
        return pipe.latent2image_grad(latents)[1].clamp(0,1)

def main(argv=None):
    a=args(argv); repository=Path(a.repository).resolve(); os.chdir(repository); sys.path.insert(0,str(repository))
    import numpy as np, torch, torchvision.transforms as transforms
    from omegaconf import OmegaConf
    from PIL import Image
    from tqdm import tqdm
    from inference import load_image
    from src.MuSA.GaMuSA import GaMuSA
    from src.module.abinet import ABINetIterModel, CharsetMapper, postprocess
    from utils import create_model, load_state_dict
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    jobs=read(a.manifest); path=Path(a.predictions); path.parent.mkdir(parents=True,exist_ok=True)
    completed={} if a.overwrite or not path.is_file() else {int(x["index"]):x for x in read(path)}
    if a.overwrite: path.write_text("",encoding="utf-8")
    model=create_model("configs/inference.yaml").cuda(); model.load_state_dict(load_state_dict(a.checkpoint),strict=False); model.eval()
    pipe=GaMuSA(model,{"max_length":25,"loss_weight":1.,"attention":"position","backbone":"transformer","backbone_ln":3,"checkpoint":"weights/vision_model.pth","charset_path":"src/module/abinet/data/charset_36.txt"})
    cfg=OmegaConf.load("configs/inference.yaml").model.params.base_config.ocr_model
    ocr=ABINetIterModel(cfg).cuda(); ocr.load_state_dict(torch.load(a.ocr_checkpoint,map_location="cpu")); ocr.eval()
    charset=CharsetMapper(filename=str(cfg.charset_path),max_length=int(cfg.max_length)+1); resize=transforms.Resize([int(cfg.height),int(cfg.width)]); tensor=transforms.ToTensor()
    with path.open("a",encoding="utf-8",buffering=1) as output:
        for job in tqdm(jobs,desc="Character masking"):
            index=int(job["index"]); destination=Path(job["output_path"])
            if a.overwrite or not destination.is_file():
                generated=generate(pipe,load_image(job["input_path"]),job["source_text"],job["target_text"],job,a.num_inference_steps,a.starting_layer,a.guidance_scale)
                array=np.clip(generated.cpu().permute(1,2,0).numpy()*255,0,255).astype(np.uint8)
                with Image.open(job["input_path"]) as opened: size=opened.size
                destination.parent.mkdir(parents=True,exist_ok=True); Image.fromarray(array).resize(size,Image.Resampling.BICUBIC).save(destination)
            if not a.overwrite and index in completed: continue
            with Image.open(destination) as opened: inp=resize(tensor(opened.convert("RGB"))).unsqueeze(0).cuda()
            with torch.no_grad(): prediction=postprocess(ocr(inp,mode="test"),charset,"alignment")[0][0]
            row={"index":index,"ocr_predicted_text":prediction,"output_path":str(destination)}; output.write(json.dumps(row,sort_keys=True)+"\n"); completed[index]=row
if __name__=="__main__": main()
