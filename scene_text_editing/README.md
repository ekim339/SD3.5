### Design a Text to Image Editing repository:

1. Reference Code: 

2. Reference Code: https://github.com/weichaozeng/TextCtrl

3. Key Features to import:

- Keep hydra configs structure for network, diffusion. Note that my repository is only for inference on SDXL. Adapt this for finetuning if required.
- Introduce a new folder for dataset and capture the dataset that the reference code has used. Update networks to support any models that are used in the reference code.
- I would like you to define a folder called "tasks" inside configs. Why? Because I want to define a new task of image editing based of text image editing in the reference code. This will formulate the task of image editing.
*USE HYDRA
- The dataset would capture images that I will modify and the prompts I will use to modify them. If needed, defined datasets/ as a subfolder of tasks as each task will have its unique dataset. Correspondingly, create a 'prompts' folder which will define the prompt needed for editing.

Here is a folder structure:
"tasks"/
  "text_image_editing/"
    "datasets/"
      "dataset_a"
      "dataset_b"
    "prompts/"
      "prompt_set_a"
      "prompt_set_b"

- If there are pretrained models available, I'd like you to download them and support them during inference.

### Support for flowmatching

If there's flow matching support that in diffusion through a separate yaml file (called flow_matching.yaml)

