import argparse
import json, os, tqdm, torch

from JDiffusion.pipelines import StableDiffusionPipeline

parser = argparse.ArgumentParser(description='video-audio Dataset Config')
parser.add_argument('--mid', type=int, default=7, help='')
parser.add_argument('--choice', type=int, default=0,help='choose the range')
parser.add_argument('--output', type=str, default='./output')
parser.add_argument('--style_path', type=str, default='style')
parser.add_argument('--inference_step', type=int, default=25)
parser.add_argument('--draw_times', type=int, default=5)
parser.add_argument('--prefix', type=str, default='A')
parser.add_argument('--postfix', type=str, default='')
parser.add_argument('--guided_scale', type=float, default=7.5)


config = parser.parse_args()

max_num = 3
dataset_root = "/media/php/code/Jittor/generater/data_B"
output_root = f'{config.output}'

style_id_list = [[0,config.mid],[config.mid,max_num]]
left = style_id_list[config.choice][0]
right = style_id_list[config.choice][1]
left = 0
right = 28
prefix = config.prefix
negative_prompt  = "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry"
#test: load -> prompt -> 
each_p_nums = config.draw_times
with torch.no_grad():
    for tempid in tqdm.tqdm(range(left, right)):
        taskid = "{:0>2d}".format(tempid)
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to("cuda")
        print(pipe.scheduler)
        # pipe.scheduler = DPMSolverMultistepScheduler.
        # load json
        with open(f"{dataset_root}/{taskid}/prompt_p.json", "r") as file:
            prompts = json.load(file)
        if  config.postfix!='':
            # letter = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']
            # postfix = f"[{letters[tempid]}*]"
            postfix = " in style_{}".format(taskid) 
            print('postfix:',postfix)
            pipe.load_lora_weights(f"{config.style_path}/style_{taskid}")
        else:
            postfix = ""
            print('postfix:', postfix)
            pipe.load_lora_weights(f"{config.style_path}/style_{taskid}")
            
        print('prompt: {} xxx{}'.format(prefix,postfix))
        
        for id, prompt in prompts.items():
            # image = pipe(prompt + f" in style_{taskid}", num_inference_steps=config.inference_step, width=512, height=512).images[0]
            images = pipe(prefix +' '+ prompt + postfix, num_images_per_prompt = each_p_nums,num_inference_steps=config.inference_step, width=512, height=512, negative_prompt=negative_prompt, guidance_scale=config.guided_scale).images #7.5 -> 15
            os.makedirs(f"{output_root}/{taskid}", exist_ok=True)
            for idx in range(each_p_nums):
                image = images[idx]
                if idx==0:
                    image.save("{0}/{1}/{2}.png".format(output_root,taskid,prompt.split(' ')[0]))  
                else:
                    image.save("{0}/{1}/{2}_{3}.png".format(output_root,taskid,prompt.split(' ')[0],idx))
