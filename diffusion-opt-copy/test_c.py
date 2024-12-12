
import pandas as pd
import math
from typing import List, Tuple, Any
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from scheduling_euler_discrete import EulerDiscreteScheduler
from datasets import load_dataset
import argparse
from PIL import Image
from torchvision import models, transforms
from torch.nn.functional import adaptive_avg_pool2d
import numpy as np
from scipy import linalg
import skimage.metrics
import os
from transformers import CLIPProcessor, CLIPModel
import queue
from queue import Queue
import faiss
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
import time

parser = argparse.ArgumentParser(description="Stable diffusion test")
parser.add_argument("--bs", type=int, help="Batch size", default=0)
parser.add_argument("--np", type=bool, help="Enable negative prompt or not", default=False)
parser.add_argument("--eps", type=float, help="Enable negative prompt or not", default=0.3)


# def latents_to_rgb(latents):
#     weights = (
#         (60, -60, 25, -70),
#         (60,  -5, 15, -50),
#         (60,  10, -5, -35)
#     )

#     weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
#     biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
#     rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
#     image_array = rgb_tensor.clamp(0, 255)[0].byte().cpu().numpy()
#     image_array = image_array.transpose(1, 2, 0)

#     return Image.fromarray(image_array)

# def decode_tensors(pipe, step, timestep, callback_kwargs):
#     latents = callback_kwargs["latents"]
#     image = latents_to_rgb(latents)
#     image.save(f"{step}.png")
#     return callback_kwargs

if torch.cuda.is_available():
    print(f"GPU is available. Number of GPUs: {torch.cuda.device_count()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    print("GPU is not available.")
    
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


def get_clip_score(images, text):

    inputs = processor(text=text, images=images, return_tensors="pt", truncation=True, padding=True,max_length=77)
    # print(inputs)

    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # print(outputs)
 
    logits_per_image = outputs.logits_per_image.max(dim=1)[0]
    # print(logits_per_image, logits_per_image.shape)  # 1,4
    # probs = logits_per_image.softmax(dim=1)
    # mean_score = torch.mean(logits_per_image,dim=0)
    # print(f"average CLIP:{mean_score}")
    return logits_per_image


# Define a function to calculate FID between two sets of images
def calculate_fid(image1, image2):
    # Load the pre-trained InceptionV3 model
    model = models.inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()  # Use the pool3 layer features
    model.eval()

    # Define the preprocessing transformations (resize to 299x299 for InceptionV3)
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def get_features(image):
        # Preprocess the image and add a batch dimension
        image = preprocess(image).unsqueeze(0)

        # Extract features using the InceptionV3 model
        with torch.no_grad():
            features = model(image)
        return features.squeeze(0).numpy()

    # Get the features of both images
    features1 = get_features(image1)
    features2 = get_features(image2)
    print(features1)
    print(features2)

    # Compute FID (Fréchet Inception Distance)
    mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)
    sigma1 = np.array([[sigma1]])  # Convert scalar to 1x1 matrix

    sigma2 = np.array([[sigma2]])  # Convert scalar to 1x1 matrix
    print(sigma1)  # Should be (n_features, n_features)
    print(sigma2)  # Should be (n_features, n_features)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # If sqrtm produces complex numbers due to numerical errors, take only the real part
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid =  np.dot(diff, diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def clip_img_score(images):

    images = processor(images=images, return_tensors="pt").to(device)
    print(images['pixel_values'])
    # Calculate the embeddings for the samples_images using the CLIP model
    with torch.no_grad():

        embedding = model.get_image_features(**images)

    print(embedding.shape)
    # Calculate the cosine similarity between the embeddings

    similarity_score = torch.nn.functional.cosine_similarity(embedding[0], embedding[1], dim=0)

    return similarity_score.item()

def clip_text_score(text1, text2):
    # Process the text prompts to get their embeddings
    texts = processor(text=[text1, text2], return_tensors="pt", padding=True).to(device)

    # Calculate the embeddings for the text prompts using the CLIP model
    with torch.no_grad():
        embedding = model.get_text_features(**texts)

    # print(embedding.shape)  # Print shape for debugging

    # Calculate the cosine similarity between the embeddings
    similarity_score = torch.nn.functional.cosine_similarity(embedding[0], embedding[1], dim=0)
    
    # Return the similarity score; a higher score indicates more similarity
    return similarity_score.item()

def callback(pipe, step_index, timestep, output_type, dtype, prompt,callback_kwargs):
    latents = callback_kwargs.get("latents")
    
    with torch.no_grad():
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
        ]
        image, has_nsfw_concept = pipe.run_safety_checker(image, pipe._execution_device, dtype)
        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = pipe.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # image1 = np.array(image[0])  # Convert first image to numpy array
        # image2 = np.array(image[1])  # Convert second image to numpy array
        # print(image1.shape)
        # print(image2.shape)

        # ssim_value = skimage.metrics.structural_similarity(image1, image2, channel_axis = 2)
        # cosine = clip_img_score(image)
        # print(f"ssim iteartion {step_index}:", ssim_value)
        
        # print(f"cosine iteartion {step_index}:", cosine)
        
        clip_score = get_clip_score(image[1],prompts[1])
        print(f"clip score: {clip_score.item()}")
        # image = pipe.numpy_to_pil(image)[0]
        # print("images:", image)
        # print(f"ssim iteartion {step_index}:", ssim_value)
        # fid_score = calculate_fid(image[0], image[1])
        # print(f"FID Score iteration {step_index}: {fid_score}")
        
        # store the generated image after each iteration
        # for idx in range(len(image)):
        #     image[idx].save(f"./imgs/{prompt[idx].replace(' ', '_')}_{step_index}.png")
        
    return callback_kwargs


def precompute_timesteps_for_labels(scheduler, labels, device):
    timesteps = []
    for label in labels:
        if label == 0 or label == 2:
            # For label 0, use full 50 iterations
            scheduler.set_timesteps(num_inference_steps=50, device=device)
            timesteps.append(scheduler.timesteps.tolist())
        elif label == 1:
            # For label 1, use last 40 iterations
            scheduler.set_timesteps(num_inference_steps=50, device=device)
            timesteps.append(scheduler.timesteps[-40:].tolist())         
        else:
            timesteps.append([])  

    return timesteps


def create_full_path(example):
    """Create full path to image using `base_path` to COCO2017 folder."""
    example["image_path"] = os.path.join(PATH_TO_IMAGE_FOLDER, example["file_name"])
    return example

def request_scheduler(queues: List[Queue], current_latents: Any, cached_latents: Any, timesteps_batch: Any, last_labels: List[int], last_requests: List[str], last_queue_nums: List[int],
                        last_prompt_embeds: Any,  last_add_text_embeds: Any, last_add_time_ids: Any,
                      maximum_bs: int, generator: torch.Generator, pipe: StableDiffusionXLPipeline, negative_prompts: List[str], scheduler, queue_tail: int):
    
    num_last_requests = len(last_queue_nums)
    requests = [] 
    labels = []
    
    new_requests_r = []
    new_labels_r = []
    
    new_requests_co = []
    new_labels_co = []
    
    queue_num = []
    queue_num_r = []
    queue_num_co = []
    
    if num_last_requests:
        last_prompt_embeds_n, last_prompt_embeds_p = torch.chunk(last_prompt_embeds, 2, dim=0)
        last_add_text_embeds_n, last_add_text_embeds_p = torch.chunk(last_add_text_embeds, 2, dim=0)
        # print("cached latent size:",cached_latents.shape )
        new_cached_latents = torch.empty((0, *cached_latents.shape[1:]), dtype=cached_latents.dtype, device=cached_latents.device)
        # print("new cached latent size:",new_cached_latents.shape )
        new_current_latents = torch.empty((0, *current_latents.shape[1:]), dtype=current_latents.dtype, device=current_latents.device)
        new_timesteps_batch = []
        new_prompt_embeds_p = torch.empty((0, *last_prompt_embeds.shape[1:]), dtype=last_prompt_embeds.dtype, device=last_prompt_embeds.device)
        new_add_text_embeds_p = torch.empty((0, *last_add_text_embeds.shape[1:]), dtype=last_add_text_embeds.dtype, device=last_add_text_embeds.device)
        new_prompt_embeds_n = torch.empty((0, *last_prompt_embeds.shape[1:]), dtype=last_prompt_embeds.dtype, device=last_prompt_embeds.device)
        new_add_text_embeds_n = torch.empty((0, *last_add_text_embeds.shape[1:]), dtype=last_add_text_embeds.dtype, device=last_add_text_embeds.device)
        new_add_time_ids = torch.empty((0, *last_add_time_ids.shape[1:]), dtype=last_add_time_ids.dtype, device=last_add_time_ids.device)

        new_current_latents_r = torch.empty((0, *current_latents.shape[1:]), dtype=current_latents.dtype, device=current_latents.device)
    # new_cached_latents_r = torch.empty((0, *cached_latents.shape[1:]), dtype=cached_latents.dtype, device=cached_latents.device)
    
    # back_to_queue_latents = torch.zeros((maximum_bs, *current_latents.shape[1:]), dtype=current_latents.dtype, device=current_latents.device)
    # back_to_queue_cached_latents = torch.zeros((maximum_bs, *cached_latents.shape[1:]), dtype=cached_latents.dtype, device=cached_latents.device)
    # back_to_queue_timesteps_batch = torch.zeros((maximum_bs, *timesteps_batch.shape[1:]), dtype=timesteps_batch.dtype, device=timesteps_batch.device)
    # back_to_queue_labels = [None] * maximum_bs
    # back_to_queue_requests = [None] * maximum_bs
    
    # for latent, cached_latent, label, request, timesteps ,queue_idx in zip(current_latents, cached_latents, last_labels, last_requests, timesteps_batch, last_queue_nums):
    #     back_to_queue_latents[queue_idx] = latent
    #     back_to_queue_cached_latents[queue_idx] = cached_latent
    #     back_to_queue_timesteps_batch[queue_idx] = timesteps
    #     back_to_queue_labels[queue_idx] = label
    #     back_to_queue_requests[queue_idx] = request
    ## Beginning
    if current_latents is None and timesteps_batch is None:
        for i in range(maximum_bs):
            if not queues[i].empty():
                prompt, label = queues[i].get()  # Pop one request from the queue
                requests.append(prompt)
                labels.append(label)
                queue_num.append(i)
                queue_tail += 1
                print(f"Queue {i+1}: Popped Prompt: {prompt}, Label: {label}")
            else:
                print(f"Queue {i+1} is empty.")
        timesteps_batch = precompute_timesteps_for_labels(scheduler, labels, "cpu")
        # print("timesteps:", timesteps_batch)
        prompt_embeds, add_text_embeds, add_time_ids, current_latents = pipe.input_process(prompt = requests,negative_prompt = negative_prompts, generator=generator, callback_on_step_end=None,
            callback_on_step_end_tensor_inputs=["latents"])
        # print("current latents:", current_latents)
        # print("new_prompt_embeds", prompt_embeds)
        # Return the required parameters to pass to the pipe function
        return (prompt_embeds, add_text_embeds, add_time_ids, timesteps_batch, requests, labels,  current_latents, cached_latents, queue_num, queue_tail)
    
    else:
        for i in range(maximum_bs):
            print("tail:",queue_tail)
            try:

                last_cluster_num = last_queue_nums[i]
                # print(position)
                ## request finished 
                if not timesteps_batch[i]:
        
                    if not queues[last_cluster_num].empty():
                        prompt, label = queues[last_cluster_num].get()
                        if label == 0 or label == 2:
                            new_requests_co.append(prompt)
                            new_labels_co.append(label)
                            queue_num_co.append(last_cluster_num)
                        # reuse request    
                        elif label == 1: 
                            new_requests_r.append(prompt)
                            new_labels_r.append(label)
                            queue_num_r.append(last_cluster_num)       
                            new_current_latents_r = torch.cat((new_current_latents_r, cached_latents[i].unsqueeze(0)), dim=0)
                            # new_cached_latents_r = torch.cat((new_cached_latents_r, cached_latents[position]), dim=0)          
                    else:
                        print(f"cluster {last_cluster_num} is empty") 
                        try: 
                            present_cluster = queues[queue_tail]
                            if not present_cluster.empty():
                                prompt, label = present_cluster.get()
                                if label == 0 or label == 2:
                                    new_requests_co.append(prompt)
                                    new_labels_co.append(label)
                                    queue_num_co.append(queue_tail)
                                # reuse request    
                                elif label == 1: 
                                    new_requests_r.append(prompt)
                                    new_labels_r.append(label)
                                    queue_num_r.append(queue_tail)       
                                    new_current_latents_r = torch.cat((new_current_latents_r, cached_latents[i].unsqueeze(0)), dim=0)
                            queue_tail += 1
                                                            
                        except IndexError:
                            print(f"current cluster {queue_tail - 1}, no available cluster") 
                            
                            
                        
                # request not finished
                else:
                    new_cached_latents = torch.cat((new_cached_latents, cached_latents[i].unsqueeze(0)), dim=0)
                    new_current_latents = torch.cat((new_current_latents, current_latents[i].unsqueeze(0)), dim=0)
                    new_prompt_embeds_p = torch.cat((new_prompt_embeds_p, last_prompt_embeds_p[i].unsqueeze(0)), dim=0)
                    new_prompt_embeds_n = torch.cat((new_prompt_embeds_n, last_prompt_embeds_n[i].unsqueeze(0)), dim=0)
                    new_add_text_embeds_p = torch.cat((new_add_text_embeds_p, last_add_text_embeds_p[i].unsqueeze(0)), dim=0)
                    new_add_text_embeds_n = torch.cat((new_add_text_embeds_n, last_add_text_embeds_n[i].unsqueeze(0)), dim=0)
                    new_add_time_ids = torch.cat((new_add_time_ids, last_add_time_ids[i*2:i*2+2]), dim=0)
                    
                    requests.append(last_requests[i])
                    labels.append(last_labels[i])
                    new_timesteps_batch.append(timesteps_batch[i])
                    queue_num.append(last_cluster_num)
                    
            except IndexError:
                print(f"no past request from batch position {i}.")
                try: 
                    present_cluster = queues[queue_tail]
                    if not present_cluster.empty():
                        prompt, label = present_cluster.get()
                        if label == 0 or label == 2:
                            new_requests_co.append(prompt)
                            new_labels_co.append(label)
                            queue_num_co.append(queue_tail)
                        # reuse request    
                        elif label == 1: 
                            raise ValueError("reuse request exists, but no cached latent")
                    queue_tail += 1
                                                            
                except IndexError:
                    print(f"current cluster {queue_tail - 1}, no available cluster") 
                    
        
        
        num_of_r_requests = len(new_labels_r)
          
        new_labels_co.extend(new_labels_r)
        new_timesteps_batch_cor = precompute_timesteps_for_labels(scheduler, new_labels_co, "cpu") 
        print("timesteps cor:",new_timesteps_batch_cor )
        if new_timesteps_batch_cor:
            new_timesteps_batch.extend(new_timesteps_batch_cor)
        # have new requests
        new_requests_co.extend(new_requests_r)  
        if new_requests_co:
            prompt_embeds_cor, add_text_embeds_cor, add_time_ids_cor, current_latents_cor = pipe.input_process(prompt = new_requests_co,negative_prompt = negative_prompts, generator=generator, callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=["latents"]) 
            
            prompt_embeds_cor_n, prompt_embeds_cor_p = torch.chunk(prompt_embeds_cor, 2, dim=0)
            add_text_embeds_cor_n, add_text_embeds_cor_p = torch.chunk(add_text_embeds_cor, 2, dim=0)      
            new_prompt_embeds_p = torch.cat((new_prompt_embeds_p, prompt_embeds_cor_p), dim=0)
            new_prompt_embeds_n = torch.cat((new_prompt_embeds_n, prompt_embeds_cor_n), dim=0)
            new_add_text_embeds_p = torch.cat((new_add_text_embeds_p, add_text_embeds_cor_p), dim=0)
            new_add_text_embeds_n = torch.cat((new_add_text_embeds_n, add_text_embeds_cor_n), dim=0)
            new_add_time_ids = torch.cat((new_add_time_ids, add_time_ids_cor), dim=0)
            
            if num_of_r_requests:
                current_latents_cor[-num_of_r_requests:] = new_current_latents_r
            new_cached_latents = torch.cat((new_cached_latents, current_latents_cor), dim=0)    
            new_current_latents = torch.cat((new_current_latents, current_latents_cor), dim=0)  
        new_prompt_embeds =  torch.cat((new_prompt_embeds_n, new_prompt_embeds_p), dim=0)
        new_add_text_embeds = torch.cat((new_add_text_embeds_n, new_add_text_embeds_p), dim=0)
        requests.extend(new_requests_co)

        labels.extend(new_labels_co)

        queue_num.extend(queue_num_co)
        queue_num.extend(queue_num_r)
        
        print("timesteps:", new_timesteps_batch)
        # print("last current latents:", current_latents )
        # print("current_latents:", new_current_latents)
        # print("new_prompt_embeds", new_prompt_embeds)
        is_equal = torch.equal(current_latents, new_current_latents)
        is_equal_prompt_embedding = torch.equal(new_prompt_embeds, last_prompt_embeds)
        is_equal_add_time_ids = torch.equal(new_add_time_ids, last_add_time_ids)
        is_equal_add_text_embedding = torch.equal(new_add_text_embeds, last_add_text_embeds)
        print("Are latent tensors exactly equal?:", is_equal)
        print("Are prompt embedding exactly equal?:", is_equal_prompt_embedding)
        print("Are add_time_ids exactly equal?:", is_equal_add_time_ids)
        print("Are add_text_embeds exactly equal?:", is_equal_add_text_embedding)



        return (new_prompt_embeds, new_add_text_embeds, new_add_time_ids, new_timesteps_batch,  requests, labels, new_current_latents, new_cached_latents, queue_num, queue_tail)
    
    
# def decoder(latent_queue: Queue, pipe: StableDiffusionXLPipeline, maximum_bs : int):
#     if not latent_queue.empty():
#         prompts = []
#         output_latent, prompt = latent_queue.get()
#         prompts.append(prompt)
#         output_latents = torch.empty((0, *output_latent.shape[1:]), dtype=output_latent.dtype, device=output_latent.device)
#         output_latents = torch.cat((output_latents,output_latent), dim=0)
#         for i in range(maximum_bs - 1):
#             if not latent_queue.empty():
#                 output_latent, prompt = latent_queue.get()
#                 prompts.append(prompt)
#                 output_latents = torch.cat((output_latents,output_latent), dim=0)
        
#         images = pipe.decode(prompt = prompts, latents = output_latents).images
#         print(images) 
#         for i in range(len(images)):  
                     
#             images[i].save(f"./cluster_img_{args.eps}_xl_pipeline/{prompts[i].replace(' ', '_').replace('/', '_')}_10.png")


PATH_TO_IMAGE_FOLDER = "/data3/llama_model/yuchen/datasets/COCO2017"


maximum_bs = 8
k = 10
num_inference_steps = 50
largest_common = math.gcd(k, num_inference_steps)
print("The largest common number (GCD) is:", largest_common)

# dataset = load_dataset("phiyodr/coco2017")
# dataset = dataset.map(create_full_path)
# train_dataset = dataset['train'][:1000]

# prompts = [subarray[-1].replace('\n','') for subarray in train_dataset['captions']]
# print(prompts)


args = parser.parse_args()



metadata_df = pd.read_parquet('/home/stilex/metadata.parquet')
if 'timestamp' in metadata_df.columns:
    # print("First few entries in the 'timestamp' column:")
    # print(metadata_df['timestamp'].head())
    
    # Optionally sort by date
    # metadata_df['timestamp'] = pd.to_datetime(metadata_df['timestamp'])  # Ensure it's in datetime format
    sorted_df = metadata_df.sort_values(by='timestamp')
    # print("Sorted DataFrame by timestamp:")
    # print(sorted_df.head())
else:
    print("No timestamp column found in the DataFrame.")
# print(sorted_df['prompt'][0])
num_columns_shape = sorted_df.shape
print(f"Number of columns (using shape): {num_columns_shape}")
prompts = sorted_df['prompt'].head(1000).to_list()


texts = processor(text=prompts, return_tensors="pt",truncation=True, padding=True,max_length=77).to(device)
# print(texts)
with torch.no_grad():
    embeddings = model.get_text_features(**texts)
cosine_distance_matrix = cosine_distances(embeddings.cpu())
dbscan = DBSCAN(eps=args.eps, min_samples=2, metric="precomputed")
clusters = dbscan.fit_predict(cosine_distance_matrix)
subset_df = sorted_df.head(1000).copy()
# subset_df = pd.DataFrame(train_dataset)

subset_df['cluster'] = clusters
subset_df['prompt'] = prompts
print(subset_df.columns)

## cluster all the outliers
# for cluster_num in subset_df['cluster'].unique():
#     # Get the number of examples in this cluster
#     cluster_size = len(subset_df[subset_df['cluster'] == cluster_num])
    
#     # If the cluster has only one example, set the cluster number to -2
#     if cluster_size == 1:
#         subset_df.loc[subset_df['cluster'] == cluster_num, 'cluster'] = -1
        
for cluster_num, group in subset_df.groupby('cluster'):
    # Get the number of examples in this cluster
    
    # Print the indices of examples in each cluster
    indices = group.index.tolist()
    print(f"Cluster {cluster_num}: {indices}")
    print("cluster size:", len(group))
# print(subset_df[['prompt', 'cluster']])


# latent_queue = queue.Queue()
queues = []
core_requets = []
for i, cluster_num in enumerate(subset_df['cluster'].unique()):
    # if cluster_num != -1:  # Ignore clusters that were reassigned to -2
        # print(cluster_num)
        # print(i)
        # queue_index = i % maximum_bs  # Cyclically assign cluster to a queue
    cluster_prompts = subset_df[subset_df['cluster'] == cluster_num]['prompt'].tolist()
    cluster = queue.Queue()
    
    # Add all prompts from the current cluster to the selected queue
    for j, prompt in enumerate(cluster_prompts):
        label = 0 if j == 0 else 1  # Label the first prompt as 0, others as 1
        cluster.put((prompt, label))
        core_requets.append(prompt) if j == 0 else None
    queues.append(cluster)
print(len(core_requets))


        
        
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
print(torch.version.cuda) 
# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to(device)


print(pipe)
print(f"Using scheduler: {type(pipe.scheduler).__name__}")
scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

seed = 42 #any
clip_score_uncached = []
clip_score_cached = [[] for _ in range(21)]

requests = [None for _ in range(maximum_bs)]
labels = [None for _ in range(maximum_bs)]
negative_prompts = None

full_timesteps = precompute_timesteps_for_labels(scheduler,[0],"cpu")
cached_timestep = full_timesteps[0][k-1]
print("cached timestep:", cached_timestep)

current_latents = None
cached_latents = None
timesteps_batch = None
last_labels = None
last_requests = None
last_queue_nums = []
last_prompt_embeds = None
last_add_text_embeds = None
last_add_time_ids = None

generator = torch.Generator(device).manual_seed(seed)
queue_tail = 0
while any(not q.empty() for q in queues) or any(timesteps for timesteps in timesteps_batch):  # Continue while at least one queue has items
    # for i, q in enumerate(queues):
    #     if not q.empty():
    #         prompt, label = q.get()  # Pop one request from the queue
    #         requests[i] = prompt
    #         labels[i] = label
    #         print(f"Queue {i+1}: Popped Prompt: {prompt}, Label: {label}")
    #     else:
    #         requests[i] = None
    #         labels[i] = None
    #         print(f"Queue {i+1} is empty.")
    # print(labels)
    # timesteps_batch = precompute_timesteps_for_labels(scheduler,labels,"cpu")
    # print("timesteps:",timesteps_batch)
    
    start_time = time.time()
    model_inputs = request_scheduler(queues = queues, current_latents = current_latents, cached_latents = cached_latents, timesteps_batch = timesteps_batch, last_labels = last_labels, 
                     last_requests = last_requests, last_queue_nums = last_queue_nums, last_prompt_embeds = last_prompt_embeds, last_add_text_embeds = last_add_text_embeds,
                     last_add_time_ids = last_add_time_ids, maximum_bs = maximum_bs, generator = generator, pipe = pipe,  negative_prompts = negative_prompts, scheduler = scheduler, queue_tail = queue_tail)
    scheduler_time = time.time() - start_time
    print(f"Time for request_scheduler: {scheduler_time:.4f} seconds")
    queue_tail = model_inputs[9]
    # start_time = time.time()
    # decoder(latent_queue = latent_queue, pipe = pipe, maximum_bs = maximum_bs)
    # decoder_time = time.time() - start_time
    # print(f"Time for decoder: {decoder_time:.4f} seconds")

    # generator = torch.Generator(device).manual_seed(seed)
    
    start_time = time.time()
    model_outputs = pipe(prompt = model_inputs[4], prompt_embeds = model_inputs[0], add_text_embeds = model_inputs[1], add_time_ids = model_inputs[2], generator=generator, callback_on_step_end=None,
            callback_on_step_end_tensor_inputs=["current_latents"], largest_common=largest_common, timesteps_batch = model_inputs[3],
            cached_timestep=cached_timestep ,labels_batch = model_inputs[5],cached_latents=model_inputs[7],current_latents=model_inputs[6], queue_nums = model_inputs[8])
    pipe_time = time.time() - start_time
    print(f"Time for pipe: {pipe_time:.4f} seconds")
    
    if model_outputs[9]:
        for i in range(len(model_outputs[10])):
            model_outputs[9][i].save(f"./cluster_img_{args.eps}_xl_pipeline/{model_outputs[10][i].replace(' ', '_').replace('/', '_')}_10.png")
    
    current_latents = model_outputs[0]
    cached_latents = model_outputs[1]
    timesteps_batch = model_outputs[2]
    last_requests = model_outputs[3]
    last_labels = model_outputs[4]
    last_prompt_embeds = model_outputs[5]
    last_add_text_embeds = model_outputs[6]
    last_add_time_ids = model_outputs[7]
    last_queue_nums = model_outputs[8]
    
    
    
    
    
    
    # for i in range(cluster_size):            
    #     image[i].save(f"./cluster_img_{args.eps}_xl/{requests[i].replace(' ', '_').replace('/', '_')}_noncache.png")
    # clip_score = get_clip_score(image,requests)
    # clip_score_uncached.extend(clip_score.detach().cpu().tolist())
    # print(clip_score_uncached)
    # torch.cuda.empty_cache()






                

        # for j, timestep in enumerate(back_to_queue_timesteps_batch):
        #     # request finished 
        #     if not timestep:
        #         if not queues[j].empty():
        #             prompt, label = queues[j].get()
        #             if label == 0 or label == 2:
        #                 new_requests_co.extend(prompt)
        #                 new_labels_co.extend(label)
        #                 queue_num_co.extend(j)
                        
        #             elif label == 1: 
        #                 new_requests_r.extend(prompt)
        #                 new_labels_r.extend(label)
        #                 queue_num_r.extend(j)
                        
        #         else:
        #             print(f"queue {j} is empty")
            
        #     # request not finished
        #     else:
        #         new_cached_latents.append(cached_latents[j])
        #         new_current_latents.append(current_latents[j])
        #         requests.extend()
        #         labels.extend()
                

                






















# cs = 8
# for cluster_num in subset_df['cluster'].unique():
#     print(f"\nCluster {cluster_num}:")
#     cluster_size = len(subset_df[subset_df['cluster'] == cluster_num])
#     print(f"size of this cluster: {cluster_size}")
#     print(subset_df[subset_df['cluster'] == cluster_num][['prompt']].head(100))
#     if cluster_num != -1 and cluster_num != -2:
#         prompts = subset_df[subset_df['cluster'] == cluster_num]['prompt'].to_list()
#         if cluster_size <= cs:
#             negative_prompts = None
#             print(prompts)
#             print("cache iteration:", -1)
#             generator = torch.Generator(device).manual_seed(seed)
#             image = pipe(prompts,negative_prompt= negative_prompts, generator=generator, callback_on_step_end=None,
#                     callback_on_step_end_tensor_inputs=["latents"],cache_start_iter = 1000).images
#             for i in range(cluster_size):            
#                 image[i].save(f"./cluster_img_{args.eps}_xl/{prompts[i].replace(' ', '_').replace('/', '_')}_noncache.png")
#             clip_score = get_clip_score(image,prompts)
#             clip_score_uncached.extend(clip_score.detach().cpu().tolist())
#             print(clip_score_uncached)
#             torch.cuda.empty_cache()
            
#             for j in [5,10,15,20]:
#                 # if args.np:
#                 #     negative_prompts = ["", prompts[0]]
#                 print("cache iteration:", j)
#                 generator = torch.Generator(device).manual_seed(seed)
#                 image = pipe(prompts,negative_prompt= negative_prompts, generator=generator, callback_on_step_end=None,
#                     callback_on_step_end_tensor_inputs=["latents"],cache_start_iter = j).images  
#                 for k in range(cluster_size):        
#                     image[k].save(f"./cluster_img_{args.eps}_xl/{prompts[k].replace(' ', '_').replace('/', '_')}_{j}.png")
#                 clip_score = get_clip_score(image,prompts)
                
                
#                 clip_score_cached[j].extend(clip_score.detach().cpu().tolist())
#                 print(clip_score_cached[j])
#                 torch.cuda.empty_cache()
#         else:
#             chunk_size = math.ceil(cluster_size / cs)
#             for m in range(chunk_size):
#                 chunk_prompts = prompts[m*cs:(m+1)*cs]
#                 negative_prompts = None
#                 print(chunk_prompts)
#                 print("cache iteration:", -1)
#                 generator = torch.Generator(device).manual_seed(seed)
#                 image = pipe(chunk_prompts,negative_prompt= negative_prompts, generator=generator, callback_on_step_end=None,
#                         callback_on_step_end_tensor_inputs=["latents"],cache_start_iter = 1000).images
#                 for i in range(len(chunk_prompts)):            
#                     image[i].save(f"./cluster_img_{args.eps}_xl/{chunk_prompts[i].replace(' ', '_').replace('/', '_')}_noncache.png")
#                 clip_score = get_clip_score(image,chunk_prompts)
                
#                 clip_score_uncached.extend(clip_score.detach().cpu().tolist())
#                 print(clip_score_uncached)
#                 torch.cuda.empty_cache()
                
#                 for j in [5,10,15,20]:
#                     # if args.np:
#                     #     negative_prompts = ["", prompts[0]]
#                     print("cache iteration:", j)
#                     generator = torch.Generator(device).manual_seed(seed)
#                     image = pipe(chunk_prompts,negative_prompt= negative_prompts, generator=generator, callback_on_step_end=None,
#                         callback_on_step_end_tensor_inputs=["latents"],cache_start_iter = j).images  
#                     for k in range(len(chunk_prompts)):        
#                         image[k].save(f"./cluster_img_{args.eps}_xl/{chunk_prompts[k].replace(' ', '_').replace('/', '_')}_{j}.png")
#                     clip_score = get_clip_score(image,chunk_prompts)
#                     clip_score_cached[j].extend(clip_score.detach().cpu().tolist())
#                     print(clip_score_cached[j]) 
#                     torch.cuda.empty_cache()
                    
        
# print(f"uncached avg clip:", sum(clip_score_uncached) / len(clip_score_uncached))
# for n in [5,10,15,20]:
#     print(f"cached avg clip {n}:", sum(clip_score_cached[n]) / len(clip_score_cached[n]))
    




















# # example = train_dataset[0]
# # print(example)
# num_examples = 100
# num_cache = 1000

# examples = train_dataset[:num_examples]
# examples1 = train_dataset[num_examples:num_examples+num_cache]

# count = 0
# sum = 0
# highest = 0
# larger80 = 0
# larger90 = 0
# larger70 = 0
# larger60 = 0


# prompt_cosine = [[None for _ in range(num_cache)] for _ in range(num_examples)]


# # print(len(prompt_cosine))
# for i in range(num_examples):
#     for j in range(num_cache):
#         # print(j)
#         prompt_cosine[i][j] =clip_text_score(examples['captions'][i][-1],examples1['captions'][j][-1])
        
#         # print(prompt_cosine[i][j])
#         sum += prompt_cosine[i][j]
#         count += 1
#         if highest < prompt_cosine[i][j]:
#             highest = prompt_cosine[i][j]
#         if prompt_cosine[i][j] >= 0.8:
#             larger80 += 1
#         if prompt_cosine[i][j] >= 0.9:
#             larger90 += 1
#         if prompt_cosine[i][j] >= 0.7:
#             larger70 += 1
#         if prompt_cosine[i][j] >= 0.6:
#             larger60 += 1
# avg=sum/count
# print(count)
# print(avg)
# print(highest)

# print(">0.6", larger60)
# print(">0.7", larger70)
# print(">0.8", larger80)
# print(">0.9", larger90)

# max_values_per_row = []
# max_indices_per_row = []

# for row in prompt_cosine:
#     max_value = max(row)                # Find the max value
#     max_index = row.index(max_value)    # Find the index of the max value
#     max_values_per_row.append(max_value)
#     max_indices_per_row.append(max_index)


# seed = 1024 #any

# # Parse command-line arguments
# args = parser.parse_args()

# # ds = load_dataset("poloclub/diffusiondb", "2m_first_1k")
# # ds_prompt = ds['train']['prompt']

# model_id = "stabilityai/stable-diffusion-2-1"
# print(torch.version.cuda) 
# # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)


# pipe = pipe.to(device)


# # prompts = ["An orange tree", "A banana tree"]
# # if args.np:
# #     negative_prompts = [None, prompts[0]]
# # else:
# #     negative_prompts = [None, None]
    
# # prompt=prompts[0:args.bs]

# # image = pipe(prompts,negative_prompt= negative_prompts, generator=generator, callback_on_step_end=callback,
# #         callback_on_step_end_tensor_inputs=["latents"],cache_start_iter = 48).images
# # image[1].save(f"{prompts[1].replace(' ', '_')}_wtf.png")
# # image[0].save(f"{prompts[0].replace(' ', '_')}_wtf.png")

# # clip_score = get_clip_score(image[1],prompts[0])
# # print(f"clip: {clip_score.item()}")
# negative_prompts = None
# for i in range(num_examples):
#     if max_values_per_row[i] >= 0.8:

#         prompts = [examples1['captions'][max_indices_per_row[i]][-1],examples['captions'][i][-1]]
#         negative_prompts = None
#         print(prompts)
#         print("text cosine similarity:", max_values_per_row[i])
#         print("cache iteration:", -1)
#         generator = torch.Generator(device).manual_seed(seed)
#         image = pipe(prompts,negative_prompt= negative_prompts, generator=generator, callback_on_step_end=callback,
#                 callback_on_step_end_tensor_inputs=["latents"],cache_start_iter = 1000).images            
#         image[1].save(f"./img_cache_np_10/{prompts[1].replace(' ', '_')}_noncache.png")
#         for j in [5,10,15,20]:
#             if args.np:
#                 negative_prompts = ["", prompts[0]]
#             print("cache iteration:", j)
#             generator = torch.Generator(device).manual_seed(seed)
#             image = pipe(prompts,negative_prompt= negative_prompts, generator=generator, callback_on_step_end=callback,
#                 callback_on_step_end_tensor_inputs=["latents"],cache_start_iter = j).images            
#             image[1].save(f"./img_cache_np_10/{prompts[1].replace(' ', '_')}_{j}.png")
#         image[0].save(f"{prompts[0].replace(' ', '_')}.png")  


#     # clip_score = get_clip_score(image[1],prompts[1])

#     # print(f"clip: {clip_score.item()}")
    

    
    
# # print(image)
# # image.save("astronaut_rides_horse.png")
# # image[0].save(f"{prompts[0].replace(' ', '_')}.png")
# # image[1].save(f"{prompts[1].replace(' ', '_')}.png")