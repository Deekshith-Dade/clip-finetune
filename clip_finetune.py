import json
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# import clip
from transformers import CLIPProcessor, CLIPModel
import datasets
from tqdm import tqdm
from torchvision import transforms
from datasets import load_dataset
import matplotlib.pyplot as plt
import os

save_dir = "/scratch/general/vast/u1475870/models5"

def validate_and_plot(dataloader, processor, model, device, ep, type):
    images, text = next(iter(dataloader))
    if type == "train":
        images = images[:64]
        text = text[:64]
    inputs = processor(text=text, images=images, return_tensors="pt", padding=True, do_rescale=False)
    inputs = inputs.to(device)

    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    logits_per_text = outputs.logits_per_text

    probs = logits_per_image.softmax(dim=1)  # Softmax to get probabilities


    # Create a grid plot for the probabilities (interpreting it as a color-coded heatmap)
    plt.imshow(probs.cpu().detach().numpy(), cmap='viridis', aspect='auto')  # Reshape into a 1D grid
    plt.colorbar(label='Probability')
    plt.title("Softmax Probabilities as Colors (for a single image)")
    plt.xlabel("Classes")
    plt.ylabel("Image Index")
    dir = f"{save_dir}/images/"
    os.makedirs(dir, exist_ok=True)
    plt.savefig(f"{dir}/{type}_probabilities_{ep}.png")
    plt.close()
    print(f"Saved {type} probabilities plot at epoch {ep}")

data = load_dataset('Magneto/caption_for_mars_and_rover_image_size_1024')

train_dataset = data['train']
test_dataset = data['test']

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to CLIP input size
    transforms.ToTensor()  # Convert to torch tensor
])

model_name = "clip-vit-large-patch14"
model = CLIPModel.from_pretrained(f"openai/{model_name}")
processor = CLIPProcessor.from_pretrained(f"openai/{model_name}")

def evaluate(model, processor, dataloader):
    losses = []
    model.eval()
    pbar = tqdm(dataloader, total=len(train_dataloader))
    for batch in pbar:
        images, text = batch
        inputs = processor(text=text, images=images, return_tensors="pt", padding=True, do_rescale=False)
        inputs = inputs.to(device)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text
        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
        loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        losses.append(loss.item())
        
        pbar.set_description(f"Loss: {loss.item():.4f}")
    pbar.close()
    return losses, np.mean(losses)
    

class Image_caption_dataset():
  def __init__(self, train_dataset, caption="long"):
    self.caption = caption
    self.train_dataset = train_dataset
    self.transform = transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to CLIP input size
    transforms.ToTensor()  # Convert to torch tensor
    ])

  def __len__(self):
    return len(train_dataset)

  def __getitem__(self, idx):
    image = train_dataset[idx]["image"]

    if image.mode == "L":
      image = image.convert("RGB")

    image = transform(image)

    if self.caption == "short":
      text = train_dataset[idx]["short_caption"]
    elif self.caption == "long":
      text = train_dataset[idx]["long_caption"]
    else:
      raise "Wrong Stuff"

    return image, text

device = "cuda" if torch.cuda.is_available() else "cpu"

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5 ,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

batch_size = 64 
train_set = Image_caption_dataset(train_dataset, caption="short")
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_batch_size = 64
test_set = Image_caption_dataset(test_dataset, caption="short")
test_dataloader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True)

model = model.to(device)
torch.cuda.empty_cache()
losses = []
test_losses = []
test_losses_avg = []
num_epochs = 10
validate_and_plot(train_dataloader, processor, model, device, -1, "train")
validate_and_plot(test_dataloader, processor, model, device, -1, "test")

for epoch in range(num_epochs):
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    model.train()
    for batch in pbar:
        model.train()
        optimizer.zero_grad()

        images,texts = batch

        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, do_rescale=False, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Forward pass
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text
        # Compute loss
        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)


        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        losses.append(total_loss)
        # Backward pass
        total_loss.backward()
        optimizer.step()
        # if device == "cpu":
        #     optimizer.step()
        # else :
        #     convert_models_to_fp32(model)
        #     optimizer.step()
        #     clip.model.convert_weights(model)

        
        
        pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")
    # close the progress
    pbar.close()
    
    losses, avg_loss = evaluate(model, processor, test_dataloader)
    test_losses.append(losses)
    test_losses_avg.append(avg_loss)
    print(f"Average Test Loss @ Epoch {ep}: {avg_loss}")
    save_config = dict (
        epoch = epoch,
        model_state_dict = model.state_dict(),
        optimizer_state_dict = optimizer.state_dict(),
        losses = losses,
        test_losses = test_losses,
        test_losses_avg = test_losses_avg
    )
    torch.save(save_conf=g, f"{save_dir}/clip_finetune_{model_name}_{epoch}.pt")
    validate_and_plot(train_dataloader, processor, model, device, epoch, "train")
    validate_and_plot(test_dataloader, processor, model, device, epoch, "test")
