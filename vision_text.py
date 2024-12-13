from datasets import load_dataset
import torch
from torch import optim
from transformers import logging, VisionTextDualEncoderModel, VisionTextDualEncoderProcessor, BertTokenizer, ViTFeatureExtractor
from tqdm import tqdm
import time

logging.set_verbosity_error()

# Load the Mars Rover dataset
ds = load_dataset("Magneto/caption_for_mars_and_rover_image_size_768")

class CFG:
    max_text_tokens_length = 128
    text_backbone = 'bert-base-uncased'
    image_backbone = 'google/vit-base-patch16-224'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    max_epochs = 75
    max_bad_epochs = 9
    patience = 3
    factor = 0.1

def read_mars_rover_pairs(split='train'):
    dataset_split = ds[split]
    pairs = [{
        'caption': item['long_caption'],
        'image': item['image']
    } for item in dataset_split]
    return pairs

class DataSet(torch.utils.data.Dataset):
    def __init__(self, pairs, processor):
        super().__init__()
        self.pairs = pairs
        self.processor = processor
    
    def __getitem__(self, idx):
        try:
            caption = self.pairs[idx]['caption']
            image = self.pairs[idx]['image']
            
            # Convert grayscale to RGB if needed
            if len(image.mode) == 1 or image.mode == 'L':
                image = image.convert('RGB')
            
            encoded_pair = self.processor(
                text=[caption], 
                images=[image], 
                return_tensors="pt", 
                max_length=CFG.max_text_tokens_length, 
                padding='max_length', 
                truncation=True
            )
            
            return {
                'input_ids': encoded_pair['input_ids'].squeeze(0),
                'attention_mask': encoded_pair['attention_mask'].squeeze(0),
                'pixel_values': encoded_pair['pixel_values'].squeeze(0)
            }
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            return None
    
    def __len__(self):
        return len(self.pairs)

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def train_epoch(model, train_loader, optimizer, epoch, max_epochs):
    model.train()
    nb_batches = len(train_loader)
    tqdm_object = tqdm(train_loader, total=nb_batches)   
    epoch_loss = 0.0
    
    for i, batch in enumerate(tqdm_object):
        optimizer.zero_grad()  # Added this line
        outputs = model(
            input_ids=batch['input_ids'].to(CFG.device),
            attention_mask=batch['attention_mask'].to(CFG.device),
            pixel_values=batch['pixel_values'].to(CFG.device),
            return_loss=True
        )
        
        loss = outputs.loss
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        tqdm_object.set_postfix(
            batch=f"{i+1}/{nb_batches}",
            train_loss=loss.item(),
            lr=get_lr(optimizer)
        )
    
    return epoch_loss / nb_batches

def valid_epoch(model, dev_loader):
    model.eval()
    nb_batches = len(dev_loader)
    tqdm_object = tqdm(dev_loader, total=nb_batches)
    epoch_loss = 0.0   
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm_object):
            outputs = model(
                input_ids=batch['input_ids'].to(CFG.device),
                attention_mask=batch['attention_mask'].to(CFG.device),
                pixel_values=batch['pixel_values'].to(CFG.device),
                return_loss=True
            )
            
            loss = outputs.loss
            epoch_loss += loss.item()
            tqdm_object.set_postfix(
                batch=f"{i+1}/{nb_batches}",
                dev_loss=loss.item(),
            )
    
    return epoch_loss / nb_batches

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def main():
    # Initialize custom processor
    print("Initializing tokenizer and feature extractor...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    processor = VisionTextDualEncoderProcessor(feature_extractor, tokenizer)
    
    # Initialize CLIP model
    print("Initializing VisionTextDualEncoder model...")
    clip = VisionTextDualEncoderModel.from_vision_text_pretrained(
        CFG.image_backbone,
        CFG.text_backbone
    )
    
    # Prepare datasets
    train_pairs = read_mars_rover_pairs('train')
    val_pairs = read_mars_rover_pairs('validation')
    
    train_dataset = DataSet(train_pairs, processor)
    val_dataset = DataSet(val_pairs, processor)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Training setup
    clip.to(CFG.device)
    optimizer = torch.optim.AdamW(clip.parameters(), lr=5e-5)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode="min", 
        patience=CFG.patience, 
        factor=CFG.factor
    )
    
    best_dev_score = float('inf')
    train_history = []
    dev_history = []
    nb_bad_epochs = 0
    
    print(f"Starting training on {CFG.device}")
    
    for epoch in range(1, CFG.max_epochs + 1):
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}/{CFG.max_epochs}")
        
        if nb_bad_epochs >= CFG.max_bad_epochs:
            print("Early stopping triggered")
            break
        
        epoch_start_time = time.time()
        
        train_loss = train_epoch(clip, train_loader, optimizer, epoch, CFG.max_epochs)
        val_loss = valid_epoch(clip, val_loader)
        
        duration = time.time() - epoch_start_time
        
        train_history.append(train_loss)
        dev_history.append(val_loss)
        
        lr_scheduler.step(val_loss)
        
        if val_loss < best_dev_score:
            best_dev_score = val_loss
            torch.save(clip.state_dict(), "vision_text_best_model.pt")
            nb_bad_epochs = 0
            print(f"New best model saved! Val Loss: {val_loss:.4f}")
        else:
            nb_bad_epochs += 1
        
        print(f"Epoch {epoch} completed in {duration:.2f}s. Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return {'train': train_history, 'dev': dev_history}

if __name__ == "__main__":
    main()