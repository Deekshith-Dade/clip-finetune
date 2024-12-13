import torch
from datasets import load_dataset
import torch.nn.functional as F
from transformers import VisionTextDualEncoderModel, VisionTextDualEncoderProcessor, BertTokenizer, ViTFeatureExtractor
from pathlib import Path
import os
from PIL import Image
import json
from datetime import datetime

# Define configuration for Vision-Text model
class VisionTextConfig:
    text_backbone = 'bert-base-uncased'
    image_backbone = 'google/vit-base-patch16-224'
    max_text_tokens_length = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageSearcher:
    def __init__(self, model_path="outputs/vision_text_best_model.pt", output_dir="search_results"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        print("Loading dataset...")
        self.ds = load_dataset("Magneto/caption_for_mars_and_rover_image_size_768")
        self.dataset = self.ds['test']
        print(f"Loaded {len(self.dataset)} test images")
        
        # Initialize processors
        print("Initializing processors...")
        self.tokenizer = BertTokenizer.from_pretrained(VisionTextConfig.text_backbone)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(VisionTextConfig.image_backbone)
        self.processor = VisionTextDualEncoderProcessor(self.feature_extractor, self.tokenizer)
        
        # Initialize model
        print("Loading model...")
        self.model = VisionTextDualEncoderModel.from_vision_text_pretrained(
            VisionTextConfig.image_backbone,
            VisionTextConfig.text_backbone
        )
        
        # Load trained weights
        print(f"Loading weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Process all images
        print("Processing images...")
        self.image_features = self._preprocess_images()
        print("Ready for searching!")

    def _preprocess_images(self):
        """Process all images and store their features"""
        features = []
        total_images = len(self.dataset)
        
        print(f"Processing {total_images} images...")
        for idx, item in enumerate(self.dataset):
            if idx % 100 == 0:
                print(f"Processing image {idx}/{total_images}")
            
            image = item['image']
            # Convert grayscale to RGB if needed
            if isinstance(image, Image.Image) and (len(image.mode) == 1 or image.mode == 'L'):
                image = image.convert('RGB')
            
            try:
                inputs = self.processor(
                    images=image, 
                    return_tensors="pt",
                    padding=True
                )
                
                with torch.no_grad():
                    outputs = self.model.get_image_features(
                        inputs['pixel_values'].to(self.device)
                    )
                    features.append(outputs)
            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                continue
        
        return torch.cat(features, dim=0)

    def search(self, query, top_k=5):
        """Search for images matching the text query"""
        print(f"Searching for: '{query}'")
        inputs = self.processor(
            text=[query],  # Wrap query in list
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=VisionTextConfig.max_text_tokens_length
        )
        
        with torch.no_grad():
            text_features = self.model.get_text_features(
                input_ids=inputs['input_ids'].to(self.device),
                attention_mask=inputs['attention_mask'].to(self.device)
            )
            
            # Normalize features
            text_features = F.normalize(text_features, dim=-1)
            image_features = F.normalize(self.image_features, dim=-1)
            
            # Calculate similarity
            similarity = torch.matmul(text_features, image_features.T)[0]
            
            top_scores, top_indices = similarity.topk(top_k)
            
            return [
                (self.dataset[idx.item()], score.item()) 
                for score, idx in zip(top_scores, top_indices)
            ]

    def save_results(self, query, results):
        """Save search results to directory"""
        # Create timestamp for unique folder name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create sanitized query string for folder name
        query_slug = "".join(x if x.isalnum() else "_" for x in query)[:50]
        search_dir = self.output_dir / f"{timestamp}_{query_slug}"
        search_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "query": query,
            "timestamp": timestamp,
            "results": []
        }

        # Save images and collect metadata
        for idx, (item, score) in enumerate(results, 1):
            # Save image
            image_filename = f"result_{idx}.jpg"
            image_path = search_dir / image_filename
            
            # Convert to PIL Image if necessary and save
            if isinstance(item['image'], Image.Image):
                image = item['image']
            else:
                image = Image.fromarray(item['image'])
            image.save(image_path)

            # Collect metadata for this result
            result_metadata = {
                "filename": image_filename,
                "score": score,
                "short_caption": item['short_caption'],
                "long_caption": item['long_caption'],
                "url": item['url']
            }
            metadata["results"].append(result_metadata)

        # Save metadata to JSON file
        metadata_path = search_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {search_dir}")
        return search_dir

def main():
    # Print dataset split sizes
    ds = load_dataset("Magneto/caption_for_mars_and_rover_image_size_768")
    print(f"Dataset splits:")
    print(f"- Train: {len(ds['train'])} images")
    print(f"- Validation: {len(ds['validation'])} images")
    print(f"- Test: {len(ds['test'])} images")
    
    # Initialize the searcher
    searcher = ImageSearcher(
        model_path="/uufs/chpc.utah.edu/common/home/u1475870/clip_project/outputs/vision_text_best_model.pt",  
        output_dir="/scratch/general/vast/u1475870/clip_project/vision_text_search_results"
    )
    
    # Example searches
    queries = [
        "A large crater on Mars surface",
        "Martian rocks and sand",
        "Rover tracks in the Martian soil",
        "Close up of Martian soil",
        "Mars horizon view",
        "Rocky Martian terrain"
    ]
    
    # Perform searches and save results
    for query in queries:
        print(f"\nSearching for: {query}")
        results = searcher.search(query, top_k=5)
        
        # Save results
        output_dir = searcher.save_results(query, results)
        
        print(f"\nTop matches saved to: {output_dir}")
        print("Results:")
        for item, score in results:
            print(f"- Score: {score:.3f}")
            print(f"  Caption: {item['short_caption']}")
            print(f"  URL: {item['url']}\n")

if __name__ == "__main__":
    main()