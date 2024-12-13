import torch
from datasets import load_dataset
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, CLIPConfig, CLIPTextConfig, CLIPVisionConfig
import os
from pathlib import Path
from PIL import Image
import json
from datetime import datetime
from clip import CFG

class ImageSearcher:
    def __init__(self, model_path="outputs/clip_best_model.pt", output_dir="clip_search_outputs"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        print("Loading dataset...")
        self.ds = load_dataset("Magneto/caption_for_mars_and_rover_image_size_768")
        self.dataset = self.ds['test']
        print(f"Loaded {len(self.dataset)} test images")
        
        # Initialize processor
        print("Initializing processor...")
        self.processor = CLIPProcessor.from_pretrained(CFG.model_name)
        
        # Initialize model
        print("Loading model...")
        config_text = CLIPTextConfig.from_pretrained(CFG.model_name)
        config_vision = CLIPVisionConfig.from_pretrained(CFG.model_name)
        config = CLIPConfig.from_text_vision_configs(config_text, config_vision)
        self.model = CLIPModel(config)
        
        # Load trained weights
        print(f"Loading weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Process all images and store features
        print("Processing all images...")
        self.image_features = self._preprocess_images()
        print("Image processing complete!")
        
        # Clear previous search results
        if self.output_dir.exists():
            print("Cleaning previous search results...")
            for item in self.output_dir.glob("*"):
                if item.is_dir():
                    for subitem in item.glob("*"):
                        subitem.unlink()
                    item.rmdir()
                else:
                    item.unlink()

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
                    image_features = self.model.get_image_features(
                        pixel_values=inputs['pixel_values'].to(self.device)
                    )
                    features.append(image_features)
            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                continue
        
        return torch.cat(features, dim=0)

    def search(self, query, top_k=5):
        """Search for images matching the text query"""
        print(f"Searching for: '{query}'")
        inputs = self.processor(
            text=[query],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CFG.max_text_tokens_length
        )
        
        with torch.no_grad():
            text_features = self.model.get_text_features(
                input_ids=inputs['input_ids'].to(self.device),
                attention_mask=inputs['attention_mask'].to(self.device)
            )
            
            # Normalize features
            text_features = F.normalize(text_features, dim=-1)
            image_features = F.normalize(self.image_features, dim=-1)
            
            # Calculate similarity using CLIP's cosine similarity
            logits_per_text = torch.matmul(text_features, image_features.T)
            similarity = logits_per_text[0]  # Take first query result
            
            top_scores, top_indices = similarity.topk(top_k)
            
            return [
                (self.dataset[idx.item()], score.item()) 
                for score, idx in zip(top_scores, top_indices)
            ]

    def save_results(self, query, results):
        """Save search results to directory with unique run identifier"""
        # Create a unique run directory with timestamp
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / run_timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create query-specific directory
        query_slug = "".join(x if x.isalnum() else "_" for x in query)[:50]
        search_dir = run_dir / query_slug
        search_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "query": query,
            "timestamp": run_timestamp,
            "model_name": CFG.model_name,
            "results": []
        }

        for idx, (item, score) in enumerate(results, 1):
            image_filename = f"result_{idx}.jpg"
            image_path = search_dir / image_filename
            
            if isinstance(item['image'], Image.Image):
                image = item['image']
            else:
                image = Image.fromarray(item['image'])
            image.save(image_path)

            result_metadata = {
                "filename": image_filename,
                "score": score,
                "short_caption": item['short_caption'],
                "long_caption": item['long_caption'],
                "url": item['url']
            }
            metadata["results"].append(result_metadata)

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
    
    # Create a timestamp for this run
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize the searcher with run-specific output directory
    output_dir = Path(f"/scratch/general/vast/u1475870/clip_project/clip_search_results/{run_timestamp}")
    searcher = ImageSearcher(
        model_path="/uufs/chpc.utah.edu/common/home/u1475870/clip_project/outputs/clip_best_model.pt",
        output_dir=output_dir
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
    
    # Create a summary file for all searches in this run
    summary = {
        "run_timestamp": run_timestamp,
        "model_name": CFG.model_name,
        "queries": []
    }
    
    # Perform searches and save results
    for query in queries:
        print(f"\nSearching for: {query}")
        results = searcher.search(query, top_k=5)
        
        output_dir = searcher.save_results(query, results)
        
        # Add to summary
        query_summary = {
            "query": query,
            "output_dir": str(output_dir),
            "results": []
        }
        
        print(f"\nTop matches saved to: {output_dir}")
        print("Results:")
        for item, score in results:
            print(f"- Score: {score:.3f}")
            print(f"  Caption: {item['short_caption']}")
            print(f"  URL: {item['url']}\n")
            
            query_summary["results"].append({
                "score": score,
                "short_caption": item['short_caption'],
                "url": item['url']
            })
        
        summary["queries"].append(query_summary)
    
    # Save run summary
    summary_path = output_dir.parent / f"summary_{run_timestamp}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nSearch run complete. Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
