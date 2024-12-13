import os
import dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from typing import Set, List, Dict, Tuple
import json
from datetime import datetime
import time
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

PERSIST_DIRECTORY = "/scratch/general/vast/u1475870/clip_project/vector_db"
llm = ChatOpenAI(model="gpt-4o-mini")
dotenv.load_dotenv()

def get_all_urls_and_images(base_url: str) -> tuple[Set[str], Set[str]]:
    urls_found, image_urls = set(), set()
    urls_to_scan, scanned_urls = {base_url}, set()
    parsed_base = urlparse(base_url)
    base_domain, base_path = parsed_base.netloc, parsed_base.path.rstrip('/')
    image_dir = os.path.join(PERSIST_DIRECTORY, "images")
    os.makedirs(image_dir, exist_ok=True)
    
    while urls_to_scan:
        current_url = urls_to_scan.pop()
        if current_url in scanned_urls:
            continue
            
        try:
            response = requests.get(current_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            scanned_urls.add(current_url)
            
            if current_url.startswith(base_url):
                urls_found.add(current_url)
                for img in soup.find_all('img'):
                    img_url = img.get('src')
                    if img_url:
                        img_url = urljoin(current_url, img_url)
                        if img_url not in image_urls:
                            image_urls.add(img_url)
                            try:
                                img_response = requests.get(img_url, timeout=10)
                                if img_response.status_code == 200:
                                    img_name = os.path.basename(urlparse(img_url).path)
                                    if img_name:
                                        with open(os.path.join(image_dir, img_name), 'wb') as f:
                                            f.write(img_response.content)
                            except Exception:
                                continue
       
            for link in soup.find_all(['a']):
                href = link.get('href')
                if not href:
                    continue
                
                url = urljoin(current_url, href)
                parsed_url = urlparse(url)
                
                if (parsed_url.scheme in ['http', 'https'] and
                    parsed_url.netloc == base_domain and
                    parsed_url.path.startswith(base_path) and
                    not any(url.lower().endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.gif', '.zip'])):
                    clean_url = url.split('#')[0].rstrip('/')
                    if (clean_url.startswith(base_url) and
                        clean_url not in scanned_urls and 
                        clean_url not in urls_to_scan):
                        urls_to_scan.add(clean_url)
            
            time.sleep(1)
            
        except Exception:
            continue
    
    return urls_found, image_urls

def process_urls_to_db(urls: List[str], persist_directory: str = PERSIST_DIRECTORY):
    os.makedirs(persist_directory, exist_ok=True)
    loader = WebBaseLoader(
        web_paths=urls,
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                ["article", "main", "div", "section", "p", "h1", "h2", "h3", "h4", "h5", "h6"]
            )
        )
    )
    
    try:
        docs = loader.load()
        if not docs:
            return None
        
        splits = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        ).split_documents(docs)
        
        return Chroma.from_documents(
            documents=splits,
            embedding=OpenAIEmbeddings(),
            persist_directory=persist_directory
        )
    except Exception:
        return None

def save_urls_to_json(urls: Set[str], image_urls: Set[str], base_url: str):
    data = {
        "base_url": base_url,
        "crawl_date": datetime.now().isoformat(),
        "urls": sorted(list(urls)),
        "image_urls": sorted(list(image_urls))
    }
    with open(os.path.join(PERSIST_DIRECTORY, "crawled_urls.json"), "w") as f:
        json.dump(data, f, indent=4)

def save_query_results(question: str, answer: str, sources: List[str]):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(PERSIST_DIRECTORY, "query_results", f"query_{timestamp}.txt")
    os.makedirs(os.path.join(PERSIST_DIRECTORY, "query_results"), exist_ok=True)
    
    with open(output_file, "w") as f:
        f.write(f"Question: {question}\n\n")
        f.write(f"Answer: {answer}\n\n")
        f.write("Sources:\n")
        for source in set(sources):
            f.write(f"- {source}\n")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class MarsImageRetrieval:
    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory
        
        # Load fine-tuned CLIP model
        model_path = "/scratch/general/vast/u1475870/models2"
        latest_checkpoint = self._get_latest_checkpoint(model_path)
        
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        if latest_checkpoint:
            checkpoint = torch.load(latest_checkpoint)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        
        self.vectorstore = Chroma(persist_directory=persist_directory)
        self.image_to_text_map = self._load_image_text_mapping()
    
    def _get_latest_checkpoint(self, model_path: str) -> str:
        checkpoints = [f for f in os.listdir(model_path) if f.startswith('clip_finetune_')]
        if not checkpoints:
            return None
        latest = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return os.path.join(model_path, latest)
    
    def encode_image(self, image: Image) -> torch.Tensor:
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt", do_rescale=False)
            inputs = inputs.to(self.model.device)
            image_features = self.model.get_image_features(**inputs)
            return image_features.squeeze()
    
    def _load_image_text_mapping(self) -> Dict[str, List[Tuple[str, str]]]:
        with open(os.path.join(self.persist_directory, "crawled_urls.json"), "r") as f:
            data = json.load(f)
            
        mapping = {}
        for url in data["urls"]:
            docs = self.vectorstore.get_relevant_documents(url)
            for doc in docs:
                text_content = doc.page_content
                for img_url in data["image_urls"]:
                    if img_url in text_content:
                        if img_url not in mapping:
                            mapping[img_url] = []
                        mapping[img_url].append((text_content, url))
        return mapping
    
    def get_related_content(self, image_path: str, top_k: int = 3) -> List[Dict]:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_embedding = self.encode_image(image)
        
        results = []
        for img_url, content_list in self.image_to_text_map.items():
            try:
                img_path = os.path.join(self.persist_directory, "images", 
                    os.path.basename(urlparse(img_url).path))
                if not os.path.exists(img_path):
                    continue
                    
                stored_image = Image.open(img_path)
                if stored_image.mode != "RGB":
                    stored_image = stored_image.convert("RGB")
                stored_embedding = self.encode_image(stored_image)
                
                similarity = torch.cosine_similarity(
                    image_embedding.unsqueeze(0), 
                    stored_embedding.unsqueeze(0)
                )
                
                for text_content, source_url in content_list:
                    results.append({
                        "similarity": similarity.item(),
                        "text": text_content,
                        "source_url": source_url,
                        "image_url": img_url
                    })
            except Exception as e:
                print(f"Error processing image {img_url}: {str(e)}")
                continue
                
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

if __name__ == "__main__":
    base_url = "https://science.nasa.gov/mars/"
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    os.makedirs(os.path.join(PERSIST_DIRECTORY, "query_results"), exist_ok=True)
    
    urls, image_urls = get_all_urls_and_images(base_url)
    save_urls_to_json(urls, image_urls, base_url)
    
    vectorstore = process_urls_to_db(list(urls))
    if not vectorstore:
        exit(1)
    
    retriever = MarsImageRetrieval(PERSIST_DIRECTORY)
    
    # Example usage
    test_image_path = "/uufs/chpc.utah.edu/common/home/u1475870/clip_project/vector_db/mars.jpg"
    related_content = retriever.get_related_content(test_image_path)
    
    for content in related_content:
        print(f"\nSimilarity Score: {content['similarity']:.4f}")
        print(f"Related Text: {content['text'][:200]}...")
        print(f"Source URL: {content['source_url']}")
        print(f"Related Image: {content['image_url']}")