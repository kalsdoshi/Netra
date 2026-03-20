import cv2
from PIL import Image

class CaptionEngine:
    """
    Optional captioning module. If transformers/torch are installed,
    generates captions for images using BLIP.
    """
    def __init__(self, use_gpu=False):
        self.available = False
        self.processor = None
        self.model = None
        
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            import torch
            
            self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
            print(f"Loading BLIP Caption Model on {self.device}...")
            
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
            self.available = True
            print("BLIP Model loaded successfully.")
            
        except ImportError:
            print("⚠️ [CaptionEngine] transformers or torch not installed. Semantic graph features disabled.")
        except Exception as e:
            print(f"⚠️ [CaptionEngine] Failed to load BLIP: {e}")

    def generate_caption(self, image_bgr):
        if not self.available:
            return None
            
        try:
            # Convert OpenCV BGR to PIL RGB
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            inputs = self.processor(pil_image, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_new_tokens=30)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            print(f"Caption generation failed: {e}")
            return None
