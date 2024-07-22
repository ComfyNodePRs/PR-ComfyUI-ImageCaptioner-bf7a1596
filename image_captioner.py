import dashscope
import base64
import comfy.utils
import numpy as np

from PIL import Image
from io import BytesIO
from http import HTTPStatus

class DashscopeConfig:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""})
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "set_api_key"
    CATEGORY = "Config"
    
    def set_api_key(self, api_key):
        dashscope.api_key = api_key
        return {}

class ImageCaptioner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "user_prompt": ("STRING", {"default": "Please describe this image in 50 to 70 words.", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_image_captions"
    CATEGORY = "Image Processing"

    def post_process_prompt(self, raw_prompt):
        tags = [tag.strip().lower() for tag in raw_prompt.split(',') if tag.strip()]
        tags = ['_'.join(tag.split()) for tag in tags]
        seen = set()
        unique_tags = [tag for tag in tags if not (tag in seen or seen.add(tag))]
        final_tags = unique_tags[:70]
        return ', '.join(final_tags)

    def generate_image_captions(self, image, user_prompt):
        """Simple single round multimodal conversation call."""
        image = Image.fromarray((image.numpy() * 255).astype(np.uint8)[0])
        with BytesIO() as output:
            image.save(output, format="PNG")
            image_bytes = output.getvalue()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:image/png;base64,{base64_image}"

        messages = [
            {
                "role": "system",
                "content": "As an AI image tagging expert, please provide precise tags for these images to enhance CLIP model's understanding of the content. Employ succinct keywords or phrases, steering clear of elaborate sentences and extraneous conjunctions. Prioritize the tags by relevance. Your tags should capture key elements such as the main subject, setting, artistic style, composition, image quality, color tone, filter, and camera specifications, and any other tags crucial for the image. When tagging photos of people, include specific details like gender, nationality, attire, actions, pose, expressions, accessories, makeup, composition type, age, etc. For other image categories, apply appropriate and common descriptive tags as well. Recognize and tag any celebrities, well-known landmark or IPs if clearly featured in the image. Your tags should be accurate, non-duplicative, and within a 20-75 word count range. These tags will use for image re-creation, so the closer the resemblance to the original image, the better the tag quality. Tags should be comma-separated. Exceptional tagging will be rewarded with $10 per image."
            },
            {
                "role": "user",
                "content": [
                    {"image": {"url": image_url}},
                    {"text": user_prompt}
                ]
            }
        ]

        response = dashscope.MultiModalConversation.call(
            model="qwen-vl-plus",
            messages=messages
        )

        if response.status_code == HTTPStatus.OK:
            raw_prompt = response.output.choices[0].message.content
            processed_prompt = self.post_process_prompt(raw_prompt)
            return (processed_prompt,)
        else:
            print(f"Error: {response.code} - {response.message}")
            return ("Error generating captions.",)
