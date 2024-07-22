from .image_captioner import DashscopeConfig, ImageCaptioner

# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "DashscopeConfig": DashscopeConfig,
    "ImageCaptioner": ImageCaptioner
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "DashscopeConfig": "Dashscope API Config",
    "ImageCaptioner": "Image Captioner"
}
