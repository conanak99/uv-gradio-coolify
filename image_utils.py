from PIL import Image


MAX_DIMENSION = 2048


def resize_if_needed(image_path: str) -> str:
    with Image.open(image_path) as image:
        width, height = image.size
        if width <= MAX_DIMENSION and height <= MAX_DIMENSION:
            return image_path
        scale = MAX_DIMENSION / max(width, height)
        new_size = (int(width * scale), int(height * scale))
        resized_image = image.resize(new_size, Image.LANCZOS)

    resized_path = image_path + "_resized.png"
    try:
        resized_image.save(resized_path)
    finally:
        resized_image.close()
    return resized_path
