from PIL import Image

# Open an image file
with Image.open('08-DLSS\Extra\image.png') as img:
    # Get the size of the image
    width, height = img.size

    # Define the upscale factor
    upscale_factor = 2

    # Calculate the new size
    new_size = (width * upscale_factor, height * upscale_factor)

    # Resize the image using bilinear interpolation
    img_upscaled = img.resize(new_size, Image.BILINEAR)

    # Save the upscaled image
    img_upscaled.save('08-DLSS\Extra\output.jpg')
