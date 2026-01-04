# Transforming an image, to a gray image.

Personal project to ensure my knowledge in CUDA and GPU software development

## What is needed

CMake(with Ninja) + CudaToolKit + Nvidia GPU + stb_image

### Pipeline from the code

Basically, we use an external library call stb_image to turn our image into an array of bytes, we discover also the height, the width and the channel (RGB or RGBA) from the image chosen.
After that, we allocate the right size in the GPU VRAM to receive an array of bytes from our image.
With that, we process it with parallelism using a lot of threads from our GPU and with the CPU.

# Results:

We clearly see that amount of threads guarantees around 60 times more faster to process the image.