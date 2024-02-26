# 08-DLSS
## Introduction
Rescaling an image in something like Microsoft Paint involves some kind of interpolation that uses an algorithm to decide how to resize it. Some of the most common image interpolation techniques are:
- Nearest-neighbour
- Bilinear
- Edge-directed

If we run the following code on a small image, it will use Bilinear interpolation to upscale it by a factor of 2.
```
from PIL import Image

# Open an image file
with Image.open('input.jpg') as img:
    # Get the size of the image
    width, height = img.size

    # Define the upscale factor
    upscale_factor = 2

    # Calculate the new size
    new_size = (width * upscale_factor, height * upscale_factor)

    # Resize the image using bilinear interpolation
    img_upscaled = img.resize(new_size, Image.BILINEAR)

    # Save the upscaled image
    img_upscaled.save('output.jpg')
```

Viewing the images at their native resolutions, you can see the image is twice as big but doesn't look great.

<!-- <div align="center">
  <a href="Extra\image.png" target="_blank">
    <img src="Extra\image.png"/>
  </a>
</div>

<div align="center">
  <a href="Extra\output.jpg" target="_blank">
    <img src="Extra\output.jpg"/>
  </a>
</div> -->

In recent years, new technologies have been developed that offer much more advanced ways of upscaling images. And these techniques have been introduced into video game engines.

This technology can be referred to as Super Resolution upscaling and has been used to save computation. Here's an example of how it can be used to save computation:
- A game is being played at 4K and runs at 60 FPS.
- The game settings are changed so the game now runs at 1080p at 100 FPS.
- A Super Resolution upscaler is used to take the game's output render and upscale it back to 4K, and the game now looks like it is running at 4K but FPS is now just under 100 FPS.

This technology allows players with less-powerful hardware to run games at high resolutions. It is even employed on the Steam Deck to save on computation and battery life.

Some of the most common Super Resolution upscalers are from:

### AMD
<div align="center">
  <a href="Images\FSR1 Comparison.jpg" target="_blank">
    <img src="Images\FSR1 Comparison.jpg" style="height:600px;"/>
  </a>
</div>
<div align="center">
  <a href="https://uk.pcmag.com/graphics-cards/134066/amd-launches-dlss-competitor-fidelityfx-super-resolution">
  Image of FSR1 Compared to Traditional Upscaling Techniques
  </a>
</div>
<br>

AMD has introduced:
- June 2021: AMD FidelityFX Super Resolution v1.0 (FSR1)
- March 2022: AMD Radeon Super Resolution (RSR)
- June 2022: AMD FidelityFX Super Resolution v2.0 (FSR2): uses temporal information, meaning it can refrence previous frames to be more context aware and improve quality.
- Sept 2023: AMD FidelityFX Super Resolution v2.0 (FSR3)

I'm not going to go into detail on how AMD's systems work. The rundown is:
- They are all open-source
- None of them use Machine/Deep Learning (AI)
- FSR requires 'game integration' and very specific hardware requirements, whereas RSR is 'driver-based' making it compatible with any game running in full screen mode when using AMD RDNA™ architecture-based or newer graphics hardware.
([ref](https://www.amd.com/en/technologies/radeon-super-resolution#:~:text=A%3A%20FidelityFX%20Super%20Resolution%20requires,%E2%84%A2%20architecture%2Dbased%20or%20newer))

The site for FSR1 has some comparison sliders to compare qualities:
https://gpuopen.com/fidelityfx-superresolution/

This table shows how the FSR Quality Mode setting changes the scale factor to make the user view the game at 1440p or 4K. So in the Ultra Quality setting, to display a 4K frame to the user, the game is ran at 1970 x 1108 resolution and passed through the FSR upscaler with a Scale Factor of 1.3x. For better performance, the games are downscaled even more and use a higher Scale Factor.

| FSR Quality Mode | Scale Factor | Input Resolution for 1440p FSR | Input Resolution for 4K FSR |
| --- | --- | --- | --- |
| Ultra Quality | 1.3X per dimension | 1970 x 1108 | 2954 x 1662 |
| Quality | 1.5X per dimension | 1706 x 960 | 2560 x 1440 |
| Balanced | 1.7X per dimension | 1506 x 847 | 2259 x 1270 |
| Performance | 2.0X per dimension | 1280 x 720 | 1920 x 1080 |

Some more info if you want it:
[FSR3 Announcement Blog](https://community.amd.com/t5/gaming/amd-fsr-3-now-available/ba-p/634265) |
[FSR2 Reveal Video](https://youtu.be/JUQ8j-bpQ1Q) |
[FSR3 Reveal Video](https://www.youtube.com/watch?v=zttHxmKFpm4)

<div align="center">
  <a href="Images\AMD Radeon Super Resolution.png" target="_blank">
    <img src="Images\AMD Radeon Super Resolution.png" style="height:500px;"/>
  </a>
</div>
<div align="center">
  <a href="https://www.amd.com/en/technologies/radeon-super-resolution">
  Source (Use the Slider Here)
  </a>
</div>
<br>

### Deep Learning Super Sampling (DLSS)
DLSS, developed by Nvidia, is probably the most well known Super Resolution system as it gained popularity first.

Looking at comparison videos, you can see a clear performance improvement. It's a little difficult to compare quality considering YouTube's compression is pretty bad: https://www.youtube.com/watch?v=1NsfqJPmhYY

<div align="center">
  <a href="Images\DLSS Demo.png" target="_blank">
    <img src="Images\DLSS Demo.png" style="height:500px;"/>
  </a>
</div>
<div align="center">
  <a href="https://www.youtube.com/watch?v=1NsfqJPmhYY">
  Source
  </a>
</div>
<br>

To use DLSS requires an RTX GPU (luckily we have these!).

Newer DLSS versions don't require this anymore, but in DLSS version 1 developers had to record gameplay and then pass these images though a Deep Learning neural network which learned the relationship between low resolution and high resolution images.

## Summary
- Reminder of working with Pygame
- Learn about DLSS and other super-resolution systems
- Create an actually quite sophisticated yet simple super resolution system
- Implement DLSS in a Unity project
- Attempt to implemented this system inside a Pygame game (this would be a world first -- I haven't even attempted it yet)

## Tutorial
Let's emulate what developers had to do to make their game DLSS v1 compatible but in a simpler Pygame environment.

We'll miss out a lot of the complicated stuff, like we won't reference previous frames or implement motion vectors to do things like track motion blur, but this is ok as we'll work with a simple game. Our steps look like this:

**Part 1**

1. Capture frames from a Pygame game
2. Downscale these frames so we have the frames at native and half resolution
3. Train a Deep Learning model to learn the relationship between half resolution and native resolution images of our game so that it can upscale new images

**Part 2**

4. Implement DLSS in Unity while we wait for the model to train

**Part 3**

5. Implement the Super Resolution model we trained on our Pygame game inside the game

### Part 1
1. Using VS Code, go to the Pygame Jet Game folder and play Pygame_Jet_Game.py (just for fun). The game is the result of [a tutorial](https://realpython.com/pygame-a-primer/), so there's some extra files included.
2. We need to capture frames from the game to use to train our Deep Learning model. Create a folder called 'frame_captures_256_256' (256x256 is the resolution of the game window) in the same directory as the game. And implement frame capturing by using the code below:

```
...

# Variable to keep our main loop running
running = True

# ADD FRAME COUNT HERE
frame_count = 0

...
```


```
...

    # Ensure we maintain a 60 frames per second rate
    clock.tick(60)

    # FRAME CAPTURE HERE
    # Save the current frame 
    pygame.image.save(screen, f"frame_captures_256_256/frame_{frame_count}.png")
    frame_count += 1

# At this point, we're done, so we can stop and quit the mixer
pygame.mixer.music.stop()
pygame.mixer.quit()
```
3. Play the game for a few seconds and make sure the frames are being captured by checking inside the folder.
4. If you want to make things easier, alter the code so that the game keeps running even if you collide with a jet.
5. If everythings good, play the game for about a minute, you'll need at least 40 seconds of gameplay to get a decent enough sized dataset. Less might work (but I haven't tested it):
<br>40 s * 60 FPS = 2,400 Frames Captured
6. Remove the first x amount of images in the file explorer so you have 2000 frames left. It's important to remove the first images as a few of the first frames will be an image of just the jet, but our DL model will learn best with more varied data with missiles and clouds too.
7. Open **02 - Downscaler.ipynb** in VS Code, this is a Python Notebook file -- you should see a prompt to install Python plugins, install them (ask if you don't see this). We'll use a Python package called Pillow (PIL) for image manipulation. To install it, open the Anconda Prompt program via the Windows Search bar, and paste and run ```pip install pillow``` or ```conda install anaconda::pillow```.
8. Update the source and destination directories in the notebook. Select the root Kernel (I think it will be this on the Uni machines, please ask me to check this to confirm it) by selecting the button in the top right and selecting the root python kernel. <div align="center"><img src="Images\Kernel.png" style="height:100px;"/></div>
9. Run the code cell. This will create a downscaled copy of your frame captures. Now you will have a copy that is half the resolution of the capture (256x256 -> 128x128). It might take up to a minute to run.
10. Now we can train our super-resolution model. There's one last thing we need to install; open Anaconda Prompt again and run ```conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia```. This will install Pytorch, a popular Machine Learning and Deep Learning framework. You might need to press Y to continue the installation when prompted.
11. In 02 - Downscaler.ipynb, run through the cells. Click the first cell and press `Shift` + `Enter` to run the cell and move to the next one. I've added notes inside this file letting you know what it's doing.

### Part 2
If you have completed the previous steps, you should have an AI model training on the dataset you provided. While we wait, we can see how the most recent DLSS version can be implemented inside Unity.

This is quite straight forward.

1. Load the **3D Sample Scene (HDRP)** example scene in Unity, you will find it when selecting a new project, alternatively, you can create an empty HDRP project and use one of the HDRP projects from the [Unity Asset Store](https://assetstore.unity.com/?publisher=Unity%20Technologies%5CUnity%20Edu%5CUnity%20Education%5CUnity%20Technologies%20Japan%5Cunity-chan!%5CSpeedTree%C2%AE&free=true&q=hdrp&orderBy=1).

2. The [Unity Docs](https://docs.unity3d.com/Packages/com.unity.render-pipelines.high-definition@12.0/manual/deep-learning-super-sampling-in-hdrp.html) provide steps to implement DLSS, but to be honest you can just follow [this YouTube video](https://www.youtube.com/watch?v=OhN3BG1USVs) that can show you how to do it in under a minute! 

There is also a plugin for Unreal Engine: https://developer.nvidia.com/rtx/dlss/get-started

### Part 3
Hopefully you have finished training by now. So let's implement this model inside our Pygame.