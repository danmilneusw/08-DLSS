import numpy as np
import torch
import torch.nn as nn
from math import sqrt

class Model(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        # 1 FEATURE EXTRACTION
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=56, kernel_size=5, stride=1, padding=2),
            nn.PReLU(56))

        # 2 SHRINKING
        self.shrinking = nn.Sequential(
            nn.Conv2d(in_channels=56, out_channels=12, kernel_size=1, stride=1, padding=0),
            nn.PReLU(12))

        # 3 NON-LINEAR MAPPING
        self.mapping = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.PReLU(12),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.PReLU(12),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.PReLU(12),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.PReLU(12),
        )

        # 4 EXPANDING
        self.expanding = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=56, kernel_size=1),
            nn.PReLU(56)
        )

        # 5 DECONVOLUTION
        self.deconvolution = nn.ConvTranspose2d(in_channels=56, out_channels=3, kernel_size=9, stride=upscale_factor, padding=4, output_padding=upscale_factor-1)

        # WEIGHT INITIALISATION
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print(f'0 Input:              {x.shape}')
        x = self.feature_extraction(x)
        #print(f'1 Feature Extraction: {x.shape}')
        x = self.shrinking(x)
        #print(f'2 Shrinking:          {x.shape}')
        x = self.mapping(x)
        #print(f'2 Non-linear Mapping: {x.shape}')
        x = self.expanding(x)
        #print(f'4 Expanding:          {x.shape}')
        x = self.deconvolution(x)
        #print(f'5 Output:             {x.shape}')
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)

        nn.init.normal_(self.deconvolution.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.deconvolution.bias.data)



# Instantiate model
model = Model(2)
# Load the state dict into the model
model.load_state_dict(torch.load(r'E:\GitHub Game Engine Optimisation\08-DLSS\03 - AI Model\250224_best_model.pth'))
# Ensure the model is in evaluation mode
model.eval()


# Import the pygame module
import pygame

# Import random for random numbers
import random

# Import pygame.locals for easier access to key coordinates
# Updated to conform to flake8 and black standards
# from pygame.locals import *
from pygame.locals import (
    RLEACCEL,
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)

# If true, resolution will be halved from 256x256 to 128x128
# Sprites will also need to be halved too
make_half = True

# Define constants for the screen width and height
if make_half == True:
    SCREEN_WIDTH = 256
    SCREEN_HEIGHT = 256
else:
    SCREEN_WIDTH = 128
    SCREEN_HEIGHT = 128


# Define the Player object extending pygame.sprite.Sprite
# Instead of a surface, we use an image for a better looking sprite
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super(Player, self).__init__()        
        if make_half == False:
            self.surf = pygame.image.load(r"E:\GitHub Game Engine Optimisation\08-DLSS\04 - Pygame Jet Game with SR\jet.png").convert()
            #self.surf = pygame.image.load("jet.png").convert()
        else:
            original_surf = pygame.image.load(r"E:\GitHub Game Engine Optimisation\08-DLSS\04 - Pygame Jet Game with SR\jet.png").convert()
            width, height = original_surf.get_size()
            self.surf = pygame.transform.scale(original_surf, (width // 2, height // 2))
            
        self.surf.set_colorkey((255, 255, 255), RLEACCEL)
        self.rect = self.surf.get_rect()

    # Move the sprite based on keypresses
    def update(self, pressed_keys):
        if pressed_keys[K_UP]:
            self.rect.move_ip(0, -5)
            move_up_sound.play()
        if pressed_keys[K_DOWN]:
            self.rect.move_ip(0, 5)
            move_down_sound.play()
        if pressed_keys[K_LEFT]:
            self.rect.move_ip(-5, 0)
        if pressed_keys[K_RIGHT]:
            self.rect.move_ip(5, 0)

        # Keep player on the screen
        if self.rect.left < 0:
            self.rect.left = 0
        elif self.rect.right > SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH
        if self.rect.top <= 0:
            self.rect.top = 0
        elif self.rect.bottom >= SCREEN_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT


# Define the enemy object extending pygame.sprite.Sprite
# Instead of a surface, we use an image for a better looking sprite
class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        super(Enemy, self).__init__()
        if make_half == False:
            self.surf = pygame.image.load(r"E:\GitHub Game Engine Optimisation\08-DLSS\04 - Pygame Jet Game with SR\missile.png").convert()
            #self.surf = pygame.image.load("missile.png").convert()
        else:
            original_missile = pygame.image.load(r"E:\GitHub Game Engine Optimisation\08-DLSS\04 - Pygame Jet Game with SR\missile.png").convert()
            width, height = original_missile.get_size()
            self.surf = pygame.transform.scale(original_missile, (width // 2, height // 2))

        self.surf.set_colorkey((255, 255, 255), RLEACCEL)
        # The starting position is randomly generated, as is the speed
        self.rect = self.surf.get_rect(
            center=(
                random.randint(SCREEN_WIDTH + 20, SCREEN_WIDTH + 100),
                random.randint(0, SCREEN_HEIGHT),
            )
        )
        self.speed = random.randint(1, 3)

    # Move the enemy based on speed
    # Remove it when it passes the left edge of the screen
    def update(self):
        self.rect.move_ip(-self.speed, 0)
        if self.rect.right < 0:
            self.kill()


# Define the cloud object extending pygame.sprite.Sprite
# Use an image for a better looking sprite
class Cloud(pygame.sprite.Sprite):
    def __init__(self):
        super(Cloud, self).__init__()
        if make_half == False:
            self.surf = pygame.image.load(r"E:\GitHub Game Engine Optimisation\08-DLSS\04 - Pygame Jet Game with SR\cloud.png").convert()
            #self.surf = pygame.image.load("cloud.png").convert()
        else:
            original_cloud = pygame.image.load(r"E:\GitHub Game Engine Optimisation\08-DLSS\04 - Pygame Jet Game with SR\cloud.png").convert()
            width, height = original_cloud.get_size()
            self.surf = pygame.transform.scale(original_cloud, (width // 2, height // 2))

        self.surf.set_colorkey((0, 0, 0), RLEACCEL)
        # The starting position is randomly generated
        self.rect = self.surf.get_rect(
            center=(
                random.randint(SCREEN_WIDTH + 20, SCREEN_WIDTH + 100),
                random.randint(0, SCREEN_HEIGHT),
            )
        )

    # Move the cloud based on a constant speed
    # Remove it when it passes the left edge of the screen
    def update(self):
        self.rect.move_ip(-5, 0)
        if self.rect.right < 0:
            self.kill()


# Setup for sounds, defaults are good
pygame.mixer.init()

# Initialize pygame
pygame.init()

# Setup the clock for a decent framerate
clock = pygame.time.Clock()

# Create the screen object
# The size is determined by the constant SCREEN_WIDTH and SCREEN_HEIGHT
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Create custom events for adding a new enemy and cloud
ADDENEMY = pygame.USEREVENT + 1
pygame.time.set_timer(ADDENEMY, 500)
ADDCLOUD = pygame.USEREVENT + 2
pygame.time.set_timer(ADDCLOUD, 1000)

# Create our 'player'
player = Player()

# Create groups to hold enemy sprites, cloud sprites, and all sprites
# - enemies is used for collision detection and position updates
# - clouds is used for position updates
# - all_sprites isused for rendering
enemies = pygame.sprite.Group()
clouds = pygame.sprite.Group()
all_sprites = pygame.sprite.Group()
all_sprites.add(player)

# Load and play our background music
# Sound source: http://ccmixter.org/files/Apoxode/59262
# License: https://creativecommons.org/licenses/by/3.0/
pygame.mixer.music.load(r"E:\GitHub Game Engine Optimisation\08-DLSS\04 - Pygame Jet Game with SR\Apoxode_-_Electric_1.mp3")
#pygame.mixer.music.load("Apoxode_-_Electric_1.mp3")
pygame.mixer.music.play(loops=-1)

# Load all our sound files
# Sound sources: Jon Fincher
move_up_sound = pygame.mixer.Sound(r"E:\GitHub Game Engine Optimisation\08-DLSS\04 - Pygame Jet Game with SR\Rising_putter.ogg")
move_down_sound = pygame.mixer.Sound(r"E:\GitHub Game Engine Optimisation\08-DLSS\04 - Pygame Jet Game with SR\Falling_putter.ogg")
collision_sound = pygame.mixer.Sound(r"E:\GitHub Game Engine Optimisation\08-DLSS\04 - Pygame Jet Game with SR\Collision.ogg")
# move_up_sound = pygame.mixer.Sound("Rising_putter.ogg")
# move_down_sound = pygame.mixer.Sound("Falling_putter.ogg")
# collision_sound = pygame.mixer.Sound("Collision.ogg")

# Set the base volume for all sounds
move_up_sound.set_volume(0.)
move_down_sound.set_volume(0.)
collision_sound.set_volume(0.)

# Variable to keep our main loop running
running = True

# Our main loop
while running:
    # Look at every event in the queue
    for event in pygame.event.get():
        # Did the user hit a key?
        if event.type == KEYDOWN:
            # Was it the Escape key? If so, stop the loop
            if event.key == K_ESCAPE:
                running = False

        # Did the user click the window close button? If so, stop the loop
        elif event.type == QUIT:
            running = False

        # Should we add a new enemy?
        elif event.type == ADDENEMY:
            # Create the new enemy, and add it to our sprite groups
            new_enemy = Enemy()
            enemies.add(new_enemy)
            all_sprites.add(new_enemy)

        # Should we add a new cloud?
        elif event.type == ADDCLOUD:
            # Create the new cloud, and add it to our sprite groups
            new_cloud = Cloud()
            clouds.add(new_cloud)
            all_sprites.add(new_cloud)

    # Get the set of keys pressed and check for user input
    pressed_keys = pygame.key.get_pressed()
    player.update(pressed_keys)

    # Update the position of our enemies and clouds
    enemies.update()
    clouds.update()

    # Fill the screen with sky blue
    screen.fill((135, 206, 250))

    # Draw all our sprites
    for entity in all_sprites:
        screen.blit(entity.surf, entity.rect)

    """
    # Check if any enemies have collided with the player
    if pygame.sprite.spritecollideany(player, enemies):
        # If so, remove the player
        player.kill()

        # Stop any moving sounds and play the collision sound
        move_up_sound.stop()
        move_down_sound.stop()
        collision_sound.play()

        # Stop the loop
        running = False
    """

    # Flip everything to the display
    pygame.display.flip()

    frame = pygame.surfarray.array3d(screen)
    #frame = frame / 255.0  # Normalize pixel values
    #frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    # Convert the numpy array to a PyTorch Tensor
    frame = torch.from_numpy(frame).float()
    # PyTorch expects the input in the format (batch_size, channels, height, width)
    frame = frame.permute(2, 0, 1)
    # Add a batch dimension
    frame = frame.unsqueeze(0)

    output = model(frame)
    # Remove the batch dimension and reorder the dimensions
    output = output.squeeze(0).permute(1, 2, 0)
    # Convert the output Tensor to a numpy array
    output = output.detach().numpy()
    # Convert the numpy array to a Pygame Surface
    output_surface = pygame.surfarray.make_surface(output)
    # Blit the output surface to the screen
    screen.blit(output_surface, (0, 0))

    # Ensure we maintain a 60 frames per second rate
    clock.tick(60)
    

# At this point, we're done, so we can stop and quit the mixer
pygame.mixer.music.stop()
pygame.mixer.quit()
