from entities import rocket
from entities.core import SimEnvironment
import pygame
import numpy as np
import os
import random

class Animation:
    def __init__(self, fps: int | str = 60) -> None:
        pygame.init()
        self.infoObject = pygame.display.Info()
        self.screen = pygame.display.set_mode(
            (self.infoObject.current_w / 2, self.infoObject.current_h / 2)
        )
        self.clock = pygame.time.Clock()
        self.running = True

        # Zoom and pan variables
        self.zoom_scale = 1.0
        self.pan_offset = [0, 0]  # Additional offset for panning
        self.is_panning = False
        self.last_mouse_pos = (0, 0)
        
        self.center_offset = (
            int(self.infoObject.current_w / 4),
            int(self.infoObject.current_h / 4)
        )

        self.dt = 0

        if type(fps) == str:
            fps = int(fps)

        self.fps = fps
        
        # Load or create sprites
        self.load_sprites()
        
        # Cloud positions for background
        self.clouds = []
        self.init_clouds()
        
        # Star positions for sky
        self.stars = []
        self.init_stars()
        
        # Particle effects for rocket exhaust
        self.exhaust_particles = []
        
        # Static images to draw (new feature)
        self.static_images = []
        self.load_static_images()

    def load_sprites(self):
        """Load or create sprite images"""
        try:
            # Try to load rocket sprite from file
            self.rocket_sprite = pygame.image.load("rocket_sprite.png").convert_alpha()
        except:
            # Create a simple rocket sprite if file doesn't exist
            self.rocket_sprite = self.create_rocket_sprite()
        
        try:
            # Try to load ground texture from file
            self.ground_texture = pygame.image.load("ground_texture.png").convert()
        except:
            # Create a ground texture if file doesn't exist
            self.ground_texture = self.create_ground_texture()
            
        # Create sky gradient
        self.sky_surface = self.create_sky_gradient()

    def load_static_images(self):
        """Load static images from the folder"""
        # Create a list of supported image formats
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
        
        # Scan the current directory for image files
        for filename in os.listdir('./files'):
            # Check if file is an image
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                try:
                    # Skip images we're already using for sprites
                    if filename in ['rocket_sprite.png', 'ground_texture.png']:
                        continue
                        
                    # Load the image
                    image = pygame.image.load('./files/' + filename).convert_alpha()
                    
                    # Get the base name without extension for naming
                    name = os.path.splitext(filename)[0]
                    
                    # Add to static images list with default position (center of screen)
                    self.static_images.append({
                        'name': name,
                        'image': image,
                        'original_image': image.copy(),  # Keep original for rotation/scaling
                        'world_x': -6300,  # World X coordinate
                        'world_y': 0,  # World Y coordinate
                        'scale': 0.4,   # Scale factor
                        'angle': 0.0,   # Rotation angle in degrees
                        'visible': True
                    })
                    
                    print(f"Loaded static image: {filename}")
                    
                except Exception as e:
                    print(f"Failed to load image {filename}: {e}")

    def add_static_image(self, filename, world_x=0, world_y=0, scale=1.0, angle=0.0):
        """Add a static image at a specific world coordinate"""
        try:
            # Load the image
            image = pygame.image.load(filename).convert_alpha()
            
            # Get the base name without extension
            name = os.path.splitext(os.path.basename(filename))[0]
            
            # Add to static images list
            self.static_images.append({
                'name': name,
                'image': image,
                'original_image': image.copy(),  # Keep original for rotation/scaling
                'world_x': world_x,
                'world_y': world_y,
                'scale': scale,
                'angle': angle,
                'visible': True
            })
            
            print(f"Added static image '{name}' at position ({world_x}, {world_y})")
            return True
            
        except Exception as e:
            print(f"Failed to add image {filename}: {e}")
            return False

    def draw_static_images(self):
        """Draw all static images at their world coordinates"""
        for img_data in self.static_images:
            if not img_data['visible']:
                continue
                
            # Get image properties
            image = img_data['image']
            world_x = img_data['world_x']
            world_y = img_data['world_y']
            scale = img_data['scale'] * self.zoom_scale  # Apply current zoom
            angle = img_data['angle']
            
            # Convert world coordinates to screen coordinates
            screen_pos = self.world_to_screen((-world_x, -world_y))
            
            # Apply scale
            if scale != 1.0:
                original_size = img_data['original_image'].get_size()
                new_size = (int(original_size[0] * scale), 
                          int(original_size[1] * scale))
                image = pygame.transform.scale(img_data['original_image'], new_size)
            
            # Apply rotation if needed
            if angle != 0.0:
                image = pygame.transform.rotate(image, angle)
            
            # Get image rectangle and center it at screen position
            img_rect = image.get_rect()
            img_rect.center = screen_pos
            
            # Draw the image
            self.screen.blit(image, img_rect)

    def create_rocket_sprite(self):
        """Create a simple rocket sprite with colors and details"""
        # [Previous create_rocket_sprite code remains the same]
        width, height = 50, 100
        sprite = pygame.Surface((width, height), pygame.SRCALPHA)
        
        pygame.draw.rect(sprite, (200, 200, 200), (15, 0, 20, 70))
        pygame.draw.rect(sprite, (220, 60, 60), (15, 70, 20, 20))
        pygame.draw.polygon(sprite, (180, 180, 180), 
                           [(25, 0), (40, 15), (10, 15)])
        pygame.draw.circle(sprite, (30, 144, 255), (25, 30), 4)
        pygame.draw.circle(sprite, (30, 144, 255), (25, 50), 4)
        pygame.draw.polygon(sprite, (160, 160, 160), 
                           [(15, 85), (0, 100), (15, 100)])
        pygame.draw.polygon(sprite, (160, 160, 160), 
                           [(35, 85), (50, 100), (35, 100)])
        pygame.draw.ellipse(sprite, (100, 100, 100), 
                           (18, 90, 14, 10))
        
        return sprite

    def create_ground_texture(self):
        """Create a ground texture with grass and dirt"""
        # [Previous create_ground_texture code remains the same]
        texture_size = 64
        texture = pygame.Surface((texture_size, texture_size))
        
        texture.fill((34, 139, 34))
        
        for _ in range(20):
            x = random.randint(0, texture_size - 1)
            y = random.randint(0, texture_size // 2)
            color = random.choice([(28, 120, 28), (40, 150, 40), (50, 160, 50)])
            pygame.draw.line(texture, color, (x, y), (x, y + random.randint(2, 4)), 1)
        
        for _ in range(15):
            x = random.randint(0, texture_size - 1)
            y = random.randint(texture_size // 2, texture_size - 1)
            radius = random.randint(1, 3)
            color = random.choice([(139, 69, 19), (101, 67, 33), (84, 53, 24)])
            pygame.draw.circle(texture, color, (x, y), radius)
        
        return texture

    def create_sky_gradient(self):
        """Create a sky gradient background"""
        # [Previous create_sky_gradient code remains the same]
        sky = pygame.Surface((self.screen.get_width(), self.screen.get_height()))
        
        for y in range(sky.get_height()):
            blue_value = 135 + int(60 * (y / sky.get_height()))
            green_value = 206 + int(20 * (y / sky.get_height()))
            red_value = 235 - int(20 * (y / sky.get_height()))
            
            color = (red_value, green_value, blue_value)
            pygame.draw.line(sky, color, (0, y), (sky.get_width(), y))
        
        return sky

    def init_clouds(self):
        """Initialize cloud positions"""
        # [Previous init_clouds code remains the same]
        for _ in range(5):
            x = random.randint(0, self.screen.get_width())
            y = random.randint(20, self.screen.get_height() // 3)
            speed = random.uniform(0.1, 0.3)
            size = random.randint(30, 60)
            self.clouds.append({
                'x': x, 'y': y, 'speed': speed, 
                'size': size, 'density': random.uniform(0.7, 1.0)
            })

    def init_stars(self):
        """Initialize star positions (for night mode or background)"""
        # [Previous init_stars code remains the same]
        for _ in range(50):
            x = random.randint(0, self.screen.get_width())
            y = random.randint(0, self.screen.get_height() // 2)
            brightness = random.randint(200, 255)
            size = random.choice([1, 1, 1, 2])
            self.stars.append({'x': x, 'y': y, 'brightness': brightness, 'size': size})

    def draw_cloud(self, surface, x, y, size, density=1.0):
        """Draw a fluffy cloud"""
        # [Previous draw_cloud code remains the same]
        cloud_color = (255, 255, 255)
        
        alpha = int(230 * density)
        cloud_surface = pygame.Surface((size * 2, size), pygame.SRCALPHA)
        
        pygame.draw.circle(cloud_surface, (*cloud_color, alpha), 
                          (size // 2, size // 2), size // 2)
        pygame.draw.circle(cloud_surface, (*cloud_color, alpha), 
                          (size, size // 2), size // 2)
        pygame.draw.circle(cloud_surface, (*cloud_color, alpha), 
                          (size * 3 // 2, size // 2), size // 2)
        pygame.draw.circle(cloud_surface, (*cloud_color, alpha), 
                          (size, size // 3), size // 3)
        
        surface.blit(cloud_surface, (x - size, y - size // 2))

    def update_clouds(self):
        """Update cloud positions"""
        # [Previous update_clouds code remains the same]
        for cloud in self.clouds:
            cloud['x'] += cloud['speed']
            if cloud['x'] > self.screen.get_width() + cloud['size']:
                cloud['x'] = -cloud['size']
                cloud['y'] = random.randint(20, self.screen.get_height() // 3)

    def draw_exhaust(self, rocket_screen_pos, angle, engine_power=1.0):
        """Draw rocket exhaust particles"""
        # [Previous draw_exhaust code remains the same]
        exhaust_offset = 8 * self.zoom_scale
        rad_angle = np.radians(angle)
        
        exhaust_x = rocket_screen_pos[0] + np.sin(rad_angle) * exhaust_offset
        exhaust_y = rocket_screen_pos[1] - np.cos(rad_angle) * exhaust_offset
        
        for _ in range(3):
            particle_angle = angle + random.uniform(-10, 10)
            particle_speed = random.uniform(2, 5) * engine_power
            particle_size = random.uniform(2, 4) * self.zoom_scale
            particle_life = random.randint(20, 40)
            
            self.exhaust_particles.append({
                'x': exhaust_x,
                'y': exhaust_y,
                'vx': np.sin(np.radians(particle_angle)) * particle_speed,
                'vy': -np.cos(np.radians(particle_angle)) * particle_speed,
                'size': particle_size,
                'life': particle_life,
                'color': (255, random.randint(100, 200), 0)
            })
        
        for particle in self.exhaust_particles[:]:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['life'] -= 1
            particle['size'] *= 0.95
            
            if particle['life'] <= 0:
                self.exhaust_particles.remove(particle)
            else:
                alpha = int(255 * (particle['life'] / 40))
                particle_surface = pygame.Surface((int(particle['size'] * 2), 
                                                 int(particle['size'] * 2)), 
                                                 pygame.SRCALPHA)
                pygame.draw.circle(particle_surface, 
                                  (*particle['color'], alpha),
                                  (int(particle['size']), int(particle['size'])),
                                  int(particle['size']))
                self.screen.blit(particle_surface, 
                                (int(particle['x'] - particle['size']), 
                                 int(particle['y'] - particle['size'])))

    def handle_events(self):
        """Handle pygame events for zoom and pan"""
        # [Previous handle_events code remains the same]
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            elif event.type == pygame.MOUSEWHEEL:
                mouse_pos_before = pygame.mouse.get_pos()
                
                world_pos_before = self.screen_to_world(mouse_pos_before)
                
                if event.y > 0:
                    self.zoom_scale *= 1.1
                elif event.y < 0:
                    self.zoom_scale *= 0.9
                
                self.zoom_scale = max(0.1, min(10.0, self.zoom_scale))
                
                world_pos_after = self.screen_to_world(mouse_pos_before)
                
                self.pan_offset[0] += (world_pos_after[0] - world_pos_before[0]) * self.zoom_scale
                self.pan_offset[1] += (world_pos_after[1] - world_pos_before[1]) * self.zoom_scale
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.is_panning = True
                    self.last_mouse_pos = event.pos
                    
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.is_panning = False
                    
            elif event.type == pygame.MOUSEMOTION:
                if self.is_panning:
                    dx = event.pos[0] - self.last_mouse_pos[0]
                    dy = event.pos[1] - self.last_mouse_pos[1]
                    
                    self.pan_offset[0] += dx
                    self.pan_offset[1] += dy
                    
                    self.last_mouse_pos = event.pos
        
        return True

    def screen_to_world(self, screen_pos):
        """Convert screen coordinates to world coordinates"""
        # [Previous screen_to_world code remains the same]
        x, y = screen_pos
        world_x = (x - self.center_offset[0] - self.pan_offset[0]) / self.zoom_scale
        world_y = (y - self.center_offset[1] - self.pan_offset[1]) / self.zoom_scale
        return (world_x, world_y)

    def world_to_screen(self, world_pos):
        """Convert world coordinates to screen coordinates"""
        # [Previous world_to_screen code remains the same]
        x, y = world_pos
        screen_x = x * self.zoom_scale + self.center_offset[0] + self.pan_offset[0]
        screen_y = y * self.zoom_scale + self.center_offset[1] + self.pan_offset[1]
        return (screen_x, screen_y)

    def update(self, env: SimEnvironment):
        # Handle events (zoom and pan)
        if not self.handle_events():
            return

        # Draw sky background
        self.screen.blit(self.sky_surface, (0, 0))
        
        # Draw stars
        for star in self.stars:
            pygame.draw.circle(self.screen, 
                             (star['brightness'], star['brightness'], star['brightness']),
                             (star['x'], star['y']), star['size'])
        
        # Update and draw clouds
        self.update_clouds()
        for cloud in self.clouds:
            self.draw_cloud(self.screen, int(cloud['x']), int(cloud['y']), 
                          cloud['size'], cloud['density'])

        # DRAW STATIC IMAGES (NEW FEATURE)
        self.draw_static_images()

        rockets = env.rockets
        for myrocket in rockets:
            # Convert world coordinates to screen coordinates with zoom and pan
            rocket_screen_pos = self.world_to_screen(
                (-myrocket.body_model.current_coordinates[0],
                 -myrocket.body_model.current_coordinates[1])
            )
            
            # Calculate ground position (scaled and panned)
            ground_top = self.world_to_screen((0, myrocket.body_model.dims[1]))[1]
            ground_height = self.screen.get_height() - ground_top
            
            # Draw ground with tiled texture
            if ground_height > 0:
                texture_width = self.ground_texture.get_width()
                texture_height = self.ground_texture.get_height()
                
                for x in range(0, self.screen.get_width(), texture_width):
                    scaled_texture = pygame.transform.scale(
                        self.ground_texture,
                        (texture_width, int(ground_height))
                    )
                    self.screen.blit(scaled_texture, (x, ground_top))
            
            # Scale rocket sprite based on zoom and rocket dimensions
            original_width, original_height = self.rocket_sprite.get_size()
            scale_factor = (myrocket.body_model.dims[0] / 50) * self.zoom_scale
            
            scaled_width = int(original_width * scale_factor)
            scaled_height = int(original_height * scale_factor)
            
            scaled_rocket = pygame.transform.scale(
                self.rocket_sprite, 
                (scaled_width, scaled_height)
            )
            
            angle = myrocket.body_model.tilt * 180 / np.pi
            rotated_rocket = pygame.transform.rotate(
                scaled_rocket, angle=angle * -1 % 360
            )
            
            rotated_rect = rotated_rocket.get_rect()
            rotated_rect.center = rocket_screen_pos
            
            # Draw exhaust particles
            engine_power = 0.5
            self.draw_exhaust(rocket_screen_pos, angle - 180, engine_power)
            
            # Blit the rotated rocket onto the screen
            self.screen.blit(rotated_rocket, rotated_rect)
            
            # Draw rocket shadow on ground
            shadow_offset = 5
            shadow_surface = pygame.Surface((scaled_width, scaled_height // 4), pygame.SRCALPHA)
            shadow_surface.fill((0, 0, 0, 100))
            shadow_rect = shadow_surface.get_rect()
            shadow_rect.center = (rocket_screen_pos[0], ground_top)
            self.screen.blit(shadow_surface, shadow_rect)
            
        # Draw horizon line
        if rockets:
            horizon_y = ground_top
            pygame.draw.line(self.screen, (100, 100, 150), 
                           (0, horizon_y), (self.screen.get_width(), horizon_y), 2)
            
        pygame.display.flip()

    def __del__(self):
        pygame.quit()