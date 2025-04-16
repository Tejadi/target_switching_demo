import pygame
import random
from constants import GREEN, WIDTH, HEIGHT

class TargetAgent:
    def __init__(self, x, y, radius=15):
        self.origin_x = x
        self.origin_y = y
        self.x = x
        self.y = y
        self.radius = radius
        self.vx = 50
        self.vy = 0
        self.modes = []
        self.current_mode_idx = 0
        self.switch_timer = 0
        self.switch_interval = 1.0
        self.mode_history = []
        self.position_history = []
        self.max_history = 100
        self.stopped = False

    def reset(self):
        self.x = self.origin_x
        self.y = self.origin_y
        self.vx = 50
        self.vy = 0
        self.mode_history = []
        self.position_history = []
        self.current_mode_idx = 0
        self.switch_timer = 0
        
    def add_mode(self, mode):
        self.modes.append(mode)
        
    def update(self, dt, obstacles, should_stop=False):
        if should_stop:
            self.stopped = True
            
        if self.stopped:
            self.position_history.append((self.x, self.y))
            if len(self.position_history) > self.max_history:
                self.position_history.pop(0)
            return
            
        self.switch_timer += dt
        if self.switch_timer >= self.switch_interval:
            self.switch_timer = 0
            self.current_mode_idx = random.randint(0, len(self.modes) - 1)
            self.mode_history.append(self.current_mode_idx)
        
        if self.modes:
            dx, dy = self.modes[self.current_mode_idx].update(self, dt)
            
            new_x = self.x + dx
            new_y = self.y + dy
            
            collision = False
            for obstacle in obstacles:
                if obstacle.check_collision(new_x, new_y, self.radius):
                    collision = True
                    break
            
            if not collision:
                self.x = new_x
                self.y = new_y
            else:
                self.vx *= -1
                self.vy *= -1
            
            if self.x < self.radius:
                self.x = self.radius
                self.vx *= -1
            if self.x > WIDTH - self.radius:
                self.x = WIDTH - self.radius
                self.vx *= -1
            if self.y < self.radius:
                self.y = self.radius
                self.vy *= -1
            if self.y > HEIGHT - self.radius:
                self.y = HEIGHT - self.radius
                self.vy *= -1
                
        self.position_history.append((self.x, self.y))
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
    
    def draw(self, surface):
        for i, pos in enumerate(self.position_history[::3]):
            if i > 0 and i*3 < len(self.position_history)-3:
                alpha = int(128 * (i / (len(self.position_history)/3)))
                color = (0, min(alpha+128, 255), 0)
                pygame.draw.line(surface, color, self.position_history[i*3-3], pos, 1)
        
        if self.modes:
            pygame.draw.circle(surface, GREEN, (int(self.x), int(self.y)), self.radius)