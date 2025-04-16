import pygame
from constants import DARK_GRAY, GRAY

class Obstacle:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.rect = pygame.Rect(x, y, width, height)
        
    def draw(self, surface):
        pygame.draw.rect(surface, DARK_GRAY, self.rect)
        pygame.draw.rect(surface, GRAY, self.rect, 2)
        
    def check_collision(self, x, y, radius):
        expanded_rect = pygame.Rect(
            self.x - radius, 
            self.y - radius, 
            self.width + 2 * radius, 
            self.height + 2 * radius
        )
        return expanded_rect.collidepoint(x, y)

def create_obstacles():
    obstacles = []
    
    obstacles.append(Obstacle(400, 200, 200, 40))  
    obstacles.append(Obstacle(300, 350, 40, 200)) 
    obstacles.append(Obstacle(600, 300, 40, 250)) 
    
    return obstacles