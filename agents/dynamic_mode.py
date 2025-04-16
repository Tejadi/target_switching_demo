import random
import math
import pygame

class DynamicMode:
    def __init__(self, color, update_func):
        self.color = color
        self.update_func = update_func
        
    def update(self, target, dt):
        return self.update_func(target, dt)

def spiral_motion(target, dt):
    time_val = pygame.time.get_ticks() / 1000
    radius = 50 + time_val % 10 * 5
    angular_velocity = 3.0
    target.vx = radius * angular_velocity * math.cos(angular_velocity * time_val)
    target.vy = radius * angular_velocity * math.sin(angular_velocity * time_val)
    return target.vx * dt, target.vy * dt

def bounce_motion(target, dt):
    speed = 120
    return target.vx * dt, target.vy * dt

def pursuit_motion(target, dt, ego_pos=None):
    if ego_pos is None:
        return random_walk(target, dt)
        
    pursuit_speed = 60
    dx = ego_pos[0] - target.x
    dy = ego_pos[1] - target.y
    dist = math.sqrt(dx**2 + dy**2)
    
    if dist > 0:
        dx = dx / dist * pursuit_speed
        dy = dy / dist * pursuit_speed
    
    return dx * dt, dy * dt

def evasion_motion(target, dt, ego_pos=None):
    if ego_pos is None:
        return random_walk(target, dt)
        
    evasion_speed = 90
    dx = target.x - ego_pos[0]
    dy = target.y - ego_pos[1]
    dist = math.sqrt(dx**2 + dy**2)
    
    if dist > 0:
        dx = dx / dist * evasion_speed
        dy = dy / dist * evasion_speed
    
    return dx * dt, dy * dt

def oscillating_motion(target, dt):
    time_val = pygame.time.get_ticks() / 1000
    amplitude = 30 + 20 * math.sin(time_val / 5)
    speed = 70
    target.vx = amplitude * math.sin(time_val * 2)
    return speed * dt, target.vx * dt

def linear_motion(target, dt):
    speed = 100
    return speed * dt, 0

def sine_wave_motion(target, dt):
    speed = 80
    target.vy = 50 * math.sin(pygame.time.get_ticks() / 500)
    return speed * dt, target.vy * dt

def circular_motion(target, dt):
    speed = 100
    time_val = pygame.time.get_ticks() / 1000
    target.vx = speed * math.cos(time_val)
    target.vy = speed * math.sin(time_val)
    return target.vx * dt, target.vy * dt

def random_walk(target, dt):
    if random.random() < 0.05:
        target.vx = random.uniform(-100, 100)
        target.vy = random.uniform(-100, 100)
    return target.vx * dt, target.vy * dt

def zigzag_motion(target, dt):
    speed = 120
    if int(pygame.time.get_ticks() / 1000) % 2 == 0:
        return speed * dt, speed * 0.5 * dt
    else:
        return -speed * dt, -speed * 0.5 * dt