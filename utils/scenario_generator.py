import random
from agents.dynamic_mode import (
    DynamicMode,
    linear_motion, 
    sine_wave_motion, 
    circular_motion, 
    random_walk, 
    zigzag_motion,
    spiral_motion,
    bounce_motion,
    pursuit_motion,
    evasion_motion,
    oscillating_motion
)


def setup_motion_modes(target_agent, green_color):

    target_agent.add_mode(DynamicMode(green_color, linear_motion))
    target_agent.add_mode(DynamicMode(green_color, sine_wave_motion))
    target_agent.add_mode(DynamicMode(green_color, circular_motion))
    target_agent.add_mode(DynamicMode(green_color, random_walk))
    target_agent.add_mode(DynamicMode(green_color, zigzag_motion))
    target_agent.add_mode(DynamicMode(green_color, spiral_motion))
    target_agent.add_mode(DynamicMode(green_color, bounce_motion))
    

    pursuit_wrapper = lambda target, dt: pursuit_motion(target, dt)
    evasion_wrapper = lambda target, dt: evasion_motion(target, dt)
    
    target_agent.add_mode(DynamicMode(green_color, pursuit_wrapper))
    target_agent.add_mode(DynamicMode(green_color, evasion_wrapper))
    target_agent.add_mode(DynamicMode(green_color, oscillating_motion))

def create_random_motion_set(target_agent, green_color, num_modes=5):

    all_modes = [
        linear_motion,
        sine_wave_motion,
        circular_motion,
        random_walk,
        zigzag_motion,
        spiral_motion,
        bounce_motion,
        lambda target, dt: pursuit_motion(target, dt),
        lambda target, dt: evasion_motion(target, dt),
        oscillating_motion
    ]
    
    selected_modes = random.sample(all_modes, min(num_modes, len(all_modes)))
    
    for mode in selected_modes:
        target_agent.add_mode(DynamicMode(green_color, mode))