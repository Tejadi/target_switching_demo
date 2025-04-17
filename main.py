from cgitb import reset

import pygame
import sys
import math
import random
from constants import WIDTH, HEIGHT, WHITE
from obstacles import create_obstacles
from agents.target_agent import TargetAgent
from agents.ego_agent import CasADiEgoAgent
from agents.dynamic_mode import DynamicMode
from agents.dynamic_mode import (
    linear_motion, sine_wave_motion, circular_motion, random_walk,
    zigzag_motion, spiral_motion, bounce_motion, oscillating_motion
)
from utils.visualization import (
    draw_legend, draw_estimation_stats, create_standard_legend, draw_text
)

def setup_simulation():
    obstacles = create_obstacles()
    
    target = TargetAgent(WIDTH // 2, HEIGHT // 2) #random.randint(2,4)
    target.add_mode(DynamicMode(GREEN, linear_motion))
    target.add_mode(DynamicMode(GREEN, sine_wave_motion))
    target.add_mode(DynamicMode(GREEN, circular_motion))
    target.add_mode(DynamicMode(GREEN, random_walk))
    target.add_mode(DynamicMode(GREEN, zigzag_motion))
    target.add_mode(DynamicMode(GREEN, spiral_motion))
    target.add_mode(DynamicMode(GREEN, bounce_motion))
    
    # Wrap functions that require additional parameters
    pursuit_wrapper = lambda target, dt: pursuit_motion(target, dt)
    evasion_wrapper = lambda target, dt: evasion_motion(target, dt)
    
    target.add_mode(DynamicMode(GREEN, pursuit_wrapper))
    target.add_mode(DynamicMode(GREEN, evasion_wrapper))
    target.add_mode(DynamicMode(GREEN, oscillating_motion))
    
    start_pos = (100, 100)
    goal_pos = (WIDTH - 100, HEIGHT - 100)
    
    ego = CasADiEgoAgent(start_pos, goal_pos, target, obstacles)
    
    return target, ego, obstacles

def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("CasADi MPC with Valiant Estimation and Reduced Obstacles")
    
    target, ego, obstacles = setup_simulation()

    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    

    legend_items = create_standard_legend()
    completed_runs = 0
    runtimes = []
    collision_counter = 0
    while completed_runs < 100:
        running = True
        dt = 0.016
        reset_timer = 0
        reset_interval = 30

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        ego.reset()
                        target.stopped = False
                        reset_timer = 0
                    elif event.key == pygame.K_SPACE:
                        if dt > 0:
                            dt = 0
                        else:
                            dt = 0.016

            if ego.at_goal:
                reached_goal_time = reset_timer
                runtimes.append(reached_goal_time)
                completed_runs+=1
                running = False
                ego.reset()
                target.stopped = False
                reset_timer = 0
                target.reset()

            reset_timer += dt
            if reset_timer >= reset_interval or ego.collision or ego.collision_with_obstacle:
                if ego.collision or ego.collision_with_obstacle:
                    collision_counter += 1
                    pygame.time.delay(1000)

                if not ego.at_goal:
                    ego.reset()
                    target.stopped = False
                    reset_timer = 0


            target.update(dt, obstacles, should_stop=ego.at_goal)
            ego.update(dt)

            screen.fill(WHITE)

            # Draw obstacles
            for obstacle in obstacles:
                obstacle.draw(screen)

            ego.mpc.draw_scenarios(screen)

            target.draw(screen)
            ego.draw(screen)


            draw_estimation_stats(screen, font, ego)


            legend_y = draw_legend(screen, font, legend_items, (WIDTH - 200, 10))


            draw_text(screen, font, "Magenta arrow: Average target trajectory forecast", (10 , 220), MAGENTA)

            pygame.display.flip()

            if dt > 0:
                clock.tick(60)
    print("Runtimes: " + str(runtimes))
    print("Total collisions: " + str(collision_counter))
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    from constants import GREEN, MAGENTA
    from agents.dynamic_mode import pursuit_motion, evasion_motion
    main()