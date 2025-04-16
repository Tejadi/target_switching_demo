import math
import pygame
from planning.valiant_estimator import ValiantEstimator
from planning.mpc import CasADiMPC
from constants import RED, YELLOW, BLUE

class CasADiEgoAgent:
    def __init__(self, start_pos, goal_pos, target, obstacles, radius=15, target_bound=0.92):
        self.x, self.y = start_pos
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.radius = radius
        self.speed = 80
        self.vx, self.vy = 0, 0
        self.target = target
        self.obstacles = obstacles
        self.observation_interval = 0.2
        self.observation_timer = 0
        self.estimator = ValiantEstimator(target_bound)
        
        self.mpc = CasADiMPC(target, self.estimator, obstacles)
        
        self.planned_trajectory = []
        self.position_history = []
        self.max_history = 100
        self.at_goal = False
        self.collision = False
        self.collision_with_obstacle = False
        self.planning_timer = 0
        self.planning_interval = 0.5
        self.target_bound = target_bound
        self.sufficient_samples = False
        self.using_conservative_trajectory = True
        self.min_samples_required = 20 
        
    def reset(self):
        self.x, self.y = self.start_pos
        self.vx, self.vy = 0, 0
        self.position_history = []
        self.planned_trajectory = []
        self.at_goal = False
        self.collision = False
        self.collision_with_obstacle = False
        self.sufficient_samples = False
        self.using_conservative_trajectory = True
        self.estimator = ValiantEstimator(self.target_bound)
        
    def update(self, dt):
        self.position_history.append((self.x, self.y))
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
            
        dist_to_goal = math.sqrt((self.x - self.goal_pos[0])**2 + (self.y - self.goal_pos[1])**2)
        if dist_to_goal < self.radius:
            self.at_goal = True
            return
            

        dist_to_target = math.sqrt((self.x - self.target.x)**2 + (self.y - self.target.y)**2)
        if dist_to_target < (self.radius + self.target.radius):
            self.collision = True
            

        for obstacle in self.obstacles:
            if obstacle.check_collision(self.x, self.y, self.radius):
                self.collision_with_obstacle = True
                break
            
        self.observation_timer += dt
        if self.observation_timer >= self.observation_interval:
            self.observation_timer = 0
            
            self.estimator.add_observation(self.target.current_mode_idx)
            

            current_bound = self.estimator.support_estimate_bound()
            self.sufficient_samples = (current_bound >= self.target_bound)
                
        self.planning_timer += dt
        if self.planning_timer >= self.planning_interval or not self.planned_trajectory or len(self.planned_trajectory) <= 1:
            self.planning_timer = 0
            
            current_state = [self.x, self.y, self.vx, self.vy]
            

            current_bound = self.estimator.support_estimate_bound()
            self.sufficient_samples = (current_bound >= self.target_bound)
            
            if self.sufficient_samples:
                mode_probs = self.estimator.get_mode_probabilities()
                ucb_probs = self.estimator.calculate_ucb(mode_probs)
                
                self.estimator.estimated_modes['probabilities'] = ucb_probs
                self.planned_trajectory = self.mpc.plan_trajectory(current_state, self.goal_pos)
                self.using_conservative_trajectory = False 
            else:
                self.planned_trajectory = self.mpc.plan_conservative_trajectory(current_state, self.goal_pos)
                self.using_conservative_trajectory = True  
            
        if self.planned_trajectory and len(self.planned_trajectory) > 1:
            next_pos = self.planned_trajectory[1]
            
            dx = next_pos[0] - self.x
            dy = next_pos[1] - self.y
            dist = math.sqrt(dx**2 + dy**2)
            
            if dist > 0:
                dx /= dist
                dy /= dist
                
                move_dist = min(dist, self.speed * dt)
                new_x = self.x + dx * move_dist
                new_y = self.y + dy * move_dist
                

                collision = False
                for obstacle in self.obstacles:
                    if obstacle.check_collision(new_x, new_y, self.radius):
                        collision = True
                        break
                
                if not collision:
                    self.x = new_x
                    self.y = new_y
                    self.vx = dx * self.speed
                    self.vy = dy * self.speed
                else:

                    self.planning_timer = self.planning_interval
                
            self.planned_trajectory.pop(0)
    
    def draw(self, surface):
        for i, pos in enumerate(self.position_history[::3]):
            if i > 0 and i*3 < len(self.position_history)-3:
                alpha = int(128 * (i / (len(self.position_history)/3)))
                color = (min(alpha+128, 255), 0, 0)
                pygame.draw.line(surface, color, self.position_history[i*3-3], pos, 2)
        
        for i in range(len(self.planned_trajectory) - 1):
            pygame.draw.line(surface, BLUE, self.planned_trajectory[i], self.planned_trajectory[i+1], 1)
            
        if self.collision or self.collision_with_obstacle:
            color = YELLOW
        elif self.at_goal:
            color = YELLOW
        else:
            color = RED
            
        pygame.draw.circle(surface, color, (int(self.x), int(self.y)), self.radius)
        
        pygame.draw.circle(surface, YELLOW, self.goal_pos, 15, 2)
        if self.using_conservative_trajectory:
            pygame.draw.circle(surface, (230, 230, 230), (int(self.x), int(self.y)), 200, 1)
        else:
            pygame.draw.circle(surface, (230, 230, 230), (int(self.x), int(self.y)), 100, 1)