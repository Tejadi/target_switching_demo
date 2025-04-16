import casadi as ca
import numpy as np
import pygame
import random
import math
from constants import WIDTH, HEIGHT, MAGENTA

class CasADiMPC:
    def __init__(self, target_agent, estimator, obstacles, horizon=10, dt=0.1):
        self.target_agent = target_agent
        self.estimator = estimator
        self.obstacles = obstacles
        self.horizon = horizon
        self.dt = dt
        self.max_speed = 80
        self.safety_distance = 100
        self.obstacle_safety_distance = 50
        
        self.nx = 4
        self.nu = 2
        self.nvar = self.nx * (self.horizon + 1) + self.nu * self.horizon
        
        self.x_min, self.x_max = 0, WIDTH
        self.y_min, self.y_max = 0, HEIGHT
        self.v_min, self.v_max = -100, 100
        self.a_min, self.a_max = -30, 30
        
        self.opti = ca.Opti()
        
        self.X = self.opti.variable(self.nx, self.horizon + 1)
        self.U = self.opti.variable(self.nu, self.horizon)
        
        self.P_target = self.opti.parameter(2, self.horizon)
        self.P_initial = self.opti.parameter(self.nx)
        self.P_goal = self.opti.parameter(2)
        self.P_confidence = self.opti.parameter(1)
        
        self.scenarios = []
        self.planned_trajectory = []
        self.average_target_trajectory = []
        
        self.setup_optimization_problem()
        
    def set_mode_probabilities(self, probabilities):
        pass
        
    def setup_optimization_problem(self):
        self.opti.subject_to(self.X[:, 0] == self.P_initial)
        
        for k in range(self.horizon):
            x_next = self.X[0, k] + self.X[2, k] * self.dt + 0.5 * self.U[0, k] * self.dt**2
            y_next = self.X[1, k] + self.X[3, k] * self.dt + 0.5 * self.U[1, k] * self.dt**2
            vx_next = self.X[2, k] + self.U[0, k] * self.dt
            vy_next = self.X[3, k] + self.U[1, k] * self.dt
            
            self.opti.subject_to(self.X[0, k+1] == x_next)
            self.opti.subject_to(self.X[1, k+1] == y_next)
            self.opti.subject_to(self.X[2, k+1] == vx_next)
            self.opti.subject_to(self.X[3, k+1] == vy_next)
        
        for k in range(self.horizon + 1):
            self.opti.subject_to(self.X[0, k] >= self.x_min + 10)
            self.opti.subject_to(self.X[0, k] <= self.x_max - 10)
            self.opti.subject_to(self.X[1, k] >= self.y_min + 10)
            self.opti.subject_to(self.X[1, k] <= self.y_max - 10)
            self.opti.subject_to(self.X[2, k] >= self.v_min)
            self.opti.subject_to(self.X[2, k] <= self.v_max)
            self.opti.subject_to(self.X[3, k] >= self.v_min)
            self.opti.subject_to(self.X[3, k] <= self.v_max)
        
        for k in range(self.horizon):
            self.opti.subject_to(self.U[0, k] >= self.a_min)
            self.opti.subject_to(self.U[0, k] <= self.a_max)
            self.opti.subject_to(self.U[1, k] >= self.a_min)
            self.opti.subject_to(self.U[1, k] <= self.a_max)
        
        obj = 0
        

        for k in range(self.horizon + 1):
            goal_dist = ca.sumsqr(self.X[:2, k] - self.P_goal)
            stage_weight = 1.0 + k / self.horizon
            obj += stage_weight * goal_dist
        

        control_weight = 0.1
        for k in range(self.horizon):
            obj += control_weight * ca.sumsqr(self.U[:, k])
        

        collision_weight = 10.0 * (1.0 + 5.0 * (1.0 - self.P_confidence))
        for k in range(self.horizon):
            target_dist = ca.sumsqr(self.X[:2, k] - self.P_target[:, k])
            collision_cost = ca.fmax(0, self.safety_distance**2 - target_dist)
            obj += collision_weight * collision_cost
        

        obstacle_weight = 50.0
        for obstacle in self.obstacles:
            for k in range(self.horizon + 1):
                obstacle_center_x = obstacle.x + obstacle.width / 2
                obstacle_center_y = obstacle.y + obstacle.height / 2
                
                dx = ca.fmax(0, ca.fabs(self.X[0, k] - obstacle_center_x) - obstacle.width / 2)
                dy = ca.fmax(0, ca.fabs(self.X[1, k] - obstacle_center_y) - obstacle.height / 2)
                squared_dist = dx**2 + dy**2
                
                obstacle_cost = ca.fmax(0, self.obstacle_safety_distance**2 - squared_dist)
                obj += obstacle_weight * (obstacle_cost + 0.1 * ca.exp(0.05 * obstacle_cost))
        
        self.opti.minimize(obj)
        
        p_opts = {"expand": True}
        s_opts = {"max_iter": 100, "print_level": 0}
        self.opti.solver("ipopt", p_opts, s_opts)
    
    def generate_target_scenarios(self):
        scenarios = []
        
        mode_probs = self.estimator.get_mode_probabilities()
        
        if not mode_probs:
            if self.target_agent.modes:
                uniform_prob = 1.0 / len(self.target_agent.modes)
                mode_probs = {i: uniform_prob for i in range(len(self.target_agent.modes))}
            else:
                return scenarios
                
        current_pos = (self.target_agent.x, self.target_agent.y)
        
        num_scenarios = 20
        for _ in range(num_scenarios):
            scenario = [current_pos]
            
            vx, vy = self.target_agent.vx, self.target_agent.vy
            
            x, y = current_pos
            
            for t in range(self.horizon):
                modes = list(mode_probs.keys())
                weights = list(mode_probs.values())
                
                if sum(weights) > 0:
                    weights = [w/sum(weights) for w in weights]
                else:
                    weights = [1.0/len(modes)] * len(modes)
                
                if modes:
                    mode_idx = random.choices(modes, weights=weights)[0]
                    
                    if mode_idx < len(self.target_agent.modes):
                        temp_target = type('obj', (), {'x': x, 'y': y, 'vx': vx, 'vy': vy})
                        dx, dy = self.target_agent.modes[mode_idx].update_func(temp_target, self.dt)
                        
                        new_x = x + dx
                        new_y = y + dy
                        
                        collision = False
                        for obstacle in self.obstacles:
                            expanded_rect = pygame.Rect(
                                obstacle.x - self.target_agent.radius, 
                                obstacle.y - self.target_agent.radius, 
                                obstacle.width + 2 * self.target_agent.radius, 
                                obstacle.height + 2 * self.target_agent.radius
                            )
                            if expanded_rect.collidepoint(new_x, new_y):
                                collision = True
                                break
                                
                        if not collision:
                            x = new_x
                            y = new_y
                        else:
                            vx *= -1
                            vy *= -1
                            
                        vx, vy = (x - scenario[-1][0]) / self.dt, (y - scenario[-1][1]) / self.dt
                        
                        if x < 15:
                            x = 15
                            vx *= -1
                        if x > WIDTH - 15:
                            x = WIDTH - 15
                            vx *= -1
                        if y < 15:
                            y = 15
                            vy *= -1
                        if y > HEIGHT - 15:
                            y = HEIGHT - 15
                            vy *= -1
                            
                        scenario.append((x, y))
            
            scenarios.append(scenario)
            
        self.scenarios = scenarios
        return scenarios
    
    def plan_trajectory(self, current_state, goal_pos):
        scenarios = self.generate_target_scenarios()
        
        if not scenarios:
            return self.plan_direct_trajectory(current_state, goal_pos)
            
        target_traj = np.zeros((2, self.horizon))
        avg_trajectory_points = []
        
        for t in range(self.horizon):
            positions = [s[min(t, len(s)-1)] for s in scenarios]
            x_mean = sum(p[0] for p in positions) / len(positions)
            y_mean = sum(p[1] for p in positions) / len(positions)
            target_traj[0, t] = x_mean
            target_traj[1, t] = y_mean
            avg_trajectory_points.append((x_mean, y_mean))
        
        self.average_target_trajectory = avg_trajectory_points
            
        self.opti.set_value(self.P_initial, current_state)
        self.opti.set_value(self.P_goal, goal_pos)
        self.opti.set_value(self.P_target, target_traj)
        
        confidence = self.estimator.support_estimate_bound()
        self.opti.set_value(self.P_confidence, confidence)
        
        try:
            sol = self.opti.solve()
            
            X_opt = sol.value(self.X)
            U_opt = sol.value(self.U)
            
            traj = [(X_opt[0, k], X_opt[1, k]) for k in range(self.horizon + 1)]
            self.planned_trajectory = traj
            
            return traj
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            return self.plan_direct_trajectory(current_state, goal_pos)
            
    def plan_direct_trajectory(self, current_state, goal_pos):
        x, y, vx, vy = current_state
        traj = [(x, y)]
        
        dx_goal = goal_pos[0] - x
        dy_goal = goal_pos[1] - y
        dist_to_goal = math.sqrt(dx_goal**2 + dy_goal**2)
        
        if dist_to_goal < 1:
            return traj
            
        dx_goal /= dist_to_goal
        dy_goal /= dist_to_goal
        

        for _ in range(self.horizon):

            blocked = False
            for obstacle in self.obstacles:

                expanded_rect = pygame.Rect(
                    obstacle.x - 20, 
                    obstacle.y - 20, 
                    obstacle.width + 40, 
                    obstacle.height + 40
                )
                if expanded_rect.clipline((x, y), (goal_pos[0], goal_pos[1])):
                    blocked = True
                    break
            

            if blocked:

                nearest_dist = float('inf')
                nearest_obstacle = None
                
                for obstacle in self.obstacles:

                    obs_center_x = obstacle.x + obstacle.width / 2
                    obs_center_y = obstacle.y + obstacle.height / 2
                    

                    obs_dist = math.sqrt((x - obs_center_x)**2 + (y - obs_center_y)**2)
                    
                    if obs_dist < nearest_dist:
                        nearest_dist = obs_dist
                        nearest_obstacle = obstacle
                
                if nearest_obstacle:

                    obs_center_x = nearest_obstacle.x + nearest_obstacle.width / 2
                    obs_center_y = nearest_obstacle.y + nearest_obstacle.height / 2
                    

                    avoid_dx = x - obs_center_x
                    avoid_dy = y - obs_center_y
                    

                    avoid_dist = math.sqrt(avoid_dx**2 + avoid_dy**2)
                    if avoid_dist > 0:
                        avoid_dx /= avoid_dist
                        avoid_dy /= avoid_dist
                    

                    blend_factor = min(1.0, 100 / nearest_dist) 
                    dx = dx_goal * (1 - blend_factor) + avoid_dx * blend_factor
                    dy = dy_goal * (1 - blend_factor) + avoid_dy * blend_factor
                    

                    d_norm = math.sqrt(dx**2 + dy**2)
                    if d_norm > 0:
                        dx /= d_norm
                        dy /= d_norm
                else:
                    dx, dy = dx_goal, dy_goal
            else:
                dx, dy = dx_goal, dy_goal
            
            move_dist = min(dist_to_goal, self.max_speed * self.dt)
            
            next_x = x + dx * move_dist
            next_y = y + dy * move_dist
            

            collision = False
            for obstacle in self.obstacles:
                expanded_rect = pygame.Rect(
                    obstacle.x - 20, 
                    obstacle.y - 20, 
                    obstacle.width + 40, 
                    obstacle.height + 40
                )
                if expanded_rect.collidepoint(next_x, next_y):
                    collision = True
                    break
            
            if not collision:
                x, y = next_x, next_y
                traj.append((x, y))
                

                dist_to_goal = math.sqrt((goal_pos[0] - x)**2 + (goal_pos[1] - y)**2)
                
                if dist_to_goal < 1:
                    break
            else:

                best_angle = 0
                best_dist = float('inf')
                
                for angle in range(0, 360, 30):
                    rad = math.radians(angle)
                    test_dx = math.cos(rad)
                    test_dy = math.sin(rad)
                    test_x = x + test_dx * move_dist
                    test_y = y + test_dy * move_dist
                    

                    test_collision = False
                    for obstacle in self.obstacles:
                        expanded_rect = pygame.Rect(
                            obstacle.x - 20, 
                            obstacle.y - 20, 
                            obstacle.width + 40, 
                            obstacle.height + 40
                        )
                        if expanded_rect.collidepoint(test_x, test_y):
                            test_collision = True
                            break
                    
                    if not test_collision:

                        test_dist = math.sqrt((goal_pos[0] - test_x)**2 + (goal_pos[1] - test_y)**2)
                        if test_dist < best_dist:
                            best_dist = test_dist
                            best_angle = angle
                
                if best_dist < float('inf'):
                    rad = math.radians(best_angle)
                    dx = math.cos(rad)
                    dy = math.sin(rad)
                    x += dx * move_dist
                    y += dy * move_dist
                    traj.append((x, y))
                else:

                    traj.append((x, y))
                
        return traj
        
    def plan_conservative_trajectory(self, current_state, goal_pos):
        if self.target_agent.modes:
            uniform_prob = 1.0 / len(self.target_agent.modes)
            self.estimator.estimated_modes['probabilities'] = {i: uniform_prob for i in range(len(self.target_agent.modes))}
        
        original_safety_distance = self.safety_distance
        self.safety_distance *= 2.0  
        
        traj = self.plan_trajectory(current_state, goal_pos)
        
        self.safety_distance = original_safety_distance
        
        return traj
        
    def draw_scenarios(self, surface):
        subset_size = min(5, len(self.scenarios))
        if subset_size == 0:
            return
            
        subset = random.sample(self.scenarios, subset_size)
        
        for scenario in subset:
            color = (180, 180, 180)
            
            if len(scenario) > 1:
                pygame.draw.lines(surface, color, False, scenario, 1)
        

        if self.average_target_trajectory and len(self.average_target_trajectory) > 1:

            pygame.draw.lines(surface, MAGENTA, False, self.average_target_trajectory, 2)
            

            if len(self.average_target_trajectory) > 2:
                end_point = self.average_target_trajectory[-1]
                prev_point = self.average_target_trajectory[-2]
                

                dx = end_point[0] - prev_point[0]
                dy = end_point[1] - prev_point[1]
                

                length = math.sqrt(dx*dx + dy*dy)
                if length > 0:
                    dx /= length
                    dy /= length
                

                arrow_size = 15
                

                arrow_p1 = (
                    end_point[0] - arrow_size * dx + arrow_size * 0.5 * dy,
                    end_point[1] - arrow_size * dy - arrow_size * 0.5 * dx
                )
                arrow_p2 = (
                    end_point[0] - arrow_size * dx - arrow_size * 0.5 * dy,
                    end_point[1] - arrow_size * dy + arrow_size * 0.5 * dx
                )
                

                pygame.draw.polygon(surface, MAGENTA, [end_point, arrow_p1, arrow_p2])