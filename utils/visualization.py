import pygame
from constants import BLACK, RED, GREEN, BLUE, YELLOW, MAGENTA, GRAY

def draw_text(surface, font, text, position, color=BLACK):

    text_surface = font.render(text, True, color)
    surface.blit(text_surface, position)

def draw_legend(surface, font, items, position=(0, 0)):

    legend_y = position[1]
    legend_x = position[0]
    
    for item, color in items:

        pygame.draw.rect(surface, color, (legend_x, legend_y, 20, 20))
        pygame.draw.rect(surface, BLACK, (legend_x, legend_y, 20, 20), 1)
        

        legend_text = font.render(item, True, BLACK)
        surface.blit(legend_text, (legend_x + 30, legend_y + 2))
        
        legend_y += 30
        
    return legend_y  

def draw_estimation_stats(surface, font, ego_agent, position=(10, 10)):

    y_pos = position[1]
    
    if ego_agent.estimator.estimated_modes:
        texts = [
            (f"Confidence: {ego_agent.estimator.support_estimate_bound():.2f}", BLACK),
            (f"Samples: {len(ego_agent.estimator.observations)}", BLACK),
            (f"Est. Modes: {ego_agent.estimator.estimated_modes['estimated_total']}", BLACK),
            (f"Needed: {ego_agent.estimator.sample_requirement(ego_agent.target_bound)} more samples", BLACK),
            (f"Sufficient: {'Yes' if ego_agent.sufficient_samples else 'No'}", BLACK)
        ]
        

        if ego_agent.using_conservative_trajectory:
            texts.append(("Conservative: Low Confidence", (255, 165, 0))) 
        else:
            texts.append(("Normal: High Confidence", GREEN))
        

        if ego_agent.collision:
            texts.append(("Collision with Target: Yes", RED))
        elif ego_agent.collision_with_obstacle:
            texts.append(("Collision with Obstacle: Yes", RED))
        else:
            texts.append(("Collision: No", BLACK))
        

        for text, color in texts:
            draw_text(surface, font, text, (position[0], y_pos), color)
            y_pos += 30
    
    return y_pos  

def create_standard_legend():

    return [
        ("Red Agent", RED),
        ("Green Target", GREEN),
        ("Blue Path", BLUE),
        ("Magenta Forecast", MAGENTA),
        ("Gray Scenarios", GRAY),
        ("Yellow Goal", YELLOW)
    ]