import pygame
import sys
import traceback
import time
import numpy as np
import matplotlib.pyplot as plt
from constants import WIDTH, HEIGHT, WHITE
from obstacles import create_obstacles
from agents.target_agent import TargetAgent
from agents.ego_agent import CasADiEgoAgent
from utils.scenario_generator import create_random_motion_set
from constants import GREEN

def setup_simulation(confidence_threshold):

    obstacles = create_obstacles()
    
    target = TargetAgent(WIDTH // 2, HEIGHT // 2)
    create_random_motion_set(target, GREEN, num_modes=10) 
    
    start_pos = (100, 100)
    goal_pos = (WIDTH - 100, HEIGHT - 100)
    
    ego = CasADiEgoAgent(start_pos, goal_pos, target, obstacles, target_bound=confidence_threshold)
    
    return target, ego, obstacles

def run_test(confidence_thresholds, runs_per_threshold=20, max_time_per_run=60):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 30)
    
    results = {}
    success_rates = {}
    collision_rates = {}
    

    total_runs = len(confidence_thresholds) * runs_per_threshold
    current_run = 0
    
    for threshold in confidence_thresholds:
        pygame.display.set_caption(f"Confidence Interval Test: {threshold}")
        print(f"\nTesting confidence threshold: {threshold}")
        success_runtimes = []
        all_runtimes = []
        collision_runtimes = []
        timeout_count = 0
        successes = 0
        collisions = 0
        
        for run in range(runs_per_threshold):
            current_run += 1
            print(f"  Run {run+1}/{runs_per_threshold} (Overall progress: {current_run}/{total_runs})")
            target, ego, obstacles = setup_simulation(threshold)
            
            start_time = time.time()
            running = True
            dt = 0.016
            elapsed = 0
            result = "timeout"
            
            while running and elapsed < max_time_per_run:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                

                target.update(dt, obstacles, should_stop=ego.at_goal)
                ego.update(dt)
                

                screen.fill(WHITE)
                

                progress_text = f"Confidence: {threshold} | Run: {run+1}/{runs_per_threshold} | Progress: {current_run}/{total_runs}"
                runtime_text = f"Runtime: {elapsed:.2f}s / {max_time_per_run}s"
                text_surface = font.render(progress_text, True, (0, 0, 0))
                screen.blit(text_surface, (10, 10))
                text_surface = font.render(runtime_text, True, (0, 0, 0))
                screen.blit(text_surface, (10, 40))
                
                if successes > 0 or collisions > 0:
                    status_text = f"Success: {successes} | Collisions: {collisions} | Timeouts: {timeout_count}"
                    text_surface = font.render(status_text, True, (0, 0, 0))
                    screen.blit(text_surface, (10, 70))
                

                for obstacle in obstacles:
                    obstacle.draw(screen)
                target.draw(screen)
                ego.draw(screen)
                pygame.display.flip()
                

                if ego.at_goal:
                    result = "success"
                    running = False
                elif ego.collision or ego.collision_with_obstacle:
                    result = "collision"
                    running = False
                
                elapsed = time.time() - start_time
                clock.tick(60)
            
            all_runtimes.append(elapsed)
            
            if result == "success":
                success_runtimes.append(elapsed)
                successes += 1
                print(f"    Success in {elapsed:.2f} seconds")
            elif result == "collision":
                collision_runtimes.append(elapsed)
                collisions += 1
                print(f"    Collision after {elapsed:.2f} seconds")
            else:
                timeout_count += 1
                print(f"    Timed out after {max_time_per_run} seconds")
            

            pygame.time.delay(500)
        

        success_rates[threshold] = successes / runs_per_threshold
        collision_rates[threshold] = collisions / runs_per_threshold
        timeout_rate = timeout_count / runs_per_threshold
        

        if success_runtimes:
            success_avg = np.mean(success_runtimes)
            results[threshold] = {
                'success_avg': success_avg,
                'all_avg': np.mean(all_runtimes),
                'collision_avg': np.mean(collision_runtimes) if collision_runtimes else None,
                'success_rate': success_rates[threshold],
                'collision_rate': collision_rates[threshold],
                'timeout_rate': timeout_rate
            }
            print(f"  Average runtime (success only): {success_avg:.2f} seconds")
            print(f"  Average runtime (all runs): {np.mean(all_runtimes):.2f} seconds")
            print(f"  Success rate: {success_rates[threshold]:.2f}")
            print(f"  Collision rate: {collision_rates[threshold]:.2f}")
            print(f"  Timeout rate: {timeout_rate:.2f}")
        else:
            results[threshold] = {
                'success_avg': float('inf'),
                'all_avg': np.mean(all_runtimes),
                'collision_avg': np.mean(collision_runtimes) if collision_runtimes else None,
                'success_rate': 0,
                'collision_rate': collision_rates[threshold],
                'timeout_rate': timeout_rate
            }
            print("  No successful runs")
            print(f"  Average runtime (all runs): {np.mean(all_runtimes):.2f} seconds")
            print(f"  Collision rate: {collision_rates[threshold]:.2f}")
            print(f"  Timeout rate: {timeout_rate:.2f}")
    
    pygame.quit()
    return results, success_rates, collision_rates

def plot_results(results, success_rates, collision_rates):

    thresholds = list(results.keys())
    

    success_runtimes = [results[t]['success_avg'] for t in thresholds]
    all_runtimes = [results[t]['all_avg'] for t in thresholds]
    collision_runtimes = [results[t]['collision_avg'] for t in thresholds]
    

    success_rates = [results[t]['success_rate'] for t in thresholds]
    collision_rates = [results[t]['collision_rate'] for t in thresholds]
    timeout_rates = [results[t]['timeout_rate'] for t in thresholds]
    

    valid_success_indices = [i for i, r in enumerate(success_runtimes) if r != float('inf')]
    valid_success_thresholds = [thresholds[i] for i in valid_success_indices]
    valid_success_runtimes = [success_runtimes[i] for i in valid_success_indices]
    

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, all_runtimes, 'o-', linewidth=2, markersize=8, label='All Runs')
    if valid_success_thresholds:
        plt.plot(valid_success_thresholds, valid_success_runtimes, 'o-', linewidth=2, markersize=8, label='Successful Runs')
    

    valid_collision_thresholds = []
    valid_collision_runtimes = []
    for i, r in enumerate(collision_runtimes):
        if r is not None:
            valid_collision_thresholds.append(thresholds[i])
            valid_collision_runtimes.append(r)
    
    if valid_collision_thresholds:
        plt.plot(valid_collision_thresholds, valid_collision_runtimes, 'o-', linewidth=2, markersize=8, label='Collision Runs')
    
    plt.xlabel('Confidence Threshold', fontsize=12)
    plt.ylabel('Average Runtime (seconds)', fontsize=12)
    plt.title('Average Runtime vs. Confidence Threshold', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('confidence_runtime_plot.png')
    

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, success_rates, 'o-', color='green', linewidth=2, markersize=8, label='Success Rate')
    plt.plot(thresholds, collision_rates, 'o-', color='red', linewidth=2, markersize=8, label='Collision Rate')
    plt.plot(thresholds, timeout_rates, 'o-', color='orange', linewidth=2, markersize=8, label='Timeout Rate')
    

    for i, threshold in enumerate(thresholds):
        plt.annotate(f"{success_rates[i]:.2f}", 
                    (threshold, success_rates[i]), 
                    textcoords="offset points",
                    xytext=(0,10), 
                    ha='center',
                    fontsize=9)
        plt.annotate(f"{collision_rates[i]:.2f}", 
                    (threshold, collision_rates[i]), 
                    textcoords="offset points",
                    xytext=(0,10), 
                    ha='center',
                    fontsize=9)
        plt.annotate(f"{timeout_rates[i]:.2f}", 
                    (threshold, timeout_rates[i]), 
                    textcoords="offset points",
                    xytext=(0,10), 
                    ha='center',
                    fontsize=9)
    
    plt.xlabel('Confidence Threshold', fontsize=12)
    plt.ylabel('Rate', fontsize=12)
    plt.title('Outcome Rates vs. Confidence Threshold', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('confidence_rates_plot.png')
    

    plt.figure(figsize=(10, 6))
    plt.scatter(success_rates, [results[t]['all_avg'] for t in thresholds], s=100, alpha=0.7)
    

    for i, threshold in enumerate(thresholds):
        plt.annotate(
            f"{threshold}",
            (success_rates[i], results[threshold]['all_avg']),
            textcoords="offset points",
            xytext=(0,10),
            ha='center',
            fontsize=11
        )
    
    plt.xlabel('Success Rate', fontsize=12)
    plt.ylabel('Average Runtime (seconds)', fontsize=12)
    plt.title('Trade-off: Runtime vs. Success Rate', fontsize=14, fontweight='bold')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('confidence_tradeoff_plot.png')
    

    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    

    axes[0].plot(thresholds, all_runtimes, 'o-', linewidth=2, markersize=8, label='All Runs')
    if valid_success_thresholds:
        axes[0].plot(valid_success_thresholds, valid_success_runtimes, 'o-', linewidth=2, markersize=8, label='Successful Runs')
    if valid_collision_thresholds:
        axes[0].plot(valid_collision_thresholds, valid_collision_runtimes, 'o-', linewidth=2, markersize=8, label='Collision Runs')
    axes[0].set_xlabel('Confidence Threshold', fontsize=12)
    axes[0].set_ylabel('Average Runtime (seconds)', fontsize=12)
    axes[0].set_title('Average Runtime vs. Confidence Threshold', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True)
    

    axes[1].plot(thresholds, success_rates, 'o-', color='green', linewidth=2, markersize=8, label='Success Rate')
    axes[1].plot(thresholds, collision_rates, 'o-', color='red', linewidth=2, markersize=8, label='Collision Rate')
    axes[1].plot(thresholds, timeout_rates, 'o-', color='orange', linewidth=2, markersize=8, label='Timeout Rate')
    axes[1].set_xlabel('Confidence Threshold', fontsize=12)
    axes[1].set_ylabel('Rate', fontsize=12)
    axes[1].set_title('Outcome Rates vs. Confidence Threshold', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True)
    

    axes[2].scatter(success_rates, [results[t]['all_avg'] for t in thresholds], s=100, alpha=0.7)
    for i, threshold in enumerate(thresholds):
        axes[2].annotate(
            f"{threshold}",
            (success_rates[i], results[threshold]['all_avg']),
            textcoords="offset points",
            xytext=(0,10),
            ha='center',
            fontsize=11
        )
    axes[2].set_xlabel('Success Rate', fontsize=12)
    axes[2].set_ylabel('Average Runtime (seconds)', fontsize=12)
    axes[2].set_title('Trade-off: Runtime vs. Success Rate', fontsize=14, fontweight='bold')
    axes[2].grid(True)
    
    plt.tight_layout(pad=3.0)
    plt.savefig('confidence_results.png', dpi=150)
    plt.show()

def main():
    confidence_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
    try:
        results, success_rates, collision_rates = run_test(
            confidence_thresholds,
            runs_per_threshold=1,
            max_time_per_run=30
        )


        print("\nFinal Results:")
        print("-" * 90)
        print("Confidence | Success Runtime | All Runtime | Success Rate | Collision Rate | Timeout Rate")
        print("-" * 90)
        for t in confidence_thresholds:
            sr = results[t]['success_avg']
            ar = results[t]['all_avg']
            sr_str = "N/A" if sr == float('inf') else f"{sr:.2f}s"
            print(f"{t:^10} | {sr_str:^15} | {ar:.2f}s | {results[t]['success_rate']:^12.2f} | {results[t]['collision_rate']:^14.2f} | {results[t]['timeout_rate']:^12.2f}")


        efficiencies = {t: results[t]['success_rate'] / results[t]['all_avg'] for t in confidence_thresholds}
        best_eff_threshold = max(efficiencies, key=efficiencies.get)
        best_eff = efficiencies[best_eff_threshold]

        print(f"\nHighest efficiency threshold: {best_eff_threshold:.2f} (Efficiency: {best_eff:.4f} success-rate/sec)")


        plot_results(results, success_rates, collision_rates)


        with open('confidence_test_results.csv', 'w') as f:
            f.write("Confidence,Success Runtime,All Runtime,Collision Runtime,Success Rate,Collision Rate,Timeout Rate\n")
            for t in confidence_thresholds:
                cr = results[t]['collision_avg']
                cr_str = "N/A" if cr is None else f"{cr:.2f}"
                sr = results[t]['success_avg']
                sr_str = "N/A" if sr == float('inf') else f"{sr:.2f}"
                f.write(f"{t},{sr_str},{results[t]['all_avg']:.2f},{cr_str},"
                        f"{results[t]['success_rate']:.2f},{results[t]['collision_rate']:.2f},{results[t]['timeout_rate']:.2f}\n")

        print("\nResults saved to confidence_test_results.csv")
        print("Plots saved to confidence_results.png")

    except Exception:
        traceback.print_exc()
    finally:
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()
