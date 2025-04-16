import math
from collections import Counter, defaultdict

class ValiantEstimator:
    def __init__(self, confidence_threshold=0.95, delta=0.05, c=.5):
        self.observations = []
        self.confidence_threshold = confidence_threshold
        self.delta = delta
        self.c = c
        self.estimated_modes = {
            'observed': 0,
            'estimated_total': 0,
            'confidence': 0.0,
            'samples': 0,
            'probabilities': {}
        }
        self.unseen_class_estimate = 0
        
    def add_observation(self, mode_idx):
        self.observations.append(mode_idx)
        self.update_estimate()
        
    def update_estimate(self):
        if not self.observations:
            return
            
        n = len(self.observations)
        mode_counts = Counter(self.observations)
        
        fingerprint = defaultdict(int)
        for count in mode_counts.values():
            fingerprint[count] += 1
            
        observed_unique_modes = len(mode_counts)
        
        kappa = 2 * self.delta / n
        
        unseen_mass = fingerprint[1] / n if 1 in fingerprint else 0
        
        if observed_unique_modes > 0 and unseen_mass > 0:
            avg_prob_per_known_mode = (1 - unseen_mass) / observed_unique_modes
            if avg_prob_per_known_mode > 0:
                self.unseen_class_estimate = max(0, unseen_mass / avg_prob_per_known_mode)
            else:
                self.unseen_class_estimate = 0
        else:
            self.unseen_class_estimate = 0
            
        confidence = 1.0 - math.exp(-self.c * n / (observed_unique_modes + self.unseen_class_estimate + 1))
        
        prob_estimates = {}
        for mode, count in mode_counts.items():
            if count >= 1:
                if count >= 5:
                    prob_estimates[mode] = count / n
                else:
                    r_plus_1 = fingerprint[count + 1] if count + 1 in fingerprint else 0
                    r = fingerprint[count] if count in fingerprint else 1
                    prob_estimates[mode] = ((r_plus_1 + 1) / r) * ((count + 1) / n)
        
        total_prob = sum(prob_estimates.values())
        if total_prob > 0:
            for mode in prob_estimates:
                prob_estimates[mode] /= total_prob
                
        self.estimated_modes = {
            'observed': observed_unique_modes,
            'estimated_total': observed_unique_modes + round(self.unseen_class_estimate),
            'confidence': confidence,
            'samples': n,
            'probabilities': prob_estimates
        }
    
    def calculate_ucb(self, mode_probs):
        n = len(self.observations)
        ucb_probs = {}
        
        if n == 0:
            return {}
            
        for mode, prob in mode_probs.items():
            ucb = prob + math.sqrt(math.log(1/self.delta)/(2*n))
            ucb_probs[mode] = min(1.0, ucb)
            
        return ucb_probs
    
    def support_estimate_bound(self):
        n = len(self.observations)
        if n == 0:
            return 0.0
            
        observed_modes = len(set(self.observations))
        
        bound = 1.0 - math.exp(-self.c * n / (observed_modes + self.unseen_class_estimate + 1))
        
        return bound
    
    def sample_requirement(self, target_bound):
        current_bound = self.support_estimate_bound()
        n = len(self.observations)
        observed_modes = len(set(self.observations))
        
        if current_bound >= target_bound:
            return 0
        
        estimated_total_modes = observed_modes + round(self.unseen_class_estimate)
        
        required_n = -math.log(1.0 - target_bound) * (estimated_total_modes + 1) / self.c
        
        return max(0, math.ceil(required_n - n))
        
    def sufficient_confidence(self):
        if not self.estimated_modes:
            return False
        return self.estimated_modes['confidence'] >= self.confidence_threshold
        
    def get_mode_probabilities(self):
        if not self.estimated_modes:
            return {}
        return self.estimated_modes['probabilities']