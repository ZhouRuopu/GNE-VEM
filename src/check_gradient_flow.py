import torch
import numpy as np

def check_gradient_flow(model, epoch, batch_idx, prefix=""):
    """
    Check model gradient flow
    """
    print(f"\n{prefix}Gradient Check - Epoch {epoch}, Batch {batch_idx}:")
    
    total_grad_norm = 0.0
    zero_grad_layers = 0
    has_gradient_layers = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm
            
            if grad_norm > 1e-8:  # Threshold for meaningful gradients
                has_gradient_layers += 1
                print(f"  ✓ {name}: Grad Norm = {grad_norm:.6f}")
            else:
                zero_grad_layers += 1
                print(f"  ✗ {name}: Vanishing Gradient (Norm = {grad_norm:.6e})")
        else:
            zero_grad_layers += 1
            print(f"  ✗ {name}: No Gradient")
    
    print(f"Summary: {has_gradient_layers} layers have grads, {zero_grad_layers} layers have no grads, Total Norm = {total_grad_norm:.6f}")
    return total_grad_norm > 1e-6  # Return if effective gradients exist


class ParameterMonitor:
    def __init__(self, model):
        self.model = model
        self.initial_params = {}
        self.previous_params = {}
        self._save_initial_state()
    
    def _save_initial_state(self):
        """Save initial parameter state"""
        for name, param in self.model.named_parameters():
            self.initial_params[name] = param.data.clone()
            self.previous_params[name] = param.data.clone()
    
    def check_parameter_changes(self, epoch, batch_idx):
        """Check parameter changes"""
        print(f"\nParameter Change Check - Epoch {epoch}, Batch {batch_idx}:")
        
        total_change = 0.0
        changed_layers = 0
        
        for name, param in self.model.named_parameters():
            current_param = param.data
            previous_param = self.previous_params[name]
            initial_param = self.initial_params[name]
            
            # Calculate relative change
            change_since_last = (current_param - previous_param).norm().item()
            change_since_initial = (current_param - initial_param).norm().item()
            relative_change = change_since_last / (previous_param.norm().item() + 1e-8)
            
            if change_since_last > 1e-8:
                changed_layers += 1
                print(f"  ✓ {name}: Change = {change_since_last:.6e} (Rel: {relative_change:.2%})")
            else:
                print(f"  ✗ {name}: No Change")
            
            total_change += change_since_last
            self.previous_params[name] = current_param.clone()
        
        print(f"Change Summary: {changed_layers} layers changed, Total Change = {total_change:.6e}")
        return total_change
