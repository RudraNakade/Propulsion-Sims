import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Tuple

class FrictionFactorMethod(ABC):
    """Abstract base class for friction factor calculation methods."""
    
    @abstractmethod
    def calculate(self, Re: float, rel_roughness: float) -> float:
        """Calculate friction factor for given Reynolds number and relative roughness."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the method."""
        pass

class ColebrookMethod(FrictionFactorMethod):
    """Colebrook equation - exact solution (reference)."""
    
    def calculate(self, Re: float, rel_roughness: float) -> float:
        def colebrook_equation(f: float) -> float:
            return (1 / np.sqrt(f)) + 2 * np.log10((rel_roughness/3.7) + 2.51/(Re*np.sqrt(f)))
        
        return root_scalar(colebrook_equation, bracket=[1e-6, 1]).root
    
    @property
    def name(self) -> str:
        return "Colebrook"

class SwameeJainMethod(FrictionFactorMethod):
    """Swamee-Jain approximation."""
    
    def calculate(self, Re: float, rel_roughness: float) -> float:
        return 1.325 / ((np.log((rel_roughness / 3.7) + (5.74 / Re ** 0.9))) ** 2)
    
    @property
    def name(self) -> str:
        return "Swamee-Jain"

class HaalandMethod(FrictionFactorMethod):
    """Haaland approximation."""
    
    def calculate(self, Re: float, rel_roughness: float) -> float:
        return (-1.8 * np.log10((rel_roughness/3.7)**1.11 + (6.9 / Re))) ** -2
    
    @property
    def name(self) -> str:
        return "Haaland"

class FrictionFactorComparison:
    """Class to handle friction factor comparison analysis."""
    
    def __init__(self, Re_range: Tuple[float, float], num_points: int = 100):
        self.Re_arr = np.logspace(np.log10(Re_range[0]), np.log10(Re_range[1]), num=num_points)
        self.rel_roughness_arr = [0, 1e-3, 1e-2, 1e-1]
        
        self.reference_method = ColebrookMethod()
        self.comparison_methods = [SwameeJainMethod(), HaalandMethod()]
        
        self.results = {}
        self.error_stats = {}
    
    def run_analysis(self) -> None:
        """Run the complete friction factor analysis."""
        # Initialize results storage
        n_re = len(self.Re_arr)
        n_rough = len(self.rel_roughness_arr)
        n_methods = len(self.comparison_methods) + 1
        
        self.results = {
            self.reference_method.name: np.zeros((n_re, n_rough)),
            **{method.name: np.zeros((n_re, n_rough)) for method in self.comparison_methods}
        }
        
        # Calculate friction factors
        for j, rel_roughness in enumerate(self.rel_roughness_arr):
            for i, Re in enumerate(self.Re_arr):
                # Reference method (Colebrook)
                self.results[self.reference_method.name][i, j] = self.reference_method.calculate(Re, rel_roughness)
                
                # Comparison methods
                for method in self.comparison_methods:
                    self.results[method.name][i, j] = method.calculate(Re, rel_roughness)
        
        # Calculate error statistics
        self._calculate_error_statistics()
    
    def _calculate_error_statistics(self) -> None:
        """Calculate error statistics for comparison methods."""
        self.error_stats = {}
        
        for method in self.comparison_methods:
            method_name = method.name
            errors = np.zeros((len(self.Re_arr), len(self.rel_roughness_arr)))
            
            for j in range(len(self.rel_roughness_arr)):
                for i in range(len(self.Re_arr)):
                    ref_value = self.results[self.reference_method.name][i, j]
                    comp_value = self.results[method_name][i, j]
                    errors[i, j] = 100 * (ref_value - comp_value) / ref_value
            
            self.error_stats[method_name] = {
                'errors': errors,
                'avg_error': np.array([np.mean(np.abs(errors[:, j])) for j in range(len(self.rel_roughness_arr))]),
                'min_error': np.array([np.min(np.abs(errors[:, j])) for j in range(len(self.rel_roughness_arr))]),
                'max_error': np.array([np.max(np.abs(errors[:, j])) for j in range(len(self.rel_roughness_arr))])
            }
    
    def plot_results(self) -> None:
        """Create comprehensive plots of the analysis results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Friction factor vs Reynolds number
        for j, rel_roughness in enumerate(self.rel_roughness_arr):
            label_suffix = f'ε/D = {rel_roughness:.0e}' if rel_roughness > 0 else 'Smooth pipe'
            
            # Reference method
            ax1.loglog(self.Re_arr, self.results[self.reference_method.name][:, j], 
                      '-', label=f'{self.reference_method.name} ({label_suffix})', linewidth=2)
            
            # Comparison methods
            line_styles = ['--', ':']
            for i, method in enumerate(self.comparison_methods):
                ax1.loglog(self.Re_arr, self.results[method.name][:, j], 
                          line_styles[i], label=f'{method.name} ({label_suffix})')
        
        ax1.set_xlabel('Reynolds Number')
        ax1.set_ylabel('Friction Factor')
        ax1.set_title('Friction Factor Comparison')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2 & 3: Error plots for each comparison method
        error_axes = [ax2, ax3]
        for ax_idx, method in enumerate(self.comparison_methods):
            ax = error_axes[ax_idx]
            for j, rel_roughness in enumerate(self.rel_roughness_arr):
                label = f'ε/D = {rel_roughness:.0e}' if rel_roughness > 0 else 'Smooth pipe'
                ax.semilogx(self.Re_arr, self.error_stats[method.name]['errors'][:, j], label=label)
            
            ax.set_xlabel('Reynolds Number')
            ax.set_ylabel('Percentage Error (%)')
            ax.set_title(f'{method.name} Error vs {self.reference_method.name}')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Plot 4: Error statistics comparison
        x = np.arange(len(self.rel_roughness_arr))
        width = 0.35
        
        for i, method in enumerate(self.comparison_methods):
            offset = (i - 0.5) * width
            ax4.bar(x + offset, self.error_stats[method.name]['avg_error'], 
                   width, label=f'{method.name} Avg Error', alpha=0.8)
        
        ax4.set_xlabel('Relative Roughness Cases')
        ax4.set_ylabel('Average Absolute Error (%)')
        ax4.set_title('Average Absolute Error Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'{rr:.0e}' if rr > 0 else 'Smooth' for rr in self.rel_roughness_arr])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_error_summary(self) -> None:
        """Print detailed error statistics summary."""
        print("Error Statistics Summary:")
        print("-" * 50)
        
        for j, rel_roughness in enumerate(self.rel_roughness_arr):
            roughness_label = f'ε/D = {rel_roughness:.0e}' if rel_roughness > 0 else 'Smooth pipe'
            print(f"\n{roughness_label}:")
            
            for method in self.comparison_methods:
                stats = self.error_stats[method.name]
                print(f"  {method.name:12}: Avg={stats['avg_error'][j]:.3f}%, "
                      f"Min={stats['min_error'][j]:.3f}%, Max={stats['max_error'][j]:.3f}%")

def main():
    """Main execution function."""
    # Create and run the analysis
    comparison = FrictionFactorComparison(Re_range=(1e4, 1e9), num_points=100)
    comparison.run_analysis()
    
    # Display results
    comparison.plot_results()
    comparison.print_error_summary()

if __name__ == "__main__":
    main()