from pyfluids import Fluid, Input, FluidsList
import matplotlib.pyplot as plt
import numpy as np
from os import system

system('cls')

fluids_list = [
    {'name': 'Oxygen', 'fluid_type': FluidsList.Oxygen},
    {'name': 'Nitrogen', 'fluid_type': FluidsList.Nitrogen},
    {'name': 'Propane', 'fluid_type': FluidsList.nPropane}, 
    {'name': 'Methane', 'fluid_type': FluidsList.Methane},  
    {'name': 'Hydrogen', 'fluid_type': FluidsList.Hydrogen},
]

# Initialize fluids
fluids = {}
for fluid in fluids_list:
    fluids[fluid['name']] = {
        'fluid': Fluid(fluid['fluid_type']),
        'density': None,
        'specific_heat': None,
        'conductivity': None,
        'compressibility': None,
        'vapor_pressure': None,
        }

P = np.concatenate([np.array([1, 5]), np.linspace(10, 100, 10)]) * 1e5  # Pressure range

# Calculate density data for each fluid
for name, fluid_info in fluids.items():
    fluid = fluid_info['fluid']
    
    T_fluid = np.linspace(fluid.min_temperature+1.1, 500-273.15, 500)
    
    # Initialize arrays for all properties
    fluid_info['density'] = np.zeros((len(P), len(T_fluid)))
    fluid_info['specific_heat'] = np.zeros((len(P), len(T_fluid)))
    fluid_info['conductivity'] = np.zeros((len(P), len(T_fluid)))
    fluid_info['compressibility'] = np.zeros((len(P), len(T_fluid)))
    fluid_info['vapor_pressure'] = np.zeros(len(T_fluid))
    fluid_info['temp_range'] = T_fluid
    
    # Calculate properties for all pressure-temperature combinations
    for j, pressure in enumerate(P):
        for i, temp in enumerate(T_fluid):
            fluid.update(Input.temperature(temp), Input.pressure(pressure))
            fluid_info['density'][j, i] = fluid.density
            fluid_info['specific_heat'][j, i] = fluid.specific_heat
            fluid_info['conductivity'][j, i] = fluid.conductivity
            fluid_info['compressibility'][j, i] = fluid.compressibility
            if temp < fluid.critical_temperature:
                fluid.update(Input.temperature(temp), Input.quality(0))
                fluid_info['vapor_pressure'][i] = fluid.pressure / 1e5
            else:
                fluid_info['vapor_pressure'][i] = None

# storage_pressure = 1e5 # atmospheric pressure
storage_pressure = 5e5 # 5 bar(g)



# Print critical point and storage properties
print("Critical Point Properties:")
print("-" * 70)
for name, fluid_info in fluids.items():
    fluid = fluid_info['fluid']
    fluid.update(Input.pressure(storage_pressure), Input.quality(0))
    print(f'{name:<8} - Critical: T: {(fluid.critical_temperature+273.15):.2f} K / {fluid.critical_temperature:.2f} °C, P: {(fluid.critical_pressure/1e5):.2f} Bar(a)')

print(f"\nStorage Properties at {storage_pressure/1e5:.1f} Bar(a):")
print("-" * 70)
for name, fluid_info in fluids.items():
    fluid = fluid_info['fluid']
    print(f'{name:<8}: {(fluid.temperature+273.15):.2f} K / {fluid.temperature:.2f} °C, {fluid.density:.2f} kg/m³')

print(f"\nFreezing Points at ambient:")
print("-" * 70)
for name, fluid_info in fluids.items():
    fluid = fluid_info['fluid']
    fluid.update(Input.pressure(101325), Input.temperature(fluid.min_temperature+0.025))
    freezing_point = fluid.min_temperature + 273.15 if fluid.min_temperature is not None else None
    print(f'{name:<8}: {freezing_point:.2f} K / {freezing_point - 273.15:.2f} °C, {fluid.density:.2f} kg/m³, {fluid.phase}') if freezing_point is not None else print(f'{name:<8}: No freezing point data available')

def plot_properties_2d(fluids, P, property_name, ylabel, title_suffix):
    """Create plots for a specific fluid property"""
    # Calculate subplot dimensions based on number of fluids
    num_fluids = len(fluids)
    cols = int(np.ceil(np.sqrt(num_fluids)))
    rows = int(np.ceil(num_fluids / cols))
    
    # Create subplot layout
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    
    # Handle case where there's only one subplot
    if num_fluids == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # Create plots for each fluid
    for idx, (name, fluid_info) in enumerate(fluids.items()):
        ax = axes[idx]
        
        # Prepare data for plotting
        temp_range = fluid_info['temp_range'] + 273.15
        property_data = fluid_info[property_name]
        colors = plt.cm.viridis(np.linspace(0, 1, len(P)))
        
        for i, pressure in enumerate(P):
            ax.plot(temp_range, property_data[i], 
                    label=f'{pressure/1e5:.1f} bar(a)', color=colors[i])
        
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{name} {title_suffix}')
        ax.legend(loc='best', fontsize='small')
        ax.set_xlim(min(temp_range), max(temp_range))
        ax.grid(True, alpha=0.3)
    
    # Hide any unused subplots
    for idx in range(num_fluids, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_properties_1d(fluids, property_name, ylabel, title_suffix):
    """Create plots for a specific fluid property"""
    # Calculate subplot dimensions based on number of fluids
    num_fluids = len(fluids)
    cols = int(np.ceil(np.sqrt(num_fluids)))
    rows = int(np.ceil(num_fluids / cols))
    
    # Create subplot layout
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    
    # Handle case where there's only one subplot
    if num_fluids == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # Create plots for each fluid
    for idx, (name, fluid_info) in enumerate(fluids.items()):
        ax = axes[idx]
        
        temp_range = fluid_info['temp_range'] + 273.15
        property_data = fluid_info[property_name]
        
        # Handle both 1D and 2D arrays
        if property_data.ndim == 1:
            # For 1D arrays like vapor_pressure
            # Filter out None values
            valid_mask = np.array([~np.isnan(x) for x in property_data])
            valid_temps = temp_range[valid_mask]
            valid_data = property_data[valid_mask]
            
            if len(valid_data) > 0:
                ax.plot(valid_temps, valid_data, label=name)
                ax.set_xlim(min(valid_temps), max(valid_temps))
                ax.set_ylim(min(valid_data), max(valid_data))
        else:
            # Filter out None/NaN values
            valid_mask = ~np.isnan(property_data) & (property_data != None)
            valid_temps = temp_range[valid_mask]
            valid_data = property_data[valid_mask]

            if len(valid_data) > 0:
                print(f"Plotting {name} with valid data points: {len(valid_data)}")
                ax.plot(valid_temps, valid_data, label=name)
                ax.set_xlim(min(valid_temps), max(valid_temps))
                ax.set_ylim(min(valid_data), max(valid_data))
        
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{name} {title_suffix}')
        ax.grid(True, alpha=0.3)
    
    # Hide any unused subplots
    for idx in range(num_fluids, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig

# density_fig = plot_properties_2d(fluids, P, 'density', 'Density (kg/m³)', 'Density vs Temperature')
# specific_heat_fig = plot_properties_2d(fluids, P, 'specific_heat', 'Specific Heat (J/kg·K)', 'Specific Heat vs Temperature')
# conductivity_fig = plot_properties_2d(fluids, P, 'conductivity', 'Thermal Conductivity (W/m·K)', 'Thermal Conductivity vs Temperature')
# compressibility_fig = plot_properties_2d(fluids, P, 'compressibility', 'Compressibility Factor', 'Compressibility vs Temperature')
vapor_pressure_fig = plot_properties_1d(fluids, 'vapor_pressure', 'Vapor Pressure (Bar)', 'Vapor Pressure vs Temperature')

plt.show()