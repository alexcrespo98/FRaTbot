import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import FloatSlider, FloatText, Checkbox, VBox, Layout, Output, Dropdown
from IPython.display import display

# Common pipe diameters in meters
PIPE_DIAMETERS = {
    '1/2" (0.5 in)': 0.0127,
    '3/4" (0.75 in)': 0.01905,
    '1" (1 in)': 0.0254,
    '1 1/4" (1.25 in)': 0.03175,
    '1 1/2" (1.5 in)': 0.0381,
    '2" (2 in)': 0.0508,
}

PROXIMAL_LOCATION = 1.0  # Fixed proximal sensor location (meters)

def water_thermal_diffusivity(T_C):
    """
    Returns the thermal diffusivity of water [m^2/s] as a function of temperature [°C].
    Empirical fit for 0-100°C, based on literature.
    Reference: Incropera & DeWitt, Fundamentals of Heat and Mass Transfer, 7th ed.
    """
    # Fit: alpha ~ 1.43e-7 + 3e-10*(T-20)
    return 1.43e-7 + 3e-10 * (T_C - 20)

def calc_temp_profile(
    distance,
    deltaT_amplitude,
    flow_gpm,
    sinusoidal,
    pipe_diameter,
    wavelength=1.0,
    T0=43.3,
    alpha=1.43e-7,
    # enable_pipe_wall=False,    # Uncomment to enable pipe wall modeling
    # pipe_wall_thickness=0.001, # Typical for copper (1mm)
    # pipe_wall_alpha=1.13e-4,   # Thermal diffusivity of copper [m^2/s]
    # pipe_wall_rho=8960,        # Density of copper [kg/m^3]
    # pipe_wall_cp=385,          # Specific heat capacity of copper [J/kg/K]
    # outside_temp=20            # Ambient temp for external losses [C]
):
    """
    Simulates the propagation of a temperature pulse through water in a pipe using
    1D advection-diffusion. Optionally (commented out) can be extended to include
    pipe wall heat transfer.

    To enable pipe wall modeling:
    1. Uncomment enable_pipe_wall and the pipe wall parameter arguments above.
    2. Uncomment the code sections below marked 'OPTIONAL: Pipe wall modeling'.
    3. Adjust parameters as appropriate for your pipe material and wall thickness.

    The commented block will give you a starting point for a two-body (water/pipe) simulation.
    """
    area = np.pi * (pipe_diameter / 2) ** 2
    flow_gpm = max(flow_gpm, 0.01)
    flow_m3s = flow_gpm * 0.00378541 / 60
    velocity = flow_m3s / area

    dx = 0.25  # spatial step (m)
    dt = 0.01  # time step (s)
    nx = int(distance / dx) + 1
    nt = int((distance / velocity + wavelength + 8) / dt)

    T = np.ones(nx) * T0
    outlet_temps = []

    # --- OPTIONAL: Pipe wall modeling (disabled, see docstring above) ---
    # if enable_pipe_wall:
    #     T_wall = np.ones(nx) * T0
    #     wall_area = np.pi * pipe_diameter * dx
    #     wall_mass = wall_area * pipe_wall_thickness * pipe_wall_rho
    #     wall_capacity = wall_mass * pipe_wall_cp

    for tstep in range(nt):
        time = tstep * dt
        T_new = T.copy()
        if time < wavelength:
            if not sinusoidal:
                T_new[0] = T0 + deltaT_amplitude
            else:
                T_new[0] = T0 + deltaT_amplitude * np.sin(np.pi * time / wavelength)
        else:
            T_new[0] = T0

        for i in range(1, nx - 1):
            adv = -velocity * dt / dx * (T[i] - T[i - 1])
            diff = alpha * dt / dx**2 * (T[i + 1] - 2 * T[i] + T[i - 1])
            T_new[i] = T[i] + adv + diff

            # --- OPTIONAL: Pipe wall heat exchange (disabled by default) ---
            # if enable_pipe_wall:
            #     # Simple heat transfer between water and pipe wall
            #     h = 1000  # W/m^2/K, forced convection estimate
            #     dQ = h * wall_area * dt * (T[i] - T_wall[i])
            #     # Water side (assume rho*cp=1000 J/K/L for water)
            #     T_new[i] -= dQ / (area * dx * 1000)
            #     # Wall side
            #     T_wall[i] += dQ / wall_capacity

        T_new[-1] = T_new[-2]
        T = T_new.copy()
        outlet_temps.append(T[-1])

        # --- OPTIONAL: Pipe wall update (disabled) ---
        # if enable_pipe_wall:
        #     # Pipe wall conduction (1D, along the length)
        #     for j in range(1, nx - 1):
        #         wall_diff = pipe_wall_alpha * dt / dx**2 * (T_wall[j + 1] - 2 * T_wall[j] + T_wall[j - 1])
        #         T_wall[j] += wall_diff
        #     # External losses (to ambient)
        #     for j in range(nx):
        #         ext_h = 10  # W/m^2/K, free convection/insulated
        #         ext_area = np.pi * (pipe_diameter + 2*pipe_wall_thickness) * dx
        #         dQ_ext = ext_h * ext_area * dt * (T_wall[j] - outside_temp)
        #         T_wall[j] -= dQ_ext / wall_capacity

    max_rise = max(outlet_temps) - T0
    return max_rise

def estimate_flow_from_deltaT(
    prox_loc, dist_loc, deltaT_amplitude, deltaT_sensor, pipe_diameter,
    sinusoidal=False, wavelength=1.0, T0=43.3
):
    alpha = water_thermal_diffusivity(T0)
    flow_range = np.linspace(0.1, 100, 300)
    deltaT_sim = []

    for flow in flow_range:
        T_prox = calc_temp_profile(prox_loc, deltaT_amplitude, flow, sinusoidal, pipe_diameter, wavelength, T0, alpha)
        T_dist = calc_temp_profile(dist_loc, deltaT_amplitude, flow, sinusoidal, pipe_diameter, wavelength, T0, alpha)
        deltaT_sim.append(T_prox - T_dist)  # sensor deltaT = proximal - distal

    deltaT_sim = np.array(deltaT_sim)
    idx = (np.abs(deltaT_sim - deltaT_sensor)).argmin()
    estimated_flow = flow_range[idx]
    return estimated_flow, deltaT_sim[idx], flow_range, deltaT_sim

def interactive_flowrate_ui():
    sensor_spacing_slider = FloatSlider(value=2.0, min=0.1, max=10, step=0.1,
                                        description='Sensor Spacing (m):', layout=Layout(width='450px'))
    deltaT_amplitude_slider = FloatSlider(value=3.0, min=1.0, max=10.0, step=0.1,
                                         description='ΔT Amplitude (°C):', layout=Layout(width='450px'))
    wavelength_slider = FloatSlider(value=1.0, min=0.1, max=5.0, step=0.05,
                                    description='Wavelength (s):', layout=Layout(width='450px'))
    deltaT_sensor_input = FloatText(value=0.15, description='ΔT Sensor (°C):', layout=Layout(width='250px'))
    sinusoidal_checkbox = Checkbox(value=False, description='Sinusoidal Pulse', layout=Layout(width='200px'))
    pipe_diameter_dropdown = Dropdown(options=PIPE_DIAMETERS, value=0.01905,
                                      description='Pipe Diameter:', layout=Layout(width='300px'))
    T0_slider = FloatSlider(value=43.3, min=0, max=100, step=0.1,
                            description='Baseline Temp (°C):', layout=Layout(width='450px'))
    output = Output()

    def update(_):
        with output:
            output.clear_output()
            spacing = sensor_spacing_slider.value
            prox = PROXIMAL_LOCATION
            distal = prox + spacing
            deltaT_a = deltaT_amplitude_slider.value
            wavelength = wavelength_slider.value
            deltaT_s = deltaT_sensor_input.value
            sinusoidal = sinusoidal_checkbox.value
            pipe_dia = pipe_diameter_dropdown.value
            T0 = T0_slider.value

            alpha = water_thermal_diffusivity(T0)

            est_flow, closest_deltaT, flow_vals, deltaT_vals = estimate_flow_from_deltaT(
                prox, distal, deltaT_a, deltaT_s, pipe_dia, sinusoidal, wavelength, T0
            )

            print(f"ΔT Sensor (Proximal - Distal): {deltaT_s:.3f} °C")
            print(f"Closest Simulated ΔT: {closest_deltaT:.3f} °C")
            print(f"Estimated Flow Rate: {est_flow:.2f} GPM")
            print(f"Proximal Sensor Location: {prox:.2f} m")
            print(f"Distal Sensor Location: {distal:.2f} m")
            print(f"Pipe Diameter Used: {pipe_dia*1000:.1f} mm")
            print(f"Wavelength (Time Between Peaks): {wavelength:.2f} s")
            print(f"Baseline Temp (T0): {T0:.1f} °C")
            print(f"Thermal Diffusivity Used: {alpha:.2e} m²/s")

            plt.figure(figsize=(8, 4))
            plt.plot(flow_vals, deltaT_vals, label='Simulated ΔT vs Flow Rate')
            plt.axhline(deltaT_s, color='red', linestyle='--', label='Measured ΔT Sensor')
            plt.axvline(est_flow, color='green', linestyle='--', label=f'Estimated Flow: {est_flow:.2f} GPM')
            plt.xlabel('Flow Rate (GPM)')
            plt.ylabel('ΔT (°C)')
            plt.title('ΔT (Proximal - Distal) vs Flow Rate')
            plt.legend()
            plt.grid(True)
            plt.show()

    sensor_spacing_slider.observe(update, names='value')
    deltaT_amplitude_slider.observe(update, names='value')
    wavelength_slider.observe(update, names='value')
    deltaT_sensor_input.observe(update, names='value')
    sinusoidal_checkbox.observe(update, names='value')
    pipe_diameter_dropdown.observe(update, names='value')
    T0_slider.observe(update, names='value')

    display(VBox([
        sensor_spacing_slider,
        deltaT_amplitude_slider,
        wavelength_slider,
        deltaT_sensor_input,
        sinusoidal_checkbox,
        pipe_diameter_dropdown,
        T0_slider,
        output
    ]))

    update(None)  # initial run

interactive_flowrate_ui()
