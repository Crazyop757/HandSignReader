import plotly.graph_objects as go
import numpy as np

# Futuristic Neural Pipeline Visualization - Cylinder Tubes Design
# Flowing data through transparent neural tubes

# Layer info: (name, size_text, filters/neurons, color_primary, color_secondary)
layers = [
    ("INPUT", "64×64×3", 3, "#00F5FF", "#0080FF"),
    ("CONV-32", "62×62×32", 32, "#FF00FF", "#C000FF"),
    ("ReLU", "activation", 32, "#FF66FF", "#FF33FF"),
    ("POOL", "31×31×32", 32, "#FF3366", "#FF0066"),
    ("CONV-64", "29×29×64", 64, "#FFAA00", "#FF6600"),
    ("ReLU", "activation", 64, "#FFCC44", "#FFAA00"),
    ("POOL", "14×14×64", 64, "#00FF88", "#00CC66"),
    ("CONV-128", "12×12×128", 128, "#FF0099", "#CC0077"),
    ("ReLU", "activation", 128, "#FF44BB", "#FF0099"),
    ("POOL", "6×6×128", 128, "#FFDD00", "#FFB000"),
    ("FLATTEN", "4608 units", 4608, "#9966FF", "#6633FF"),
    ("DENSE", "128 units", 128, "#FF3399", "#FF0066"),
    ("ReLU", "activation", 128, "#FF66BB", "#FF3399"),
    ("DROPOUT", "50%", 128, "#00FFCC", "#00CC99"),
    ("OUTPUT", "29 classes", 29, "#00FF00", "#00DD00"),
    ("Softmax", "activation", 29, "#44FF44", "#00FF00"),
]

fig = go.Figure()

# Create cylindrical pipeline
for i, (name, size, units, color1, color2) in enumerate(layers):
    x_pos = i * 6.5
    
    # Cylinder dimensions based on layer size
    radius = 0.5 + (np.log10(units + 1) * 0.3)
    height = 4 + (np.log10(units + 1) * 0.5)
    
    # Create cylinder using parametric equations
    theta = np.linspace(0, 2*np.pi, 30)
    z_cylinder = np.linspace(0, height, 20)
    theta_grid, z_grid = np.meshgrid(theta, z_cylinder)
    
    x_cylinder = radius * np.cos(theta_grid) + x_pos
    y_cylinder = radius * np.sin(theta_grid)
    
    # Main cylinder
    fig.add_trace(go.Surface(
        x=x_cylinder,
        y=y_cylinder,
        z=z_grid,
        colorscale=[[0, color2], [0.5, color1], [1, color2]],
        showscale=False,
        opacity=0.7,
        name=name,
        hovertemplate=f"<b>{name}</b><br>Size: {size}<extra></extra>",
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.9, roughness=0.05),
        lightposition=dict(x=0, y=0, z=1000)
    ))
    
    # Energy ring at top
    ring_z = height + 0.2
    ring_radius = radius + 0.3
    fig.add_trace(go.Scatter3d(
        x=ring_radius * np.cos(theta) + x_pos,
        y=ring_radius * np.sin(theta),
        z=[ring_z] * len(theta),
        mode='lines',
        line=dict(color=color1, width=15),
        opacity=0.9,
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Inner glow ring
    inner_ring_radius = radius * 0.6
    fig.add_trace(go.Scatter3d(
        x=inner_ring_radius * np.cos(theta) + x_pos,
        y=inner_ring_radius * np.sin(theta),
        z=[height/2] * len(theta),
        mode='lines',
        line=dict(color=color1, width=8),
        opacity=0.5,
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Label above cylinder
    fig.add_trace(go.Scatter3d(
        x=[x_pos],
        y=[0],
        z=[height + 1.5],
        mode='text',
        text=[f"<b>{name}</b>"],
        textfont=dict(size=16, color=color1, family="Orbitron, monospace"),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Size label
    fig.add_trace(go.Scatter3d(
        x=[x_pos],
        y=[0],
        z=[height + 0.8],
        mode='text',
        text=[size],
        textfont=dict(size=11, color='white', family="Courier New, monospace"),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Connection beam to next layer
    if i < len(layers) - 1:
        next_x = (i + 1) * 6.5
        beam_points = 30
        beam_t = np.linspace(0, 1, beam_points)
        
        # Bezier curve for beam
        beam_x = x_pos + (next_x - x_pos) * beam_t
        beam_y = np.sin(beam_t * np.pi) * 0.8
        beam_z = height/2 + np.cos(beam_t * np.pi * 2) * 0.3
        
        # Main energy beam
        fig.add_trace(go.Scatter3d(
            x=beam_x,
            y=beam_y,
            z=beam_z,
            mode='lines',
            line=dict(
                color=color1,
                width=10
            ),
            opacity=0.6,
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Secondary glow beam
        fig.add_trace(go.Scatter3d(
            x=beam_x,
            y=beam_y * 1.2,
            z=beam_z,
            mode='lines',
            line=dict(
                color='white',
                width=4
            ),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip'
        ))

# Add flowing data particles
particle_count = 100
np.random.seed(123)
total_length = len(layers) * 6.5
particle_x = np.random.uniform(0, total_length - 6.5, particle_count)
particle_y = np.random.uniform(-3, 3, particle_count)
particle_z = np.random.uniform(1, 5, particle_count)
particle_colors = np.random.rand(particle_count)

fig.add_trace(go.Scatter3d(
    x=particle_x,
    y=particle_y,
    z=particle_z,
    mode='markers',
    marker=dict(
        size=4,
        color=particle_colors,
        colorscale='Viridis',
        opacity=0.6,
        symbol='diamond',
        line=dict(color='white', width=1)
    ),
    showlegend=False,
    hoverinfo='skip'
))

# Add grid floor for depth perception
grid_size = 20
grid_x = np.linspace(-5, total_length, grid_size)
grid_y = np.linspace(-4, 4, grid_size)

# Horizontal grid lines
for y_val in np.linspace(-4, 4, 10):
    fig.add_trace(go.Scatter3d(
        x=grid_x,
        y=[y_val] * grid_size,
        z=[-0.5] * grid_size,
        mode='lines',
        line=dict(color='rgba(0, 255, 255, 0.1)', width=1),
        showlegend=False,
        hoverinfo='skip'
    ))

# Vertical grid lines
for x_val in np.linspace(0, total_length - 6.5, 12):
    fig.add_trace(go.Scatter3d(
        x=[x_val] * grid_size,
        y=grid_y,
        z=[-0.5] * grid_size,
        mode='lines',
        line=dict(color='rgba(0, 255, 255, 0.1)', width=1),
        showlegend=False,
        hoverinfo='skip'
    ))

# Layout
fig.update_layout(
    title=dict(
        text="<b>⚡ ASL NEURAL PIPELINE ⚡</b><br><sub>Deep Learning Architecture Visualization</sub>",
        font=dict(size=32, color='#00FFFF', family="Orbitron, sans-serif"),
        x=0.5,
        y=0.97,
        xanchor='center'
    ),
    scene=dict(
        xaxis=dict(
            showbackground=False,
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            visible=False
        ),
        yaxis=dict(
            showbackground=False,
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            visible=False
        ),
        zaxis=dict(
            showbackground=False,
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            visible=False
        ),
        bgcolor='#000000',
        camera=dict(
            eye=dict(x=1.5, y=1.8, z=0.8),
            center=dict(x=0.3, y=0, z=0),
            up=dict(x=0, y=0, z=1)
        ),
        aspectmode='manual',
        aspectratio=dict(x=3, y=1, z=0.8)
    ),
    paper_bgcolor='#000000',
    plot_bgcolor='#000000',
    showlegend=False,
    margin=dict(l=10, r=10, t=120, b=10),
    width=1920,
    height=1080,
    font=dict(family="Orbitron, monospace", color='#00FFFF')
)

# Save
fig.write_html("asl_cnn_3d_architecture_interactive.html", config={'displayModeBar': True})

print("\n" + "="*70)
print("⚡ FUTURISTIC NEURAL PIPELINE GENERATED! ⚡")
print("="*70)
print("\n🔮 NEW DESIGN FEATURES:")
print("  ┃ ")
print("  ┣━ 🌀 Cylindrical tube design representing neural data flow")
print("  ┣━ 💫 Energy rings pulsing at each layer node")
print("  ┣━ ⚡ Glowing connection beams between pipeline segments")
print("  ┣━ 🎨 Gradient colors flowing through each cylinder")
print("  ┣━ 💎 100 floating data particles with viridis colormap")
print("  ┣━ 🌐 Holographic grid floor for depth perception")
print("  ┣━ 🔵 Radius & height scaled by layer complexity")
print("  ┣━ ✨ Sci-fi aesthetic with Orbitron font")
print("  ┗━ 🖥️  1920×1080 Full HD resolution")
print("\n" + "="*70)
print("� File: asl_cnn_3d_architecture_interactive.html")
print("🚀 Opening in browser...")
print("="*70 + "\n")

fig.show()
