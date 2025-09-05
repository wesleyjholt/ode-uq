"""
Interactive plotting functions for ODE uncertainty quantification results.
"""

import numpy as np
try:
    import plotly.graph_objects as go
    import plotly.subplots as sp
    from plotly.offline import plot
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import dash
    from dash import dcc, html, Input, Output
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

from .up import SimulationResults


def create_sobol_dashboard(sobol_results, dynamical_system, port=8051):
    """
    Create an interactive Dash dashboard for visualizing Sobol sensitivity analysis results.
    
    Args:
        sobol_results: dict - Output from analyze_sobol_results function containing 'S1' and 'ST' keys
        dynamical_system: DynamicalSystem instance
        port: int - Port to run the dashboard on
        
    Returns:
        Dash app instance
    """
    if not DASH_AVAILABLE:
        raise ImportError("dash is required for interactive plotting. Install with: pip install dash")
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for plotting. Install with: pip install plotly")
    
    app = dash.Dash(__name__)
    
    # Extract available outputs
    s1_results = sobol_results['S1']
    st_results = sobol_results['ST']
    
    # Get available state names, pointwise outputs, and functional outputs
    state_names = list(s1_results.states.keys()) if s1_results.states else []
    pointwise_names = list(s1_results.pointwise_outputs.keys()) if s1_results.pointwise_outputs else []
    functional_names = list(s1_results.functional_outputs.keys()) if s1_results.functional_outputs else []
    
    # Get input names for the plots
    if state_names:
        sample_state_result = s1_results.states[state_names[0]]
        param_names = list(sample_state_result.params.keys()) if sample_state_result.params else []
        init_state_names = [k for k, v in sample_state_result.init_state.items() if v is not None] if sample_state_result.init_state else []
        input_names = param_names + init_state_names
    else:
        input_names = []
    
    # Dashboard layout
    app.layout = html.Div([
        html.H1("Sobol Sensitivity Analysis Dashboard", style={'textAlign': 'center'}),
        
        # Controls
        html.Div([
            html.Div([
                html.Label("Sobol Index Type:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.RadioItems(
                    id='index-type',
                    options=[
                        {'label': 'First-order (S1)', 'value': 'S1'},
                        {'label': 'Total-effect (ST)', 'value': 'ST'}
                    ],
                    value='S1',
                    inline=True,
                    style={'marginTop': '5px'}
                )
            ], style={'width': '30%', 'display': 'inline-block', 'paddingRight': '20px'}),
            
            html.Div([
                html.Label("Output to Display:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='output-selector',
                    options=[
                        {'label': f'State: {name}', 'value': f'state_{name}'} for name in state_names
                    ] + [
                        {'label': f'Pointwise: {name}', 'value': f'pointwise_{name}'} for name in pointwise_names
                    ],
                    value=f'state_{state_names[0]}' if state_names else None,
                    style={'fontSize': '14px'}
                )
            ], style={'width': '33%', 'display': 'inline-block', 'paddingRight': '20px'}),
            
            html.Div([
                html.Label("Inputs to Show:", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                dcc.Checklist(
                    id='input-selector',
                    options=[
                        {'label': f'Param: {name}', 'value': f'param_{name}'} for name in param_names
                    ] + [
                        {'label': f'Init: {name}', 'value': f'init_{name}'} for name in init_state_names
                    ],
                    value=[f'param_{name}' for name in param_names] + [f'init_{name}' for name in init_state_names],  # All selected by default
                    inline=False,  # Stack vertically for better spacing
                    style={
                        'fontSize': '14px',
                        'lineHeight': '1.8',  # Better spacing between items
                        'marginTop': '5px'
                    },
                    inputStyle={'marginRight': '8px', 'marginLeft': '0px'},  # Space between checkbox and label
                    labelStyle={'marginBottom': '8px', 'display': 'block'}  # Space between each option
                )
            ], style={'width': '33%', 'display': 'inline-block'})
        ], style={'margin': '20px', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
        
        # Time series plot
        html.Div([
            dcc.Graph(id='sobol-timeseries')
        ], style={'width': '100%', 'margin': '20px'}),
        
        # Functional outputs control section
        html.Div([
            html.Div([
                html.Label("Functional Output to Display:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='functional-output-selector',
                    options=[
                        {'label': f'Functional: {name}', 'value': name} for name in functional_names
                    ],
                    value=functional_names[0] if functional_names else None,
                    style={'fontSize': '14px'}
                )
            ], style={'width': '40%', 'display': 'inline-block'})
        ], style={'margin': '20px', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}) if functional_names else html.Div(),
        
        # Functional outputs bar chart
        html.Div([
            dcc.Graph(id='functional-outputs-bar')
        ], style={'width': '100%', 'margin': '20px'}) if functional_names else html.Div()
    ])
    
    # Callback for updating plots
    @app.callback(
        [Output('sobol-timeseries', 'figure'),
         Output('functional-outputs-bar', 'figure')],
        [Input('index-type', 'value'),
         Input('output-selector', 'value'),
         Input('input-selector', 'value'),
         Input('functional-output-selector', 'value')]
    )
    def update_plots(index_type, selected_output, selected_inputs, selected_functional_output):
        results = sobol_results[index_type]
        
        # Time series plot
        timeseries_fig = create_sobol_timeseries_plot(
            results, selected_output, dynamical_system.times, input_names, index_type, selected_inputs
        )
        
        # Functional outputs bar chart - pass the selected functional output
        functional_fig = create_functional_outputs_bar_chart(
            results, functional_names, input_names, index_type, selected_functional_output
        )
        
        return timeseries_fig, functional_fig
    
    return app


def create_sobol_timeseries_plot(results, selected_output, times, input_names, index_type, selected_inputs=None):
    """Create time series plot of Sobol indices."""
    if not selected_output:
        return go.Figure()
    
    # If no inputs selected, show all
    if selected_inputs is None:
        param_names = list(results.states[list(results.states.keys())[0]].params.keys()) if results.states else []
        init_state_names = [k for k, v in results.states[list(results.states.keys())[0]].init_state.items() if v is not None] if results.states else []
        selected_inputs = [f'param_{name}' for name in param_names] + [f'init_{name}' for name in init_state_names]
    
    output_type, output_name = selected_output.split('_', 1)
    
    if output_type == 'state':
        if not results.states or output_name not in results.states:
            return go.Figure()
        output_results = results.states[output_name]
    elif output_type == 'pointwise':
        if not results.pointwise_outputs or output_name not in results.pointwise_outputs:
            return go.Figure()
        output_results = results.pointwise_outputs[output_name]
    else:
        return go.Figure()
    
    fig = go.Figure()
    
    # Color palette for different inputs
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Plot parameters
    if output_results.params:
        for i, param_name in enumerate(output_results.params.keys()):
            param_key = f'param_{param_name}'
            if param_name in input_names and param_key in selected_inputs:
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    x=times,
                    y=output_results.params[param_name],
                    mode='lines',
                    name=f'param: {param_name}',
                    line=dict(color=color, width=2)
                ))
    
    # Plot initial states
    if output_results.init_state:
        param_count = len(output_results.params) if output_results.params else 0
        for i, (init_name, init_values) in enumerate(output_results.init_state.items()):
            init_key = f'init_{init_name}'
            if init_values is not None and init_name in input_names and init_key in selected_inputs:
                color = colors[(param_count + i) % len(colors)]
                fig.add_trace(go.Scatter(
                    x=times,
                    y=init_values,
                    mode='lines',
                    name=f'init: {init_name}',
                    line=dict(color=color, width=2, dash='dash')
                ))
    
    fig.update_layout(
        title=f'{index_type} Sobol Indices vs Time for {output_name}',
        xaxis_title='Time',
        yaxis_title=f'{index_type} Sobol Index',
        legend=dict(x=1.05, y=1),
        height=500
    )
    
    return fig


def create_functional_outputs_bar_chart(results, functional_names, input_names, index_type, selected_functional_output=None):
    """Create bar chart of Sobol indices for functional outputs."""
    if not functional_names or not results.functional_outputs:
        return go.Figure()
    
    # If no specific functional output selected, use the first one or show all
    if selected_functional_output is None:
        target_functional_names = functional_names
    else:
        target_functional_names = [selected_functional_output] if selected_functional_output in functional_names else functional_names
    
    fig = go.Figure()
    
    # Colors for different inputs
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for func_name in target_functional_names:
        if func_name not in results.functional_outputs:
            continue
            
        func_results = results.functional_outputs[func_name]
        
        # Collect all input values for this functional output
        input_values = []
        input_labels = []
        input_colors = []
        
        # Parameters
        if func_results.params:
            for i, (param_name, param_value) in enumerate(func_results.params.items()):
                if param_name in input_names:
                    input_values.append(float(param_value))
                    input_labels.append(f'param: {param_name}')
                    input_colors.append(colors[i % len(colors)])
        
        # Initial states
        if func_results.init_state:
            param_count = len(func_results.params) if func_results.params else 0
            for i, (init_name, init_value) in enumerate(func_results.init_state.items()):
                if init_value is not None and init_name in input_names:
                    input_values.append(float(init_value))
                    input_labels.append(f'init: {init_name}')
                    input_colors.append(colors[(param_count + i) % len(colors)])
        
        # Create grouped bar chart
        x_positions = np.arange(len(input_labels))
        
        fig.add_trace(go.Bar(
            x=input_labels,
            y=input_values,
            name=func_name,
            marker_color=input_colors,
            text=[f'{val:.3f}' for val in input_values],
            textposition='auto'
        ))
    
    # Create title based on selection
    if selected_functional_output and len(target_functional_names) == 1:
        title = f'{index_type} Sobol Indices for {selected_functional_output}'
    else:
        title = f'{index_type} Sobol Indices for Functional Outputs'
    
    fig.update_layout(
        title=title,
        xaxis_title='Input Variables',
        yaxis_title=f'{index_type} Sobol Index',
        height=400,
        showlegend=True if len(target_functional_names) > 1 else False
    )
    
    return fig


def run_sobol_dashboard(sobol_results, dynamical_system, port=8051, debug=False):
    """
    Run the Sobol sensitivity analysis dashboard.
    
    Args:
        sobol_results: dict - Output from analyze_sobol_results function
        dynamical_system: DynamicalSystem instance
        port: int - Port to run the dashboard on
        debug: bool - Whether to run in debug mode
    """
    app = create_sobol_dashboard(sobol_results, dynamical_system, port)
    print(f"Starting Sobol dashboard on http://localhost:{port}")
    app.run(port=port, debug=debug)


def plot_state_trajectories_static(times, state_dict, max_samples=100, opacity=0.3):
    """
    Create static interactive plots for sampled state trajectories.
    
    Args:
        times: array-like, shape (num_times,) - time points
        state_dict: dict of state_name -> array of shape (num_samples, num_times)
        max_samples: int - maximum number of samples to plot (for performance)
        opacity: float - opacity of trajectory lines
    
    Returns:
        plotly Figure object with subplots for each state variable
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for plotting. Install with: pip install plotly")
    
    state_names = list(state_dict.keys())
    num_states = len(state_names)
    
    # Get the actual number of samples to plot
    actual_samples = min(max_samples, min([state_dict[name].shape[0] for name in state_names]))
    
    # Generate a color palette for consistent coloring across subplots
    import plotly.colors as colors
    color_palette = colors.qualitative.Plotly  # Default Plotly color sequence
    # Extend the palette if we have more samples than colors
    if actual_samples > len(color_palette):
        color_palette = color_palette * (actual_samples // len(color_palette) + 1)
    
    # Create subplots
    fig = sp.make_subplots(
        rows=num_states, 
        cols=1,
        subplot_titles=state_names,
        shared_xaxes=True,
        vertical_spacing=0.1
    )
    
    for i, state_name in enumerate(state_names):
        samples = state_dict[state_name]
        
        # Add trajectory lines with consistent colors
        for j in range(actual_samples):
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=samples[j],
                    mode='lines',
                    line=dict(width=1, color=color_palette[j]),
                    opacity=opacity,
                    name=f'Sample {j+1}',
                    showlegend=False,  # No legend for trajectory plots
                    legendgroup=f'sample_{j}',  # Group legend entries
                    hovertemplate=f'{state_name}<br>Sample {j+1}<br>Time: %{{x}}<br>Value: %{{y}}<extra></extra>'
                ),
                row=i+1, col=1
            )
        
        # Update y-axis labels (extract just the name from title format)
        axis_label = state_name.split('[')[1].rstrip(']') if '[' in state_name else state_name
        fig.update_yaxes(title_text=axis_label, row=i+1, col=1)
    
    # Update x-axis label for bottom plot
    fig.update_xaxes(title_text="Time", row=num_states, col=1)
    
    # Update layout
    fig.update_layout(
        height=300 * num_states,
        hovermode='closest'
    )
    
    return fig


def plot_trajectories_from_mc_results(mc_results, max_samples=100, opacity=0.3):
    """
    Create static interactive plots for trajectories from SimulationResults.
    
    Args:
        mc_results: SimulationResults object
        max_samples: int - maximum number of samples to plot (for performance)
        opacity: float - opacity of trajectory lines
    
    Returns:
        plotly Figure object with subplots for states and pointwise outputs
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for plotting. Install with: pip install plotly")
    
    # Combine states and pointwise outputs with proper titles
    all_trajectories = {}
    
    # Add states
    if mc_results.states is not None:
        for state_name, state_data in mc_results.states.items():
            all_trajectories[f"State[{state_name}]"] = state_data
    
    # Add pointwise outputs
    if mc_results.pointwise_outputs is not None:
        for output_name, output_data in mc_results.pointwise_outputs.items():
            all_trajectories[f"Output[{output_name}]"] = output_data
    
    if not all_trajectories:
        raise ValueError("No trajectory data found in SimulationResults")
    
    return plot_state_trajectories_static(mc_results.times, all_trajectories, max_samples, opacity)


def plot_histograms_from_mc_results(mc_results, bins=30):
    """
    Create histograms for functional outputs and parameters from SimulationResults.
    
    Args:
        mc_results: SimulationResults object
        bins: int - number of bins for histograms
    
    Returns:
        plotly Figure object with histograms
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for plotting. Install with: pip install plotly")
    
    # Collect all histogram data with proper titles
    histogram_data = {}
    
    # Add functional outputs
    if mc_results.functional_outputs is not None:
        for output_name, output_data in mc_results.functional_outputs.items():
            histogram_data[f"Output[{output_name}]"] = np.array(output_data)
    
    # Add parameters
    if mc_results.params is not None:
        for param_name, param_data in mc_results.params.items():
            histogram_data[f"Parameter[{param_name}]"] = np.array(param_data)
    
    # Add initial states
    if mc_results.init_state is not None:
        for state_name, state_data in mc_results.init_state.items():
            histogram_data[f"InitialState[{state_name}]"] = np.array(state_data)
    
    if not histogram_data:
        return None
    
    # Calculate subplot layout
    num_hists = len(histogram_data)
    cols = min(3, num_hists)
    rows = (num_hists + cols - 1) // cols
    
    # Create subplots
    fig = sp.make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=list(histogram_data.keys()),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # Add histograms
    for i, (name, data) in enumerate(histogram_data.items()):
        row = i // cols + 1
        col = i % cols + 1
        
        # Determine color based on data type
        if name.startswith('Output['):
            color = '#1f77b4'  # Blue for outputs
        elif name.startswith('Parameter['):
            color = '#ff7f0e'  # Orange for parameters
        elif name.startswith('InitialState['):
            color = '#2ca02c'  # Green for initial states
        else:
            color = '#d62728'  # Red for unknown types
        
        fig.add_trace(
            go.Histogram(
                x=data,
                nbinsx=bins,
                name=name,
                showlegend=False,
                marker_color=color,
                hovertemplate=f'{name}<br>Value: %{{x}}<br>Count: %{{y}}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Update axes labels (extract just the name from title format)
        axis_label = name.split('[')[1].rstrip(']') if '[' in name else name
        fig.update_xaxes(title_text=axis_label, row=row, col=col)
        fig.update_yaxes(title_text="Count", row=row, col=col)
    
    fig.update_layout(
        height=300 * rows,
        showlegend=False
    )
    
    return fig


def create_comprehensive_dash_app(mc_results, max_samples=100):
    """
    Create a comprehensive Dash web application for SimulationResults visualization.
    
    Args:
        mc_results: SimulationResults object
        max_samples: int - maximum number of samples to plot
    
    Returns:
        Dash app instance
    """
    if not DASH_AVAILABLE:
        raise ImportError("dash is required for web apps. Install with: pip install dash")
    
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for plotting. Install with: pip install plotly")
    
    app = dash.Dash(__name__)
    
    # Create initial figures
    trajectory_fig = plot_trajectories_from_mc_results(mc_results, max_samples)
    histogram_fig = plot_histograms_from_mc_results(mc_results)
    statistics_fig = plot_statistics_from_mc_results(mc_results)
    
    # Count total trajectories for slider
    total_trajectories = 0
    if mc_results.states is not None:
        total_trajectories = max([data.shape[0] for data in mc_results.states.values()])
    elif mc_results.pointwise_outputs is not None:
        total_trajectories = max([data.shape[0] for data in mc_results.pointwise_outputs.values()])
    
    # App layout
    components = [
        html.H1("ODE Uncertainty Quantification Dashboard", 
                style={'textAlign': 'center', 'marginBottom': 30}),
        
        # Controls
        html.Div([
            html.Div([
                html.Label("Number of samples to display:"),
                dcc.Slider(
                    id='sample-slider',
                    min=1,
                    max=min(max_samples, total_trajectories),
                    value=min(50, total_trajectories),
                    marks={i: str(i) for i in range(0, min(max_samples, total_trajectories)+1, 10)},
                    step=1
                )
            ], style={'margin': '20px', 'width': '32%', 'display': 'inline-block'}),
            
            html.Div([
                html.Label("Line opacity:"),
                dcc.Slider(
                    id='opacity-slider',
                    min=0.1,
                    max=1.0,
                    value=0.3,
                    marks={i/10: f'{i/10:.1f}' for i in range(1, 11, 2)},
                    step=0.1
                )
            ], style={'margin': '20px', 'width': '32%', 'display': 'inline-block'}),
            
            html.Div([
                html.Label("Display mode:"),
                dcc.RadioItems(
                    id='display-mode',
                    options=[
                        {'label': 'Sampled Trajectories', 'value': 'trajectories'},
                        {'label': 'Statistics', 'value': 'statistics'}
                    ],
                    value='trajectories',
                    style={'marginTop': '10px'}
                )
            ], style={'margin': '20px', 'width': '32%', 'display': 'inline-block'}),
        ], style={'clearfix': 'both'}),
        
        # Trajectory/Statistics plot (toggleable)
        html.H2(id='trajectory-title', children="State and Output Trajectories", style={'marginTop': 40}),
        dcc.Graph(
            id='trajectories-plot',
            figure=trajectory_fig,
            config={
                'scrollZoom': True,
                'displayModeBar': True,
                'displaylogo': False
            }
        )
    ]
    
    # Add histogram plot if data exists
    if histogram_fig is not None:
        components.extend([
            html.H2("Parameter and Functional Output Distributions", style={'marginTop': 40}),
            dcc.Graph(
                id='histograms-plot',
                figure=histogram_fig,
                config={
                    'displayModeBar': True,
                    'displaylogo': False
                }
            )
        ])
    
    app.layout = html.Div(components)
    
    # Callback for updating trajectory plot and title
    @app.callback(
        [Output('trajectories-plot', 'figure'),
         Output('trajectory-title', 'children')],
        [Input('sample-slider', 'value'),
         Input('opacity-slider', 'value'),
         Input('display-mode', 'value')]
    )
    def update_trajectory_plot(num_samples, opacity, display_mode):
        if display_mode == 'trajectories':
            fig = plot_trajectories_from_mc_results(mc_results, num_samples, opacity)
            title = "State and Output Trajectories"
        else:  # statistics
            fig = plot_statistics_from_mc_results(mc_results, max_samples=None)
            title = "Trajectory Statistics"
        return fig, title
    
    return app


def plot_statistics_from_mc_results(mc_results, max_samples=None, percentiles=[5, 25, 50, 75, 95]):
    """
    Plot statistics (percentiles) of trajectories from SimulationResults.
    
    Args:
        mc_results: SimulationResults object
        max_samples: int - maximum number of samples to use for statistics
        percentiles: list of percentiles to compute and plot
    
    Returns:
        plotly Figure object with statistical summaries
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for plotting. Install with: pip install plotly")
    
    # Combine states and pointwise outputs with proper titles
    all_trajectories = {}
    
    # Add states
    if mc_results.states is not None:
        for state_name, state_data in mc_results.states.items():
            data = state_data if max_samples is None else state_data[:max_samples]
            all_trajectories[f"State[{state_name}]"] = data
    
    # Add pointwise outputs
    if mc_results.pointwise_outputs is not None:
        for output_name, output_data in mc_results.pointwise_outputs.items():
            data = output_data if max_samples is None else output_data[:max_samples]
            all_trajectories[f"Output[{output_name}]"] = data
    
    if not all_trajectories:
        raise ValueError("No trajectory data found in SimulationResults")
    
    return plot_state_statistics(mc_results.times, all_trajectories, percentiles)
    """
    Create static interactive plots for sampled state trajectories.
    
    Args:
        times: array-like, shape (num_times,) - time points
        state_dict: dict of state_name -> array of shape (num_samples, num_times)
        max_samples: int - maximum number of samples to plot (for performance)
        opacity: float - opacity of trajectory lines
    
    Returns:
        plotly Figure object with subplots for each state variable
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for plotting. Install with: pip install plotly")
    
    state_names = list(state_dict.keys())
    num_states = len(state_names)
    
    # Create subplots
    fig = sp.make_subplots(
        rows=num_states, 
        cols=1,
        subplot_titles=state_names,
        shared_xaxes=True,
        vertical_spacing=0.1
    )
    
    for i, state_name in enumerate(state_names):
        samples = state_dict[state_name]
        num_samples = min(samples.shape[0], max_samples)
        
        # Add trajectory lines
        for j in range(num_samples):
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=samples[j],
                    mode='lines',
                    line=dict(width=1),
                    opacity=opacity,
                    name=f'{state_name}_sample_{j+1}',
                    showlegend=False,
                    hovertemplate=f'{state_name}<br>Time: %{{x}}<br>Value: %{{y}}<extra></extra>'
                ),
                row=i+1, col=1
            )
        
        # Update y-axis labels
        fig.update_yaxes(title_text=state_name, row=i+1, col=1)
    
    # Update x-axis label for bottom plot
    fig.update_xaxes(title_text="Time", row=num_states, col=1)
    
    # Update layout
    fig.update_layout(
        height=300 * num_states,
        hovermode='closest'
    )
    
    return fig


def create_dash_app(times, state_dict, max_samples=100):
    """
    Create a Dash web application for interactive visualization of state trajectories.
    
    Args:
        times: array-like, shape (num_times,) - time points
        state_dict: dict of state_name -> array of shape (num_samples, num_times)
        max_samples: int - maximum number of samples to plot
    
    Returns:
        Dash app instance
    """
    if not DASH_AVAILABLE:
        raise ImportError("dash is required for web apps. Install with: pip install dash")
    
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for plotting. Install with: pip install plotly")
    
    app = dash.Dash(__name__)
    
    state_names = list(state_dict.keys())
    
    # Create initial figure
    fig = plot_state_trajectories_static(times, state_dict, max_samples)
    
    # App layout
    app.layout = html.Div([
        html.H1("ODE Uncertainty Quantification - State Trajectories", 
                style={'textAlign': 'center', 'marginBottom': 30}),
        
        html.Div([
            html.Label("Number of samples to display:"),
            dcc.Slider(
                id='sample-slider',
                min=1,
                max=min(max_samples, max([state_dict[s].shape[0] for s in state_names])),
                value=min(50, max([state_dict[s].shape[0] for s in state_names])),
                marks={i: str(i) for i in range(0, min(max_samples, max([state_dict[s].shape[0] for s in state_names]))+1, 10)},
                step=1
            )
        ], style={'margin': '20px', 'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Line opacity:"),
            dcc.Slider(
                id='opacity-slider',
                min=0.1,
                max=1.0,
                value=0.3,
                marks={i/10: f'{i/10:.1f}' for i in range(1, 11, 2)},
                step=0.1
            )
        ], style={'margin': '20px', 'width': '48%', 'float': 'right', 'display': 'inline-block'}),
        
        dcc.Graph(
            id='trajectories-plot',
            figure=fig,
            config={
                'scrollZoom': True,
                'displayModeBar': True,
                'displaylogo': False
            }
        )
    ])
    
    # Callback for updating plot
    @app.callback(
        Output('trajectories-plot', 'figure'),
        [Input('sample-slider', 'value'),
         Input('opacity-slider', 'value')]
    )
    def update_plot(num_samples, opacity):
        # Create subset of state_dict with limited samples
        subset_state_dict = {}
        for state_name in state_names:
            subset_state_dict[state_name] = state_dict[state_name][:num_samples]
        
        return plot_state_trajectories_static(times, subset_state_dict, num_samples, opacity)
    
    return app


def plot_state_statistics(times, state_dict, percentiles=[5, 25, 50, 75, 95]):
    """
    Plot statistics (percentiles) of the state trajectories over time.
    
    Args:
        times: array-like, shape (num_times,) - time points
        state_dict: dict of state_name -> array of shape (num_samples, num_times)
        percentiles: list of percentiles to compute and plot
    
    Returns:
        plotly Figure object with statistical summaries
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for plotting. Install with: pip install plotly")
    
    state_names = list(state_dict.keys())
    num_states = len(state_names)
    
    # Create subplots
    fig = sp.make_subplots(
        rows=num_states, 
        cols=1,
        subplot_titles=state_names,
        shared_xaxes=True,
        vertical_spacing=0.1
    )
    
    colors = ['rgba(0,100,80,0.2)', 'rgba(0,176,246,0.2)', 'rgba(231,107,243,1)', 
              'rgba(0,176,246,0.2)', 'rgba(0,100,80,0.2)']
    
    for i, state_name in enumerate(state_names):
        samples = state_dict[state_name]
        
        # Compute percentiles
        perc_values = np.percentile(samples, percentiles, axis=0)
        
        # Plot percentile bands
        for j in range(len(percentiles)//2):
            lower_idx = j
            upper_idx = len(percentiles) - 1 - j
            
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([times, times[::-1]]),
                    y=np.concatenate([perc_values[lower_idx], perc_values[upper_idx][::-1]]),
                    fill='toself',
                    fillcolor=colors[j] if j < len(colors) else 'rgba(128,128,128,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{percentiles[lower_idx]}-{percentiles[upper_idx]}%',
                    showlegend=(i == 0),
                    hoverinfo='skip'
                ),
                row=i+1, col=1
            )
        
        # Plot median
        median_idx = len(percentiles) // 2
        fig.add_trace(
            go.Scatter(
                x=times,
                y=perc_values[median_idx],
                mode='lines',
                line=dict(color='rgba(231,107,243,1)', width=2),
                name='Median' if i == 0 else None,
                showlegend=(i == 0),
                hovertemplate=f'{state_name} Median<br>Time: %{{x}}<br>Value: %{{y}}<extra></extra>'
            ),
            row=i+1, col=1
        )
        
        # Update y-axis labels (extract just the name from title format)
        axis_label = state_name.split('[')[1].rstrip(']') if '[' in state_name else state_name
        fig.update_yaxes(title_text=axis_label, row=i+1, col=1)
    
    # Update x-axis label for bottom plot
    fig.update_xaxes(title_text="Time", row=num_states, col=1)
    
    # Update layout
    fig.update_layout(
        height=300 * num_states,
        hovermode='closest'
    )
    
    return fig


def save_plot_html(fig, filename="trajectories.html"):
    """
    Save a plotly figure as an HTML file.
    
    Args:
        fig: plotly Figure object
        filename: str - output filename
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for saving plots. Install with: pip install plotly")
    
    plot(fig, filename=filename, auto_open=False)
    print(f"Plot saved as {filename}")


def run_dash_app(mc_results_or_times, state_dict=None, max_samples=100, debug=True, port=8050):
    """
    Convenience function to create and run a Dash app.
    
    Args:
        mc_results_or_times: SimulationResults object OR array-like times (for backward compatibility)
        state_dict: dict (only used if first arg is times, for backward compatibility)
        max_samples: int - maximum number of samples to plot
        debug: bool - run in debug mode
        port: int - port to run the server on
    """
    # Check if first argument is SimulationResults
    if hasattr(mc_results_or_times, 'times') and hasattr(mc_results_or_times, 'states'):
        # New interface: SimulationResults object
        app = create_comprehensive_dash_app(mc_results_or_times, max_samples)
    else:
        # Backward compatibility: times and state_dict
        if state_dict is None:
            raise ValueError("state_dict must be provided when using times array")
        app = create_dash_app(mc_results_or_times, state_dict, max_samples)
    
    app.run(debug=debug, port=port)
