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
        timeseries_fig = _create_sobol_timeseries_plot(
            results, selected_output, dynamical_system.times, input_names, index_type, selected_inputs
        )
        
        # Functional outputs bar chart - pass the selected functional output
        functional_fig = _create_functional_outputs_bar_chart(
            results, functional_names, input_names, index_type, selected_functional_output
        )
        
        return timeseries_fig, functional_fig
    
    return app


def _create_sobol_timeseries_plot(results, selected_output, times, input_names, index_type, selected_inputs=None):
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


def _create_functional_outputs_bar_chart(results, functional_names, input_names, index_type, selected_functional_output=None):
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


def _plot_trajectories_from_simulation_results(mc_results, selected_trajectories=None, max_samples=100, opacity=0.3):
    """
    Create static interactive plots for sampled state trajectories from SimulationResults.
    Each trajectory gets its own subplot, ordered by the trajectory_options order.
    
    Args:
        mc_results: SimulationResults object
        selected_trajectories: list of trajectory keys to plot (if None, plots all)
        max_samples: int - maximum number of samples to plot (for performance)
        opacity: float - opacity of trajectory lines
    
    Returns:
        plotly Figure object with subplots for each selected trajectory
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for plotting. Install with: pip install plotly")
    
    # Build the complete trajectory order (states first, then outputs)
    trajectory_order = []
    trajectory_data = {}
    
    # Add states in their natural order
    if mc_results.states is not None:
        for state_name, state_data in mc_results.states.items():
            key = f"state_{state_name}"
            trajectory_order.append(key)
            trajectory_data[key] = {'data': state_data, 'title': f"State: {state_name}"}
    
    # Add pointwise outputs in their natural order
    if mc_results.pointwise_outputs is not None:
        for output_name, output_data in mc_results.pointwise_outputs.items():
            key = f"output_{output_name}"
            trajectory_order.append(key)
            trajectory_data[key] = {'data': output_data, 'title': f"Output: {output_name}"}
    
    # Filter to selected trajectories if specified, maintaining order
    if selected_trajectories is not None:
        selected_keys = [key for key in trajectory_order if key in selected_trajectories]
    else:
        selected_keys = trajectory_order
    
    if not selected_keys:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title="No trajectories selected",
            xaxis_title="Time",
            yaxis_title="Value",
            template="plotly_white"
        )
        return fig
    
    num_plots = len(selected_keys)
    
    # Create subplots
    subplot_titles = [trajectory_data[key]['title'] for key in selected_keys]
    fig = sp.make_subplots(
        rows=num_plots, 
        cols=1,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        vertical_spacing=0.15 / max(1, num_plots - 1) if num_plots > 1 else 0.1
    )
    
    # Get the actual number of samples to plot
    actual_samples = min(max_samples, min([trajectory_data[key]['data'].shape[0] for key in selected_keys]))
    
    # Generate a color palette for consistent coloring across subplots
    import plotly.colors as colors
    color_palette = colors.qualitative.Plotly  # Default Plotly color sequence
    # Extend the palette if we have more samples than colors
    if actual_samples > len(color_palette):
        color_palette = color_palette * (actual_samples // len(color_palette) + 1)
    
    for i, key in enumerate(selected_keys):
        samples = trajectory_data[key]['data']
        title = trajectory_data[key]['title']
        
        # Add trajectory lines with consistent colors
        for j in range(actual_samples):
            fig.add_trace(
                go.Scatter(
                    x=mc_results.times,
                    y=samples[j],
                    mode='lines',
                    line=dict(width=1, color=color_palette[j]),
                    opacity=opacity,
                    name=f'Sample {j+1}',
                    showlegend=False,  # No legend for trajectory plots
                    legendgroup=f'sample_{j}',  # Group legend entries
                    hovertemplate=f'{title}<br>Sample {j+1}<br>Time: %{{x}}<br>Value: %{{y}}<extra></extra>'
                ),
                row=i+1, col=1
            )
        
        # Update y-axis labels (extract just the name from title)
        axis_label = title.split(': ')[1] if ': ' in title else title
        fig.update_yaxes(title_text=axis_label, row=i+1, col=1)
    
    # Update x-axis label for bottom plot
    fig.update_xaxes(title_text="Time", row=num_plots, col=1)
    
    # Update layout
    fig.update_layout(
        height=300 * num_plots,
        hovermode='closest'
    )
    
    return fig


def _plot_trajectories_from_mc_results(mc_results, max_samples=100, opacity=0.3):
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
    
    # Use the new function that handles SimulationResults directly
    return _plot_trajectories_from_simulation_results(mc_results, None, max_samples, opacity)


def _plot_histograms_from_mc_results(mc_results, bins=30, selected_histograms=None):
    """
    Create histograms for functional outputs and parameters from SimulationResults.
    
    Args:
        mc_results: SimulationResults object
        bins: int - number of bins for histograms
        selected_histograms: list of str - keys of histograms to plot (if None, plot all)
    
    Returns:
        plotly Figure object with histograms
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for plotting. Install with: pip install plotly")
    
    # Collect all histogram data with proper titles
    all_histogram_data = {}
    
    # Add functional outputs
    if mc_results.functional_outputs is not None:
        for output_name, output_data in mc_results.functional_outputs.items():
            key = f"output_{output_name}"
            all_histogram_data[key] = {
                'data': np.array(output_data),
                'title': f"Output[{output_name}]",
                'color': '#1f77b4'  # Blue for outputs
            }
    
    # Add parameters
    if mc_results.params is not None:
        for param_name, param_data in mc_results.params.items():
            key = f"param_{param_name}"
            all_histogram_data[key] = {
                'data': np.array(param_data),
                'title': f"Parameter[{param_name}]",
                'color': '#ff7f0e'  # Orange for parameters
            }
    
    # Add initial states
    if mc_results.init_state is not None:
        for state_name, state_data in mc_results.init_state.items():
            key = f"init_{state_name}"
            all_histogram_data[key] = {
                'data': np.array(state_data),
                'title': f"InitialState[{state_name}]",
                'color': '#2ca02c'  # Green for initial states
            }
    
    if not all_histogram_data:
        return None
    
    # Filter to selected histograms if specified
    if selected_histograms is not None:
        histogram_data = {k: v for k, v in all_histogram_data.items() if k in selected_histograms}
    else:
        histogram_data = all_histogram_data
    
    if not histogram_data:
        return go.Figure()  # Return empty figure if no selected histograms
    
    # Calculate subplot layout
    num_hists = len(histogram_data)
    cols = min(3, num_hists)
    rows = (num_hists + cols - 1) // cols
    
    # Ensure vertical spacing doesn't exceed the limit
    max_vertical_spacing = 1.0 / (rows - 1) if rows > 1 else 0.15
    vertical_spacing = min(0.15, max_vertical_spacing * 0.8)  # Use 80% of max to be safe
    
    # Create subplots
    fig = sp.make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[item['title'] for item in histogram_data.values()],
        vertical_spacing=vertical_spacing,
        horizontal_spacing=0.1
    )
    
    # Add histograms
    for i, (key, item) in enumerate(histogram_data.items()):
        row = i // cols + 1
        col = i % cols + 1
        
        fig.add_trace(
            go.Histogram(
                x=item['data'],
                nbinsx=bins,
                name=item['title'],
                showlegend=False,
                marker_color=item['color'],
                hovertemplate=f'{item["title"]}<br>Value: %{{x}}<br>Count: %{{y}}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Update axes labels (extract just the name from title format)
        axis_label = item['title'].split('[')[1].rstrip(']') if '[' in item['title'] else key
        fig.update_xaxes(title_text=axis_label, row=row, col=col)
        fig.update_yaxes(title_text="Count", row=row, col=col)
    
    fig.update_layout(
        height=max(350 * rows, 400),  # Minimum height of 400
        showlegend=False
    )
    
    return fig


def _get_histogram_options(mc_results):
    """
    Get all available histogram options from SimulationResults.
    
    Args:
        mc_results: SimulationResults object
    
    Returns:
        tuple: (options_list, all_keys_list) where options_list contains dicts for Dash dropdown
               and all_keys_list contains the keys for default selection
    """
    options = []
    all_keys = []
    
    # Add functional outputs
    if mc_results.functional_outputs is not None:
        for output_name in mc_results.functional_outputs.keys():
            key = f"output_{output_name}"
            label = f"Output: {output_name}"
            options.append({'label': label, 'value': key})
            all_keys.append(key)
    
    # Add parameters  
    if mc_results.params is not None:
        for param_name in mc_results.params.keys():
            key = f"param_{param_name}"
            label = f"Parameter: {param_name}"
            options.append({'label': label, 'value': key})
            all_keys.append(key)
    
    # Add initial states
    if mc_results.init_state is not None:
        for state_name in mc_results.init_state.keys():
            key = f"init_{state_name}"
            label = f"Initial State: {state_name}"
            options.append({'label': label, 'value': key})
            all_keys.append(key)
    
    return options, all_keys


def _organize_options_by_type(mc_results):
    """
    Organize all trajectory and histogram options by value type.
    
    Returns:
        Dictionary with separate lists for each type: states, pointwise_outputs, 
        functional_outputs, parameters, initial_states
    """
    organized = {
        'states': {'trajectory': [], 'histogram': []},
        'pointwise_outputs': {'trajectory': [], 'histogram': []},
        'functional_outputs': {'trajectory': [], 'histogram': []},
        'parameters': {'trajectory': [], 'histogram': []},
        'initial_states': {'trajectory': [], 'histogram': []}
    }
    
    # Add states
    if mc_results.states is not None:
        for state_name in mc_results.states.keys():
            traj_key = f"state_{state_name}"
            organized['states']['trajectory'].append({'label': state_name, 'value': traj_key})
    
    # Add pointwise outputs
    if mc_results.pointwise_outputs is not None:
        for output_name in mc_results.pointwise_outputs.keys():
            traj_key = f"output_{output_name}"
            organized['pointwise_outputs']['trajectory'].append({'label': output_name, 'value': traj_key})
    
    # Add functional outputs (only in histograms)
    if mc_results.functional_outputs is not None:
        for output_name in mc_results.functional_outputs.keys():
            hist_key = f"output_{output_name}"
            organized['functional_outputs']['histogram'].append({'label': output_name, 'value': hist_key})
    
    # Add parameters (only in histograms)
    if mc_results.params is not None:
        for param_name in mc_results.params.keys():
            hist_key = f"param_{param_name}"
            organized['parameters']['histogram'].append({'label': param_name, 'value': hist_key})
    
    # Add initial states (only in histograms)
    if mc_results.init_state is not None:
        for state_name in mc_results.init_state.keys():
            hist_key = f"init_{state_name}"
            organized['initial_states']['histogram'].append({'label': state_name, 'value': hist_key})
    
    return organized


def create_comprehensive_dashboard(mc_results, max_samples=100, max_plots=8, max_histograms=6):
    """
    Create a comprehensive Dash web application for SimulationResults visualization.
    
    Args:
        mc_results: SimulationResults object
        max_samples: int - maximum number of samples to plot
        max_plots: int - maximum number of trajectory plots to show at once
        max_histograms: int - maximum number of histograms to show at once
    
    Returns:
        Dash app instance
    """
    if not DASH_AVAILABLE:
        raise ImportError("dash is required for web apps. Install with: pip install dash")
    
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for plotting. Install with: pip install plotly")
    
    app = dash.Dash(__name__)
    
    # Helper function to create plot data for selected trajectories
    def plot_selected_trajectories(selected_keys):
        """Create trajectory plot for selected trajectories using the new plotting function"""
        return _plot_trajectories_from_simulation_results(mc_results, selected_keys, max_samples, 0.3)
    
    # Organize all options by type
    organized_options = _organize_options_by_type(mc_results)
    
    # Collect trajectory options and data in original format for compatibility
    trajectory_options = []
    trajectory_data = {}
    
    # Collect all trajectory options
    for category in ['states', 'pointwise_outputs']:
        for option in organized_options[category]['trajectory']:
            key = option['value']
            name = option['label']
            
            if category == 'states':
                full_label = f"State: {name}"
                trajectory_data[key] = {'data': mc_results.states[name], 'label': full_label}
            elif category == 'pointwise_outputs':
                full_label = f"Output: {name}"  
                trajectory_data[key] = {'data': mc_results.pointwise_outputs[name], 'label': full_label}
            
            trajectory_options.append({'label': full_label, 'value': key})
    
    # Select first few items by default (up to max_plots)
    default_selection = [opt['value'] for opt in trajectory_options[:max_plots]]
    
    # Collect all histogram options
    all_histogram_keys = []
    for category in ['functional_outputs', 'parameters', 'initial_states']:
        for option in organized_options[category]['histogram']:
            all_histogram_keys.append(option['value'])
    
    # Select first few histograms by default (up to max_histograms)
    default_histogram_selection = all_histogram_keys[:max_histograms]
    
    # Get histogram options for backward compatibility
    histogram_options, _ = _get_histogram_options(mc_results)
    
    # Create initial figures
    histogram_fig = _plot_histograms_from_mc_results(mc_results, selected_histograms=default_histogram_selection)
    
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
        
        # Main controls
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
            ], style={'margin': '20px', 'width': '32%', 'display': 'inline-block'})
        ], style={'clearfix': 'both'}),
        
        # Selection controls organized by type
        html.H2("Selection Controls", style={'marginTop': 30, 'marginBottom': 10}),
        html.Div([
            # Trajectory selections
            html.Div([
                # States column
                html.Div([
                    html.H4("States", style={'textAlign': 'center', 'marginBottom': 10, 'color': '#1f77b4'}),
                    dcc.Checklist(
                        id='states-trajectory-selector',
                        options=organized_options['states']['trajectory'],
                        value=[opt['value'] for opt in organized_options['states']['trajectory'][:max_plots//2]],
                        style={'fontSize': '14px'},
                        inputStyle={'marginRight': '5px'},
                        labelStyle={'display': 'block', 'marginBottom': '3px'}
                    )
                ], style={'width': '18%', 'display': 'inline-block', 'verticalAlign': 'top', 'margin': '10px'}),
                
                # Pointwise outputs column  
                html.Div([
                    html.H4("Pointwise Outputs", style={'textAlign': 'center', 'marginBottom': 10, 'color': '#ff7f0e'}),
                    dcc.Checklist(
                        id='pointwise-trajectory-selector',
                        options=organized_options['pointwise_outputs']['trajectory'],
                        value=[opt['value'] for opt in organized_options['pointwise_outputs']['trajectory'][:max_plots//2]],
                        style={'fontSize': '14px'},
                        inputStyle={'marginRight': '5px'},
                        labelStyle={'display': 'block', 'marginBottom': '3px'}
                    )
                ], style={'width': '18%', 'display': 'inline-block', 'verticalAlign': 'top', 'margin': '10px'}),
                
                # Functional outputs column
                html.Div([
                    html.H4("Functional Outputs", style={'textAlign': 'center', 'marginBottom': 10, 'color': '#2ca02c'}),
                    dcc.Checklist(
                        id='functional-histogram-selector',
                        options=organized_options['functional_outputs']['histogram'],
                        value=[opt['value'] for opt in organized_options['functional_outputs']['histogram'][:max_histograms//3]],
                        style={'fontSize': '14px'},
                        inputStyle={'marginRight': '5px'},
                        labelStyle={'display': 'block', 'marginBottom': '3px'}
                    )
                ], style={'width': '18%', 'display': 'inline-block', 'verticalAlign': 'top', 'margin': '10px'}),
                
                # Parameters column
                html.Div([
                    html.H4("Parameters", style={'textAlign': 'center', 'marginBottom': 10, 'color': '#d62728'}),
                    dcc.Checklist(
                        id='parameters-histogram-selector',
                        options=organized_options['parameters']['histogram'],
                        value=[opt['value'] for opt in organized_options['parameters']['histogram'][:max_histograms//3]],
                        style={'fontSize': '14px'},
                        inputStyle={'marginRight': '5px'},
                        labelStyle={'display': 'block', 'marginBottom': '3px'}
                    )
                ], style={'width': '18%', 'display': 'inline-block', 'verticalAlign': 'top', 'margin': '10px'}),
                
                # Initial states column
                html.Div([
                    html.H4("Initial States", style={'textAlign': 'center', 'marginBottom': 10, 'color': '#9467bd'}),
                    dcc.Checklist(
                        id='initial-histogram-selector',
                        options=organized_options['initial_states']['histogram'],
                        value=[opt['value'] for opt in organized_options['initial_states']['histogram'][:max_histograms//3]],
                        style={'fontSize': '14px'},
                        inputStyle={'marginRight': '5px'},
                        labelStyle={'display': 'block', 'marginBottom': '3px'}
                    )
                ], style={'width': '18%', 'display': 'inline-block', 'verticalAlign': 'top', 'margin': '10px'})
            ], style={'border': '1px solid #ddd', 'padding': '15px', 'borderRadius': '5px', 'backgroundColor': '#f9f9f9'}),
            
            # Selection info
            html.Div([
                html.Div(id='trajectory-selection-info', style={'fontSize': '12px', 'color': '#666', 'display': 'inline-block', 'marginRight': '20px'}),
                html.Div(id='histogram-selection-info', style={'fontSize': '12px', 'color': '#666', 'display': 'inline-block'})
            ], style={'marginTop': '10px', 'textAlign': 'center'})
        ])
    ]
    
    # Add trajectory plot
    components.extend([
        html.H2(id='trajectory-title', children="State and Output Trajectories", style={'marginTop': 40}),
        dcc.Graph(
            id='trajectories-plot',
            figure=go.Figure(),  # Empty figure initially, will be populated by callback
            config={
                'scrollZoom': True,
                'displayModeBar': True,
                'displaylogo': False
            }
        )
    ])
    
    # Add histogram plot if data exists
    if all_histogram_keys:
        components.extend([
            html.H2("Parameter and Functional Output Distributions", style={'marginTop': 40}),
            dcc.Graph(
                id='histograms-plot',
                figure=_plot_histograms_from_mc_results(mc_results, selected_histograms=default_histogram_selection),
                config={
                    'displayModeBar': True,
                    'displaylogo': False
                }
            )
        ])
    
    app.layout = html.Div(components)
    
    # Callback for collecting all selected trajectories and updating info
    @app.callback(
        [Output('trajectory-selection-info', 'children'),
         Output('states-trajectory-selector', 'options'),
         Output('pointwise-trajectory-selector', 'options')],
        [Input('states-trajectory-selector', 'value'),
         Input('pointwise-trajectory-selector', 'value')]
    )
    def update_trajectory_selection_info(selected_states, selected_pointwise):
        all_selected = (selected_states or []) + (selected_pointwise or [])
        num_selected = len(all_selected)
        info_text = f"Trajectories selected: {num_selected}/{max_plots}"
        
        # Update options to disable/enable based on selection limit
        def update_options(options, selected):
            updated_options = []
            for opt in options:
                is_selected = opt['value'] in (selected or [])
                is_disabled = not is_selected and num_selected >= max_plots
                
                updated_opt = {
                    'label': opt['label'],
                    'value': opt['value'],
                    'disabled': is_disabled
                }
                updated_options.append(updated_opt)
            return updated_options
        
        updated_states = update_options(organized_options['states']['trajectory'], selected_states)
        updated_pointwise = update_options(organized_options['pointwise_outputs']['trajectory'], selected_pointwise)
        
        return info_text, updated_states, updated_pointwise
    
    # Callback for collecting all selected histograms and updating info
    @app.callback(
        [Output('histogram-selection-info', 'children'),
         Output('functional-histogram-selector', 'options'),
         Output('parameters-histogram-selector', 'options'),
         Output('initial-histogram-selector', 'options')],
        [Input('functional-histogram-selector', 'value'),
         Input('parameters-histogram-selector', 'value'),
         Input('initial-histogram-selector', 'value')]
    )
    def update_histogram_selection_info(selected_functional, selected_params, selected_initial):
        all_selected = (selected_functional or []) + (selected_params or []) + (selected_initial or [])
        num_selected = len(all_selected)
        info_text = f"Histograms selected: {num_selected}/{max_histograms}"
        
        # Update options to disable/enable based on selection limit
        def update_options(options, selected):
            updated_options = []
            for opt in options:
                is_selected = opt['value'] in (selected or [])
                is_disabled = not is_selected and num_selected >= max_histograms
                
                updated_opt = {
                    'label': opt['label'],
                    'value': opt['value'],
                    'disabled': is_disabled
                }
                updated_options.append(updated_opt)
            return updated_options
        
        updated_functional = update_options(organized_options['functional_outputs']['histogram'], selected_functional)
        updated_params = update_options(organized_options['parameters']['histogram'], selected_params)
        updated_initial = update_options(organized_options['initial_states']['histogram'], selected_initial)
        
        return info_text, updated_functional, updated_params, updated_initial
    
    # Callback for updating trajectory plot and title
    @app.callback(
        [Output('trajectories-plot', 'figure'),
         Output('trajectory-title', 'children')],
        [Input('sample-slider', 'value'),
         Input('opacity-slider', 'value'),
         Input('display-mode', 'value'),
         Input('states-trajectory-selector', 'value'),
         Input('pointwise-trajectory-selector', 'value')]
    )
    def update_trajectory_plot(num_samples, opacity, display_mode, selected_states, selected_pointwise):
        # Combine all selected trajectories
        selected_trajectories = (selected_states or []) + (selected_pointwise or [])
        
        if not selected_trajectories:
            # Return empty figure if nothing selected
            return go.Figure(), "No Trajectories Selected"
        
        if display_mode == 'trajectories':
            # Use the new plotting function directly
            fig = _plot_trajectories_from_simulation_results(mc_results, selected_trajectories, num_samples, opacity)
            title = f"Selected Trajectories ({len(selected_trajectories)} plots)"
        else:  # statistics
            # For statistics mode, create a custom plot with selected trajectories
            # Create a temporary SimulationResults object with selected data
            temp_states = {}
            temp_pointwise = {}
            
            for key in selected_trajectories:
                if key.startswith('state_'):
                    state_name = key.replace('state_', '')
                    if mc_results.states and state_name in mc_results.states:
                        temp_states[state_name] = mc_results.states[state_name]
                elif key.startswith('output_'):
                    output_name = key.replace('output_', '')
                    if mc_results.pointwise_outputs and output_name in mc_results.pointwise_outputs:
                        temp_pointwise[output_name] = mc_results.pointwise_outputs[output_name]
            
            if temp_states or temp_pointwise:
                # Create temporary SimulationResults object
                from .up import SimulationResults
                temp_results = SimulationResults(
                    times=mc_results.times,
                    states=temp_states if temp_states else None,
                    pointwise_outputs=temp_pointwise if temp_pointwise else None,
                    params=mc_results.params,
                    init_state=mc_results.init_state,
                    functional_outputs=mc_results.functional_outputs
                )
                
                fig = _plot_statistics_from_mc_results(temp_results, max_samples=None)
                title = f"Selected Trajectory Statistics ({len(selected_trajectories)} plots)"
            else:
                fig = go.Figure()
                title = "No Valid Trajectories Selected"
        
        return fig, title
    
    # Add histogram callback if histogram options exist
    if all_histogram_keys:
        # Callback for updating histogram plot
        @app.callback(
            Output('histograms-plot', 'figure'),
            [Input('functional-histogram-selector', 'value'),
             Input('parameters-histogram-selector', 'value'),
             Input('initial-histogram-selector', 'value')]
        )
        def update_histogram_plot(selected_functional, selected_params, selected_initial):
            # Combine all selected histograms
            selected_histograms = (selected_functional or []) + (selected_params or []) + (selected_initial or [])
            
            if not selected_histograms:
                return go.Figure()
            
            return _plot_histograms_from_mc_results(mc_results, selected_histograms=selected_histograms)
    
    return app


def _plot_statistics_from_mc_results(mc_results, max_samples=None, percentiles=[5, 25, 50, 75, 95]):
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
    
    return _plot_state_statistics(mc_results.times, all_trajectories, percentiles)


def _plot_state_statistics(times, state_dict, percentiles=[5, 25, 50, 75, 95]):
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


def run_comprehensive_dashboard(mc_results, max_samples=100, debug=True, port=8050):
    """
    Convenience function to create and run a Dash app.
    
    Args:
        mc_results: SimulationResults object
        max_samples: int - maximum number of samples to plot
        debug: bool - run in debug mode
        port: int - port to run the server on
    """
    if not hasattr(mc_results, 'times') or not hasattr(mc_results, 'states'):
        raise ValueError("mc_results must be a SimulationResults object")
    
    app = create_comprehensive_dashboard(mc_results, max_samples)
    app.run(debug=debug, port=port)
