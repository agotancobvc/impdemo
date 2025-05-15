import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

#CSV handling


def process_csv_file(uploaded_file):
    
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Validate required columns
        required_columns = [
            'Right Step Time (sec)', 'Right Step Length (meters)', 'Right Cadence (steps/min)',
            'Right Swing Time (sec)', 'Right Stance Time (sec)', 'GaitSpeed Rtable (mph*10)',
            'Right Stride Time (sec)', 'Right Stride Length (meters)', 'Left Step Time (sec)',
            'Left Step Length (meters)', 'Left Cadence (steps/min)', 'Left Swing Time (sec)',
            'Left Stance Time (sec)', 'GaitSpeed Ltable (mph*10)', 'Left Stride Time (sec)',
            'Left Stride Length (meters)'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return None, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Basic data cleaning
        # 1. Remove any rows with all NaN values
        df = df.dropna(how='all')
        
        # 2. Handle missing values in important columns (you can adjust this strategy as needed)
        for col in required_columns:
            if df[col].isna().any():
                # Fill with column median or use another appropriate strategy
                df[col] = df[col].fillna(df[col].median())
        
        # 3. Remove outliers (optional - use with caution)
        # For example, remove rows where values are more than 3 standard deviations from the mean
        for col in required_columns:
            mean = df[col].mean()
            std = df[col].std()
            df = df[(df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)]
        
        # 4. Add a Time column if it doesn't exist
        if 'Time' not in df.columns:
            df['Time'] = range(len(df))
            
        return df, None  # Return the dataframe and no error message
        
    except Exception as e:
        return None, f"Error processing CSV file: {str(e)}"
    


# Set page configuration
st.set_page_config(
    page_title="Summary",
    page_icon="👣",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add a summary section at the top of the page
st.markdown("""
<div class="main-header">Summary</div>
""", unsafe_allow_html=True)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #3366ff;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0055cc;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .info-text {
        font-size: 1rem;
        color: #555;
    }
    .highlight {
        background-color: #e6f2ff;
        padding: 0.2rem;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def calculate_statistics(df):
    """Calculate key statistics from gait data."""
    stats = {}
    
    # Right side statistics
    stats['avg_right_cadence'] = df['Right Cadence (steps/min)'].mean()
    stats['avg_right_speed'] = df['GaitSpeed Rtable (mph*10)'].mean() / 10  # Convert to mph
    stats['avg_right_step_length'] = df['Right Step Length (meters)'].mean()
    stats['avg_right_stride_length'] = df['Right Stride Length (meters)'].mean()
    
    # Left side statistics
    stats['avg_left_cadence'] = df['Left Cadence (steps/min)'].mean()
    stats['avg_left_speed'] = df['GaitSpeed Ltable (mph*10)'].mean() / 10  # Convert to mph
    stats['avg_left_step_length'] = df['Left Step Length (meters)'].mean()
    stats['avg_left_stride_length'] = df['Left Stride Length (meters)'].mean()
    
    # Average  statistics
    stats['avg_speed'] = (stats['avg_right_speed'] + stats['avg_left_speed']) / 2
    
    # Calculate swing to stance ratio
    stats['right_swing_stance_ratio'] = df['Right Swing Time (sec)'].mean() / df['Right Stance Time (sec)'].mean()
    stats['left_swing_stance_ratio'] = df['Left Swing Time (sec)'].mean() / df['Left Stance Time (sec)'].mean()
    
    return stats

# Main application
def main():
    
    # Sidebar for controls
    st.sidebar.title("Patient Dashboard")

    # Initialize dataframe
    df = None
    error_message = None

    # Sidebar for uploading CSV
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df, error_message = process_csv_file(uploaded_file)
        if error_message:
            st.error(error_message)

    # Help & Information section in the sidebar
    with st.sidebar.expander("Help & Information"):
        st.write("""Help & Information section here
            """)

    # Continue only if we have valid data
    if df is not None:
        # Session management
        if 'patient_id' not in st.session_state:
            st.session_state.patient_id = "PT-" + str(np.random.randint(1000, 9999))
        
        # Display patient ID in sidebar
        st.sidebar.markdown("---")
        patient_id = st.sidebar.text_input("Patient ID", value=st.session_state.patient_id)
        st.session_state.patient_id = patient_id
        
        # Time range selection for filtering data
        time_range = st.sidebar.slider(
            "Select Time Range",
            0, len(df)-1, (0, len(df)-1)
        )
        
        # Filter data based on selected range
        filtered_df = df.iloc[time_range[0]:time_range[1]+1].copy()
        
        # Calculate statistics from filtered data
        stats = calculate_statistics(filtered_df)
        
        # Main content area

        # Create three columns for centering
        acol1, acol2, acol3 = st.columns([1, 2, 1])  # Adjust column widths for centering

        # Create three columns for centering
        scol1, scol2, scol3 = st.columns([1, 2, 1])  # Adjust column widths for centering

        st.markdown("### Analysis", unsafe_allow_html=True)

        # Create three columns for the metrics
        acol1, acol2, acol3 = st.columns(3)

        # Add metrics to each column
        with acol1:
            st.write("Assessment 1")
            st.write("Based on cadence, speed, and step length, patient is x.")
        with acol2:
            st.write("Assessment 2")
            st.write("Based on cadence, speed, and step length, patient is y.")
        with acol3:
            st.write("Assessment 3")
            st.write("Based on cadence, speed, and step length, patient is z.")

        st.markdown("### Statistics", unsafe_allow_html=True)

        # Create three columns for the metrics
        scol1, scol2, scol3 = st.columns(3)

        # Add metrics to each column
        with scol1:
            st.metric("Average Gait Speed (mph)", f"{stats.get('avg_gait', (stats['avg_left_speed'] + stats['avg_right_speed']) / 2):.2f}")
        with scol2:
            st.metric("Right Cadence (steps/min)", f"{stats.get('right_cadence', stats['avg_right_cadence']):.2f}")
        with scol3:
            st.metric("Left Cadence (steps/min)", f"{stats.get('left_cadence', stats['avg_left_cadence']):.2f}")
            
        
        # Visualizations
        st.markdown('<h2 class="sub-header">Trends</h2>', unsafe_allow_html=True)
        
        # Tab-based visualization layout
        tab1, tab2, tab3, tab4 = st.tabs(["Speed & Cadence", "Step Parameters", "Stance vs. Swing", "Compare Left/Right"])
        
        with tab1:
            st.subheader("Speed and Cadence Over Time")        

            # SPEED
            filtered_df['Right Speed (mph)'] = filtered_df['GaitSpeed Rtable (mph*10)'] / 10
            filtered_df['Left Speed (mph)'] = filtered_df['GaitSpeed Ltable (mph*10)'] / 10
            
            # Create index for plotting (assumes time series)
            filtered_df['Sample Index'] = range(len(filtered_df))
            
            # Plot speed over time
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add speed traces
            fig.add_trace(
                go.Scatter(x=filtered_df['Sample Index'], y=filtered_df['Right Speed (mph)'], 
                          name="Right Speed", line=dict(color="blue")),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(x=filtered_df['Sample Index'], y=filtered_df['Left Speed (mph)'], 
                          name="Left Speed", line=dict(color="lightblue")),
                secondary_y=False
            )

            # Update axes labels
            fig.update_xaxes(title_text="Time (s)")
            fig.update_yaxes(title_text="Speed (mph)", secondary_y=False)

            fig.update_layout(
                title="Speed Trends",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )

            
            # CADENCE
            filtered_df['Right Cadence (steps/min)'] = filtered_df['Left Cadence (steps/min)']
            filtered_df['Left Speed (steps/min)'] = filtered_df['Right Cadence (steps/min)']
            
            # Create index for plotting (assumes time series)
            filtered_df['Index'] = range(len(filtered_df))
            
            # Plot cadence over time
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])

            # Add cadence traces
            fig2.add_trace(
                go.Scatter(x=filtered_df['Sample Index'], y=filtered_df['Right Cadence (steps/min)'], 
                          name="Right Cadence", line=dict(color="red", dash="dash")),
                secondary_y=True
            )
            
            fig2.add_trace(
                go.Scatter(x=filtered_df['Sample Index'], y=filtered_df['Left Cadence (steps/min)'], 
                          name="Left Cadence", line=dict(color="lightcoral", dash="dash")),
                secondary_y=True
            )
            
            # Update axes labels
            fig2.update_xaxes(title_text="Time (s)")
            fig2.update_yaxes(title_text="Cadence (steps/min)", secondary_y=False)
            
            fig2.update_layout(
                title="Cadence Trends",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            st.subheader("Step and Stride Parameters")
            
            # Create multi-line chart for step and stride parameters
            fig = go.Figure()
            
            # Step length
            fig.add_trace(go.Scatter(
                x=filtered_df['Sample Index'], 
                y=filtered_df['Right Step Length (meters)'],
                mode='lines',
                name='Right Step Length',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=filtered_df['Sample Index'], 
                y=filtered_df['Left Step Length (meters)'],
                mode='lines',
                name='Left Step Length',
                line=dict(color='red')
            ))
            
            # Stride length
            fig.add_trace(go.Scatter(
                x=filtered_df['Sample Index'], 
                y=filtered_df['Right Stride Length (meters)'],
                mode='lines',
                name='Right Stride Length',
                line=dict(color='blue', dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=filtered_df['Sample Index'], 
                y=filtered_df['Left Stride Length (meters)'],
                mode='lines',
                name='Left Stride Length',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title='Step and Stride Lengths Over Time',
                xaxis_title='Sample Index',
                yaxis_title='Length (meters)',
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add step time visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Step time histogram
                fig = px.histogram(
                    filtered_df, 
                    x=['Right Step Time (sec)', 'Left Step Time (sec)'],
                    nbins=20,
                    barmode='overlay',
                    opacity=0.7,
                    color_discrete_map={'Right Step Time (sec)': 'blue', 'Left Step Time (sec)': 'red'}
                )
                
                fig.update_layout(
                    title="Step Time Distribution",
                    xaxis_title="Step Time (seconds)",
                    yaxis_title="Frequency",
                    legend_title="Side"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Step length histogram
                fig = px.histogram(
                    filtered_df, 
                    x=['Right Step Length (meters)', 'Left Step Length (meters)'],
                    nbins=20,
                    barmode='overlay',
                    opacity=0.7,
                    color_discrete_map={'Right Step Length (meters)': 'blue', 'Left Step Length (meters)': 'red'}
                )
                
                fig.update_layout(
                    title="Step Length Distribution",
                    xaxis_title="Step Length (meters)",
                    yaxis_title="Frequency",
                    legend_title="Side"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Left vs. Right Comparison")
            
            # Parameter selection
            parameter_options = [
                "Step Time (sec)", "Step Length (meters)", "Cadence (steps/min)",
                "Swing Time (sec)", "Stance Time (sec)", "Stride Time (sec)", "Stride Length (meters)"
            ]
            
            selected_parameter = st.selectbox("Select parameter to compare:", parameter_options)
            
            # Map selected parameter to dataframe columns
            left_param = f"Left {selected_parameter}"
            right_param = f"Right {selected_parameter}"
            
            # Create scatter plot with diagonal reference line
            min_val = min(filtered_df[left_param].min(), filtered_df[right_param].min())
            max_val = max(filtered_df[left_param].max(), filtered_df[right_param].max())
            
            fig = px.scatter(
                filtered_df, 
                x=left_param, 
                y=right_param,
                trendline="ols",
                marginal_x="histogram",
                marginal_y="histogram"
            )
            
            # Add diagonal reference line (perfect symmetry)
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Symmetry',
                    line=dict(color='gray', dash='dash')
                )
            )
            
            fig.update_layout(
                title=f"Left vs. Right {selected_parameter}",
                xaxis_title=f"Left {selected_parameter}",
                yaxis_title=f"Right {selected_parameter}",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display statistical comparison
            col1, col2 = st.columns(2)
            
            with col1:
                # Statistics table
                stats_df = pd.DataFrame({
                    'Parameter': [selected_parameter],
                    'Left (Mean)': [filtered_df[left_param].mean()],
                    'Right (Mean)': [filtered_df[right_param].mean()],
                    'Difference': [filtered_df[right_param].mean() - filtered_df[left_param].mean()],
                    'Difference (%)': [(filtered_df[right_param].mean() / filtered_df[left_param].mean() - 1) * 100],
                    'Correlation': [filtered_df[left_param].corr(filtered_df[right_param])]
                })
                
                # Format the statistics table
                stats_df['Left (Mean)'] = stats_df['Left (Mean)'].map('{:.3f}'.format)
                stats_df['Right (Mean)'] = stats_df['Right (Mean)'].map('{:.3f}'.format)
                stats_df['Difference'] = stats_df['Difference'].map('{:.3f}'.format)
                stats_df['Difference (%)'] = stats_df['Difference (%)'].map('{:.1f}%'.format)
                stats_df['Correlation'] = stats_df['Correlation'].map('{:.2f}'.format)
                
                st.write("Statistical Comparison")
                st.dataframe(stats_df, hide_index=True)
        
        # Export data options
        st.markdown('<h2 class="sub-header">Export Options</h2>', unsafe_allow_html=True)
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            if st.button("Export Statistics Summary"):
                # Create a summary dataframe
                summary_stats = pd.DataFrame({
                    'Parameter': [
                        'Average Gait Speed (mph)',
                        'Right Cadence (steps/min)',
                        'Left Cadence (steps/min)',
                        'Right Step Length (m)',
                        'Left Step Length (m)',
                        'Right Stride Length (m)',
                        'Left Stride Length (m)',
                        'Right Swing/Stance Ratio',
                        'Left Swing/Stance Ratio',
                    ],
                    'Value': [
                        f"{stats['avg_speed']:.2f}",
                        f"{stats['avg_right_cadence']:.1f}",
                        f"{stats['avg_left_cadence']:.1f}",
                        f"{stats['avg_right_step_length']:.3f}",
                        f"{stats['avg_left_step_length']:.3f}",
                        f"{stats['avg_right_stride_length']:.3f}",
                        f"{stats['avg_left_stride_length']:.3f}",
                        f"{stats['right_swing_stance_ratio']:.2f}",
                        f"{stats['left_swing_stance_ratio']:.2f}",
                    ]
                })
                
                # Convert to CSV
                csv_stats = summary_stats.to_csv(index=False)
                
                # Create download link
                st.download_button(
                    label="Download Statistics CSV",
                    data=csv_stats,
                    file_name=f"patient_{st.session_state.patient_id}_gait_stats.csv",
                    mime="text/csv",
                )
                
                # Also display the stats
                st.dataframe(summary_stats, hide_index=True)
        
        with export_col2:
            if st.button("Export Full Report"):
                # Create buffer
                report_buffer = StringIO()
                
                # Write report header
                report_buffer.write(f"# Patient Gait Analysis Report\n\n")
                report_buffer.write(f"**Patient ID:** {st.session_state.patient_id}\n")
                report_buffer.write(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
                
                # Write key statistics
                report_buffer.write("## Key Statistics\n\n")
                report_buffer.write(f"- Average Gait Speed: {stats['avg_speed']:.2f} mph\n")
                report_buffer.write(f"- Right Cadence: {stats['avg_right_cadence']:.1f} steps/min\n")
                report_buffer.write(f"- Left Cadence: {stats['avg_left_cadence']:.1f} steps/min\n")
                report_buffer.write(f"- Right Step Length: {stats['avg_right_step_length']:.3f} m\n")
                report_buffer.write(f"- Left Step Length: {stats['avg_left_step_length']:.3f} m\n")
              
                # Write recommendations
                report_buffer.write("\n## Recommendations\n\n")
                if stats['avg_speed'] < 2.5:
                    report_buffer.write("- Consider physical therapy to improve overall gait speed\n")
                
                report_text = report_buffer.getvalue()
                
                # Create download button
                st.download_button(
                    label="Download Full Report",
                    data=report_text,
                    file_name=f"patient_{st.session_state.patient_id}_gait_report.md",
                    mime="text/markdown",
                )
                
                # Display preview
                with st.expander("Preview Report", expanded=True):
                    st.markdown(report_text)
        
        # Raw data section
        st.markdown('<h2 class="sub-header">Raw Data</h2>', unsafe_allow_html=True)
        
        if st.checkbox("Show Raw Data Table"):
            st.dataframe(filtered_df)
            
            # Export raw data option
            if st.button("Export Raw Data"):
                csv_data = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download Raw Data CSV",
                    data=csv_data,
                    file_name=f"patient_{st.session_state.patient_id}_raw_gait_data.csv",
                    mime="text/csv",
                )
        
        # Progress tracking section
        st.markdown('<h2 class="sub-header">Progress Tracking</h2>', unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()