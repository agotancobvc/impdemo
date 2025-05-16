import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import fpdf
from io import StringIO
from io import BytesIO
from fpdf import FPDF

def process_csv_file(uploaded_file):
    
    try:
        df = pd.read_csv(uploaded_file)
        
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
        
        df = df.dropna(how='all')
        
        for col in required_columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

        for col in required_columns:
            mean = df[col].mean()
            std = df[col].std()
            df = df[(df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)]

        if 'Time' not in df.columns:
            df['Time'] = range(len(df))
            
        return df, None
        
    except Exception as e:
        return None, f"Error processing CSV file: {str(e)}"
    
st.set_page_config(
    page_title="Summary",
    page_icon="👣",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<div class="main-header"; color:#d32f2f; >Summary</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5em;
        color: #d32f2f;
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

st.markdown("""
    <style>
        .assessment-box {
            background-color: transparent; 
            color: #d32f2f; /* Bright neon red text */
            border: 1px solid #d32f2f; 
            border-radius: 5px;
            padding: 14px;
            margin-bottom: 10px;
        }
        
        .assessment-box p {
            color: #FFFFFF;
        }
            
        /* Style for statistics group outline */
        .statistics-group {
            border: 2px solid #6c757d; /* Grey border */
            border-radius: 5px;
            padding: 15px;
            margin-top: 20px;
        }
            
        /* Style for separating lines between metrics */
        .metric-separator {
            border-top: 1px solid #000; /* Black line */
            margin: 10px 0;
        }
""", unsafe_allow_html=True)


st.markdown('<div class="metric-separator"></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True) 

def calculate_statistics(df):
    stats = {}
    
    stats['avg_right_cadence'] = df['Right Cadence (steps/min)'].mean()
    stats['avg_right_speed'] = df['GaitSpeed Rtable (mph*10)'].mean() / 10  
    stats['avg_right_step_length'] = df['Right Step Length (meters)'].mean()
    stats['avg_right_stride_length'] = df['Right Stride Length (meters)'].mean()
    
    stats['avg_left_cadence'] = df['Left Cadence (steps/min)'].mean()
    stats['avg_left_speed'] = df['GaitSpeed Ltable (mph*10)'].mean() / 10 
    stats['avg_left_step_length'] = df['Left Step Length (meters)'].mean()
    stats['avg_left_stride_length'] = df['Left Stride Length (meters)'].mean()
    
    stats['avg_speed'] = (stats['avg_right_speed'] + stats['avg_left_speed']) / 2
    
    stats['right_swing_stance_ratio'] = df['Right Swing Time (sec)'].mean() / df['Right Stance Time (sec)'].mean()
    stats['left_swing_stance_ratio'] = df['Left Swing Time (sec)'].mean() / df['Left Stance Time (sec)'].mean()
    
    return stats

def main():
    
    st.sidebar.title("Patient Dashboard")

    df = None
    error_message = None

    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df, error_message = process_csv_file(uploaded_file)
        if error_message:
            st.error(error_message)

    with st.sidebar.expander("Help & Information"):
        st.write("""Help & Information section here
            """)

    if df is not None:
        if 'patient_id' not in st.session_state:
            st.session_state.patient_id = "PT-" + str(np.random.randint(1000, 9999))
        
        st.sidebar.markdown("---")
        patient_id = st.sidebar.text_input("Patient ID", value=st.session_state.patient_id)
        st.session_state.patient_id = patient_id
        
        time_range = st.sidebar.slider(
            "Select Time Range",
            0, len(df)-1, (0, len(df)-1)
        )
        
        filtered_df = df.iloc[time_range[0]:time_range[1]+1].copy()
        
        stats = calculate_statistics(filtered_df)
        
        acol1, acol2, acol3 = st.columns([1, 2, 1]) 

        scol1, scol2, scol3 = st.columns([1, 2, 1]) 

        st.markdown('<span style="color:#d32f2f; font-size:32px; ">Analysis</span>', unsafe_allow_html=True)

        acol1, acol2, acol3 = st.columns(3)

        with acol1:
             st.markdown("""
                            <div class="assessment-box">
                            <h3>Assessment 1</h3>
                            <p>Based on X, patient is Y.</p>
                            </div>
                         """, unsafe_allow_html=True)
        with acol2:
            st.markdown("""
                            <div class="assessment-box">
                            <h3>Assessment 2</h3>
                            <p>Based on A, patient is B.</p>
                            </div>
                         """, unsafe_allow_html=True)
        with acol3:
            st.markdown("""
                            <div class="assessment-box">
                            <h3>Assessment 3</h3>
                            <p>Based on D, patient is E.</p>
                            </div>
                         """, unsafe_allow_html=True)

        st.markdown('<span style="color:#d32f2f; font-size:32px; ">Statistics</span>', unsafe_allow_html=True)

  
        scol1, scol2, scol3 = st.columns(3)

  
        with scol1:
            st.metric("Average Gait Speed (mph)", f"{stats.get('avg_gait', (stats['avg_left_speed'] + stats['avg_right_speed']) / 2):.2f}")
        with scol2:
            st.metric("Right Cadence (steps/min)", f"{stats.get('right_cadence', stats['avg_right_cadence']):.2f}")
        with scol3:
            st.metric("Left Cadence (steps/min)", f"{stats.get('left_cadence', stats['avg_left_cadence']):.2f}")
        
  
        st.markdown('<span style="color:#d32f2f; font-size:32px; ">Trends</span>', unsafe_allow_html=True)
        
  
        tab1, tab2, tab3 = st.tabs(["Speed & Cadence", "Step Parameters", "Stance vs. Swing"])
        
        with tab1:
            st.subheader("Speed and Cadence Over Time")        

  
            filtered_df['Right Speed (mph)'] = filtered_df['GaitSpeed Rtable (mph*10)'] / 10
            filtered_df['Left Speed (mph)'] = filtered_df['GaitSpeed Ltable (mph*10)'] / 10
            
  
            filtered_df['Sample Index'] = range(len(filtered_df))
            
  
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
  
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

  
            fig.update_xaxes(title_text="Time (s)")
            fig.update_yaxes(title_text="Speed (mph)", secondary_y=False)

            fig.update_layout(
                title="Speed Trends",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )

            
  
            filtered_df['Right Cadence (steps/min)'] = filtered_df['Left Cadence (steps/min)']
            filtered_df['Left Speed (steps/min)'] = filtered_df['Right Cadence (steps/min)']
            
  
            filtered_df['Index'] = range(len(filtered_df))
            
  
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])

  
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
            
  
            fig = go.Figure()
            
  
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
            
  
            col1, col2 = st.columns(2)
            
            with col1:
  
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
            
  
            parameter_options = [
                "Step Time (sec)", "Step Length (meters)", "Cadence (steps/min)",
                "Swing Time (sec)", "Stance Time (sec)", "Stride Time (sec)", "Stride Length (meters)"
            ]
            
            selected_parameter = st.selectbox("Select parameter to compare:", parameter_options)
            
  
            left_param = f"Left {selected_parameter}"
            right_param = f"Right {selected_parameter}"
            
  
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
            
  
            col1, col2 = st.columns(2)
            
            with col1:
  
                stats_df = pd.DataFrame({
                    'Parameter': [selected_parameter],
                    'Left (Mean)': [filtered_df[left_param].mean()],
                    'Right (Mean)': [filtered_df[right_param].mean()],
                    'Difference': [filtered_df[right_param].mean() - filtered_df[left_param].mean()],
                    'Difference (%)': [(filtered_df[right_param].mean() / filtered_df[left_param].mean() - 1) * 100],
                    'Correlation': [filtered_df[left_param].corr(filtered_df[right_param])]
                })
                
  
                stats_df['Left (Mean)'] = stats_df['Left (Mean)'].map('{:.3f}'.format)
                stats_df['Right (Mean)'] = stats_df['Right (Mean)'].map('{:.3f}'.format)
                stats_df['Difference'] = stats_df['Difference'].map('{:.3f}'.format)
                stats_df['Difference (%)'] = stats_df['Difference (%)'].map('{:.1f}%'.format)
                stats_df['Correlation'] = stats_df['Correlation'].map('{:.2f}'.format)
                
                st.write("Statistical Comparison")
                st.dataframe(stats_df, hide_index=True)
        
  
        st.markdown('<span style="color:#d32f2f; font-size:32px; ">Export Options</span>', unsafe_allow_html=True)
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            if st.button("Export Statistics Summary"):
  
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
                
  
                csv_stats = summary_stats.to_csv(index=False)
                
  
                st.download_button(
                    label="Download Statistics CSV",
                    data=csv_stats,
                    file_name=f"patient_{st.session_state.patient_id}_gait_stats.csv",
                    mime="text/csv",
                )
                
  
                st.dataframe(summary_stats, hide_index=True)
        
        with export_col2:
            def export_full_report_to_pdf(stats, patient_id):
  
                pdf = FPDF()
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.add_page()
                pdf.set_font("Arial", size=12)

  
                pdf.set_font("Arial", style="B", size=16)
                pdf.cell(0, 10, "Patient Gait Analysis Report", ln=True, align="C")
                pdf.ln(10)

                pdf.set_font("Arial", size=12)
                pdf.cell(0, 10, f"Patient ID: {patient_id}", ln=True)
                pdf.cell(0, 10, f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}", ln=True)
                pdf.ln(10)

  
                pdf.set_font("Arial", style="B", size=14)
                pdf.cell(0, 10, "Key Statistics", ln=True)
                pdf.ln(5)

                pdf.set_font("Arial", size=12)
                pdf.cell(0, 10, f"- Average Gait Speed: {stats['avg_speed']:.2f} mph", ln=True)
                pdf.cell(0, 10, f"- Right Cadence: {stats['avg_right_cadence']:.1f} steps/min", ln=True)
                pdf.cell(0, 10, f"- Left Cadence: {stats['avg_left_cadence']:.1f} steps/min", ln=True)
                pdf.cell(0, 10, f"- Right Step Length: {stats['avg_right_step_length']:.3f} m", ln=True)
                pdf.cell(0, 10, f"- Left Step Length: {stats['avg_left_step_length']:.3f} m", ln=True)
                pdf.ln(10)

  
                pdf.set_font("Arial", style="B", size=14)
                pdf.cell(0, 10, "Recommendations", ln=True)
                pdf.ln(5)

  
                pdf_buffer = BytesIO()  
                pdf_content = pdf.output(dest='S').encode('latin1')  
                pdf_buffer.write(pdf_content)  
                pdf_buffer.seek(0)  

                return pdf_buffer
            
        if st.button("Export Full Report"):
            pdf_buffer = export_full_report_to_pdf(stats, st.session_state.patient_id)
            st.download_button(
                label="Download Full Report as PDF",
                data=pdf_buffer,
                file_name=f"patient_{st.session_state.patient_id}_gait_report.pdf",
                mime="application/pdf",
            )
        
  
        st.markdown('<span style="color:#d32f2f; font-size:32px; ">Raw Data</span>', unsafe_allow_html=True)
        
        if st.checkbox("Show Raw Data Table"):
            st.dataframe(filtered_df)
            
  
            if st.button("Export Raw Data"):
                csv_data = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download Raw Data CSV",
                    data=csv_data,
                    file_name=f"patient_{st.session_state.patient_id}_raw_gait_data.csv",
                    mime="text/csv",
                )
        
  
        st.markdown('<span style="color:#d32f2f; font-size:32px; ">Progress Tracking</span>', unsafe_allow_html=True)

  
if __name__ == "__main__":
    main()
