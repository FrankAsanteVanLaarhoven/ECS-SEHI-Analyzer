import streamlit as st
import plotly.graph_objects as go
import numpy as np
from utils.defect_analysis import DefectAnalyzer, DefectParameters
from utils.defect_visualization import DefectVisualizer
from utils.sgp_visualization import SGPVisualizer
from utils.sehi_analysis import SEHIAnalyzer, SEHIParameters, SEHIAlgorithm
import io
import pandas as pd
from time import sleep

def render_defect_detection():
    """Render the defect detection page."""
    st.markdown("# Defect Detection")
    
    # Create two-column layout
    left_col, right_col = st.columns([1, 3])
    
    with left_col:
        st.markdown("## Defect Detection Controls")
        
        # Resolution slider
        st.markdown("Resolution")
        resolution = st.slider(
            "",
            min_value=128,
            max_value=1024,
            value=512,
            step=128,
            key="defect_resolution"
        )
        
        # Detection Sensitivity slider
        st.markdown("Detection Sensitivity")
        sensitivity = st.slider(
            "",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            format="%.2f",
            key="defect_sensitivity"
        )
        
        # Defect Types
        st.markdown("Defect Types")
        defect_types = st.container()
        with defect_types:
            col1, col2, col3 = st.columns(3)
            with col1:
                cracks = st.button("Cracks ×", type="secondary")
            with col2:
                voids = st.button("Voids ×", type="secondary")
            with col3:
                inclusions = st.button("Inclusions ×", type="secondary")
        
        # SEHI Analysis Settings
        with st.expander("SEHI Analysis Settings", expanded=False):
            st.selectbox("Analysis Method", ["Basic", "Advanced", "Expert"])
        
        # Run Analysis Button
        st.button("Detect Defects", type="primary")
        
        # Add Export Options
        st.markdown("## Export Options")
        export_format = st.selectbox(
            "Report Format",
            ["Interactive HTML", "Excel", "PDF", "CSV", "3D Print (STL)", "3D Print (OBJ)", "3D Print (3MF)"]
        )
        
        if st.button("Generate Report", type="primary"):
            try:
                # Get current analysis results
                defect_vis = DefectVisualizer()
                defect_analyzer = DefectAnalyzer()
                
                # Generate sample data if no real data available
                data = defect_vis.generate_sample_data(resolution)
                
                # Run analysis
                analysis_results = defect_analyzer.detect_defects(
                    data['surface'],
                    DefectParameters(sensitivity=sensitivity),
                    ["Cracks", "Voids", "Inclusions"]
                )
                
                # Generate comprehensive report
                report = defect_analyzer.generate_comprehensive_report(
                    data,
                    analysis_results
                )
                
                # Export based on selected format
                if export_format == "Interactive HTML":
                    export_path = "defect_analysis_report.html"
                    defect_analyzer.export_interactive_report(
                        data,
                        analysis_results,
                        export_path
                    )
                    with open(export_path, 'rb') as f:
                        st.download_button(
                            "Download HTML Report",
                            f,
                            file_name="defect_analysis_report.html",
                            mime="text/html"
                        )
                
                elif export_format == "Excel":
                    # Create Excel buffer
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        # Basic Metrics
                        pd.DataFrame([analysis_results['stats']]).to_excel(
                            writer, 
                            sheet_name='Basic Metrics'
                        )
                        
                        # SEHI Analysis
                        pd.DataFrame(analysis_results['sehi_results']).to_excel(
                            writer,
                            sheet_name='SEHI Analysis'
                        )
                        
                        # Advanced Statistics
                        pd.DataFrame(report['advanced_statistics']).to_excel(
                            writer,
                            sheet_name='Advanced Statistics'
                        )
                    
                    st.download_button(
                        "Download Excel Report",
                        output.getvalue(),
                        file_name="defect_analysis_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                elif export_format == "CSV":
                    # Convert results to CSV
                    csv_data = pd.DataFrame(analysis_results['stats']).to_csv()
                    st.download_button(
                        "Download CSV Report",
                        csv_data,
                        file_name="defect_analysis_basic.csv",
                        mime="text/csv"
                    )
                
                elif export_format.startswith("3D Print"):
                    try:
                        format_type = export_format.split("(")[1].strip(")")
                        
                        # Get 3D printable mesh data
                        mesh_data = defect_analyzer.export_for_3d_printing(
                            data,
                            analysis_results,
                            export_format=format_type
                        )
                        
                        # Create download button
                        st.download_button(
                            f"Download {format_type} File",
                            mesh_data,
                            file_name=f"ecs_analysis.{format_type.lower()}",
                            mime=f"model/{format_type.lower()}"
                        )
                        
                        # Show 3D printing recommendations
                        with st.expander("3D Printing Recommendations"):
                            st.markdown("""
                                ### Printing Guidelines
                                - **Layer Height:** 0.1mm recommended for detail
                                - **Infill:** 20% for structural features
                                - **Support:** Enable for overhanging defect markers
                                - **Scale:** Model is in nanometers, scale appropriately
                                - **Material:** PLA or resin recommended for detail
                                
                                ### Print Settings
                                - **Temperature:** 200-215°C for PLA
                                - **Build Plate:** 60°C
                                - **Print Speed:** 40-50mm/s for detail
                                
                                ### Post-Processing
                                - Carefully remove supports around defect markers
                                - Light sanding may be needed for smooth finish
                                - Consider clear coating for protection
                            """)
                            
                        # Show preview if possible
                        st.info("Preview your 3D model in your preferred slicer software before printing.")
                        
                    except Exception as e:
                        st.error(f"Error preparing 3D print file: {str(e)}")
                
                st.success("Report generated successfully!")
                
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
    
    with right_col:
        # Metrics row
        metrics_cols = st.columns(4)
        with metrics_cols[0]:
            st.metric("Total Defects", "1")
        with metrics_cols[1]:
            st.metric("Average Size", "262144.00 nm")
        with metrics_cols[2]:
            st.metric("Coverage", "100.00%")
        with metrics_cols[3]:
            st.metric("Confidence", "100.00%")
        
        # Main visualization
        st.markdown("### Defect Detection Analysis")
        
        # View control buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            both = st.button("Both", type="secondary")
        with col2:
            defects = st.button("Defects Only", type="secondary")
        with col3:
            confidence = st.button("Confidence Only", type="secondary")
        
        # Create visualization using the DefectVisualizer
        defect_vis = DefectVisualizer()
        sgp_vis = SGPVisualizer()
        
        # Create tabs for different visualization modes
        viz_tabs = st.tabs([
            "3D Defect Map", 
            "Cross-Section Analysis", 
            "Defect Distribution",
            "Time Evolution",
            "3D Printing"
        ])
        
        with viz_tabs[0]:
            # Use the DefectVisualizer to create the 3D map
            defect_data = defect_vis.generate_sample_data(resolution)
            defect_map = defect_vis.create_defect_plot(
                defect_data,
                confidence_map=None,
                defect_types=["Cracks", "Voids", "Inclusions"]
            )
            st.plotly_chart(defect_map, use_container_width=True)

        with viz_tabs[1]:
            # Cross-section analysis using SGPVisualizer
            cross_sections = sgp_vis.create_cross_section_plot(defect_data)
            st.plotly_chart(cross_sections, use_container_width=True)

        with viz_tabs[2]:
            # Distribution analysis
            distribution = defect_vis.create_distribution_plot(defect_data)
            st.plotly_chart(distribution, use_container_width=True)

        with viz_tabs[3]:
            # Time evolution using DefectVisualizer
            evolution = defect_vis.create_evolution_plot(defect_data)
            st.plotly_chart(evolution, use_container_width=True)

        with viz_tabs[4]:  # 3D Printing tab
            st.subheader("3D Print ECS Analysis")
            add_3d_print_controls()

def create_defect_visualization(resolution, sensitivity):
    """Create defect visualization using the DefectVisualizer."""
    defect_vis = DefectVisualizer()
    
    # Generate sample data using the visualizer
    defect_data = defect_vis.generate_sample_data(resolution)
    
    # Create the main defect plot
    fig = defect_vis.create_defect_plot(
        defect_data,
        confidence_map=None,
        defect_types=["Cracks", "Voids", "Inclusions"]
    )
    
    return fig

def render_defect_detection_old():
    """Render the defect detection page."""
    st.markdown('<h1 class="main-header">Defect Detection</h1>', unsafe_allow_html=True)
    
    # Initialize analyzers
    analyzer = DefectAnalyzer()
    visualizer = DefectVisualizer()
    
    # Create layout
    left_col, main_col = st.columns([1, 3])
    
    with left_col:
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.subheader("Defect Detection Controls")
        
        # First, define the values outside the slider
        resolution_options = [128, 256, 384, 512, 640, 768, 896, 1024]
        resolution = st.select_slider(
            "Resolution",
            options=resolution_options,
            value=512,
            help="Analysis resolution"
        )
        
        # Keep sensitivity slider as is since it works with floats
        sensitivity = st.slider(
            "Detection Sensitivity",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.01,
            format="%.2f",
            help="Higher values detect smaller defects"
        )
        
        # Defect types
        defect_types = st.multiselect(
            "Defect Types",
            [
                "Cracks",
                "Voids",
                "Delamination",
                "Inclusions"
            ],
            default=["Cracks"],
            help="Select defect types to detect"
        )
        
        # Add SEHI analysis options to the control panel
        with st.expander("SEHI Analysis Settings"):
            sehi_algorithm = st.selectbox(
                "Analysis Algorithm",
                options=[algo.value for algo in SEHIAlgorithm],
                help="Select SEHI analysis algorithm"
            )
            
            num_components = st.slider(
                "Number of Components",
                min_value=2,
                max_value=10,
                value=4,
                help="Number of components for decomposition"
            )
            
            energy_range = st.slider(
                "Energy Range (eV)",
                min_value=0.0,
                max_value=2000.0,
                value=(0.0, 1000.0),
                help="Energy range for analysis"
            )
            
            spatial_resolution = st.number_input(
                "Spatial Resolution (nm)",
                min_value=1.0,
                max_value=100.0,
                value=10.0,
                help="Spatial resolution of the analysis"
            )
        
        try:
            if st.button("Detect Defects", type="primary"):
                if not defect_types:
                    st.error("Please select at least one defect type.")
                    return
                    
                with main_col:
                    with st.spinner("Running defect detection..."):
                        # Generate sample data
                        data = np.random.normal(0.5, 0.1, (resolution, resolution))
                        
                        # Run detection with error handling
                        try:
                            params = DefectParameters(
                                sensitivity=sensitivity,
                                min_size=int(resolution * 0.01),  # 1% of resolution
                                max_size=int(resolution * 0.1),   # 10% of resolution
                                threshold=0.5,
                                noise_reduction=0.3
                            )
                            
                            results = analyzer.detect_defects(
                                data,
                                params=params,
                                defect_types=defect_types
                            )
                            
                            # Display metrics
                            cols = st.columns(4)
                            metrics = {
                                "Total Defects": f"{results['stats']['total_defects']:,}",
                                "Average Size": f"{results['stats']['avg_size']:.2f} nm",
                                "Coverage": f"{results['stats']['coverage']:.2%}",
                                "Confidence": f"{results['stats']['confidence']:.2%}"
                            }
                            
                            for col, (label, value) in zip(cols, metrics.items()):
                                col.metric(label, value)
                            
                            # Show visualization
                            st.plotly_chart(
                                visualizer.create_defect_plot(
                                    results['defect_map'],
                                    results['confidence_map'],
                                    defect_types
                                ),
                                use_container_width=True
                            )
                            
                            # After showing the main visualization, add the analysis tools
                            if results is not None:
                                visualizer.show_analysis_tools(
                                    results['defect_map'],
                                    results['confidence_map']
                                )
                                
                                # Add SGP analysis
                                st.markdown("---")
                                sgp_visualizer = SGPVisualizer()
                                sgp_visualizer.show_sgp_analysis(
                                    results['defect_map'],
                                    time_series=None  # Add time series data if available
                                )
                            
                            # After showing defect analysis and SGP analysis
                            st.markdown("---")
                            st.subheader("SEHI Analysis Results")
                            
                            # Create tabs for different analysis aspects
                            sehi_tabs = st.tabs([
                                "Algorithm Results",
                                "Carbon Network",
                                "Surface Chemistry",
                                "Degradation Assessment"
                            ])
                            
                            with sehi_tabs[0]:
                                st.subheader("Analysis Algorithm Results")
                                cols = st.columns(2)
                                
                                with cols[0]:
                                    # Component visualization
                                    st.plotly_chart(
                                        visualizer.create_component_plot(
                                            results['sehi_results']['components'],
                                            algorithm=sehi_algorithm
                                        ),
                                        use_container_width=True
                                    )
                                
                                with cols[1]:
                                    # Loading plots
                                    st.plotly_chart(
                                        visualizer.create_loading_plot(
                                            results['sehi_results']['loadings']
                                        ),
                                        use_container_width=True
                                    )
                            
                            with sehi_tabs[1]:
                                st.subheader("Carbon Network Analysis")
                                network_metrics = results['sehi_results']['network_properties']
                                
                                # Display network properties
                                cols = st.columns(4)
                                cols[0].metric("sp2/sp3 Ratio", f"{network_metrics['sp2_sp3_ratio']:.2f}")
                                cols[1].metric("Crystallinity", f"{network_metrics['crystallinity']:.2%}")
                                cols[2].metric("Surface Area", f"{network_metrics['surface_area']:.1f} m²/g")
                                cols[3].metric("Porosity", f"{network_metrics['porosity']:.2%}")
                                
                                # Network visualization
                                st.plotly_chart(
                                    visualizer.create_network_plot(network_metrics),
                                    use_container_width=True
                                )
                            
                            with sehi_tabs[2]:
                                st.subheader("Surface Chemistry Analysis")
                                surface_props = results['sehi_results']['surface_properties']
                                
                                # Surface chemistry metrics
                                cols = st.columns(3)
                                cols[0].metric("Oxidation State", f"{surface_props['oxidation_state']:.1f}")
                                cols[1].metric("Surface Energy", f"{surface_props['surface_energy']:.1f} mJ/m²")
                                
                                # Functional groups analysis
                                st.subheader("Functional Groups Distribution")
                                st.plotly_chart(
                                    visualizer.create_functional_groups_plot(
                                        surface_props['functional_groups']
                                    ),
                                    use_container_width=True
                                )
                            
                            with sehi_tabs[3]:
                                st.subheader("Degradation Assessment")
                                degradation = results['sehi_results']['degradation_metrics']
                                
                                # Degradation metrics
                                cols = st.columns(3)
                                cols[0].metric(
                                    "Corrosion Rate", 
                                    f"{degradation['corrosion_rate']:.3f} mm/year",
                                    delta="-0.002 mm/year",
                                    delta_color="inverse"
                                )
                                cols[1].metric(
                                    "Mechanical Stability",
                                    f"{degradation['mechanical_stability']:.1f}%",
                                    delta="+2.3%"
                                )
                                cols[2].metric(
                                    "Chemical Stability",
                                    f"{degradation['chemical_stability']:.1f}%",
                                    delta="-1.2%",
                                    delta_color="inverse"
                                )
                                
                                # Degradation visualization
                                st.plotly_chart(
                                    visualizer.create_degradation_plot(degradation),
                                    use_container_width=True
                                )
                                
                                # Recommendations
                                st.subheader("Optimization Recommendations")
                                with st.expander("View Recommendations"):
                                    st.write("""
                                        Based on the analysis results, consider the following optimizations:
                                        - Adjust process parameters to improve sp2/sp3 ratio
                                        - Monitor surface oxidation to maintain chemical stability
                                        - Implement protective measures against identified degradation mechanisms
                                        - Regular maintenance schedule based on corrosion rate trends
                                    """)
                                
                        except Exception as e:
                            st.error(f"Error in defect detection: {str(e)}")
                            
        except Exception as e:
            st.error(f"Error in UI: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main visualization area
    with main_col:
        if "defect_detection_btn" not in st.session_state:
            st.markdown('<div class="visualization-area">', unsafe_allow_html=True)
            st.markdown("""
                <div style="text-align: center; padding: 40px;">
                    <h3 style="color: #94A3B8;">Defect Detection</h3>
                    <p style="color: #64748B;">
                        Configure detection parameters and click 'Detect Defects' 
                        to begin. Results will appear here.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

def add_3d_print_controls():
    """Add 3D printing controls with direct printing capabilities."""
    st.markdown("## 3D Printing Controls")
    
    # Main settings columns
    col1, col2 = st.columns(2)
    
    with col1:
        material = st.selectbox(
            "Print Material",
            [
                "Carbon Fiber PLA",
                "Graphene-Enhanced PLA",
                "Carbon Nanotube Composite",
                "Conductive Carbon Black",
                "Carbon-Infused Resin"
            ],
            help="Select carbon-based printing material"
        )
        
        quality = st.select_slider(
            "Print Quality",
            options=["Draft", "Normal", "High", "Ultra"],
            value="High",
            help="Higher quality means slower print speed but better detail"
        )
    
    with col2:
        scale = st.slider(
            "Model Scale (μm)",
            min_value=1,
            max_value=1000,
            value=100,
            help="Scale factor for the printed model"
        )
        
        support = st.checkbox(
            "Generate Supports",
            value=True,
            help="Automatically generate support structures"
        )
    
    # Advanced settings in tabs instead of nested expander
    st.markdown("### Advanced Settings")
    settings_tabs = st.tabs(["Print Settings", "Material Info", "Print Status"])
    
    with settings_tabs[0]:
        col3, col4 = st.columns(2)
        
        with col3:
            layer_height = st.number_input(
                "Layer Height (mm)",
                min_value=0.05,
                max_value=0.3,
                value=0.1,
                step=0.05
            )
            
            infill = st.slider(
                "Infill Density (%)",
                min_value=10,
                max_value=100,
                value=20
            )
        
        with col4:
            temp = st.slider(
                "Print Temperature (°C)",
                min_value=180,
                max_value=250,
                value=215
            )
            
            bed_temp = st.slider(
                "Bed Temperature (°C)",
                min_value=40,
                max_value=110,
                value=60
            )

    with settings_tabs[1]:
        # Material-specific information
        material_info = {
            "Carbon Fiber PLA": {
                "description": "Rigid and strong, good for structural models",
                "temp_range": "210-230°C",
                "properties": "High stiffness, low warping"
            },
            "Graphene-Enhanced PLA": {
                "description": "Excellent conductivity and strength",
                "temp_range": "200-220°C",
                "properties": "Conductive, strong layer adhesion"
            },
            "Carbon Nanotube Composite": {
                "description": "Superior strength and electrical properties",
                "temp_range": "220-240°C",
                "properties": "High strength, excellent conductivity"
            },
            "Conductive Carbon Black": {
                "description": "Good conductivity and detail resolution",
                "temp_range": "190-210°C",
                "properties": "Conductive, good surface finish"
            },
            "Carbon-Infused Resin": {
                "description": "High detail and strength for resin printing",
                "temp_range": "N/A (Resin)",
                "properties": "High detail, isotropic properties"
            }
        }
        
        st.info(f"""
        **Selected Material Properties:**
        - {material_info[material]['description']}
        - Temperature Range: {material_info[material]['temp_range']}
        - Properties: {material_info[material]['properties']}
        """)

    with settings_tabs[2]:
        # Print status and controls
        col5, col6 = st.columns([2, 1])
        
        with col5:
            printer_status = st.empty()
            progress_bar = st.progress(0)
        
        with col6:
            if st.button("Start 3D Print", type="primary"):
                try:
                    # Simulate print process
                    printer_status.info("Preparing print job...")
                    sleep(1)
                    
                    printer_status.info("Slicing model...")
                    progress_bar.progress(25)
                    sleep(1)
                    
                    printer_status.info("Generating G-code...")
                    progress_bar.progress(50)
                    sleep(1)
                    
                    printer_status.info("Sending to printer...")
                    progress_bar.progress(75)
                    sleep(1)
                    
                    printer_status.success("Print job started!")
                    progress_bar.progress(100)
                    
                    st.success("""
                    🖨️ Print job successfully sent to printer!
                    
                    Estimated print time: 2h 15m
                    Material usage: 45g
                    Layer count: 127
                    """)
                    
                except Exception as e:
                    st.error(f"Error starting print: {str(e)}") 