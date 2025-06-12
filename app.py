import streamlit as st
import streamlit.components.v1 as components
import json
import pandas as pd
from enhanced_data_processor import EnhancedDataProcessor
from enhanced_visualizer import EnhancedPacketVisualizer
from clustering_analyzer import ITSClusteringAnalyzer
import traceback
import numpy as np

def main():
    st.set_page_config(
        page_title="Comprehensive ITS Packet Analyzer",
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("🚗 Comprehensive ITS Vehicle Communication Analyzer")
    st.markdown("Advanced analysis of Intelligent Transport Systems (ITS) packet data from PCAP files")
    
    # Sidebar for file upload and controls
    with st.sidebar:
        st.header("📁 Data Input")
        uploaded_file = st.file_uploader(
            "Choose a JSON file",
            type=['json'],
            help="Upload a JSON file converted from PCAP data containing ITS/CAM messages"
        )
        
        if uploaded_file is not None:
            try:
                # Display file info
                file_size = uploaded_file.size
                st.info(f"File size: {file_size:,} bytes ({file_size/(1024*1024):.1f} MB)")
                
                # Load and process the JSON data with progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Load JSON data
                    status_text.text("Loading JSON file...")
                    progress_bar.progress(10)
                    content = uploaded_file.read()
                    
                    status_text.text("Parsing JSON data...")
                    progress_bar.progress(20)
                    json_data = json.loads(content)
                    
                    # Initialize enhanced processors
                    data_processor = EnhancedDataProcessor()
                    visualizer = EnhancedPacketVisualizer()
                    clustering_analyzer = ITSClusteringAnalyzer()
                    
                    # Step 2: Process data in chunks for large files
                    status_text.text("Processing packet data...")
                    progress_bar.progress(40)
                    
                    # Handle different JSON structures with memory optimization
                    if isinstance(json_data, list) and len(json_data) > 500:
                        # Process large files in smaller chunks to prevent memory issues
                        chunk_size = 250  # Smaller chunks for better memory management
                        processed_chunks = []
                        total_chunks = (len(json_data) + chunk_size - 1) // chunk_size
                        
                        for i in range(0, len(json_data), chunk_size):
                            chunk = json_data[i:i+chunk_size]
                            
                            # Process chunk with error handling
                            try:
                                chunk_df = data_processor.process_json_data(chunk)
                                if not chunk_df.empty:
                                    processed_chunks.append(chunk_df)
                                
                                # Force garbage collection for large datasets
                                if len(processed_chunks) > 10:
                                    import gc
                                    gc.collect()
                                    
                            except Exception as chunk_error:
                                st.warning(f"Error processing chunk {i//chunk_size + 1}: {str(chunk_error)}")
                                continue
                            
                            # Update progress
                            chunk_progress = 40 + (40 * (i // chunk_size + 1) / total_chunks)
                            progress_bar.progress(int(chunk_progress))
                            status_text.text(f"Processing chunk {i//chunk_size + 1}/{total_chunks}...")
                            
                            # Allow UI to update
                            if i % (chunk_size * 5) == 0:  # Every 5 chunks
                                import time
                                time.sleep(0.1)
                        
                        # Combine all chunks with memory management
                        if processed_chunks:
                            status_text.text("Combining processed data...")
                            try:
                                processed_data = pd.concat(processed_chunks, ignore_index=True)
                                # Clear chunk data from memory
                                del processed_chunks
                                import gc
                                gc.collect()
                            except Exception as concat_error:
                                st.error(f"Error combining data chunks: {str(concat_error)}")
                                return
                        else:
                            processed_data = pd.DataFrame()
                    else:
                        # Regular processing for smaller files
                        processed_data = data_processor.process_json_data(json_data)
                    
                    progress_bar.progress(90)
                    status_text.text("Finalizing data processing...")
                    
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON format: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()
                    return
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()
                    return
                
                progress_bar.progress(100)
                status_text.text("Processing complete!")
                
                if processed_data.empty:
                    st.error("No valid packet data found in the uploaded file.")
                    progress_bar.empty()
                    status_text.empty()
                    return
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"✅ Successfully processed {len(processed_data):,} packets")
                
                # Generate comprehensive analysis
                analysis = data_processor.get_comprehensive_analysis(processed_data)
                # Display key metrics in sidebar
                st.header("📊 Dataset Overview")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Packets", f"{analysis.get('total_packets', 0):,}")
                    st.metric("Unique Stations", analysis.get('unique_stations', 0))
                
                with col2:
                    st.metric("Time Span", f"{analysis.get('time_span_hours', 0):.1f} hours")
                    moving_count = analysis.get('speed_stats', {}).get('moving_vehicles', 0)
                    st.metric("Moving Vehicles", moving_count)
                
                # Filter controls
                st.header("🔍 Filters")
                
                # Station type filter
                if 'station_type_name' in processed_data.columns:
                    station_types = processed_data['station_type_name'].dropna().unique()
                    selected_types = st.multiselect(
                        "Vehicle Types",
                        options=station_types,
                        default=station_types
                    )
                else:
                    selected_types = []
                
                # Speed filter
                if 'speed' in processed_data.columns:
                    speed_data = processed_data['speed'].dropna()
                    if len(speed_data) > 0:
                        min_speed, max_speed = float(speed_data.min()), float(speed_data.max())
                        
                        # Handle case where all speeds are identical
                        if max_speed - min_speed < 0.01:
                            # Expand range slightly for identical values
                            if min_speed == 0:
                                speed_range = (0.0, 1.0)
                            else:
                                margin = max(0.1, min_speed * 0.1)
                                speed_range = (max(0, min_speed - margin), min_speed + margin)
                            st.info(f"All vehicles have similar speed ({min_speed:.3f} km/h). Filter range expanded.")
                        else:
                            speed_range = st.slider(
                                "Speed Range (km/h)",
                                min_value=min_speed,
                                max_value=max_speed,
                                value=(min_speed, max_speed),
                                step=0.001
                            )
                    else:
                        speed_range = (0.0, 100.0)
                else:
                    speed_range = (0.0, 100.0)
                
                # Apply filters
                filtered_data = processed_data.copy()
                
                if selected_types and 'station_type_name' in filtered_data.columns:
                    filtered_data = filtered_data[filtered_data['station_type_name'].isin(selected_types)]
                
                if 'speed' in filtered_data.columns:
                    speed_mask = (filtered_data['speed'] >= speed_range[0]) & (filtered_data['speed'] <= speed_range[1])
                    filtered_data = filtered_data[speed_mask]
                
                st.info(f"Filtered to {len(filtered_data):,} packets")
                
            except json.JSONDecodeError as e:
                st.error(f"❌ Error parsing JSON file: {str(e)}")
                return
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())
                return
        else:
            st.info("👆 Please upload a JSON file to begin analysis")
            st.markdown("""
            ### Expected Data Format
            The application expects JSON data containing ITS (Intelligent Transport Systems) 
            packet captures with CAM (Cooperative Awareness Messages) from vehicle communications.
            
            **Supported protocols:**
            - Ethernet (eth)
            - GeoNetworking (gnw)  
            - Basic Transport Protocol (btpb)
            - ITS Application Layer
            - CAM Messages
            """)
            return
    
    # Main content area with comprehensive analysis
    if 'filtered_data' in locals() and not filtered_data.empty:
        
        # Create comprehensive tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "🎯 Executive Summary", "🗺️ Geographic Analysis", "🚗 Vehicle Behavior", 
            "📡 Communication Patterns", "🔬 Clustering Analysis", "⚡ Performance Metrics", "📊 Data Quality", "📋 Raw Data"
        ])
        
        with tab1:
            st.header("🎯 Executive Summary")
            
            try:
                # Key insights
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_packets = len(filtered_data)
                    st.metric("Total Messages", f"{total_packets:,}")
                    
                    if 'station_id' in filtered_data.columns:
                        unique_vehicles = filtered_data['station_id'].nunique()
                        st.metric("Unique Vehicles", unique_vehicles)
                
                with col2:
                    if 'speed' in filtered_data.columns:
                        speeds = filtered_data['speed'].dropna()
                        if len(speeds) > 0:
                            avg_speed = speeds.mean()
                            max_speed = speeds.max()
                            st.metric("Avg Speed", f"{avg_speed:.1f} km/h")
                            st.metric("Max Speed", f"{max_speed:.1f} km/h")
                
                with col3:
                    if 'timestamp' in filtered_data.columns:
                        time_span = (filtered_data['timestamp'].max() - filtered_data['timestamp'].min())
                        hours = time_span.total_seconds() / 3600
                        st.metric("Duration", f"{hours:.1f} hours")
                        
                        msg_rate = total_packets / hours if hours > 0 else 0
                        st.metric("Message Rate", f"{msg_rate:.0f}/hour")
                
                with col4:
                    if 'has_position' in filtered_data.columns:
                        geo_coverage = filtered_data['has_position'].sum()
                        geo_percent = (geo_coverage / len(filtered_data)) * 100
                        st.metric("With GPS", f"{geo_percent:.1f}%")
                    
                    if 'is_moving' in filtered_data.columns:
                        moving_vehicles = filtered_data['is_moving'].sum()
                        moving_percent = (moving_vehicles / len(filtered_data)) * 100
                        st.metric("Moving", f"{moving_percent:.1f}%")
                
                # Comprehensive overview visualization
                st.subheader("System Overview Dashboard")
                overview_fig = visualizer.create_comprehensive_overview(filtered_data)
                st.plotly_chart(overview_fig, use_container_width=True)
                
                # Key findings
                st.subheader("📈 Key Findings")
                
                findings = []
                
                if 'station_type_name' in filtered_data.columns:
                    vehicle_distribution = filtered_data['station_type_name'].value_counts()
                    most_common = vehicle_distribution.index[0] if len(vehicle_distribution) > 0 else "Unknown"
                    findings.append(f"**Most common vehicle type:** {most_common} ({vehicle_distribution.iloc[0]:,} messages)")
                
                if 'speed' in filtered_data.columns:
                    speeds = filtered_data['speed'].dropna()
                    if len(speeds) > 0:
                        stationary = (speeds == 0).sum()
                        moving = (speeds > 0).sum()
                        findings.append(f"**Vehicle activity:** {moving:,} moving, {stationary:,} stationary")
                
                if 'timestamp' in filtered_data.columns:
                    hourly_activity = filtered_data['timestamp'].dt.hour.value_counts()
                    peak_hour = hourly_activity.index[0] if len(hourly_activity) > 0 else "Unknown"
                    findings.append(f"**Peak activity hour:** {peak_hour}:00 ({hourly_activity.iloc[0]:,} messages)")
                
                for finding in findings:
                    st.markdown(finding)
                    
            except Exception as e:
                st.error(f"Error in executive summary: {str(e)}")
        
        with tab2:
            st.header("🗺️ Geographic Analysis")
            
            try:
                geo_data = filtered_data.dropna(subset=['latitude', 'longitude']) if 'latitude' in filtered_data.columns else pd.DataFrame()
                
                if not geo_data.empty:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("Vehicle Trajectory Map")
                        trajectory_map = visualizer.create_vehicle_trajectory_map(geo_data)
                        components.html(trajectory_map._repr_html_(), height=600)
                    
                    with col2:
                        st.subheader("Geographic Statistics")
                        
                        # Coverage area
                        lat_range = geo_data['latitude'].max() - geo_data['latitude'].min()
                        lon_range = geo_data['longitude'].max() - geo_data['longitude'].min()
                        
                        # Approximate area calculation (rough)
                        area_km2 = lat_range * lon_range * 111.32 * 111.32  # Very rough approximation
                        
                        st.metric("Coverage Area", f"{area_km2:.2f} km²")
                        st.metric("Position Updates", f"{len(geo_data):,}")
                        
                        # Geographic bounds
                        st.markdown("**Boundaries:**")
                        st.text(f"North: {geo_data['latitude'].max():.6f}°")
                        st.text(f"South: {geo_data['latitude'].min():.6f}°")
                        st.text(f"East: {geo_data['longitude'].max():.6f}°")
                        st.text(f"West: {geo_data['longitude'].min():.6f}°")
                        
                        # Density analysis
                        if 'station_id' in geo_data.columns:
                            stations_with_pos = geo_data['station_id'].nunique()
                            st.metric("Vehicles Tracked", stations_with_pos)
                            
                            avg_messages_per_vehicle = len(geo_data) / stations_with_pos if stations_with_pos > 0 else 0
                            st.metric("Avg Msgs/Vehicle", f"{avg_messages_per_vehicle:.1f}")
                
                else:
                    st.warning("No geographic data available for mapping")
                    
            except Exception as e:
                st.error(f"Error in geographic analysis: {str(e)}")
        
        with tab3:
            st.header("🚗 Vehicle Behavior Analysis")
            
            try:
                # Speed analysis
                st.subheader("Speed and Movement Analysis")
                speed_fig = visualizer.create_speed_analysis(filtered_data)
                st.plotly_chart(speed_fig, use_container_width=True)
                
                # Behavior analysis
                st.subheader("Vehicle Behavior Patterns")
                behavior_fig = visualizer.create_vehicle_behavior_analysis(filtered_data)
                st.plotly_chart(behavior_fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in vehicle behavior analysis: {str(e)}")
        
        with tab4:
            st.header("📡 Communication Patterns")
            
            try:
                comm_fig = visualizer.create_communication_analysis(filtered_data)
                st.plotly_chart(comm_fig, use_container_width=True)
                
                # Protocol statistics
                if 'protocols' in filtered_data.columns:
                    st.subheader("Protocol Distribution")
                    all_protocols = filtered_data['protocols'].explode().value_counts()
                    st.bar_chart(all_protocols)
                
            except Exception as e:
                st.error(f"Error in communication analysis: {str(e)}")
        
        with tab5:
            st.header("🔬 Clustering Analysis")
            
            try:
                # Prepare clustering features
                features_df = clustering_analyzer.prepare_clustering_features(filtered_data)
                
                if not features_df.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("K-means Clustering")
                        
                        # K-means parameters
                        n_clusters = st.slider("Number of clusters (K-means)", 2, 8, 4)
                        
                        # Perform K-means clustering
                        kmeans_results = clustering_analyzer.perform_kmeans_clustering(features_df, n_clusters)
                        
                        if 'error' not in kmeans_results:
                            st.success(f"K-means completed: {kmeans_results['n_clusters']} clusters found")
                            
                            # Display cluster statistics
                            cluster_sizes = kmeans_results.get('cluster_sizes', {})
                            for cluster_id, size in cluster_sizes.items():
                                st.metric(f"Cluster {cluster_id}", f"{size} vehicles")
                        else:
                            st.error(f"K-means failed: {kmeans_results['error']}")
                    
                    with col2:
                        st.subheader("DBSCAN Clustering")
                        
                        # DBSCAN parameters
                        eps = st.slider("Epsilon (DBSCAN)", 0.1, 2.0, 0.5, 0.1)
                        min_samples = st.slider("Min samples (DBSCAN)", 2, 10, 5)
                        
                        # Perform DBSCAN clustering
                        dbscan_results = clustering_analyzer.perform_dbscan_clustering(features_df, eps, min_samples)
                        
                        if 'error' not in dbscan_results:
                            st.success(f"DBSCAN completed: {dbscan_results['n_clusters']} clusters found")
                            st.info(f"Noise points: {dbscan_results['n_noise']} ({(dbscan_results['n_noise']/len(filtered_data)*100):.1f}%)")
                            
                            # Display cluster statistics
                            cluster_sizes = dbscan_results.get('cluster_sizes', {})
                            for cluster_id, size in cluster_sizes.items():
                                label = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
                                st.metric(label, f"{size} vehicles")
                        else:
                            st.error(f"DBSCAN failed: {dbscan_results['error']}")
                    
                    # Clustering visualizations
                    if 'error' not in kmeans_results and 'error' not in dbscan_results:
                        st.subheader("Clustering Visualizations")
                        clustering_fig = clustering_analyzer.create_clustering_visualizations(
                            filtered_data, features_df, kmeans_results, dbscan_results
                        )
                        st.plotly_chart(clustering_fig, use_container_width=True)
                        
                        # Analysis report
                        st.subheader("Clustering Analysis Report")
                        analysis_report = clustering_analyzer.create_cluster_analysis_report(
                            filtered_data, features_df, kmeans_results, dbscan_results
                        )
                        
                        if 'error' not in analysis_report:
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.write("**Dataset Summary**")
                                summary = analysis_report['summary']
                                st.text(f"Data points: {summary['total_data_points']:,}")
                                st.text(f"Features used: {summary['feature_count']}")
                                st.text(f"Features: {', '.join(summary['clustering_features'][:3])}...")
                            
                            with col2:
                                st.write("**K-means Results**")
                                if 'kmeans_analysis' in analysis_report:
                                    kmeans_analysis = analysis_report['kmeans_analysis']
                                    st.text(f"Clusters: {kmeans_analysis.get('n_clusters', 0)}")
                                    st.text(f"Largest cluster: {kmeans_analysis.get('largest_cluster', 0)}")
                                    st.text(f"Smallest cluster: {kmeans_analysis.get('smallest_cluster', 0)}")
                            
                            with col3:
                                st.write("**DBSCAN Results**")
                                if 'dbscan_analysis' in analysis_report:
                                    dbscan_analysis = analysis_report['dbscan_analysis']
                                    st.text(f"Clusters: {dbscan_analysis.get('n_clusters', 0)}")
                                    st.text(f"Noise: {dbscan_analysis.get('noise_percentage', 0):.1f}%")
                                    st.text(f"Parameters: eps={dbscan_analysis.get('eps', 0)}, min_samples={dbscan_analysis.get('min_samples', 0)}")
                            
                            # Feature importance
                            if features_df.columns.tolist():
                                st.subheader("Clustering Features Used")
                                feature_info = pd.DataFrame({
                                    'Feature': features_df.columns,
                                    'Data Type': [str(features_df[col].dtype) for col in features_df.columns],
                                    'Non-null Count': [features_df[col].notna().sum() for col in features_df.columns],
                                    'Coverage %': [(features_df[col].notna().sum() / len(features_df)) * 100 for col in features_df.columns]
                                })
                                st.dataframe(feature_info, use_container_width=True)
                        
                else:
                    st.warning("Insufficient data for clustering analysis. Need more diverse vehicle data with geographic positions, speeds, or behavioral metrics.")
                    
            except Exception as e:
                st.error(f"Error in clustering analysis: {str(e)}")
        
        with tab6:
            st.header("⚡ Performance Metrics")
            
            try:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Message Frequency")
                    if 'timestamp' in filtered_data.columns:
                        freq_data = filtered_data.copy()
                        freq_data['minute'] = freq_data['timestamp'].dt.floor('min')
                        freq_counts = freq_data.groupby('minute').size()
                        
                        st.line_chart(freq_counts)
                        
                        st.metric("Peak Rate", f"{freq_counts.max()} msg/min")
                        st.metric("Avg Rate", f"{freq_counts.mean():.1f} msg/min")
                
                with col2:
                    st.subheader("System Health")
                    
                    # Data freshness
                    if 'timestamp' in filtered_data.columns:
                        latest_msg = filtered_data['timestamp'].max()
                        oldest_msg = filtered_data['timestamp'].min()
                        
                        st.text(f"Latest: {latest_msg}")
                        st.text(f"Oldest: {oldest_msg}")
                    
                    # Station activity
                    if 'station_id' in filtered_data.columns:
                        active_stations = filtered_data['station_id'].nunique()
                        total_messages = len(filtered_data)
                        
                        st.metric("Active Stations", active_stations)
                        st.metric("Total Messages", f"{total_messages:,}")
                        
                        if active_stations > 0:
                            avg_per_station = total_messages / active_stations
                            st.metric("Avg Msgs/Station", f"{avg_per_station:.1f}")
                
            except Exception as e:
                st.error(f"Error in performance metrics: {str(e)}")
        
        with tab6:
            st.header("📊 Data Quality Report")
            
            try:
                quality_fig = visualizer.create_data_quality_report(filtered_data)
                st.plotly_chart(quality_fig, use_container_width=True)
                
                # Data completeness table
                st.subheader("Field Completeness Analysis")
                
                completeness_data = []
                important_fields = [
                    'station_id', 'timestamp', 'latitude', 'longitude', 'speed', 
                    'heading', 'station_type_name', 'message_type'
                ]
                
                for field in important_fields:
                    if field in filtered_data.columns:
                        total = len(filtered_data)
                        valid = filtered_data[field].notna().sum()
                        percentage = (valid / total) * 100
                        completeness_data.append({
                            'Field': field,
                            'Valid Values': valid,
                            'Total': total,
                            'Completeness': f"{percentage:.1f}%"
                        })
                
                if completeness_data:
                    completeness_df = pd.DataFrame(completeness_data)
                    st.dataframe(completeness_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in data quality report: {str(e)}")
        
        with tab7:
            st.header("📋 Raw Data Explorer")
            
            try:
                # Search functionality
                search_term = st.text_input("🔍 Search packets (station ID, message type, etc.)")
                
                # Column selection
                available_columns = filtered_data.columns.tolist()
                display_columns = st.multiselect(
                    "Select columns to display",
                    options=available_columns,
                    default=available_columns[:10] if len(available_columns) > 10 else available_columns
                )
                
                # Apply search filter
                display_data = filtered_data[display_columns] if display_columns else filtered_data
                
                if search_term:
                    mask = display_data.astype(str).apply(
                        lambda x: x.str.contains(search_term, case=False, na=False)
                    ).any(axis=1)
                    display_data = display_data[mask]
                
                st.subheader(f"Packet Data ({len(display_data):,} packets)")
                st.dataframe(display_data, use_container_width=True, height=400)
                
                # Export functionality
                if st.button("📥 Export Current View as CSV"):
                    csv = display_data.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"its_packet_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
            except Exception as e:
                st.error(f"Error in raw data explorer: {str(e)}")
    
    else:
        # Welcome screen when no data is loaded
        st.markdown("""
        ## Welcome to the Comprehensive ITS Packet Analyzer
        
        This advanced tool provides in-depth analysis of Intelligent Transport Systems (ITS) 
        communication data, specifically designed for:
        
        ### 📊 **Comprehensive Analysis Features:**
        - **Executive Summary**: Key insights and system overview
        - **Geographic Analysis**: Vehicle trajectories and coverage mapping
        - **Vehicle Behavior**: Speed patterns, movement analysis, and vehicle characteristics  
        - **Communication Patterns**: Message types, frequency, and protocol distribution
        - **Performance Metrics**: System health and message rates
        - **Data Quality**: Completeness analysis and validation
        - **Raw Data Explorer**: Detailed packet inspection and export
        
        ### 🚗 **Supported Vehicle Data:**
        - Cooperative Awareness Messages (CAM)
        - Vehicle positions, speed, and heading
        - Vehicle dimensions and characteristics
        - Turn signals and lighting status
        - Acceleration and movement patterns
        
        ### 🔧 **Technical Capabilities:**
        - Handles large datasets (200,000+ packets)
        - Real-time filtering and analysis
        - Interactive visualizations
        - Geographic mapping with trajectories
        - Data export functionality
        
        **Upload your ITS packet data JSON file to begin comprehensive analysis.**
        """)

if __name__ == "__main__":
    main()