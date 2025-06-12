import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import folium
from folium import plugins
from typing import Dict, Any, List, Optional

class EnhancedPacketVisualizer:
    """
    Creates comprehensive interactive visualizations for ITS/CAM packet data analysis.
    """
    
    def __init__(self):
        self.colors = px.colors.qualitative.Set1
        self.vehicle_colors = {
            'PassengerCar': '#1f77b4',
            'Bus': '#ff7f0e', 
            'LightTruck': '#2ca02c',
            'HeavyTruck': '#d62728',
            'Motorcycle': '#9467bd',
            'Cyclist': '#8c564b',
            'Pedestrian': '#e377c2',
            'RoadSideUnit': '#7f7f7f',
            'Unknown': '#bcbd22'
        }
    
    def create_comprehensive_overview(self, df: pd.DataFrame) -> go.Figure:
        """Create comprehensive overview dashboard."""
        try:
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=(
                    'Vehicle Type Distribution', 'Speed Distribution', 'Message Frequency Over Time',
                    'Geographic Coverage', 'Acceleration vs Speed', 'Heading Distribution'
                ),
                specs=[
                    [{"type": "pie"}, {"type": "histogram"}, {"type": "scatter"}],
                    [{"type": "scattermapbox"}, {"type": "scatter"}, {"type": "bar"}]
                ]
            )
            
            # Vehicle type distribution
            if 'station_type_name' in df.columns:
                type_counts = df['station_type_name'].value_counts()
                fig.add_trace(
                    go.Pie(labels=type_counts.index, values=type_counts.values, name="Vehicle Types"),
                    row=1, col=1
                )
            
            # Speed distribution
            if 'speed' in df.columns:
                speeds = df['speed'].dropna()
                fig.add_trace(
                    go.Histogram(x=speeds, nbinsx=30, name="Speed Distribution"),
                    row=1, col=2
                )
            
            # Message frequency over time
            if 'timestamp' in df.columns:
                df_time = df.copy()
                df_time['time_bin'] = df_time['timestamp'].dt.floor('min')
                freq_data = df_time.groupby('time_bin').size().reset_index()
                freq_data.columns = ['time_bin', 'count']
                fig.add_trace(
                    go.Scatter(x=freq_data['time_bin'], y=freq_data['count'], 
                              mode='lines', name="Messages/Min"),
                    row=1, col=3
                )
            
            # Geographic coverage
            if 'latitude' in df.columns and 'longitude' in df.columns:
                geo_data = df.dropna(subset=['latitude', 'longitude'])
                if len(geo_data) > 0:
                    fig.add_trace(
                        go.Scattermapbox(
                            lat=geo_data['latitude'], lon=geo_data['longitude'],
                            mode='markers', name="Vehicle Positions"
                        ),
                        row=2, col=1
                    )
            
            # Acceleration vs Speed
            if 'speed' in df.columns and 'longitudinal_acceleration' in df.columns:
                valid_data = df.dropna(subset=['speed', 'longitudinal_acceleration'])
                fig.add_trace(
                    go.Scatter(x=valid_data['speed'], y=valid_data['longitudinal_acceleration'],
                              mode='markers', name="Accel vs Speed"),
                    row=2, col=2
                )
            
            # Heading distribution
            if 'heading' in df.columns:
                headings = df['heading'].dropna()
                if len(headings) > 0:
                    heading_bins = pd.cut(headings, bins=8, labels=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
                    heading_counts = heading_bins.value_counts()
                    fig.add_trace(
                        go.Bar(x=heading_counts.index.astype(str), y=heading_counts.values, name="Heading"),
                        row=2, col=3
                    )
            
            fig.update_layout(height=800, showlegend=False, title_text="Comprehensive Vehicle Communication Analysis")
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating overview: {str(e)}", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
    
    def create_vehicle_trajectory_map(self, df: pd.DataFrame) -> folium.Map:
        """Create detailed vehicle trajectory map."""
        try:
            geo_data = df.dropna(subset=['latitude', 'longitude'])
            if geo_data.empty:
                m = folium.Map(location=[51.5074, -0.1278], zoom_start=10)
                folium.Marker([51.5074, -0.1278], popup="No geographic data available").add_to(m)
                return m
            
            # Calculate map center
            center_lat = float(geo_data['latitude'].mean())
            center_lon = float(geo_data['longitude'].mean())
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
            
            # Group by station_id to create trajectories
            if 'station_id' in geo_data.columns:
                for station_id, group in geo_data.groupby('station_id'):
                    if len(group) > 1:
                        # Sort by timestamp to create trajectory
                        if 'timestamp' in group.columns:
                            group = group.sort_values('timestamp')
                        
                        # Create trajectory line
                        coordinates = [[row['latitude'], row['longitude']] for _, row in group.iterrows()]
                        folium.PolyLine(
                            coordinates,
                            weight=3,
                            color=self._get_vehicle_color(group.iloc[0].get('station_type_name', 'Unknown')),
                            opacity=0.7,
                            popup=f"Station {station_id} trajectory"
                        ).add_to(m)
                        
                        # Add start and end markers
                        start_row = group.iloc[0]
                        end_row = group.iloc[-1]
                        
                        folium.Marker(
                            [start_row['latitude'], start_row['longitude']],
                            popup=f"Start: Station {station_id}",
                            icon=folium.Icon(color='green', icon='play')
                        ).add_to(m)
                        
                        folium.Marker(
                            [end_row['latitude'], end_row['longitude']],
                            popup=f"End: Station {station_id}",
                            icon=folium.Icon(color='red', icon='stop')
                        ).add_to(m)
            
            # Add speed-based markers
            for _, row in geo_data.iterrows():
                speed = row.get('speed', 0)
                color = 'green' if speed < 30 else 'orange' if speed < 60 else 'red'
                
                popup_text = f"""
                Station: {row.get('station_id', 'Unknown')}<br>
                Type: {row.get('station_type_name', 'Unknown')}<br>
                Speed: {speed:.1f} km/h<br>
                Heading: {row.get('heading', 0):.1f}°<br>
                Time: {row.get('timestamp', 'Unknown')}
                """
                
                folium.CircleMarker(
                    [row['latitude'], row['longitude']],
                    radius=5,
                    popup=popup_text,
                    color=color,
                    fill=True,
                    fillOpacity=0.7
                ).add_to(m)
            
            return m
            
        except Exception as e:
            m = folium.Map(location=[51.5074, -0.1278], zoom_start=10)
            folium.Marker([51.5074, -0.1278], popup=f"Error creating map: {str(e)}").add_to(m)
            return m
    
    def create_speed_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Create comprehensive speed analysis."""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Speed Over Time', 'Speed by Vehicle Type', 
                              'Speed Distribution', 'Acceleration vs Time'),
                specs=[[{"secondary_y": True}, {"type": "box"}],
                       [{"type": "histogram"}, {"secondary_y": True}]]
            )
            
            # Speed over time
            if 'timestamp' in df.columns and 'speed' in df.columns:
                speed_data = df.dropna(subset=['timestamp', 'speed']).sort_values('timestamp')
                fig.add_trace(
                    go.Scatter(x=speed_data['timestamp'], y=speed_data['speed'],
                              mode='markers', name='Speed', opacity=0.6),
                    row=1, col=1
                )
                
                # Add moving average
                if len(speed_data) > 10:
                    speed_data['speed_ma'] = speed_data['speed'].rolling(window=10).mean()
                    fig.add_trace(
                        go.Scatter(x=speed_data['timestamp'], y=speed_data['speed_ma'],
                                  mode='lines', name='Speed Trend', line=dict(color='red')),
                        row=1, col=1
                    )
            
            # Speed by vehicle type
            if 'station_type_name' in df.columns and 'speed' in df.columns:
                for vehicle_type in df['station_type_name'].unique():
                    if pd.notna(vehicle_type):
                        speeds = df[df['station_type_name'] == vehicle_type]['speed'].dropna()
                        fig.add_trace(
                            go.Box(y=speeds, name=vehicle_type),
                            row=1, col=2
                        )
            
            # Speed distribution
            if 'speed' in df.columns:
                speeds = df['speed'].dropna()
                fig.add_trace(
                    go.Histogram(x=speeds, nbinsx=50, name='Speed Distribution'),
                    row=2, col=1
                )
            
            # Acceleration over time
            if 'timestamp' in df.columns and 'longitudinal_acceleration' in df.columns:
                accel_data = df.dropna(subset=['timestamp', 'longitudinal_acceleration']).sort_values('timestamp')
                fig.add_trace(
                    go.Scatter(x=accel_data['timestamp'], y=accel_data['longitudinal_acceleration'],
                              mode='markers', name='Acceleration', opacity=0.6),
                    row=2, col=2
                )
            
            fig.update_layout(height=800, title_text="Comprehensive Speed Analysis")
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating speed analysis: {str(e)}",
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
    
    def create_communication_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Create communication pattern analysis."""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Message Types', 'Messages per Station', 
                              'Communication Timeline', 'Protocol Distribution')
            )
            
            # Message types
            if 'message_type' in df.columns:
                msg_counts = df['message_type'].value_counts()
                fig.add_trace(
                    go.Bar(x=msg_counts.index, y=msg_counts.values, name='Message Types'),
                    row=1, col=1
                )
            
            # Messages per station
            if 'station_id' in df.columns:
                station_counts = df['station_id'].value_counts().head(20)
                fig.add_trace(
                    go.Bar(x=station_counts.index.astype(str), y=station_counts.values, 
                          name='Top 20 Stations'),
                    row=1, col=2
                )
            
            # Communication timeline
            if 'timestamp' in df.columns:
                df_time = df.copy()
                df_time['hour'] = df_time['timestamp'].dt.hour
                hourly_counts = df_time['hour'].value_counts().sort_index()
                fig.add_trace(
                    go.Scatter(x=hourly_counts.index, y=hourly_counts.values,
                              mode='lines+markers', name='Messages by Hour'),
                    row=2, col=1
                )
            
            # Protocol distribution
            if 'protocols' in df.columns:
                all_protocols = df['protocols'].explode().value_counts()
                fig.add_trace(
                    go.Pie(labels=all_protocols.index, values=all_protocols.values, 
                          name='Protocols'),
                    row=2, col=2
                )
            
            fig.update_layout(height=800, title_text="Communication Pattern Analysis")
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating communication analysis: {str(e)}",
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
    
    def create_vehicle_behavior_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Create vehicle behavior analysis."""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Turn Signal Usage', 'Light Status', 
                              'Vehicle Dimensions', 'Movement Patterns'),
                specs=[[{"type": "pie"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "scatter"}]]
            )
            
            # Turn signal usage
            turn_signals = []
            if 'left_turn_signal' in df.columns and 'right_turn_signal' in df.columns:
                left_on = df['left_turn_signal'].sum()
                right_on = df['right_turn_signal'].sum()
                no_signal = len(df) - left_on - right_on
                
                fig.add_trace(
                    go.Pie(labels=['No Signal', 'Left Turn', 'Right Turn'], 
                          values=[no_signal, left_on, right_on], name='Turn Signals'),
                    row=1, col=1
                )
            
            # Light status
            light_columns = ['low_beam_on', 'high_beam_on', 'daytime_lights_on', 'fog_light_on']
            light_data = []
            light_labels = []
            for col in light_columns:
                if col in df.columns:
                    light_data.append(df[col].sum())
                    light_labels.append(col.replace('_', ' ').title())
            
            if light_data:
                fig.add_trace(
                    go.Bar(x=light_labels, y=light_data, name='Light Usage'),
                    row=1, col=2
                )
            
            # Vehicle dimensions
            if 'vehicle_length' in df.columns and 'vehicle_width' in df.columns:
                valid_dims = df.dropna(subset=['vehicle_length', 'vehicle_width'])
                fig.add_trace(
                    go.Scatter(x=valid_dims['vehicle_length'], y=valid_dims['vehicle_width'],
                              mode='markers', text=valid_dims.get('station_type_name', ''),
                              name='Dimensions'),
                    row=2, col=1
                )
            
            # Movement patterns (speed vs heading)
            if 'speed' in df.columns and 'heading' in df.columns:
                movement_data = df.dropna(subset=['speed', 'heading'])
                fig.add_trace(
                    go.Scatter(x=movement_data['heading'], y=movement_data['speed'],
                              mode='markers', name='Speed vs Heading',
                              text=movement_data.get('station_type_name', '')),
                    row=2, col=2
                )
            
            fig.update_layout(height=800, title_text="Vehicle Behavior Analysis")
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating behavior analysis: {str(e)}",
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
    
    def _get_vehicle_color(self, vehicle_type: str) -> str:
        """Get color for vehicle type."""
        return self.vehicle_colors.get(vehicle_type, '#bcbd22')
    
    def create_data_quality_report(self, df: pd.DataFrame) -> go.Figure:
        """Create data quality and coverage report."""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Data Completeness', 'Value Ranges', 
                              'Temporal Coverage', 'Station Activity'),
                specs=[[{"type": "bar"}, {"type": "box"}],
                       [{"type": "scatter"}, {"type": "bar"}]]
            )
            
            # Data completeness
            completeness = {}
            key_columns = ['latitude', 'longitude', 'speed', 'heading', 'station_id', 'timestamp']
            for col in key_columns:
                if col in df.columns:
                    completeness[col] = (df[col].notna().sum() / len(df)) * 100
            
            if completeness:
                fig.add_trace(
                    go.Bar(x=list(completeness.keys()), y=list(completeness.values()),
                          name='Completeness %'),
                    row=1, col=1
                )
            
            # Value ranges for numeric columns
            numeric_cols = ['speed', 'heading', 'longitudinal_acceleration', 'vehicle_length']
            for col in numeric_cols:
                if col in df.columns:
                    values = df[col].dropna()
                    if len(values) > 0:
                        fig.add_trace(go.Box(y=values, name=col), row=1, col=2)
            
            # Temporal coverage
            if 'timestamp' in df.columns:
                df_time = df.dropna(subset=['timestamp']).copy()
                df_time['date'] = df_time['timestamp'].dt.date
                daily_counts = df_time['date'].value_counts().sort_index()
                fig.add_trace(
                    go.Scatter(x=daily_counts.index, y=daily_counts.values,
                              mode='lines+markers', name='Daily Message Count'),
                    row=2, col=1
                )
            
            # Station activity
            if 'station_id' in df.columns:
                station_activity = df['station_id'].value_counts().head(10)
                fig.add_trace(
                    go.Bar(x=station_activity.index.astype(str), y=station_activity.values,
                          name='Top 10 Active Stations'),
                    row=2, col=2
                )
            
            fig.update_layout(height=800, title_text="Data Quality and Coverage Report")
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating quality report: {str(e)}",
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig