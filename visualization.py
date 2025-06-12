import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import folium
from folium import plugins
from typing import Dict, Any, List, Optional

class PacketVisualizer:
    """
    Creates interactive visualizations for network packet data analysis.
    """
    
    def __init__(self):
        # Color palette for consistent styling
        self.colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
    
    def create_protocol_distribution(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a pie chart showing protocol distribution.
        
        Args:
            df: DataFrame containing processed packet data
            
        Returns:
            Plotly figure object
        """
        try:
            if 'protocols' not in df.columns:
                # Return empty figure with message
                fig = go.Figure()
                fig.add_annotation(
                    text="No protocol data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=16)
                )
                return fig
            
            # Count protocol occurrences
            protocol_counts = df['protocols'].explode().value_counts()
            
            if protocol_counts.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text="No protocol data found",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=16)
                )
                return fig
            
            # Create pie chart
            fig = px.pie(
                values=protocol_counts.values,
                names=protocol_counts.index,
                title="Protocol Distribution",
                color_discrete_sequence=self.colors
            )
            
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )
            
            fig.update_layout(
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5),
                margin=dict(t=50, b=50, l=50, r=50)
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating protocol distribution: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
    
    def create_packet_size_distribution(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a histogram showing packet size distribution.
        
        Args:
            df: DataFrame containing processed packet data
            
        Returns:
            Plotly figure object
        """
        try:
            if 'frame_len' not in df.columns or df['frame_len'].empty:
                fig = go.Figure()
                fig.add_annotation(
                    text="No packet size data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=16)
                )
                return fig
            
            fig = px.histogram(
                df,
                x='frame_len',
                nbins=30,
                title="Packet Size Distribution",
                labels={'frame_len': 'Packet Size (bytes)', 'count': 'Frequency'},
                color_discrete_sequence=[self.colors[0]]
            )
            
            fig.update_layout(
                xaxis_title="Packet Size (bytes)",
                yaxis_title="Frequency",
                showlegend=False,
                margin=dict(t=50, b=50, l=50, r=50)
            )
            
            # Add statistics annotation
            mean_size = df['frame_len'].mean()
            median_size = df['frame_len'].median()
            
            fig.add_annotation(
                text=f"Mean: {mean_size:.1f} bytes<br>Median: {median_size:.1f} bytes",
                xref="paper", yref="paper",
                x=0.8, y=0.8,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating packet size distribution: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
    
    def create_protocol_timeline(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a timeline showing protocol activity over time.
        
        Args:
            df: DataFrame containing processed packet data
            
        Returns:
            Plotly figure object
        """
        try:
            if 'timestamp' not in df.columns or 'protocols' not in df.columns:
                fig = go.Figure()
                fig.add_annotation(
                    text="Timestamp or protocol data not available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=16)
                )
                return fig
            
            # Prepare data
            df_copy = df.copy()
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            
            # Explode protocols and create timeline
            protocol_timeline = []
            for idx, row in df_copy.iterrows():
                if isinstance(row['protocols'], list):
                    for protocol in row['protocols']:
                        protocol_timeline.append({
                            'timestamp': row['timestamp'],
                            'protocol': protocol
                        })
            
            if not protocol_timeline:
                fig = go.Figure()
                fig.add_annotation(
                    text="No protocol timeline data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=16)
                )
                return fig
            
            timeline_df = pd.DataFrame(protocol_timeline)
            
            # Group by time intervals (e.g., 1-minute intervals)
            timeline_df['time_bin'] = timeline_df['timestamp'].dt.floor('T')
            protocol_counts = timeline_df.groupby(['time_bin', 'protocol']).size().reset_index(name='count')
            
            fig = px.line(
                protocol_counts,
                x='time_bin',
                y='count',
                color='protocol',
                title="Protocol Activity Timeline",
                labels={'time_bin': 'Time', 'count': 'Packet Count'},
                color_discrete_sequence=self.colors
            )
            
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Packet Count",
                legend_title="Protocol",
                margin=dict(t=50, b=50, l=50, r=50)
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating protocol timeline: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
    
    def create_geographic_map(self, geo_df: pd.DataFrame) -> folium.Map:
        """
        Create a geographic map showing vehicle positions.
        
        Args:
            geo_df: DataFrame containing geographic data
            
        Returns:
            Folium map object
        """
        try:
            if geo_df.empty or 'latitude' not in geo_df.columns or 'longitude' not in geo_df.columns:
                # Create a default map
                m = folium.Map(location=[51.5074, -0.1278], zoom_start=10)
                folium.Marker(
                    [51.5074, -0.1278],
                    popup="No geographic data available",
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m)
                return m
            
            # Calculate map center
            center_lat = geo_df['latitude'].mean()
            center_lon = geo_df['longitude'].mean()
            
            # Create map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12,
                tiles='OpenStreetMap'
            )
            
            # Add markers for each position
            for idx, row in geo_df.iterrows():
                if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                    popup_text = f"Station ID: {row.get('station_id', 'Unknown')}<br>"
                    popup_text += f"Latitude: {row['latitude']:.6f}<br>"
                    popup_text += f"Longitude: {row['longitude']:.6f}<br>"
                    
                    if 'speed' in row and pd.notna(row['speed']):
                        popup_text += f"Speed: {row['speed']}<br>"
                    
                    if 'heading' in row and pd.notna(row['heading']):
                        popup_text += f"Heading: {row['heading']}<br>"
                    
                    # Color-code by speed if available
                    color = 'blue'
                    if 'speed' in row and pd.notna(row['speed']):
                        if row['speed'] > 50:
                            color = 'red'
                        elif row['speed'] > 25:
                            color = 'orange'
                        else:
                            color = 'green'
                    
                    folium.Marker(
                        [row['latitude'], row['longitude']],
                        popup=popup_text,
                        icon=folium.Icon(color=color, icon='car', prefix='fa')
                    ).add_to(m)
            
            # Add a heat map if there are multiple points
            if len(geo_df) > 1:
                heat_data = [[row['latitude'], row['longitude']] for idx, row in geo_df.iterrows() 
                           if pd.notna(row['latitude']) and pd.notna(row['longitude'])]
                
                if heat_data:
                    plugins.HeatMap(heat_data).add_to(m)
            
            return m
            
        except Exception as e:
            # Return default map with error message
            m = folium.Map(location=[51.5074, -0.1278], zoom_start=10)
            folium.Marker(
                [51.5074, -0.1278],
                popup=f"Error creating map: {str(e)}",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
            return m
    
    def create_position_scatter(self, geo_df: pd.DataFrame) -> go.Figure:
        """
        Create a scatter plot of vehicle positions.
        
        Args:
            geo_df: DataFrame containing geographic data
            
        Returns:
            Plotly figure object
        """
        try:
            if geo_df.empty or 'latitude' not in geo_df.columns or 'longitude' not in geo_df.columns:
                fig = go.Figure()
                fig.add_annotation(
                    text="No geographic position data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=16)
                )
                return fig
            
            # Create scatter plot
            fig = px.scatter(
                geo_df,
                x='longitude',
                y='latitude',
                color='speed' if 'speed' in geo_df.columns else None,
                size='speed' if 'speed' in geo_df.columns else None,
                hover_data=['station_id'] if 'station_id' in geo_df.columns else None,
                title="Vehicle Positions",
                labels={'longitude': 'Longitude', 'latitude': 'Latitude'},
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                xaxis_title="Longitude",
                yaxis_title="Latitude",
                margin=dict(t=50, b=50, l=50, r=50)
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating position scatter: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
    
    def create_time_frequency_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a chart showing packet frequency over time.
        
        Args:
            df: DataFrame containing processed packet data
            
        Returns:
            Plotly figure object
        """
        try:
            if 'timestamp' not in df.columns:
                fig = go.Figure()
                fig.add_annotation(
                    text="No timestamp data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=16)
                )
                return fig
            
            df_copy = df.copy()
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            
            # Group by time intervals (1-second intervals)
            df_copy['time_bin'] = df_copy['timestamp'].dt.floor('S')
            frequency_counts = df_copy.groupby('time_bin').size().reset_index(name='count')
            
            fig = px.line(
                frequency_counts,
                x='time_bin',
                y='count',
                title="Packet Frequency Over Time",
                labels={'time_bin': 'Time', 'count': 'Packets per Second'},
                line_shape='linear'
            )
            
            fig.update_traces(line_color=self.colors[0])
            
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Packets per Second",
                margin=dict(t=50, b=50, l=50, r=50)
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating time frequency chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
    
    def create_time_delta_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a chart showing inter-packet time deltas.
        
        Args:
            df: DataFrame containing processed packet data
            
        Returns:
            Plotly figure object
        """
        try:
            if 'time_delta' not in df.columns:
                fig = go.Figure()
                fig.add_annotation(
                    text="No time delta data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=16)
                )
                return fig
            
            # Filter out extreme values for better visualization
            time_deltas = df['time_delta'].dropna()
            q99 = time_deltas.quantile(0.99)
            filtered_deltas = time_deltas[time_deltas <= q99]
            
            fig = px.histogram(
                x=filtered_deltas,
                nbins=50,
                title="Inter-Packet Time Distribution",
                labels={'x': 'Time Delta (seconds)', 'count': 'Frequency'},
                color_discrete_sequence=[self.colors[1]]
            )
            
            fig.update_layout(
                xaxis_title="Time Delta (seconds)",
                yaxis_title="Frequency",
                showlegend=False,
                margin=dict(t=50, b=50, l=50, r=50)
            )
            
            # Add statistics
            mean_delta = filtered_deltas.mean()
            median_delta = filtered_deltas.median()
            
            fig.add_annotation(
                text=f"Mean: {mean_delta:.3f}s<br>Median: {median_delta:.3f}s",
                xref="paper", yref="paper",
                x=0.8, y=0.8,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating time delta chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
