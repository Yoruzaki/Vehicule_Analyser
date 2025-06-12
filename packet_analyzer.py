import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

class PacketAnalyzer:
    """
    Analyzes network packet data and generates statistics and insights.
    """
    
    def __init__(self):
        pass
    
    def generate_overview_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate overview statistics for the packet data.
        
        Args:
            df: DataFrame containing processed packet data
            
        Returns:
            Dictionary containing overview statistics
        """
        stats = {
            'total_packets': len(df),
            'unique_protocols': 0,
            'total_bytes': 0,
            'avg_packet_size': 0.0,
            'time_span': 0.0,
            'unique_sources': 0,
            'unique_destinations': 0
        }
        
        try:
            # Protocol analysis
            if 'protocols' in df.columns:
                all_protocols = df['protocols'].explode().dropna()
                stats['unique_protocols'] = len(all_protocols.unique())
            
            # Size analysis
            if 'frame_len' in df.columns:
                stats['total_bytes'] = df['frame_len'].sum()
                stats['avg_packet_size'] = df['frame_len'].mean()
            
            # Time span analysis
            if 'timestamp' in df.columns:
                timestamps = pd.to_datetime(df['timestamp'])
                time_span = (timestamps.max() - timestamps.min()).total_seconds()
                stats['time_span'] = time_span
            
            # Source/destination analysis
            if 'src_mac' in df.columns:
                stats['unique_sources'] = df['src_mac'].nunique()
            
            if 'dst_mac' in df.columns:
                stats['unique_destinations'] = df['dst_mac'].nunique()
                
        except Exception as e:
            print(f"Error generating overview stats: {e}")
        
        return stats
    
    def analyze_protocol_distribution(self, df: pd.DataFrame) -> pd.Series:
        """
        Analyze the distribution of protocols in the packet data.
        
        Args:
            df: DataFrame containing processed packet data
            
        Returns:
            Series with protocol counts
        """
        if 'protocols' not in df.columns:
            return pd.Series()
        
        try:
            # Explode the protocols list and count occurrences
            protocol_counts = df['protocols'].explode().value_counts()
            return protocol_counts
        except Exception as e:
            print(f"Error analyzing protocol distribution: {e}")
            return pd.Series()
    
    def analyze_traffic_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze traffic patterns including peak times and communication flows.
        
        Args:
            df: DataFrame containing processed packet data
            
        Returns:
            Dictionary containing traffic pattern analysis
        """
        patterns = {
            'peak_hour': None,
            'avg_packets_per_minute': 0.0,
            'busiest_communication_pairs': [],
            'protocol_trends': {}
        }
        
        try:
            if 'timestamp' in df.columns:
                df_copy = df.copy()
                df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
                df_copy['hour'] = df_copy['timestamp'].dt.hour
                df_copy['minute'] = df_copy['timestamp'].dt.floor('T')
                
                # Find peak hour
                hourly_counts = df_copy['hour'].value_counts()
                if not hourly_counts.empty:
                    patterns['peak_hour'] = hourly_counts.index[0]
                
                # Calculate average packets per minute
                minute_counts = df_copy['minute'].value_counts()
                patterns['avg_packets_per_minute'] = minute_counts.mean()
                
                # Analyze communication pairs
                if 'src_mac' in df_copy.columns and 'dst_mac' in df_copy.columns:
                    comm_pairs = df_copy.groupby(['src_mac', 'dst_mac']).size().sort_values(ascending=False)
                    patterns['busiest_communication_pairs'] = comm_pairs.head(10).to_dict()
                
        except Exception as e:
            print(f"Error analyzing traffic patterns: {e}")
        
        return patterns
    
    def analyze_its_messages(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze ITS (Intelligent Transport Systems) specific message patterns.
        
        Args:
            df: DataFrame containing processed packet data
            
        Returns:
            Dictionary containing ITS analysis results
        """
        its_analysis = {
            'cam_messages': 0,
            'station_types': {},
            'position_updates': 0,
            'speed_distribution': {},
            'heading_distribution': {}
        }
        
        try:
            # Filter for ITS messages
            its_data = df[df['protocols'].apply(lambda x: 'its' in x if isinstance(x, list) else False)]
            
            if its_data.empty:
                return its_analysis
            
            # Count CAM messages
            its_analysis['cam_messages'] = len(its_data)
            
            # Analyze station types
            if 'station_type' in its_data.columns:
                station_counts = its_data['station_type'].value_counts().to_dict()
                its_analysis['station_types'] = station_counts
            
            # Count position updates
            position_data = its_data.dropna(subset=['latitude', 'longitude'] if 'latitude' in its_data.columns else [])
            its_analysis['position_updates'] = len(position_data)
            
            # Analyze speed distribution
            if 'speed' in its_data.columns:
                speed_bins = pd.cut(its_data['speed'], bins=5, labels=['Very Slow', 'Slow', 'Medium', 'Fast', 'Very Fast'])
                its_analysis['speed_distribution'] = speed_bins.value_counts().to_dict()
            
            # Analyze heading distribution
            if 'heading' in its_data.columns:
                heading_bins = pd.cut(its_data['heading'], bins=8, labels=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
                its_analysis['heading_distribution'] = heading_bins.value_counts().to_dict()
                
        except Exception as e:
            print(f"Error analyzing ITS messages: {e}")
        
        return its_analysis
    
    def detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect potential anomalies in the packet data.
        
        Args:
            df: DataFrame containing processed packet data
            
        Returns:
            Dictionary containing anomaly detection results
        """
        anomalies = {
            'oversized_packets': [],
            'undersized_packets': [],
            'time_gaps': [],
            'unusual_protocols': [],
            'duplicate_packets': 0
        }
        
        try:
            # Detect oversized/undersized packets
            if 'frame_len' in df.columns:
                size_q1 = df['frame_len'].quantile(0.25)
                size_q3 = df['frame_len'].quantile(0.75)
                size_iqr = size_q3 - size_q1
                
                size_lower_bound = size_q1 - 1.5 * size_iqr
                size_upper_bound = size_q3 + 1.5 * size_iqr
                
                oversized = df[df['frame_len'] > size_upper_bound]
                undersized = df[df['frame_len'] < size_lower_bound]
                
                anomalies['oversized_packets'] = oversized.index.tolist()
                anomalies['undersized_packets'] = undersized.index.tolist()
            
            # Detect time gaps
            if 'timestamp' in df.columns:
                df_sorted = df.sort_values('timestamp')
                time_diffs = pd.to_datetime(df_sorted['timestamp']).diff()
                
                # Find gaps larger than 10 seconds
                large_gaps = time_diffs[time_diffs > pd.Timedelta(seconds=10)]
                anomalies['time_gaps'] = large_gaps.index.tolist()
            
            # Detect unusual protocols
            if 'protocols' in df.columns:
                protocol_counts = df['protocols'].explode().value_counts()
                # Protocols that appear less than 1% of the time
                rare_threshold = len(df) * 0.01
                unusual_protocols = protocol_counts[protocol_counts < rare_threshold]
                anomalies['unusual_protocols'] = unusual_protocols.index.tolist()
            
            # Detect potential duplicates
            duplicate_columns = ['src_mac', 'dst_mac', 'frame_len']
            available_columns = [col for col in duplicate_columns if col in df.columns]
            
            if available_columns:
                duplicates = df.duplicated(subset=available_columns)
                anomalies['duplicate_packets'] = duplicates.sum()
                
        except Exception as e:
            print(f"Error detecting anomalies: {e}")
        
        return anomalies
    
    def generate_summary_report(self, df: pd.DataFrame) -> str:
        """
        Generate a comprehensive summary report of the packet analysis.
        
        Args:
            df: DataFrame containing processed packet data
            
        Returns:
            String containing the formatted summary report
        """
        try:
            overview = self.generate_overview_stats(df)
            patterns = self.analyze_traffic_patterns(df)
            its_analysis = self.analyze_its_messages(df)
            anomalies = self.detect_anomalies(df)
            
            report = f"""
# Network Packet Analysis Summary

## Overview Statistics
- Total Packets: {overview['total_packets']:,}
- Unique Protocols: {overview['unique_protocols']}
- Total Data: {overview['total_bytes']:,} bytes
- Average Packet Size: {overview['avg_packet_size']:.2f} bytes
- Time Span: {overview['time_span']:.2f} seconds
- Unique Sources: {overview['unique_sources']}
- Unique Destinations: {overview['unique_destinations']}

## Traffic Patterns
- Peak Hour: {patterns['peak_hour'] if patterns['peak_hour'] is not None else 'N/A'}
- Average Packets per Minute: {patterns['avg_packets_per_minute']:.2f}
- Top Communication Pairs: {len(patterns['busiest_communication_pairs'])}

## ITS Analysis
- CAM Messages: {its_analysis['cam_messages']}
- Position Updates: {its_analysis['position_updates']}
- Station Types: {len(its_analysis['station_types'])} different types

## Anomalies Detected
- Oversized Packets: {len(anomalies['oversized_packets'])}
- Undersized Packets: {len(anomalies['undersized_packets'])}
- Time Gaps: {len(anomalies['time_gaps'])}
- Unusual Protocols: {len(anomalies['unusual_protocols'])}
- Duplicate Packets: {anomalies['duplicate_packets']}
"""
            
            return report
            
        except Exception as e:
            return f"Error generating summary report: {e}"
