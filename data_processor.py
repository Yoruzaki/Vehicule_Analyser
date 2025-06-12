import pandas as pd
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import re

class DataProcessor:
    """
    Processes JSON packet data and converts it into structured formats for analysis.
    """
    
    def __init__(self):
        pass
    
    def process_json_data(self, json_data: Union[Dict, List]) -> pd.DataFrame:
        """
        Process JSON data and convert it to a structured DataFrame.
        
        Args:
            json_data: JSON data (either a single packet or list of packets)
            
        Returns:
            DataFrame containing processed packet data
        """
        try:
            # Handle both single packet and list of packets
            if isinstance(json_data, dict):
                if '_source' in json_data:
                    packets = [json_data]
                else:
                    # Assume it's already a packet structure
                    packets = [{'_source': json_data}]
            elif isinstance(json_data, list):
                packets = json_data
            else:
                return pd.DataFrame()
            
            processed_packets = []
            
            for packet in packets:
                try:
                    processed_packet = self._extract_packet_info(packet)
                    if processed_packet:
                        processed_packets.append(processed_packet)
                except Exception as e:
                    print(f"Error processing packet: {e}")
                    continue
            
            if not processed_packets:
                return pd.DataFrame()
            
            df = pd.DataFrame(processed_packets)
            
            # Convert data types
            df = self._convert_data_types(df)
            
            return df
            
        except Exception as e:
            print(f"Error processing JSON data: {e}")
            return pd.DataFrame()
    
    def _extract_packet_info(self, packet: Dict) -> Optional[Dict[str, Any]]:
        """
        Extract relevant information from a single packet.
        
        Args:
            packet: Single packet data
            
        Returns:
            Dictionary containing extracted packet information
        """
        try:
            source = packet.get('_source', {})
            layers = source.get('layers', {})
            
            packet_info = {}
            
            # Extract frame information
            frame = layers.get('frame', {})
            packet_info.update(self._extract_frame_info(frame))
            
            # Extract Ethernet information
            eth = layers.get('eth', {})
            packet_info.update(self._extract_ethernet_info(eth))
            
            # Extract GeoNetworking information
            gnw = layers.get('gnw', {})
            packet_info.update(self._extract_geonetworking_info(gnw))
            
            # Extract BTP information
            btpb = layers.get('btpb', {})
            packet_info.update(self._extract_btp_info(btpb))
            
            # Extract ITS information
            its = layers.get('its', {})
            packet_info.update(self._extract_its_info(its))
            
            # Extract CAM information
            cam = layers.get('cam', {})
            packet_info.update(self._extract_cam_info(cam))
            
            return packet_info
            
        except Exception as e:
            print(f"Error extracting packet info: {e}")
            return None
    
    def _extract_frame_info(self, frame: Dict) -> Dict[str, Any]:
        """Extract frame layer information."""
        info = {}
        
        try:
            info['timestamp'] = frame.get('frame.time_utc')
            info['time_epoch'] = self._safe_float(frame.get('frame.time_epoch'))
            info['time_delta'] = self._safe_float(frame.get('frame.time_delta'))
            info['frame_number'] = self._safe_int(frame.get('frame.number'))
            info['frame_len'] = self._safe_int(frame.get('frame.len'))
            info['cap_len'] = self._safe_int(frame.get('frame.cap_len'))
            
            # Extract protocols
            protocols_str = frame.get('frame.protocols', '')
            if protocols_str:
                info['protocols'] = protocols_str.split(':')
            else:
                info['protocols'] = []
                
        except Exception as e:
            print(f"Error extracting frame info: {e}")
        
        return info
    
    def _extract_ethernet_info(self, eth: Dict) -> Dict[str, Any]:
        """Extract Ethernet layer information."""
        info = {}
        
        try:
            info['src_mac'] = eth.get('eth.src')
            info['dst_mac'] = eth.get('eth.dst')
            info['eth_type'] = eth.get('eth.type')
            
        except Exception as e:
            print(f"Error extracting ethernet info: {e}")
        
        return info
    
    def _extract_geonetworking_info(self, gnw: Dict) -> Dict[str, Any]:
        """Extract GeoNetworking layer information."""
        info = {}
        
        try:
            # Basic header info
            bh = gnw.get('geonw.bh', {})
            info['gnw_version'] = self._safe_int(bh.get('geonw.bh.version'))
            info['gnw_next_header'] = self._safe_int(bh.get('geonw.bh.nh'))
            info['gnw_lifetime'] = self._safe_int(bh.get('geonw.bh.lt'))
            
            # Common header info
            ch = gnw.get('geonw.ch', {})
            info['gnw_header_type'] = ch.get('geonw.ch.htype')
            info['gnw_traffic_class'] = self._safe_int(ch.get('geonw.ch.tclass'))
            info['gnw_payload_length'] = self._safe_int(ch.get('geonw.ch.plength'))
            
            # Position info
            tsb = gnw.get('geonw.tsb', {})
            if tsb:
                src_pos_tree = tsb.get('geonw.src_pos_tree', {})
                info['gnw_latitude'] = self._safe_int(src_pos_tree.get('geonw.src_pos.lat'))
                info['gnw_longitude'] = self._safe_int(src_pos_tree.get('geonw.src_pos.long'))
                info['gnw_speed'] = self._safe_int(src_pos_tree.get('geonw.src_pos.speed'))
                info['gnw_heading'] = self._safe_int(src_pos_tree.get('geonw.src_pos.hdg'))
                
        except Exception as e:
            print(f"Error extracting geonetworking info: {e}")
        
        return info
    
    def _extract_btp_info(self, btpb: Dict) -> Dict[str, Any]:
        """Extract BTP layer information."""
        info = {}
        
        try:
            info['btp_dst_port'] = self._safe_int(btpb.get('btpb.dstport'))
            info['btp_dst_port_info'] = btpb.get('btpb.dstportinf')
            
        except Exception as e:
            print(f"Error extracting BTP info: {e}")
        
        return info
    
    def _extract_its_info(self, its: Dict) -> Dict[str, Any]:
        """Extract ITS layer information."""
        info = {}
        
        try:
            # ITS PDU Header
            header = its.get('its.ItsPduHeader_element', {})
            info['its_protocol_version'] = self._safe_int(header.get('its.protocolVersion'))
            info['message_id'] = self._safe_int(header.get('its.messageId'))
            info['station_id'] = self._safe_int(header.get('its.stationId'))
            
        except Exception as e:
            print(f"Error extracting ITS info: {e}")
        
        return info
    
    def _extract_cam_info(self, cam: Dict) -> Dict[str, Any]:
        """Extract CAM (Cooperative Awareness Message) information."""
        info = {}
        
        try:
            # CAM payload
            payload = cam.get('cam.CamPayload_element', {})
            info['generation_delta_time'] = self._safe_int(payload.get('cam.generationDeltaTime'))
            
            # CAM parameters
            cam_params = payload.get('cam.camParameters_element', {})
            if cam_params:
                # Basic container
                basic_container = cam_params.get('cam.basicContainer_element', {})
                if basic_container:
                    info['station_type'] = self._safe_int(basic_container.get('its.stationType'))
                    
                    # Reference position
                    ref_pos = basic_container.get('its.referencePosition_element', {})
                    if ref_pos:
                        info['latitude'] = self._safe_int(ref_pos.get('its.latitude'))
                        info['longitude'] = self._safe_int(ref_pos.get('its.longitude'))
                        
                        # Altitude
                        altitude = ref_pos.get('its.altitude_element', {})
                        if altitude:
                            info['altitude'] = self._safe_int(altitude.get('its.altitudeValue'))
                
                # High frequency container
                hf_container = cam_params.get('cam.highFrequencyContainer_tree', {})
                if hf_container:
                    basic_vehicle = hf_container.get('cam.basicVehicleContainerHighFrequency_element', {})
                    if basic_vehicle:
                        # Heading
                        heading = basic_vehicle.get('cam.heading_element', {})
                        if heading:
                            info['heading'] = self._safe_int(heading.get('its.headingValue'))
                        
                        # Speed
                        speed = basic_vehicle.get('cam.speed_element', {})
                        if speed:
                            info['speed'] = self._safe_int(speed.get('its.speedValue'))
                        
                        # Vehicle dimensions
                        info['drive_direction'] = self._safe_int(basic_vehicle.get('cam.driveDirection'))
                        
                        vehicle_length = basic_vehicle.get('cam.vehicleLength_element', {})
                        if vehicle_length:
                            info['vehicle_length'] = self._safe_int(vehicle_length.get('its.vehicleLengthValue'))
                        
                        info['vehicle_width'] = self._safe_int(basic_vehicle.get('cam.vehicleWidth'))
                
                # Low frequency container
                lf_container = cam_params.get('cam.lowFrequencyContainer_tree', {})
                if lf_container:
                    basic_vehicle_lf = lf_container.get('cam.basicVehicleContainerLowFrequency_element', {})
                    if basic_vehicle_lf:
                        info['vehicle_role'] = self._safe_int(basic_vehicle_lf.get('cam.vehicleRole'))
                        info['exterior_lights'] = basic_vehicle_lf.get('cam.exteriorLights')
                        
        except Exception as e:
            print(f"Error extracting CAM info: {e}")
        
        return info
    
    def _safe_int(self, value: Any) -> Optional[int]:
        """Safely convert value to integer."""
        if value is None or value == '':
            return None
        try:
            if isinstance(value, str):
                # Handle hex values
                if value.startswith('0x'):
                    return int(value, 16)
                # Handle regular numbers
                return int(float(value))
            return int(value)
        except (ValueError, TypeError):
            return None
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float."""
        if value is None or value == '':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert DataFrame columns to appropriate data types."""
        try:
            # Convert timestamp columns
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            # Convert coordinate values to actual lat/lon
            if 'latitude' in df.columns:
                df['latitude'] = df['latitude'] / 10000000.0  # Convert to decimal degrees
            
            if 'longitude' in df.columns:
                df['longitude'] = df['longitude'] / 10000000.0  # Convert to decimal degrees
            
            if 'gnw_latitude' in df.columns:
                df['gnw_latitude'] = df['gnw_latitude'] / 10000000.0
            
            if 'gnw_longitude' in df.columns:
                df['gnw_longitude'] = df['gnw_longitude'] / 10000000.0
            
            # Convert speed values (ITS speed is in 0.01 m/s units)
            if 'speed' in df.columns:
                df['speed'] = df['speed'] * 0.01 * 3.6  # Convert from 0.01 m/s to km/h
            
            return df
            
        except Exception as e:
            print(f"Error converting data types: {e}")
            return df
    
    def apply_filters(self, df: pd.DataFrame, protocols: Optional[List[str]] = None, 
                     time_range: Optional[tuple] = None, size_range: Optional[tuple] = None) -> pd.DataFrame:
        """
        Apply filters to the DataFrame.
        
        Args:
            df: DataFrame to filter
            protocols: List of protocols to include
            time_range: Tuple of (start_time, end_time)
            size_range: Tuple of (min_size, max_size)
            
        Returns:
            Filtered DataFrame
        """
        try:
            filtered_df = df.copy()
            
            # Protocol filter
            if protocols and 'protocols' in filtered_df.columns:
                mask = filtered_df['protocols'].apply(
                    lambda x: any(p in protocols for p in x) if isinstance(x, list) else False
                )
                filtered_df = filtered_df[mask]
            
            # Time range filter
            if time_range and 'timestamp' in filtered_df.columns:
                start_time, end_time = time_range
                mask = (filtered_df['timestamp'] >= start_time) & (filtered_df['timestamp'] <= end_time)
                filtered_df = filtered_df[mask]
            
            # Size range filter
            if size_range and 'frame_len' in filtered_df.columns:
                min_size, max_size = size_range
                mask = (filtered_df['frame_len'] >= min_size) & (filtered_df['frame_len'] <= max_size)
                filtered_df = filtered_df[mask]
            
            return filtered_df
            
        except Exception as e:
            print(f"Error applying filters: {e}")
            return df
    
    def extract_geographic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract geographic data from the DataFrame.
        
        Args:
            df: DataFrame containing packet data
            
        Returns:
            DataFrame containing only geographic information
        """
        try:
            geo_columns = ['station_id', 'latitude', 'longitude', 'altitude', 'speed', 'heading']
            available_columns = [col for col in geo_columns if col in df.columns]
            
            if not available_columns or 'latitude' not in available_columns or 'longitude' not in available_columns:
                return pd.DataFrame()
            
            geo_df = df[available_columns].copy()
            
            # Filter out invalid coordinates
            geo_df = geo_df.dropna(subset=['latitude', 'longitude'])
            geo_df = geo_df[(geo_df['latitude'] != 0) | (geo_df['longitude'] != 0)]
            
            # Filter reasonable coordinate ranges
            geo_df = geo_df[
                (geo_df['latitude'].between(-90, 90)) & 
                (geo_df['longitude'].between(-180, 180))
            ]
            
            return geo_df
            
        except Exception as e:
            print(f"Error extracting geographic data: {e}")
            return pd.DataFrame()
    
    def extract_its_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract ITS-specific data from the DataFrame.
        
        Args:
            df: DataFrame containing packet data
            
        Returns:
            DataFrame containing ITS-specific information
        """
        try:
            # Filter for ITS packets
            its_mask = df['protocols'].apply(
                lambda x: 'its' in x if isinstance(x, list) else False
            )
            its_df = df[its_mask].copy()
            
            if its_df.empty:
                return pd.DataFrame()
            
            # Select relevant ITS columns
            its_columns = [
                'station_id', 'message_id', 'station_type', 'latitude', 'longitude',
                'speed', 'heading', 'vehicle_length', 'vehicle_width', 'vehicle_role'
            ]
            
            available_columns = [col for col in its_columns if col in its_df.columns]
            
            if available_columns:
                return its_df[available_columns]
            else:
                return its_df
            
        except Exception as e:
            print(f"Error extracting ITS data: {e}")
            return pd.DataFrame()
