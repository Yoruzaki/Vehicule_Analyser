import pandas as pd
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import re

class EnhancedDataProcessor:
    """
    Enhanced processor for comprehensive analysis of ITS/CAM packet data.
    """
    
    def __init__(self):
        self.station_types = {
            0: "Unknown", 1: "Pedestrian", 2: "Cyclist", 3: "Moped", 4: "Motorcycle",
            5: "PassengerCar", 6: "Bus", 7: "LightTruck", 8: "HeavyTruck", 
            9: "Trailer", 10: "SpecialVehicle", 11: "Tram", 15: "RoadSideUnit"
        }
    
    def process_json_data(self, json_data: Union[Dict, List]) -> pd.DataFrame:
        """Process comprehensive JSON packet data into structured DataFrame."""
        try:
            # Handle different JSON structures
            if isinstance(json_data, dict):
                # Single packet case
                if '_source' in json_data:
                    packets = [json_data]
                elif 'layers' in json_data:
                    packets = [{'_source': json_data}]
                else:
                    packets = [{'_source': {'layers': json_data}}]
            elif isinstance(json_data, list):
                # Multiple packets case - detect format
                if len(json_data) > 0:
                    first_item = json_data[0]
                    if isinstance(first_item, dict):
                        if '_source' in first_item:
                            # Elasticsearch format
                            packets = json_data
                        elif 'layers' in first_item:
                            # Direct layers format
                            packets = [{'_source': item} for item in json_data]
                        else:
                            # Raw packet data
                            packets = [{'_source': {'layers': item}} for item in json_data]
                    else:
                        return pd.DataFrame()
                else:
                    return pd.DataFrame()
            else:
                return pd.DataFrame()
            
            processed_packets = []
            total_packets = len(packets)
            
            for i, packet in enumerate(packets):
                try:
                    # Process packet and validate structure
                    if not isinstance(packet, dict):
                        continue
                        
                    # Ensure _source exists and has proper structure
                    source = packet.get('_source', {})
                    if not source:
                        continue
                        
                    # Handle cases where layers might be nested differently
                    if 'layers' not in source:
                        if any(key in source for key in ['frame', 'eth', 'gnw', 'btpb', 'its']):
                            source = {'layers': source}
                        else:
                            continue
                    
                    processed_packet = self._extract_comprehensive_packet_info({'_source': source}, i)
                    if processed_packet:
                        processed_packets.append(processed_packet)
                        
                except Exception as e:
                    # Log specific parsing errors for debugging
                    print(f"Error processing packet {i}/{total_packets}: {str(e)}")
                    continue
            
            if not processed_packets:
                return pd.DataFrame()
            
            df = pd.DataFrame(processed_packets)
            df = self._convert_and_enhance_data_types(df)
            
            return df
            
        except Exception as e:
            print(f"Error processing JSON data: {e}")
            return pd.DataFrame()
    
    def _extract_comprehensive_packet_info(self, packet: Dict, index: int) -> Optional[Dict[str, Any]]:
        """Extract comprehensive information from a single packet."""
        try:
            source = packet.get('_source', {})
            layers = source.get('layers', {})
            
            packet_info = {'packet_index': index}
            
            # Frame information
            frame = layers.get('frame', {})
            packet_info.update(self._extract_enhanced_frame_info(frame))
            
            # Ethernet information
            eth = layers.get('eth', {})
            packet_info.update(self._extract_enhanced_ethernet_info(eth))
            
            # GeoNetworking information
            gnw = layers.get('gnw', {})
            packet_info.update(self._extract_enhanced_geonetworking_info(gnw))
            
            # BTP information
            btpb = layers.get('btpb', {})
            packet_info.update(self._extract_enhanced_btp_info(btpb))
            
            # ITS information
            its = layers.get('its', {})
            packet_info.update(self._extract_enhanced_its_info(its))
            
            # CAM information (most detailed)
            cam = layers.get('cam', {}) if 'cam' in layers else layers.get('its', {})
            packet_info.update(self._extract_enhanced_cam_info(cam))
            
            return packet_info
            
        except Exception as e:
            print(f"Error extracting packet info: {e}")
            return None
    
    def _extract_enhanced_frame_info(self, frame: Dict) -> Dict[str, Any]:
        """Extract enhanced frame layer information."""
        info = {}
        
        try:
            info['timestamp'] = frame.get('frame.time_utc')
            info['time_epoch'] = self._safe_float(frame.get('frame.time_epoch'))
            info['time_delta'] = self._safe_float(frame.get('frame.time_delta'))
            info['time_relative'] = self._safe_float(frame.get('frame.time_relative'))
            info['frame_number'] = self._safe_int(frame.get('frame.number'))
            info['frame_len'] = self._safe_int(frame.get('frame.len'))
            info['cap_len'] = self._safe_int(frame.get('frame.cap_len'))
            
            # Extract and clean protocols
            protocols_str = frame.get('frame.protocols', '')
            if protocols_str:
                protocols = protocols_str.split(':')
                info['protocols'] = protocols
                info['protocol_count'] = len(protocols)
                info['has_its'] = 'its' in protocols
                info['has_cam'] = any('cam' in p.lower() for p in protocols)
            else:
                info['protocols'] = []
                info['protocol_count'] = 0
                info['has_its'] = False
                info['has_cam'] = False
                
        except Exception as e:
            print(f"Error extracting frame info: {e}")
        
        return info
    
    def _extract_enhanced_ethernet_info(self, eth: Dict) -> Dict[str, Any]:
        """Extract enhanced Ethernet layer information."""
        info = {}
        
        try:
            info['src_mac'] = eth.get('eth.src')
            info['dst_mac'] = eth.get('eth.dst')
            info['eth_type'] = eth.get('eth.type')
            info['is_broadcast'] = eth.get('eth.dst') == 'ff:ff:ff:ff:ff:ff'
            
            # Extract OUI information for manufacturer identification
            src_tree = eth.get('eth.src_tree', {})
            info['src_oui'] = src_tree.get('eth.src.oui')
            
        except Exception as e:
            print(f"Error extracting ethernet info: {e}")
        
        return info
    
    def _extract_enhanced_geonetworking_info(self, gnw: Dict) -> Dict[str, Any]:
        """Extract enhanced GeoNetworking layer information."""
        info = {}
        
        try:
            # Basic header
            bh = gnw.get('geonw.bh', {})
            info['gnw_version'] = self._safe_int(bh.get('geonw.bh.version'))
            info['gnw_next_header'] = self._safe_int(bh.get('geonw.bh.nh'))
            info['gnw_lifetime'] = self._safe_int(bh.get('geonw.bh.lt'))
            info['gnw_hop_limit'] = self._safe_int(bh.get('geonw.bh.rhl'))
            
            # Common header
            ch = gnw.get('geonw.ch', {})
            info['gnw_header_type'] = ch.get('geonw.ch.htype')
            info['gnw_traffic_class'] = self._safe_int(ch.get('geonw.ch.tclass'))
            info['gnw_payload_length'] = self._safe_int(ch.get('geonw.ch.plength'))
            info['gnw_max_hop_limit'] = self._safe_int(ch.get('geonw.ch.mhl'))
            
            # Mobility flags
            flags = ch.get('geonw.ch.flags', {})
            info['gnw_mobile'] = flags.get('geonw.ch.flags.mob') == '1'
            
            # Position information from TSB
            tsb = gnw.get('geonw.tsb', {})
            if tsb:
                src_pos_tree = tsb.get('geonw.src_pos_tree', {})
                info['gnw_timestamp'] = self._safe_int(src_pos_tree.get('geonw.src_pos.tst'))
                info['gnw_latitude'] = self._safe_int(src_pos_tree.get('geonw.src_pos.lat'))
                info['gnw_longitude'] = self._safe_int(src_pos_tree.get('geonw.src_pos.long'))
                info['gnw_speed'] = self._safe_int(src_pos_tree.get('geonw.src_pos.speed'))
                info['gnw_heading'] = self._safe_int(src_pos_tree.get('geonw.src_pos.hdg'))
                info['gnw_position_accuracy'] = self._safe_int(src_pos_tree.get('geonw.src_pos.pai'))
                
        except Exception as e:
            print(f"Error extracting geonetworking info: {e}")
        
        return info
    
    def _extract_enhanced_btp_info(self, btpb: Dict) -> Dict[str, Any]:
        """Extract enhanced BTP layer information."""
        info = {}
        
        try:
            info['btp_dst_port'] = self._safe_int(btpb.get('btpb.dstport'))
            info['btp_dst_port_info'] = btpb.get('btpb.dstportinf')
            
            # Classify BTP destination port
            dst_port = info.get('btp_dst_port')
            if dst_port == 2001:
                info['btp_service'] = 'CAM'
            elif dst_port == 2002:
                info['btp_service'] = 'DENM'
            elif dst_port == 2003:
                info['btp_service'] = 'SPAT'
            elif dst_port == 2004:
                info['btp_service'] = 'MAP'
            else:
                info['btp_service'] = 'Other'
                
        except Exception as e:
            print(f"Error extracting BTP info: {e}")
        
        return info
    
    def _extract_enhanced_its_info(self, its: Dict) -> Dict[str, Any]:
        """Extract enhanced ITS layer information."""
        info = {}
        
        try:
            # ITS PDU Header
            header = its.get('its.ItsPduHeader_element', {})
            info['its_protocol_version'] = self._safe_int(header.get('its.protocolVersion'))
            info['message_id'] = self._safe_int(header.get('its.messageId'))
            info['station_id'] = self._safe_int(header.get('its.stationId'))
            
            # Message type classification
            msg_id = info.get('message_id')
            if msg_id == 1:
                info['message_type'] = 'DENM'
            elif msg_id == 2:
                info['message_type'] = 'CAM'
            elif msg_id == 3:
                info['message_type'] = 'POI'
            elif msg_id == 4:
                info['message_type'] = 'SPATEM'
            elif msg_id == 5:
                info['message_type'] = 'MAPEM'
            elif msg_id == 6:
                info['message_type'] = 'IVIM'
            elif msg_id == 7:
                info['message_type'] = 'EV-RSR'
            else:
                info['message_type'] = 'Unknown'
                
        except Exception as e:
            print(f"Error extracting ITS info: {e}")
        
        return info
    
    def _extract_enhanced_cam_info(self, cam_or_its: Dict) -> Dict[str, Any]:
        """Extract comprehensive CAM information."""
        info = {}
        
        try:
            # Handle both direct CAM structure and ITS structure containing CAM
            cam_payload = None
            if 'cam.CamPayload_element' in cam_or_its:
                cam_payload = cam_or_its['cam.CamPayload_element']
            elif 'cam.CamPayload_element' in cam_or_its.get('cam', {}):
                cam_payload = cam_or_its['cam']['cam.CamPayload_element']
            
            if not cam_payload:
                return info
            
            # Generation delta time
            info['generation_delta_time'] = self._safe_int(cam_payload.get('cam.generationDeltaTime'))
            
            # CAM parameters
            cam_params = cam_payload.get('cam.camParameters_element', {})
            if not cam_params:
                return info
            
            # Basic container - vehicle identification and position
            basic_container = cam_params.get('cam.basicContainer_element', {})
            if basic_container:
                station_type_raw = self._safe_int(basic_container.get('its.stationType'))
                info['station_type'] = station_type_raw
                if station_type_raw is not None:
                    info['station_type_name'] = self.station_types.get(station_type_raw, 'Unknown')
                else:
                    info['station_type_name'] = 'Unknown'
                
                # Reference position
                ref_pos = basic_container.get('its.referencePosition_element', {})
                if ref_pos:
                    info['latitude'] = self._safe_int(ref_pos.get('its.latitude'))
                    info['longitude'] = self._safe_int(ref_pos.get('its.longitude'))
                    
                    # Position confidence
                    pos_confidence = ref_pos.get('its.positionConfidenceEllipse_element', {})
                    if pos_confidence:
                        info['pos_semi_major'] = self._safe_int(pos_confidence.get('its.semiMajorAxisLength'))
                        info['pos_semi_minor'] = self._safe_int(pos_confidence.get('its.semiMinorAxisLength'))
                        info['pos_orientation'] = self._safe_int(pos_confidence.get('its.semiMajorAxisOrientation'))
                    
                    # Altitude
                    altitude = ref_pos.get('its.altitude_element', {})
                    if altitude:
                        info['altitude'] = self._safe_int(altitude.get('its.altitudeValue'))
                        info['altitude_confidence'] = self._safe_int(altitude.get('its.altitudeConfidence'))
            
            # High frequency container - dynamic vehicle data
            hf_container = cam_params.get('cam.highFrequencyContainer_tree', {})
            if hf_container:
                basic_vehicle = hf_container.get('cam.basicVehicleContainerHighFrequency_element', {})
                if basic_vehicle:
                    # Heading
                    heading = basic_vehicle.get('cam.heading_element', {})
                    if heading:
                        info['heading'] = self._safe_int(heading.get('its.headingValue'))
                        info['heading_confidence'] = self._safe_int(heading.get('its.headingConfidence'))
                    
                    # Speed
                    speed = basic_vehicle.get('cam.speed_element', {})
                    if speed:
                        info['speed'] = self._safe_int(speed.get('its.speedValue'))
                        info['speed_confidence'] = self._safe_int(speed.get('its.speedConfidence'))
                    
                    # Vehicle characteristics
                    info['drive_direction'] = self._safe_int(basic_vehicle.get('cam.driveDirection'))
                    
                    vehicle_length = basic_vehicle.get('cam.vehicleLength_element', {})
                    if vehicle_length:
                        info['vehicle_length'] = self._safe_int(vehicle_length.get('its.vehicleLengthValue'))
                        info['vehicle_length_confidence'] = self._safe_int(vehicle_length.get('its.vehicleLengthConfidenceIndication'))
                    
                    info['vehicle_width'] = self._safe_int(basic_vehicle.get('cam.vehicleWidth'))
                    
                    # Longitudinal acceleration
                    long_accel = basic_vehicle.get('cam.longitudinalAcceleration_element', {})
                    if long_accel:
                        info['longitudinal_acceleration'] = self._safe_int(long_accel.get('its.value'))
                        info['acceleration_confidence'] = self._safe_int(long_accel.get('its.confidence'))
                    
                    # Curvature
                    curvature = basic_vehicle.get('cam.curvature_element', {})
                    if curvature:
                        info['curvature_value'] = self._safe_int(curvature.get('its.curvatureValue'))
                        info['curvature_confidence'] = self._safe_int(curvature.get('its.curvatureConfidence'))
                    
                    info['curvature_calculation_mode'] = self._safe_int(basic_vehicle.get('cam.curvatureCalculationMode'))
                    
                    # Yaw rate
                    yaw_rate = basic_vehicle.get('cam.yawRate_element', {})
                    if yaw_rate:
                        info['yaw_rate_value'] = self._safe_int(yaw_rate.get('its.yawRateValue'))
                        info['yaw_rate_confidence'] = self._safe_int(yaw_rate.get('its.yawRateConfidence'))
            
            # Low frequency container - static vehicle data
            lf_container = cam_params.get('cam.lowFrequencyContainer_tree', {})
            if lf_container:
                basic_vehicle_lf = lf_container.get('cam.basicVehicleContainerLowFrequency_element', {})
                if basic_vehicle_lf:
                    info['vehicle_role'] = self._safe_int(basic_vehicle_lf.get('cam.vehicleRole'))
                    info['exterior_lights'] = basic_vehicle_lf.get('cam.exteriorLights')
                    
                    # Parse exterior lights
                    lights_tree = basic_vehicle_lf.get('cam.exteriorLights_tree', {})
                    if lights_tree:
                        info['low_beam_on'] = lights_tree.get('its.ExteriorLights.lowBeamHeadlightsOn') == '1'
                        info['high_beam_on'] = lights_tree.get('its.ExteriorLights.highBeamHeadlightsOn') == '1'
                        info['left_turn_signal'] = lights_tree.get('its.ExteriorLights.leftTurnSignalOn') == '1'
                        info['right_turn_signal'] = lights_tree.get('its.ExteriorLights.rightTurnSignalOn') == '1'
                        info['daytime_lights_on'] = lights_tree.get('its.ExteriorLights.daytimeRunningLightsOn') == '1'
                        info['reverse_light_on'] = lights_tree.get('its.ExteriorLights.reverseLightOn') == '1'
                        info['fog_light_on'] = lights_tree.get('its.ExteriorLights.fogLightOn') == '1'
                        info['parking_lights_on'] = lights_tree.get('its.ExteriorLights.parkingLightsOn') == '1'
                    
                    # Path history
                    path_history_length = self._safe_int(basic_vehicle_lf.get('per.sequence_of_length'))
                    info['path_history_length'] = path_history_length
                    
        except Exception as e:
            print(f"Error extracting CAM info: {e}")
        
        return info
    
    def _safe_int(self, value: Any) -> Optional[int]:
        """Safely convert value to integer."""
        if value is None or value == '':
            return None
        try:
            if isinstance(value, str):
                if value.startswith('0x'):
                    return int(value, 16)
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
    
    def _convert_and_enhance_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert DataFrame columns to appropriate data types with enhanced processing."""
        try:
            # Convert timestamp columns
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df['hour'] = df['timestamp'].dt.hour
                df['minute'] = df['timestamp'].dt.minute
                df['second'] = df['timestamp'].dt.second
                df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            # Convert coordinate values to actual lat/lon (from 1/10^7 degrees)
            for coord_col in ['latitude', 'longitude', 'gnw_latitude', 'gnw_longitude']:
                if coord_col in df.columns:
                    df[coord_col] = df[coord_col] / 10000000.0
            
            # Convert speed values (ITS speed is in 0.01 m/s units)
            for speed_col in ['speed', 'gnw_speed']:
                if speed_col in df.columns:
                    df[speed_col] = df[speed_col] * 0.01 * 3.6  # Convert to km/h
            
            # Convert heading values (from 0.1 degree units)
            for heading_col in ['heading', 'gnw_heading']:
                if heading_col in df.columns:
                    df[heading_col] = df[heading_col] * 0.1
            
            # Convert vehicle dimensions (from 0.1 meter units)
            for dimension_col in ['vehicle_length', 'vehicle_width']:
                if dimension_col in df.columns:
                    df[dimension_col] = df[dimension_col] * 0.1
            
            # Convert acceleration (from 0.1 m/s² units)
            if 'longitudinal_acceleration' in df.columns:
                df['longitudinal_acceleration'] = df['longitudinal_acceleration'] * 0.1
            
            # Convert curvature (from 1/30000 per meter)
            if 'curvature_value' in df.columns:
                df['curvature_value'] = df['curvature_value'] / 30000.0
            
            # Convert yaw rate (from 0.01 degree/s)
            if 'yaw_rate_value' in df.columns:
                df['yaw_rate_value'] = df['yaw_rate_value'] * 0.01
            
            # Add derived columns
            df['is_moving'] = (df.get('speed', 0) > 0.1)  # Moving if speed > 0.1 km/h
            df['has_position'] = df['latitude'].notna() & df['longitude'].notna()
            
            return df
            
        except Exception as e:
            print(f"Error converting data types: {e}")
            return df
    
    def get_comprehensive_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive analysis of the dataset."""
        analysis = {}
        
        try:
            # Basic statistics
            analysis['total_packets'] = len(df)
            analysis['unique_stations'] = df['station_id'].nunique() if 'station_id' in df.columns else 0
            analysis['time_span_hours'] = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600 if 'timestamp' in df.columns else 0
            
            # Vehicle type distribution
            if 'station_type_name' in df.columns:
                analysis['vehicle_types'] = df['station_type_name'].value_counts().to_dict()
            
            # Geographic coverage
            if 'latitude' in df.columns and 'longitude' in df.columns:
                valid_positions = df.dropna(subset=['latitude', 'longitude'])
                if len(valid_positions) > 0:
                    analysis['geographic_bounds'] = {
                        'min_lat': valid_positions['latitude'].min(),
                        'max_lat': valid_positions['latitude'].max(),
                        'min_lon': valid_positions['longitude'].min(),
                        'max_lon': valid_positions['longitude'].max()
                    }
            
            # Speed statistics
            if 'speed' in df.columns:
                speeds = df['speed'].dropna()
                if len(speeds) > 0:
                    analysis['speed_stats'] = {
                        'mean_speed': speeds.mean(),
                        'max_speed': speeds.max(),
                        'moving_vehicles': (speeds > 0.1).sum(),
                        'stationary_vehicles': (speeds <= 0.1).sum()
                    }
            
            # Message frequency
            if 'timestamp' in df.columns:
                df_sorted = df.sort_values('timestamp')
                time_diffs = df_sorted['timestamp'].diff().dt.total_seconds()
                analysis['message_frequency'] = {
                    'mean_interval': time_diffs.mean(),
                    'min_interval': time_diffs.min(),
                    'max_interval': time_diffs.max()
                }
            
            return analysis
            
        except Exception as e:
            print(f"Error in comprehensive analysis: {e}")
            return analysis