import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, Any, Tuple, Optional

class ITSClusteringAnalyzer:
    """
    Performs K-means and DBSCAN clustering analysis on ITS vehicle communication data.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        
    def prepare_clustering_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for clustering analysis."""
        try:
            features = []
            feature_names = []
            
            # Geographic features
            if 'latitude' in df.columns and 'longitude' in df.columns:
                geo_data = df[['latitude', 'longitude']].dropna()
                if len(geo_data) > 0:
                    features.append(geo_data['latitude'].values)
                    features.append(geo_data['longitude'].values)
                    feature_names.extend(['latitude', 'longitude'])
            
            # Vehicle behavior features
            behavior_cols = ['speed', 'heading', 'longitudinal_acceleration', 'curvature_value', 'yaw_rate_value']
            for col in behavior_cols:
                if col in df.columns:
                    col_data = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
                    features.append(col_data.values)
                    feature_names.append(col)
            
            # Vehicle characteristics
            char_cols = ['vehicle_length', 'vehicle_width']
            for col in char_cols:
                if col in df.columns:
                    col_data = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
                    features.append(col_data.values)
                    feature_names.append(col)
            
            # Communication features
            if 'generation_delta_time' in df.columns:
                time_data = df['generation_delta_time'].fillna(df['generation_delta_time'].median() if df['generation_delta_time'].notna().any() else 0)
                features.append(time_data.values)
                feature_names.append('generation_delta_time')
            
            # Message frequency (packets per station)
            if 'station_id' in df.columns:
                station_counts = df['station_id'].value_counts()
                df_with_counts = df.copy()
                df_with_counts['message_frequency'] = df_with_counts['station_id'].map(station_counts)
                features.append(df_with_counts['message_frequency'].values)
                feature_names.append('message_frequency')
            
            # Path history length
            if 'path_history_length' in df.columns:
                path_data = df['path_history_length'].fillna(0)
                features.append(path_data.values)
                feature_names.append('path_history_length')
            
            if not features:
                return pd.DataFrame()
            
            # Ensure all features have the same length
            min_length = min(len(f) for f in features)
            features = [f[:min_length] for f in features]
            
            # Create feature DataFrame
            feature_df = pd.DataFrame(np.column_stack(features), columns=feature_names)
            
            # Remove any infinite or extreme values
            feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
            feature_df = feature_df.fillna(feature_df.median())
            
            return feature_df
            
        except Exception as e:
            print(f"Error preparing clustering features: {e}")
            return pd.DataFrame()
    
    def perform_kmeans_clustering(self, features_df: pd.DataFrame, n_clusters: int = 5) -> Dict[str, Any]:
        """Perform K-means clustering analysis."""
        try:
            if features_df.empty or len(features_df) < n_clusters:
                return {'error': 'Insufficient data for clustering'}
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features_df)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            # Calculate inertia and silhouette score
            inertia = kmeans.inertia_
            
            # PCA for visualization
            features_pca = self.pca.fit_transform(features_scaled)
            
            results = {
                'cluster_labels': cluster_labels,
                'cluster_centers': kmeans.cluster_centers_,
                'inertia': inertia,
                'features_pca': features_pca,
                'pca_variance_ratio': self.pca.explained_variance_ratio_,
                'n_clusters': n_clusters,
                'algorithm': 'K-means'
            }
            
            # Cluster statistics
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            results['cluster_sizes'] = dict(zip(unique_labels, counts))
            
            return results
            
        except Exception as e:
            return {'error': f'K-means clustering failed: {str(e)}'}
    
    def perform_dbscan_clustering(self, features_df: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> Dict[str, Any]:
        """Perform DBSCAN clustering analysis."""
        try:
            if features_df.empty:
                return {'error': 'No data for clustering'}
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features_df)
            
            # Perform DBSCAN clustering
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(features_scaled)
            
            # PCA for visualization
            features_pca = self.pca.fit_transform(features_scaled)
            
            # Calculate statistics
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = counts[unique_labels == -1][0] if -1 in unique_labels else 0
            
            results = {
                'cluster_labels': cluster_labels,
                'features_pca': features_pca,
                'pca_variance_ratio': self.pca.explained_variance_ratio_,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'eps': eps,
                'min_samples': min_samples,
                'algorithm': 'DBSCAN'
            }
            
            # Cluster statistics
            results['cluster_sizes'] = dict(zip(unique_labels, counts))
            
            return results
            
        except Exception as e:
            return {'error': f'DBSCAN clustering failed: {str(e)}'}
    
    def optimize_kmeans_clusters(self, features_df: pd.DataFrame, max_clusters: int = 10) -> Dict[str, Any]:
        """Find optimal number of clusters using elbow method."""
        try:
            if features_df.empty:
                return {'error': 'No data for optimization'}
            
            features_scaled = self.scaler.fit_transform(features_df)
            
            inertias = []
            k_range = range(2, min(max_clusters + 1, len(features_df)))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(features_scaled)
                inertias.append(kmeans.inertia_)
            
            return {
                'k_values': list(k_range),
                'inertias': inertias,
                'optimal_k': self._find_elbow_point(list(k_range), inertias)
            }
            
        except Exception as e:
            return {'error': f'K-means optimization failed: {str(e)}'}
    
    def _find_elbow_point(self, k_values: list, inertias: list) -> int:
        """Find elbow point using simple difference method."""
        if len(inertias) < 3:
            return k_values[0] if k_values else 2
        
        # Calculate differences
        diffs = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
        
        # Find point where difference starts to decrease significantly
        max_diff_idx = np.argmax(diffs)
        return k_values[max_diff_idx]
    
    def create_clustering_visualizations(self, df: pd.DataFrame, features_df: pd.DataFrame, 
                                       kmeans_results: Dict[str, Any], 
                                       dbscan_results: Dict[str, Any]) -> go.Figure:
        """Create comprehensive clustering visualizations."""
        try:
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=(
                    'K-means Clusters (PCA)', 'DBSCAN Clusters (PCA)', 'Cluster Comparison',
                    'K-means Geographic', 'DBSCAN Geographic', 'Feature Importance'
                ),
                specs=[
                    [{"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}],
                    [{"type": "scattermapbox"}, {"type": "scattermapbox"}, {"type": "bar"}]
                ]
            )
            
            # K-means PCA visualization
            if 'features_pca' in kmeans_results and 'cluster_labels' in kmeans_results:
                kmeans_pca = kmeans_results['features_pca']
                kmeans_labels = kmeans_results['cluster_labels']
                
                for cluster_id in np.unique(kmeans_labels):
                    mask = kmeans_labels == cluster_id
                    fig.add_trace(
                        go.Scatter(
                            x=kmeans_pca[mask, 0], 
                            y=kmeans_pca[mask, 1],
                            mode='markers',
                            name=f'K-means Cluster {cluster_id}',
                            showlegend=False
                        ),
                        row=1, col=1
                    )
            
            # DBSCAN PCA visualization
            if 'features_pca' in dbscan_results and 'cluster_labels' in dbscan_results:
                dbscan_pca = dbscan_results['features_pca']
                dbscan_labels = dbscan_results['cluster_labels']
                
                for cluster_id in np.unique(dbscan_labels):
                    mask = dbscan_labels == cluster_id
                    cluster_name = 'Noise' if cluster_id == -1 else f'DBSCAN Cluster {cluster_id}'
                    color = 'gray' if cluster_id == -1 else None
                    
                    fig.add_trace(
                        go.Scatter(
                            x=dbscan_pca[mask, 0], 
                            y=dbscan_pca[mask, 1],
                            mode='markers',
                            name=cluster_name,
                            marker=dict(color=color) if color else None,
                            showlegend=False
                        ),
                        row=1, col=2
                    )
            
            # Cluster size comparison
            if 'cluster_sizes' in kmeans_results and 'cluster_sizes' in dbscan_results:
                algorithms = ['K-means'] * len(kmeans_results['cluster_sizes']) + ['DBSCAN'] * len(dbscan_results['cluster_sizes'])
                cluster_names = (list(kmeans_results['cluster_sizes'].keys()) + 
                               list(dbscan_results['cluster_sizes'].keys()))
                sizes = list(kmeans_results['cluster_sizes'].values()) + list(dbscan_results['cluster_sizes'].values())
                
                fig.add_trace(
                    go.Bar(x=[f"{alg} C{name}" for alg, name in zip(algorithms, cluster_names)], 
                          y=sizes, name='Cluster Sizes', showlegend=False),
                    row=1, col=3
                )
            
            # Geographic visualizations
            if 'latitude' in df.columns and 'longitude' in df.columns:
                geo_data = df[['latitude', 'longitude']].dropna()
                if len(geo_data) > 0:
                    # K-means geographic
                    if 'cluster_labels' in kmeans_results:
                        geo_kmeans_labels = kmeans_results['cluster_labels'][:len(geo_data)]
                        fig.add_trace(
                            go.Scattermapbox(
                                lat=geo_data['latitude'], 
                                lon=geo_data['longitude'],
                                mode='markers',
                                marker=dict(color=geo_kmeans_labels, colorscale='Viridis'),
                                name='K-means Geographic',
                                showlegend=False
                            ),
                            row=2, col=1
                        )
                    
                    # DBSCAN geographic
                    if 'cluster_labels' in dbscan_results:
                        geo_dbscan_labels = dbscan_results['cluster_labels'][:len(geo_data)]
                        fig.add_trace(
                            go.Scattermapbox(
                                lat=geo_data['latitude'], 
                                lon=geo_data['longitude'],
                                mode='markers',
                                marker=dict(color=geo_dbscan_labels, colorscale='Plasma'),
                                name='DBSCAN Geographic',
                                showlegend=False
                            ),
                            row=2, col=2
                        )
            
            # Feature importance (PCA components)
            if 'pca_variance_ratio' in kmeans_results:
                variance_ratio = kmeans_results['pca_variance_ratio']
                fig.add_trace(
                    go.Bar(x=['PC1', 'PC2'], y=variance_ratio, 
                          name='PCA Variance Explained', showlegend=False),
                    row=2, col=3
                )
            
            fig.update_layout(
                height=800,
                title_text="ITS Vehicle Communication Clustering Analysis",
                mapbox=dict(style="open-street-map", zoom=10)
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating clustering visualizations: {str(e)}", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
    
    def create_cluster_analysis_report(self, df: pd.DataFrame, features_df: pd.DataFrame,
                                     kmeans_results: Dict[str, Any], 
                                     dbscan_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive cluster analysis report."""
        try:
            report = {
                'summary': {},
                'kmeans_analysis': {},
                'dbscan_analysis': {},
                'comparison': {}
            }
            
            # Summary statistics
            report['summary'] = {
                'total_data_points': len(df),
                'clustering_features': list(features_df.columns) if not features_df.empty else [],
                'feature_count': len(features_df.columns) if not features_df.empty else 0
            }
            
            # K-means analysis
            if 'error' not in kmeans_results:
                report['kmeans_analysis'] = {
                    'n_clusters': kmeans_results.get('n_clusters', 0),
                    'inertia': kmeans_results.get('inertia', 0),
                    'cluster_sizes': kmeans_results.get('cluster_sizes', {}),
                    'largest_cluster': max(kmeans_results.get('cluster_sizes', {}).values()) if kmeans_results.get('cluster_sizes') else 0,
                    'smallest_cluster': min(kmeans_results.get('cluster_sizes', {}).values()) if kmeans_results.get('cluster_sizes') else 0
                }
            
            # DBSCAN analysis
            if 'error' not in dbscan_results:
                report['dbscan_analysis'] = {
                    'n_clusters': dbscan_results.get('n_clusters', 0),
                    'n_noise_points': dbscan_results.get('n_noise', 0),
                    'noise_percentage': (dbscan_results.get('n_noise', 0) / len(df)) * 100 if len(df) > 0 else 0,
                    'cluster_sizes': dbscan_results.get('cluster_sizes', {}),
                    'eps': dbscan_results.get('eps', 0),
                    'min_samples': dbscan_results.get('min_samples', 0)
                }
            
            # Comparison
            if 'error' not in kmeans_results and 'error' not in dbscan_results:
                report['comparison'] = {
                    'kmeans_clusters': kmeans_results.get('n_clusters', 0),
                    'dbscan_clusters': dbscan_results.get('n_clusters', 0),
                    'dbscan_found_noise': dbscan_results.get('n_noise', 0) > 0,
                    'clustering_agreement': self._calculate_clustering_agreement(
                        kmeans_results.get('cluster_labels', []), 
                        dbscan_results.get('cluster_labels', [])
                    )
                }
            
            return report
            
        except Exception as e:
            return {'error': f'Failed to generate cluster analysis report: {str(e)}'}
    
    def _calculate_clustering_agreement(self, labels1: np.ndarray, labels2: np.ndarray) -> float:
        """Calculate agreement between two clustering results."""
        try:
            if len(labels1) != len(labels2) or len(labels1) == 0:
                return 0.0
            
            # Simple agreement measure: percentage of points in same relative grouping
            agreement_count = 0
            total_pairs = 0
            
            for i in range(len(labels1)):
                for j in range(i + 1, min(i + 100, len(labels1))):  # Sample to avoid O(n²) complexity
                    total_pairs += 1
                    # Check if both algorithms group these points together or separately
                    same_cluster_1 = labels1[i] == labels1[j]
                    same_cluster_2 = labels2[i] == labels2[j]
                    if same_cluster_1 == same_cluster_2:
                        agreement_count += 1
            
            return (agreement_count / total_pairs) * 100 if total_pairs > 0 else 0.0
            
        except Exception:
            return 0.0