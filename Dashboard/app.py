import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class LogisticsRouteOptimizer:
    """
    A comprehensive system for finding optimal indirect air-road routes
    when direct air connections are not available.
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize the optimizer with scoring weights.
        
        Args:
            weights: Dictionary with keys 'volume', 'time', 'distance' for scoring
        """
        self.weights = weights or {'volume': 0.4, 'time': 0.4, 'distance': 0.2}
        
        # Data containers
        self.shipment_data = None
        self.airport_volume = None
        self.air_routes = None
        self.road_matrix = None
        self.no_flight_pairs = None
        self.city_coords = None
        
        # Processed containers
        self.valid_air_routes = set()
        self.road_connections = {}
        self.volume_matrix = {}
        
    def load_data(self, file_paths: Dict[str, str]):
        """
        Load all required CSV files.
        
        Args:
            file_paths: Dictionary mapping data type to file path
        """
        try:
            print("Loading data files...")
            
            # Load main datasets
            self.shipment_data = pd.read_csv(file_paths['shipment_data'])
            self.airport_volume = pd.read_csv(file_paths['airport_volume'])
            self.air_routes = pd.read_csv(file_paths['air_routes'])
            self.road_matrix = pd.read_csv(file_paths['road_matrix'])
            self.no_flight_pairs = pd.read_csv(file_paths['no_flight_pairs'])
            
            # Optional coordinates file
            if 'city_coordinates' in file_paths:
                self.city_coords = pd.read_csv(file_paths['city_coordinates'])
            
            print(f"✅ Loaded {len(self.shipment_data)} shipment records")
            print(f"✅ Loaded {len(self.airport_volume)} airport volume pairs")
            print(f"✅ Loaded {len(self.air_routes)} air routes")
            print(f"✅ Loaded {len(self.road_matrix)} road connections")
            print(f"✅ Loaded {len(self.no_flight_pairs)} no-flight OD pairs")
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self):
        """
        Clean and preprocess all loaded data for optimization.
        """
        print("\nPreprocessing data...")
        
        # 1. Create valid air routes set
        self._build_air_routes_set()
        
        # 2. Build road connections lookup
        self._build_road_connections()
        
        # 3. Create volume matrix
        self._build_volume_matrix()
        
        print("✅ Data preprocessing completed")
    
    def _build_air_routes_set(self):
        """Build a set of valid air route pairs."""
        # From air_routes.csv
        if 'origin_city' in self.air_routes.columns and 'destination_city' in self.air_routes.columns:
            air_pairs = self.air_routes[['origin_city', 'destination_city']].dropna()
            for _, row in air_pairs.iterrows():
                self.valid_air_routes.add((row['origin_city'], row['destination_city']))
        
        # From shipment data (PRIME + GCR modes)
        flight_shipments = self.shipment_data[
            self.shipment_data['Flight_Mode'].isin(['PRIME', 'GCR'])
        ].dropna(subset=['Origin Airport City', 'Destination Airport City'])
        
        for _, row in flight_shipments.iterrows():
            origin_air = row['Origin Airport City']
            dest_air = row['Destination Airport City']
            if pd.notna(origin_air) and pd.notna(dest_air):
                self.valid_air_routes.add((origin_air, dest_air))
        
        print(f"   ✓ Found {len(self.valid_air_routes)} valid air routes")
    
    def _build_road_connections(self):
        """Build road connections lookup with times and distances."""
        for _, row in self.road_matrix.iterrows():
            origin = row['origin']  # Updated column name
            dest = row['destination']  # Updated column name
            
            if origin not in self.road_connections:
                self.road_connections[origin] = {}
            
            self.road_connections[origin][dest] = {
                'distance': row.get('distance_km', 0),  # Updated column name
                'duration': row.get('duration_hr', 0)   # Updated column name
            }
        
        print(f"   ✓ Built road network with {len(self.road_connections)} origin cities")
    
    def _build_volume_matrix(self):
        """Build volume lookup from both shipment data and airport volume data."""
        # From airport_to_airport_volume.csv
        for _, row in self.airport_volume.iterrows():
            origin = row['Origin Airport City']
            dest = row['Destination Airport City']  
            volume = row.get('Weight', 0)  # Updated column name from Total_Weight to Weight
            
            key = (origin, dest)
            self.volume_matrix[key] = self.volume_matrix.get(key, 0) + volume
        
        # From shipment_data.csv (aggregate weights by airport pairs)
        shipment_volumes = self.shipment_data.groupby([
            'Origin Airport City', 'Destination Airport City'
        ])['Weights'].sum().reset_index()
        
        for _, row in shipment_volumes.iterrows():
            origin = row['Origin Airport City']
            dest = row['Destination Airport City']
            volume = row['Weights']
            
            if pd.notna(origin) and pd.notna(dest):
                key = (origin, dest)
                self.volume_matrix[key] = self.volume_matrix.get(key, 0) + volume
        
        print(f"   ✓ Built volume matrix with {len(self.volume_matrix)} airport pairs")
    
    def find_optimal_routes(self, top_n_per_pair: int = 5) -> pd.DataFrame:
        """
        Find optimal indirect routes for all no-flight OD pairs.
        
        Args:
            top_n_per_pair: Number of top routes to return per OD pair
            
        Returns:
            DataFrame with ranked route recommendations
        """
        print(f"\nFinding optimal indirect routes...")
        
        all_routes = []
        
        for idx, od_pair in self.no_flight_pairs.iterrows():
            origin_branch = od_pair['Origin Branch']
            dest_branch = od_pair['Destination Branch']
            origin_airport = od_pair['Origin Airport City']
            dest_airport = od_pair['Destination Airport City']
            
            print(f"Processing {origin_branch} → {dest_branch} (via {origin_airport} → {dest_airport})")
            
            # Find all possible middle airports
            routes = self._find_routes_via_middle_airports(
                origin_branch, dest_branch, origin_airport, dest_airport
            )
            
            # Add to results
            all_routes.extend(routes)
        
        if not all_routes:
            print("❌ No valid indirect routes found")
            return pd.DataFrame()
        
        # Convert to DataFrame and rank
        results_df = pd.DataFrame(all_routes)
        
        # Calculate composite scores and rank
        results_df = self._calculate_scores(results_df)
        
        # Sort by composite score (descending)
        results_df = results_df.sort_values('Composite Score', ascending=False)
        
        # Add suggestion flags (top routes per OD pair)
        results_df = self._add_suggestion_flags(results_df, top_n_per_pair)
        
        print(f"✅ Found {len(results_df)} total route options")
        print(f"✅ Marked {len(results_df[results_df['Suggested?'] == 'Yes'])} as suggested routes")
        
        return results_df
    
    def _find_routes_via_middle_airports(self, origin_branch: str, dest_branch: str, 
                                       origin_airport: str, dest_airport: str) -> List[Dict]:
        """Find all valid routes via middle airports for a given OD pair."""
        routes = []
        
        # Get all potential middle airports (cities with both air and road connections)
        potential_middle_airports = self._get_potential_middle_airports(origin_airport)
        
        for middle_airport in potential_middle_airports:
            # Check if this creates a valid route
            route_info = self._validate_and_score_route(
                origin_branch, dest_branch, origin_airport, 
                dest_airport, middle_airport
            )
            
            if route_info:
                routes.append(route_info)
        
        return routes
    
    def _get_potential_middle_airports(self, origin_airport: str) -> List[str]:
        """Get all airports that have air connections from origin airport."""
        middle_airports = []
        
        for origin, dest in self.valid_air_routes:
            if origin == origin_airport:
                middle_airports.append(dest)
        
        return list(set(middle_airports))  # Remove duplicates
    
    def _validate_and_score_route(self, origin_branch: str, dest_branch: str,
                                 origin_airport: str, dest_airport: str, 
                                 middle_airport: str) -> Optional[Dict]:
        """Validate and score a specific indirect route."""
        
        # 1. Check air route feasibility: Origin Airport → Middle Airport
        if (origin_airport, middle_airport) not in self.valid_air_routes:
            return None
        
        # 2. Find destination hub from shipment data
        dest_hub = self._get_destination_hub(dest_branch)
        if not dest_hub:
            return None
        
        # 3. Check road feasibility: Middle Airport → Destination Hub
        road_info = self._get_road_info(middle_airport, dest_hub)
        if not road_info:
            return None
        
        # 4. Calculate volumes
        volume_air = self.volume_matrix.get((origin_airport, middle_airport), 0)
        volume_road = self.volume_matrix.get((middle_airport, dest_hub), 0)
        total_volume = volume_air + volume_road
        
        # 5. Calculate times
        air_time = self._get_air_time(origin_airport, middle_airport)
        road_time = road_info['duration']
        full_route_time = air_time + road_time
        
        # 6. Get direct road time for comparison
        direct_road_time = self._get_direct_road_time(origin_branch, dest_branch)
        time_saved = max(0, direct_road_time - full_route_time)
        
        # 7. Build suggested route string
        suggested_route = f"{origin_branch} → {origin_airport} → {middle_airport} (air) → {dest_hub} → {dest_branch}"
        
        return {
            'Origin Branch': origin_branch,
            'Destination Branch': dest_branch,
            'Origin Airport': origin_airport,
            'Middle Airport': middle_airport,
            'Destination Hub': dest_hub,
            'Suggested Route': suggested_route,
            'Total Air Time': air_time,
            'Road Time (M-Air to D-Hub)': road_time,
            'Full Route Time': full_route_time,
            'Road-only Time': direct_road_time,
            'Time Saved (hrs)': time_saved,
            'Total Combined Volume': total_volume,
            'Volume Air Segment': volume_air,
            'Volume Road Segment': volume_road,
            'Distance Saved (km)': self._calculate_distance_saved(origin_branch, dest_branch, middle_airport, dest_hub)
        }
    
    def _get_destination_hub(self, dest_branch: str) -> Optional[str]:
        """Get destination hub for a given destination branch."""
        # Look up in shipment data
        hub_matches = self.shipment_data[
            self.shipment_data['Destination Branch'] == dest_branch
        ]['Destination Hub'].dropna().unique()
        
        if len(hub_matches) > 0:
            return hub_matches[0]
        
        # If not found in shipment data, try to extract from destination hub city
        hub_city_matches = self.shipment_data[
            self.shipment_data['Destination Branch'] == dest_branch
        ]['Destination Hub City'].dropna().unique()
        
        return hub_city_matches[0] if len(hub_city_matches) > 0 else None
    
    def _get_road_info(self, origin: str, destination: str) -> Optional[Dict]:
        """Get road connection info between two cities."""
        if origin in self.road_connections and destination in self.road_connections[origin]:
            return self.road_connections[origin][destination]
        return None
    
    def _get_air_time(self, origin_airport: str, dest_airport: str) -> float:
        """Estimate air travel time (including 4hr buffer)."""
        # Simple estimation: base flight time + 4hr buffer
        # In real implementation, you might have actual flight times
        base_flight_time = 2.0  # Assume 2 hours average domestic flight
        buffer_time = 4.0       # 4 hour buffer as specified
        return base_flight_time + buffer_time
    
    def _get_direct_road_time(self, origin_branch: str, dest_branch: str) -> float:
        """Get direct road time between origin and destination branches."""
        # Extract city names from branch codes to match with road matrix
        origin_city = self._extract_city_from_branch(origin_branch)
        dest_city = self._extract_city_from_branch(dest_branch)
        
        # Try direct lookup first
        road_info = self._get_road_info(origin_city, dest_city)
        if road_info:
            return road_info['duration']
        
        # Try reverse lookup
        road_info = self._get_road_info(dest_city, origin_city)
        if road_info:
            return road_info['duration']
            
        return 24.0  # Default 24hrs if no data found
    
    def _extract_city_from_branch(self, branch_name: str) -> str:
        """Extract city name from branch code for road matrix lookup."""
        # Common city extractions based on your data patterns
        if 'AHMEDABAD' in branch_name.upper():
            return 'AHMEDABAD'
        elif 'BARODA' in branch_name.upper():
            return 'BARODA'
        elif 'SURAT' in branch_name.upper():
            return 'SURAT'
        elif 'RAJKOT' in branch_name.upper():
            return 'RAJKOT'
        elif 'VAPI' in branch_name.upper():
            return 'VAPI'
        elif 'ANKLESHWAR' in branch_name.upper():
            return 'ANKLESHWAR'
        
        # If no match found, try to extract from shipment data
        city_matches = self.shipment_data[
            self.shipment_data['Origin Branch'] == branch_name
        ]['Origin Hub City'].dropna().unique()
        
        if len(city_matches) > 0:
            return city_matches[0]
            
        # Last resort: try destination branch
        dest_city_matches = self.shipment_data[
            self.shipment_data['Destination Branch'] == branch_name
        ]['Destination Hub City'].dropna().unique()
        
        return dest_city_matches[0] if len(dest_city_matches) > 0 else branch_name
    
    def _calculate_distance_saved(self, origin_branch: str, dest_branch: str, 
                                middle_airport: str, dest_hub: str) -> float:
        """Calculate distance saved compared to direct road route."""
        # Extract cities from branch names
        origin_city = self._extract_city_from_branch(origin_branch)
        dest_city = self._extract_city_from_branch(dest_branch)
        
        # Get direct road distance
        direct_road_info = self._get_road_info(origin_city, dest_city)
        if not direct_road_info:
            direct_road_info = self._get_road_info(dest_city, origin_city)
        direct_distance = direct_road_info['distance'] if direct_road_info else 1000
        
        # Get indirect route distance (road segment only: middle_airport to dest_hub)
        indirect_road_info = self._get_road_info(middle_airport, dest_hub)
        if not indirect_road_info:
            indirect_road_info = self._get_road_info(dest_hub, middle_airport)
        indirect_distance = indirect_road_info['distance'] if indirect_road_info else 500
        
        # Add origin to middle airport road distance (approximation)
        origin_to_middle_info = self._get_road_info(origin_city, middle_airport)
        if not origin_to_middle_info:
            origin_to_middle_info = self._get_road_info(middle_airport, origin_city)
        origin_to_middle_distance = origin_to_middle_info['distance'] if origin_to_middle_info else 300
        
        total_indirect_distance = origin_to_middle_distance + indirect_distance
        
        return max(0, direct_distance - total_indirect_distance)
    
    def _calculate_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate normalized scores and composite score."""
        
        # Normalize volume scores (0-1)
        max_volume = df['Total Combined Volume'].max() if df['Total Combined Volume'].max() > 0 else 1
        df['Volume_Score'] = df['Total Combined Volume'] / max_volume
        
        # Normalize time saving scores (0-1)
        max_time_saved = df['Time Saved (hrs)'].max() if df['Time Saved (hrs)'].max() > 0 else 1
        df['Time_Saving_Score'] = df['Time Saved (hrs)'] / max_time_saved
        
        # Normalize distance saving scores (0-1)
        max_distance_saved = df['Distance Saved (km)'].max() if df['Distance Saved (km)'].max() > 0 else 1
        df['Distance_Saving_Score'] = df['Distance Saved (km)'] / max_distance_saved
        
        # Calculate composite score
        df['Composite Score'] = (
            self.weights['volume'] * df['Volume_Score'] +
            self.weights['time'] * df['Time_Saving_Score'] +
            self.weights['distance'] * df['Distance_Saving_Score']
        )
        
        return df
    
    def _add_suggestion_flags(self, df: pd.DataFrame, top_n: int) -> pd.DataFrame:
        """Add suggestion flags for top N routes per OD pair."""
        df['Suggested?'] = 'No'
        
        # Group by OD pair and mark top N routes
        for (origin, dest), group in df.groupby(['Origin Branch', 'Destination Branch']):
            top_indices = group.nlargest(top_n, 'Composite Score').index
            df.loc[top_indices, 'Suggested?'] = 'Yes'
        
        return df
    
    def export_results(self, results_df: pd.DataFrame, output_path: str):
        """Export results to CSV file."""
        # Select and order final columns
        final_columns = [
            'Origin Branch', 'Destination Branch', 'Origin Airport', 'Middle Airport',
            'Destination Hub', 'Suggested Route', 'Total Air Time', 'Road Time (M-Air to D-Hub)',
            'Full Route Time', 'Road-only Time', 'Time Saved (hrs)', 'Total Combined Volume',
            'Composite Score', 'Suggested?'
        ]
        
        # Only include columns that exist in the DataFrame
        available_columns = [col for col in final_columns if col in results_df.columns]
        
        export_df = results_df[available_columns].round(2)
        export_df.to_csv(output_path, index=False)
        print(f"✅ Results exported to {output_path}")
        
        return export_df

# Usage Example
def main():
    """
    Main function demonstrating how to use the LogisticsRouteOptimizer
    """
    
    # Initialize optimizer with custom weights if needed
    optimizer = LogisticsRouteOptimizer(
        weights={'volume': 0.4, 'time': 0.4, 'distance': 0.2}
    )
    
    # Define file paths (update these with your actual file paths)
    file_paths = {
        'shipment_data': '/Users/devashish/sample/Alternate_rotues/new_to_look.csv',
        'airport_volume': '/Users/devashish/sample/Alternate_rotues/airport_volume_to_volume.csv',
        'air_routes': '/Users/devashish/sample/Alternate_rotues/air_routes.csv',
        'road_matrix': '/Users/devashish/sample/Alternate_rotues/city_distances_google.csv',
        'no_flight_pairs': '/Users/devashish/sample/Alternate_rotues/no_flight_data.csv',  # Updated filename
        'city_coordinates': '/Users/devashish/sample/Alternate_rotues/city_coordinate.csv'  # Optional
    }
    
    try:
        # Step 1: Load all data
        optimizer.load_data(file_paths)
        
        # Step 2: Preprocess data
        optimizer.preprocess_data()
        
        # Step 3: Find optimal routes
        results = optimizer.find_optimal_routes(top_n_per_pair=3)
        
        # Step 4: Export results
        if not results.empty:
            final_results = optimizer.export_results(results, 'optimal_indirect_routes.csv')
            
            # Display summary
            print(f"\n📊 SUMMARY:")
            print(f"Total route options found: {len(results)}")
            print(f"Suggested routes: {len(results[results['Suggested?'] == 'Yes'])}")
            print(f"Average composite score: {results['Composite Score'].mean():.3f}")
            print(f"Top route: {results.iloc[0]['Suggested Route']}")
            
            return final_results
        else:
            print("❌ No viable routes found")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Error in optimization process: {str(e)}")
        return pd.DataFrame()

if __name__ == "__main__":
    results = main()