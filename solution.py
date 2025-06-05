import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class CGPoint:
    mac: float  # mac (mean aerodynamic chord)
    weight: float  # weight
    env: int
    
    def __repr__(self):
        return f"CGPoint(weight={self.weight:.0f}, mac={self.mac:.2f}, env={self.env:.2f})"
    
    def __hash__(self):
        return hash((round(self.mac, 2), round(self.weight, 0), round(self.env, 0)))
    
    def __eq__(self, other):
        if not isinstance(other, CGPoint):
            return False
        return (round(self.mac, 2) == round(other.mac, 2) and 
                round(self.weight, 0) == round(other.weight, 0))

class EnvelopeIntersectionAnalyzer:
    """Main class for analyzing envelope intersections and finding optimal paths."""
    
    def __init__(self, env1: List[CGPoint], env2: List[CGPoint]):
        self.env1_original = env1.copy()
        self.env2_original = env2.copy()
        self.intersections = []
        self.env1_segments = []
        self.env2_segments = []
        self.connections = {}
        self.interpolated_points = []  # Track interpolated points
        
        # Automatically analyze intersections and create connection graph
        self._analyze_intersections()
        self._add_weight_interpolations()  # New method for weight interpolations
        self._create_connection_graph()
    
    @staticmethod
    def line_intersection(p1: CGPoint, p2: CGPoint, p3: CGPoint, p4: CGPoint) -> Optional[CGPoint]:
        """Find intersection point between two line segments if it exists."""
        mac1, weight1 = p1.mac, p1.weight
        mac2, weight2 = p2.mac, p2.weight
        mac3, weight3 = p3.mac, p3.weight
        mac4, weight4 = p4.mac, p4.weight

        # Calculate the direction vectors
        denom = (mac1 - mac2) * (weight3 - weight4) - (weight1 - weight2) * (mac3 - mac4)

        # Lines are parallel if denominator is 0
        if abs(denom) < 1e-10:
            return None

        # Calculate parameters
        t = ((mac1 - mac3) * (weight3 - weight4) - (weight1 - weight3) * (mac3 - mac4)) / denom
        u = -((mac1 - mac2) * (weight1 - weight3) - (weight1 - weight2) * (mac1 - mac3)) / denom

        # Check if intersection is within both line segments
        if 0 <= t <= 1 and 0 <= u <= 1:
            # Calculate intersection point
            i_mac = mac1 + t * (mac2 - mac1)
            i_weight = weight1 + t * (weight2 - weight1)
            return CGPoint(i_mac, i_weight, 0)

        return None

    def find_all_intersections(self, env1: List[CGPoint], env2: List[CGPoint]) -> List[Tuple[int, int, CGPoint]]:
        """Find all intersections between segments of two envelopes."""
        intersections = []

        for i in range(len(env1) - 1):
            for j in range(len(env2) - 1):
                intersection = self.line_intersection(env1[i], env1[i+1], env2[j], env2[j+1])
                if intersection:
                    intersections.append((i, j, intersection))

        return intersections

    def interpolate_mac_at_weight(self, target_weight: float, point1: CGPoint, point2: CGPoint, epsilon=1e-6) -> Optional[float]:
        """Interpolate MAC value at a specific weight between two points."""
        x1, y1 = point1.mac, point1.weight
        x2, y2 = point2.mac, point2.weight
        
        # Check if target weight is within the segment range
        if not (min(y1, y2) <= target_weight <= max(y1, y2)):
            return None
            
        # Handle vertical line case (same weight)
        if abs(y2 - y1) < epsilon:
            return x1 if abs(target_weight - y1) < epsilon else None
        
        # Linear interpolation: x = x1 + (x2-x1) * (target_weight-y1)/(y2-y1)
        interpolated_mac = x1 + (x2 - x1) * (target_weight - y1) / (y2 - y1)
        return interpolated_mac

    def find_interpolation_points(self, env1: List[CGPoint], env2: List[CGPoint]) -> List[Tuple[int, int, CGPoint, str]]:
        """Find points where we need to interpolate to connect envelopes at specific weights."""
        interpolation_points = []
        
        # Get all unique weights from both envelopes
        env1_weights = set(point.weight for point in env1)
        env2_weights = set(point.weight for point in env2)
        
        # Get weight ranges for each envelope
        env1_min_weight = min(point.weight for point in env1)
        env1_max_weight = max(point.weight for point in env1)
        env2_min_weight = min(point.weight for point in env2)
        env2_max_weight = max(point.weight for point in env2)
        
        # Find weights that exist in one envelope but not the other, within valid ranges
        weights_only_in_env1 = env1_weights - env2_weights
        weights_only_in_env2 = env2_weights - env1_weights
        
        # For each weight only in env2, try to interpolate in env1 (if within env1's range)
        for target_weight in weights_only_in_env2:
            if env1_min_weight <= target_weight <= env1_max_weight:
                for i in range(len(env1) - 1):
                    interpolated_mac = self.interpolate_mac_at_weight(target_weight, env1[i], env1[i+1])
                    if interpolated_mac is not None:
                        interpolated_point = CGPoint(interpolated_mac, target_weight, 1)  # Keep env1's envelope number
                        interpolation_points.append((i, -1, interpolated_point, "env1_interpolated"))
        
        # For each weight only in env1, try to interpolate in env2 (if within env2's range)
        for target_weight in weights_only_in_env1:
            if env2_min_weight <= target_weight <= env2_max_weight:
                for j in range(len(env2) - 1):
                    interpolated_mac = self.interpolate_mac_at_weight(target_weight, env2[j], env2[j+1])
                    if interpolated_mac is not None:
                        interpolated_point = CGPoint(interpolated_mac, target_weight, 2)  # Keep env2's envelope number
                        interpolation_points.append((-1, j, interpolated_point, "env2_interpolated"))
        
        # Add interpolation points at envelope boundaries to enable cross-envelope connections
        # If env1 starts later, interpolate env2 at env1's starting weight
        if env1_min_weight > env2_min_weight and env1_min_weight <= env2_max_weight:
            if env1_min_weight not in env2_weights:
                for j in range(len(env2) - 1):
                    interpolated_mac = self.interpolate_mac_at_weight(env1_min_weight, env2[j], env2[j+1])
                    if interpolated_mac is not None:
                        interpolated_point = CGPoint(interpolated_mac, env1_min_weight, 2)
                        interpolation_points.append((-1, j, interpolated_point, "env2_boundary"))
        
        # If env2 starts later, interpolate env1 at env2's starting weight
        if env2_min_weight > env1_min_weight and env2_min_weight <= env1_max_weight:
            if env2_min_weight not in env1_weights:
                for i in range(len(env1) - 1):
                    interpolated_mac = self.interpolate_mac_at_weight(env2_min_weight, env1[i], env1[i+1])
                    if interpolated_mac is not None:
                        interpolated_point = CGPoint(interpolated_mac, env2_min_weight, 1)
                        interpolation_points.append((i, -1, interpolated_point, "env1_boundary"))
        
        # Add interpolation points at ending boundaries
        # If env1 ends earlier, interpolate env2 at env1's ending weight
        if env1_max_weight < env2_max_weight and env1_max_weight >= env2_min_weight:
            if env1_max_weight not in env2_weights:
                for j in range(len(env2) - 1):
                    interpolated_mac = self.interpolate_mac_at_weight(env1_max_weight, env2[j], env2[j+1])
                    if interpolated_mac is not None:
                        interpolated_point = CGPoint(interpolated_mac, env1_max_weight, 2)
                        interpolation_points.append((-1, j, interpolated_point, "env2_end_boundary"))
        
        # If env2 ends earlier, interpolate env1 at env2's ending weight
        if env2_max_weight < env1_max_weight and env2_max_weight >= env1_min_weight:
            if env2_max_weight not in env1_weights:
                for i in range(len(env1) - 1):
                    interpolated_mac = self.interpolate_mac_at_weight(env2_max_weight, env1[i], env1[i+1])
                    if interpolated_mac is not None:
                        interpolated_point = CGPoint(interpolated_mac, env2_max_weight, 1)
                        interpolation_points.append((i, -1, interpolated_point, "env1_end_boundary"))
        
        return interpolation_points

    @staticmethod
    def distance_along_segment(start: CGPoint, end: CGPoint, point: CGPoint) -> float:
        """Calculate the distance from start to point along the segment from start to end."""
        # Use parametric distance (t parameter from line equation)
        if abs(end.mac - start.mac) > 1e-10:
            t = (point.mac - start.mac) / (end.mac - start.mac)
        else:
            t = (point.weight - start.weight) / (end.weight - start.weight)
        return t

    @staticmethod
    def split_envelope_at_intersections(envelope: List[CGPoint], intersections: List[Tuple[int, CGPoint]]) -> List[Tuple[CGPoint, CGPoint, str]]:
        """Split envelope segments at intersection points and return list of segments."""
        segments = []

        # Group intersections by segment index
        intersections_by_segment = {}
        for seg_idx, point in intersections:
            if seg_idx not in intersections_by_segment:
                intersections_by_segment[seg_idx] = []
            intersections_by_segment[seg_idx].append(point)

        for current_seg in range(len(envelope) - 1):
            start_point = envelope[current_seg]
            end_point = envelope[current_seg + 1]

            if current_seg not in intersections_by_segment:
                # No intersections, add segment as is
                segments.append((start_point, end_point, "direct"))
            else:
                # Sort intersections by their position along the segment
                seg_intersections = intersections_by_segment[current_seg]
                seg_intersections.sort(key=lambda p: EnvelopeIntersectionAnalyzer.distance_along_segment(start_point, end_point, p))

                # Add segments split by intersections
                prev_point = start_point
                for intersection in seg_intersections:
                    segments.append((prev_point, intersection, "to_intersection"))
                    prev_point = intersection
                segments.append((prev_point, end_point, "from_intersection"))

        return segments
    
    def _analyze_intersections(self):
        """Internal method to find intersections and split envelopes."""
        # Find all intersections
        self.intersections = self.find_all_intersections(self.env1_original, self.env2_original)
        
        # Prepare intersections for each envelope
        env1_intersections = [(seg1, point) for seg1, seg2, point in self.intersections]
        env2_intersections = [(seg2, point) for seg1, seg2, point in self.intersections]
        
        # Split envelopes at intersections
        self.env1_segments = self.split_envelope_at_intersections(self.env1_original, env1_intersections)
        self.env2_segments = self.split_envelope_at_intersections(self.env2_original, env2_intersections)
    
    def _add_weight_interpolations(self):
        """Add interpolated points at weights where envelopes don't naturally intersect."""
        # Find interpolation points
        interpolation_points = self.find_interpolation_points(self.env1_original, self.env2_original)
        
        # Get existing intersection points for re-splitting
        existing_env1_intersections = [(seg1, point) for seg1, seg2, point in self.intersections]
        existing_env2_intersections = [(seg2, point) for seg1, seg2, point in self.intersections]
        
        # Process interpolation points for env1
        env1_interpolations = [(seg1, point) for seg1, seg2, point, interp_type in interpolation_points 
                              if "env1" in interp_type and seg1 >= 0]
        
        # Process interpolation points for env2
        env2_interpolations = [(seg2, point) for seg1, seg2, point, interp_type in interpolation_points 
                              if "env2" in interp_type and seg2 >= 0]
        
        # Re-split envelopes including both intersection and interpolation points
        if env1_interpolations or existing_env1_intersections:
            all_env1_points = existing_env1_intersections + env1_interpolations
            self.env1_segments = self.split_envelope_at_intersections(self.env1_original, all_env1_points)
        
        if env2_interpolations or existing_env2_intersections:
            all_env2_points = existing_env2_intersections + env2_interpolations
            self.env2_segments = self.split_envelope_at_intersections(self.env2_original, all_env2_points)
        
        # Store interpolated points for visualization
        self.interpolated_points = [point for seg1, seg2, point, interp_type in interpolation_points]
        
        print(f"Added {len(self.interpolated_points)} interpolated points for weight connections")
        
        # Print interpolation details
        env1_min = min(p.weight for p in self.env1_original)
        env1_max = max(p.weight for p in self.env1_original)
        env2_min = min(p.weight for p in self.env2_original)
        env2_max = max(p.weight for p in self.env2_original)
        print(f"Env1 weight range: {env1_min} - {env1_max}")
        print(f"Env2 weight range: {env2_min} - {env2_max}")
        print(f"Overall weight range: {min(env1_min, env2_min)} - {max(env1_max, env2_max)}")
    
    def _create_connection_graph(self):
        """Create a dictionary mapping each point to the points it connects to."""
        self.connections = {}

        # Add connections from envelope segments
        for segments in self.env1_segments:
            start, end, _ = segments
            if start not in self.connections:
                self.connections[start] = set()
            self.connections[start].add(end)
            if end not in self.connections:
                self.connections[end] = set()

        for segments in self.env2_segments:
            start, end, _ = segments
            if start not in self.connections:
                self.connections[start] = set()
            self.connections[start].add(end)
            if end not in self.connections:
                self.connections[end] = set()

        # Add cross-envelope connections at same weights
        all_points = list(self.connections.keys())
        for i, point1 in enumerate(all_points):
            for j, point2 in enumerate(all_points):
                if i != j and abs(point1.weight - point2.weight) < 0.01:  # Same weight
                    if point1.env != point2.env or point1.env == 0 or point2.env == 0:  # Different envelopes or intersection points
                        self.connections[point1].add(point2)
                        self.connections[point2].add(point1)

        # Convert sets to sorted lists
        for key in self.connections:
            self.connections[key] = sorted(list(self.connections[key]), 
                                         key=lambda p: (p.weight, -p.mac))
        
        print(f"Created connection graph with {len(self.connections)} nodes")
    
    def get_points_at_weight(self, target_weight: float) -> List[CGPoint]:
        """Get all points at a specific weight, sorted by MAC descending."""
        points_at_weight = []
        
        for point in self.connections.keys():
            if abs(point.weight - target_weight) < 0.01:
                points_at_weight.append(point)
        
        points_at_weight.sort(key=lambda p: p.mac, reverse=True)
        return points_at_weight
    
    def find_closest_point(self, target_point: CGPoint) -> CGPoint:
        """Find the closest point in the connections dictionary to the target point."""
        if target_point in self.connections:
            return target_point
        
        min_distance = float('inf')
        closest_point = None
        
        for point in self.connections.keys():
            distance = ((point.mac - target_point.mac) ** 2 + 
                       (point.weight - target_point.weight) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_point = point
        
        return closest_point
    
    def find_optimal_path(self, start_point: CGPoint, end_point: CGPoint) -> List[CGPoint]:
        """
        Find the optimal path (highest MAC values) from start_point to end_point.
        """
        path = []
        
        # Find the actual starting point - lowest weight across all envelopes
        all_points = list(self.connections.keys())
        min_weight = min(point.weight for point in all_points)
        max_weight = max(point.weight for point in all_points)
        
        # Get all points at minimum weight and choose the one with highest MAC
        starting_candidates = [point for point in all_points if abs(point.weight - min_weight) < 0.01]
        current_point = max(starting_candidates, key=lambda p: p.mac)
        
        print(f"Starting from point: {current_point} (min weight: {min_weight})")
        
        # Find the closest actual point to our start point
        current_env = current_point.env
        path.append(current_point)
        
        while current_point.weight < max_weight:
            # Get outgoing connections from current point
            outgoing = self.connections.get(current_point, [])
            print()
            print("CURRENT WEIGHT", current_point.weight)
            print("outgoing", outgoing)
            print("current_point", current_point)
            
            if not outgoing:
                break
            
            # Find connections that lead to higher weight
            next_candidates = [p for p in outgoing if p.weight > current_point.weight]
            
            if not next_candidates:
                # Edge case where if we start with same macs or if we interpolated to this point
                for point in outgoing:
                    for pt in self.connections.get(point, []):
                        if pt.weight > current_point.weight:
                            next_candidates.append(point)
                
                if not next_candidates:
                    break
            
            # Sort by MAC descending
            next_candidates.sort(key=lambda p: -p.mac)
            next_point = next_candidates[0]
            current_env = next_point.env
            
            # Add the connected point to path
            path.append(next_point)
            current_point = next_point
            
            # Check for better MAC options at this weight level
            points_at_weight = self.get_points_at_weight(next_point.weight)

            # Look for higher MAC option at this weight
            for point in points_at_weight:
                # Only consider points of intersection or in the same envelope
                if current_env == 0 or point.env == current_env or point.env == 0:
                    # Check if this higher MAC point can progress further
                    higher_outgoing = self.connections.get(point, [])
                    can_progress = any(p.weight > current_point.weight for p in higher_outgoing)
                    if current_point.weight >= max_weight or can_progress:
                        path.append(point)
                        current_point = point
                        break
                        
            # Check to see if the current weight is max. If not add mac and add outgoing again for the same env
            filtered_points = [point for point in points_at_weight
                               if current_env == 0 or point.env == current_env or point.env == 0]
            max_at_weight = max(filtered_points, key=lambda point: point.mac, default=current_point)
            if max_at_weight == current_point:
                continue
            else:
                # Add max point at that weight
                path.append(max_at_weight)
                # add the current_point back
                path.append(current_point)
            
            if current_point.weight >= max_weight:
                break
        
        return path
    
    def plot_analysis(self, optimal_path: List[CGPoint] = None, title: str = "Envelope Analysis"):
        """Plot the envelopes, intersections, and optimal path."""
        plt.figure(figsize=(14, 10))
        
        # Plot original envelopes
        env1_mac = [point.mac for point in self.env1_original]
        env1_weight = [point.weight for point in self.env1_original]
        env2_mac = [point.mac for point in self.env2_original]
        env2_weight = [point.weight for point in self.env2_original]
        
        plt.plot(env1_mac, env1_weight, 'b-', linewidth=2, label='Envelope 1', 
                marker='o', markersize=6)
        plt.plot(env2_mac, env2_weight, 'g-', linewidth=2, label='Envelope 2', 
                marker='s', markersize=6)
        
        # Plot intersection points
        if self.intersections:
            intersection_points = [point for _, _, point in self.intersections]
            intersection_mac = [point.mac for point in intersection_points]
            intersection_weight = [point.weight for point in intersection_points]
            plt.plot(intersection_mac, intersection_weight, 'rx', markersize=12, 
                    markeredgewidth=3, label='Intersections', alpha=0.7)
        
        # Plot interpolated points
        if self.interpolated_points:
            interp_mac = [point.mac for point in self.interpolated_points]
            interp_weight = [point.weight for point in self.interpolated_points]
            plt.plot(interp_mac, interp_weight, 'mo', markersize=10, 
                    alpha=0.7, label='Interpolated Points')
        
        # Plot optimal path if provided
        if optimal_path:
            path_mac = [point.mac for point in optimal_path]
            path_weight = [point.weight for point in optimal_path]
            plt.plot(path_mac, path_weight, 'ro-', linewidth=3, markersize=8, 
                    alpha=0.7, label='Limiting')
        
        plt.xlabel('MAC (Mean Aerodynamic Chord)')
        plt.ylabel('Weight')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def print_summary(self, optimal_path: List[CGPoint] = None):
        """Print a comprehensive summary of the analysis."""
        print(f"\n" + "="*80)
        print("ENVELOPE INTERSECTION ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"Envelope 1: {len(self.env1_original)} points")
        print(f"Envelope 2: {len(self.env2_original)} points")
        print(f"Intersections found: {len(self.intersections)}")
        print(f"Interpolated points added: {len(self.interpolated_points)}")
        print(f"Envelope 1 split into: {len(self.env1_segments)} segments")
        print(f"Envelope 2 split into: {len(self.env2_segments)} segments")
        print(f"Total unique points in graph: {len(self.connections)}")
        
        if self.intersections:
            print(f"\nIntersection Points:")
            for i, (seg1, seg2, point) in enumerate(self.intersections):
                print(f"  {i+1}: {point} (Env1 seg {seg1} ↔ Env2 seg {seg2})")
        
        if self.interpolated_points:
            print(f"\nInterpolated Points:")
            for i, point in enumerate(self.interpolated_points):
                print(f"  {i+1}: {point}")
        
        if optimal_path:
            print(f"\nOptimal Path ({len(optimal_path)} points):")
            for i, point in enumerate(optimal_path):
                print(f"  Step {i+1}: {point} (MAC: {point.mac:.2f})")
            
            total_mac = sum(point.mac for point in optimal_path)
            avg_mac = total_mac / len(optimal_path)
            print(f"\nPath Statistics:")
            print(f"  Average MAC: {avg_mac:.2f}")
            print(f"  Weight range: {optimal_path[0].weight:.0f} → {optimal_path[-1].weight:.0f}")


# Test with different weight ranges - env1 no longer has 150000
env1 = [
    CGPoint(weight=150000, mac=21.00, env=1),
    CGPoint(weight=200000, mac=21.00, env=1),
    CGPoint(weight=200000, mac=16.00, env=1),
    CGPoint(weight=250000, mac=17.00, env=1),
    CGPoint(weight=250000, mac=21.00, env=1),
    CGPoint(weight=300000, mac=22.00, env=1),
    CGPoint(weight=300000, mac=17.00, env=1),
]

env2 = [
    CGPoint(weight=100000, mac=15.00, env=2),
    CGPoint(weight=200000, mac=18.00, env=2),
    CGPoint(weight=250000, mac=14.00, env=2),
    CGPoint(weight=250000, mac=26.00, env=2),
    CGPoint(weight=250000, mac=19.00, env=2),
    CGPoint(weight=300000, mac=22.00, env=2)
]

# Find optimal path
start_point = env1[0]
end_point = env1[-1]

# Create analyzer instance
analyzer = EnvelopeIntersectionAnalyzer(env1, env2)

# Print basic analysis
print(f"Found {len(analyzer.intersections)} intersections:")
for i, (seg1, seg2, point) in enumerate(analyzer.intersections):
    print(f"Intersection {i+1}: Env1 segment {seg1} ↔ Env2 segment {seg2} at {point}")
print()

# Print segments
print("Envelope 1 Segments:")
for i, (start, end, seg_type) in enumerate(analyzer.env1_segments):
    print(f"Envelope1 Segment {i}: {start} -> {end} ({seg_type})")

print("\nEnvelope 2 Segments:")
for i, (start, end, seg_type) in enumerate(analyzer.env2_segments):
    print(f"Envelope2 Segment {i}: {start} -> {end} ({seg_type})")

# Print all connections
print(f"\nAll Point Connections:")
print("=" * 50)
for point, connected_points in sorted(analyzer.connections.items(), key=lambda x: (x[0].weight, x[0].mac)):
    print(f"{point} -> {len(connected_points)} connections:")
    for connected in connected_points:
        print(f"    {connected}")
    print()

optimal_path = analyzer.find_optimal_path(start_point, end_point)

# Print summary
analyzer.print_summary(optimal_path)

# Plot the analysis
analyzer.plot_analysis(optimal_path, "Aircraft CG Envelope Analysis with Weight Interpolation")

print(f"\nScript completed successfully!")
print(f"Total intersections found: {len(analyzer.intersections)}")
print(f"Total interpolated points: {len(analyzer.interpolated_points)}")
print(f"Envelope 1 split into {len(analyzer.env1_segments)} segments")
print(f"Envelope 2 split into {len(analyzer.env2_segments)} segments")
print(f"Total unique points in network: {len(analyzer.connections)}")

# env1 = [
#     CGPoint(weight=150000, mac=15,env=1),
#     CGPoint(weight=200000, mac=15,env=1), 
#     CGPoint(weight=200000, mac=26,env=1),
#     CGPoint(weight=300000, mac=25,env=1)
# ]

# env2 = [
#     CGPoint(weight=150000, mac=21,env=2),
#     CGPoint(weight=300000, mac=25,env=2)
# ]

# Find optimal path
# start_point = env2[0]
# end_point = env1[-1]

# env1 = [
#     CGPoint(weight=150000, mac=21.00,env=1),
#     CGPoint(weight=225000, mac=26.00,env=1),
#     CGPoint(weight=300000, mac=16.00,env=1)
# ]

# env2 = [
#     CGPoint(weight=150000, mac=15.00,env=2),
#     CGPoint(weight=200000, mac=18.00,env=2),
#     CGPoint(weight=250000, mac=14.00,env=2),
#     CGPoint(weight=250000, mac=26.00,env=2),
#     CGPoint(weight=250000, mac=19.00,env=2),
#     CGPoint(weight=300000, mac=22.00,env=2)
# ]
# # Find optimal path
# start_point = env1[0]
# end_point = env1[-1]

# env1 = [
#     CGPoint(weight=1200,mac=15,env=1),
#     CGPoint(weight=1500,mac=17,env=1),
#     CGPoint(weight=1500,mac=16,env=1),
#     CGPoint(weight=1500,mac=18,env=1),
#     CGPoint(weight=2200,mac=22,env=1),
#     CGPoint(weight=2800,mac=30,env=1),
# ]

# env2 = [
#      CGPoint(weight=1200,mac=16,env=2),
#      CGPoint(weight=1500,mac=15,env=2),
#      CGPoint(weight=2500,mac=25,env=2),
#      CGPoint(weight=2800,mac=31,env=2),
# ]
# # Find optimal path
# start_point = env2[0]
# end_point = env1[-1]

# env1 = [
#     CGPoint(weight=150000, mac=21.00,env=1),
#     CGPoint(weight=200000, mac=21.00,env=1),
#     CGPoint(weight=200000, mac=16.00,env=1),
#     CGPoint(weight=250000, mac=17.00,env=1),
#     CGPoint(weight=250000, mac=21.00,env=1),
#     CGPoint(weight=300000, mac=22.00,env=1),
#     CGPoint(weight=300000, mac=17.00,env=1),
# ]

# env2 = [
#     CGPoint(weight=150000, mac=15.00,env=2),
#     CGPoint(weight=200000, mac=18.00,env=2),
#     CGPoint(weight=250000, mac=14.00,env=2),
#     CGPoint(weight=250000, mac=26.00,env=2),
#     CGPoint(weight=250000, mac=19.00,env=2),
#     CGPoint(weight=300000, mac=22.00,env=2)
# ]

# # Find optimal path
# start_point = env1[0]
# end_point = env1[-1]


# env1 = [
#     CGPoint(weight=200000, mac=21.00,env=1),
#     CGPoint(weight=200000, mac=16.00,env=1),
#     CGPoint(weight=250000, mac=17.00,env=1),
#     CGPoint(weight=250000, mac=21.00,env=1),
#     CGPoint(weight=300000, mac=22.00,env=1),
#     CGPoint(weight=300000, mac=17.00,env=1),
# ]

# env2 = [
#     CGPoint(weight=200000, mac=15.00,env=2),
#     CGPoint(weight=300000, mac=20.00,env=2),
# ]

# # Find optimal path
# start_point = env1[0]
# end_point = env1[-1]

# env1 = [
# CGPoint(weight=160000, mac=16,env=1),
# CGPoint(weight=188500, mac=16,env=1),
# CGPoint(weight=197800, mac=15,env=1),
# CGPoint(weight=233000, mac=15,env=1),
# CGPoint(weight=245000, mac=15,env=1),
# CGPoint(weight=270000, mac=15,env=1),
# CGPoint(weight=280000, mac=19,env=1),
# CGPoint(weight=300000, mac=18,env=1),
# CGPoint(weight=300000, mac=17,env=1),
# CGPoint(weight=350000, mac=17,env=1),
# CGPoint(weight=354600, mac=18,env=1),
# CGPoint(weight=368000, mac=15,env=1),
# ]

# env2 = [
# CGPoint(weight=160000, mac=16,env=2),
# CGPoint(weight=188500, mac=16,env=2),
# CGPoint(weight=197800, mac=15,env=2),
# CGPoint(weight=233000, mac=15,env=2),
# CGPoint(weight=245000, mac=15,env=2),
# CGPoint(weight=270000, mac=15,env=2),
# CGPoint(weight=280000, mac=19,env=2),
# CGPoint(weight=300000, mac=21,env=2),
# CGPoint(weight=300000, mac=24,env=2),
# CGPoint(weight=350000, mac=17,env=2),
# CGPoint(weight=354600, mac=18,env=2),
# CGPoint(weight=368000, mac=23,env=2),
# ]

# # Find optimal path
# start_point = env1[0]
# end_point = env1[-1]

# env1 = [
# CGPoint(weight=160000, mac=16,env=1),
# CGPoint(weight=188500, mac=16,env=1),
# CGPoint(weight=197800, mac=15,env=1),
# CGPoint(weight=233000, mac=15,env=1),
# CGPoint(weight=245000, mac=15,env=1),
# CGPoint(weight=270000, mac=15,env=1),
# CGPoint(weight=280000, mac=19,env=1),
# CGPoint(weight=320000, mac=18,env=1),
# CGPoint(weight=320001, mac=15,env=1),
# CGPoint(weight=350000, mac=17,env=1),
# CGPoint(weight=354600, mac=18,env=1),
# CGPoint(weight=368000, mac=15,env=1),
# ]

# env2 = [
# CGPoint(weight=160000, mac=16,env=2),
# CGPoint(weight=188500, mac=16,env=2),
# CGPoint(weight=197800, mac=15,env=2),
# CGPoint(weight=233000, mac=15,env=2),
# CGPoint(weight=300000, mac=19,env=2),
# CGPoint(weight=300000, mac=21,env=2),
# CGPoint(weight=300000, mac=16,env=2),
# CGPoint(weight=350000, mac=17,env=2),
# CGPoint(weight=354600, mac=18,env=2),
# CGPoint(weight=368000, mac=15,env=2),
# ]

# # Find optimal path
# start_point = env1[0]
# end_point = env1[-1]
