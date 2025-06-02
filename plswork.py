import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class CGPoint:
    mac: float  # mac (mean aerodynamic chord)
    weight: float  # weight
    
    def __repr__(self):
        return f"CGPoint(weight={self.weight:.0f}, mac={self.mac:.2f})"
    
    def __hash__(self):
        return hash((round(self.mac, 2), round(self.weight, 0)))
    
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
        
        # Automatically analyze intersections and create connection graph
        self._analyze_intersections()
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
            return CGPoint(i_mac, i_weight)

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
    
    def _create_connection_graph(self):
        """Create a dictionary mapping each point to the points it connects to."""
        self.connections = {}
        
        for segments in [self.env1_segments, self.env2_segments]:
            for start, end, _ in segments:
                # Use CGPoint objects directly as keys
                if start not in self.connections:
                    self.connections[start] = set()
                
                self.connections[start].add(end)
                
                # Ensure end point exists in dictionary
                if end not in self.connections:
                    self.connections[end] = set()
        
        # Convert sets to sorted lists
        for key in self.connections:
            self.connections[key] = sorted(list(self.connections[key]), 
                                         key=lambda p: (p.weight, -p.mac))
    
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
    
    def find_optimal_path(self, start_point: CGPoint, end_point: CGPoint, 
                         verbose: bool = True) -> List[CGPoint]:
        """
        Find the optimal path (highest MAC values) from start_point to end_point.
        """
        path = []
        
        if verbose:
            print(f"\nFinding optimal path from {start_point} to {end_point}")
            print("=" * 60)
        
        # Find the closest actual point to our start point
        current_point = self.find_closest_point(start_point)
        if current_point != start_point and verbose:
            print(f"Start point adjusted to closest available: {current_point}")
        
        if verbose:
            print(f"Starting at: {current_point} (MAC: {current_point.mac:.2f}, Weight: {current_point.weight:.0f})")
        
        path.append(current_point)
        if verbose:
            print("connections", self.connections)
        
        while current_point.weight < end_point.weight:
            # Get outgoing connections from current point
            outgoing = self.connections.get(current_point, [])
            if verbose:
                print("current_point", current_point)
                print("outgoing", outgoing)
            
            if not outgoing:
                if verbose:
                    print(f"No outgoing connections from {current_point}")
                break
            
            # Find connections that lead to higher weight
            next_candidates = [p for p in outgoing if p.weight > current_point.weight]
            
            if not next_candidates:
                if verbose:
                    print(f"No connections to higher weight from {current_point}")
                break
            
            # Sort by weight first, then by MAC descending
            next_candidates.sort(key=lambda p: -p.mac)
            if verbose:
                print("next_candidates", next_candidates)
            next_point = next_candidates[0]
            
            if verbose:
                print(f"Moving to weight {next_point.weight:.0f}")
                print(f"Following connection: {current_point} → {next_point}")
            
            # Add the connected point to path
            path.append(next_point)
            current_point = next_point
            
            # Check for better MAC options at this weight level
            points_at_weight = self.get_points_at_weight(next_point.weight)
            
            if verbose and len(points_at_weight) > 1:
                print(f"Points available at weight {next_point.weight:.0f}:")
                for point in points_at_weight:
                    marker = "→ CURRENT" if point == next_point else "  Available"
                    print(f"  {marker}: {point} (MAC: {point.mac:.2f})")
            
            # Look for higher MAC option at this weight
            for point in points_at_weight:
                if point.mac > current_point.mac and point != current_point:
                    # Check if this higher MAC point can progress further
                    higher_outgoing = self.connections.get(point, [])
                    can_progress = any(p.weight > current_point.weight for p in higher_outgoing)
                    
                    if current_point.weight >= end_point.weight or can_progress:
                        if verbose:
                            print(f"  ↑ UPGRADING: Found better MAC option {point} (MAC: {point.mac:.2f})")
                        path.append(point)
                        current_point = point
                        break
                        
            #Check to see if the current weight is max. If not add mac and add outgoing again
            max_at_weight = max(points_at_weight, key=lambda x:x.mac)
            if max_at_weight == current_point:
                continue
            else:
                path.append(max_at_weight)
                path.append(current_point)
            
            if current_point.weight >= end_point.weight:
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
                    markeredgewidth=3, label='Intersections')
        
        # Plot optimal path if provided
        if optimal_path:
            path_mac = [point.mac for point in optimal_path]
            path_weight = [point.weight for point in optimal_path]
            plt.plot(path_mac, path_weight, 'ro-', linewidth=3, markersize=8, 
                    alpha=0.7, label='Optimal Path')
        
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
        print(f"Envelope 1 split into: {len(self.env1_segments)} segments")
        print(f"Envelope 2 split into: {len(self.env2_segments)} segments")
        print(f"Total unique points in graph: {len(self.connections)}")
        
        if self.intersections:
            print(f"\nIntersection Points:")
            for i, (seg1, seg2, point) in enumerate(self.intersections):
                print(f"  {i+1}: {point} (Env1 seg {seg1} ↔ Env2 seg {seg2})")
        
        if optimal_path:
            print(f"\nOptimal Path ({len(optimal_path)} points):")
            for i, point in enumerate(optimal_path):
                print(f"  Step {i+1}: {point} (MAC: {point.mac:.2f})")
            
            total_mac = sum(point.mac for point in optimal_path)
            avg_mac = total_mac / len(optimal_path)
            print(f"\nPath Statistics:")
            print(f"  Average MAC: {avg_mac:.2f}")
            print(f"  Weight range: {optimal_path[0].weight:.0f} → {optimal_path[-1].weight:.0f}")


# Define the envelopes

env1 = [
    CGPoint(weight=1200,mac=15),
    CGPoint(weight=1500,mac=17),
    CGPoint(weight=1500,mac=16),
    CGPoint(weight=1500,mac=18),
    CGPoint(weight=2200,mac=22),
    CGPoint(weight=2800,mac=30),
]

env2 = [
    CGPoint(weight=1200,mac=16),
    CGPoint(weight=1500,mac=15),
    CGPoint(weight=2500,mac=25),
    CGPoint(weight=2800,mac=31),
]

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

# Find optimal path
start_point = env2[0]
end_point = env2[-1]
optimal_path = analyzer.find_optimal_path(start_point, end_point, verbose=True)

# Print summary
analyzer.print_summary(optimal_path)

# Plot the analysis
analyzer.plot_analysis(optimal_path, "Aircraft CG Envelope Analysis")

print(f"\nScript completed successfully!")
print(f"Total intersections found: {len(analyzer.intersections)}")
print(f"Envelope 1 split into {len(analyzer.env1_segments)} segments")
print(f"Envelope 2 split into {len(analyzer.env2_segments)} segments")
print(f"Total unique points in network: {len(analyzer.connections)}")
# -- env1 = [
# --         CGPoint(weight=150000, mac=15),
# --         CGPoint(weight=200000, mac=15), 
# --         CGPoint(weight=200000, mac=26),
# --         CGPoint(weight=300000, mac=25)
# -- ]

# -- env2 = [
# --         CGPoint(weight=150000, mac=21),
# --         CGPoint(weight=300000, mac=25)
# -- ]



# -- env1 = [
# --     CGPoint(weight=150000, mac=21.00),
# --     CGPoint(weight=225000, mac=26.00),
# --     CGPoint(weight=300000, mac=16.00)
# -- ]

# -- env2 = [
# --     CGPoint(weight=150000, mac=15.00),
# --     CGPoint(weight=200000, mac=18.00),
# --     CGPoint(weight=250000, mac=14.00),
# --     CGPoint(weight=250000, mac=26.00),
# --     CGPoint(weight=250000, mac=19.00),
# --     CGPoint(weight=300000, mac=22.00)
# -- ]


# -- env1 = [
# --     CGPoint(weight=1200,mac=15),
# --     CGPoint(weight=1500,mac=17),
# --     CGPoint(weight=1500,mac=16),
# --     CGPoint(weight=1500,mac=18),
# --     CGPoint(weight=2200,mac=22),
# --     CGPoint(weight=2800,mac=30),
# -- ]

# -- env2 = [
# --     CGPoint(weight=1200,mac=16),
# --     CGPoint(weight=1500,mac=15),
# --     CGPoint(weight=2500,mac=25),
# --     CGPoint(weight=2800,mac=31),
# -- ]