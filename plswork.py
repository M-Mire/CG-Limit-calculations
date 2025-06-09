import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class CGPoint:
    mac: float
    weight: float 
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
            return CGPoint(i_mac, i_weight,0)

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


    def interpolate_mac(self, target_weight: float, point1: CGPoint, point2: CGPoint, epsilon=1e-6) -> Optional[float]:
        """Interpolate MAC value at a specific weight between two points."""
        w1, w2 = point1.weight, point2.weight
        mac1, mac2 = point1.mac, point2.mac
        
        # Handle vertical line (same weight)
        if abs(w2 - w1) < epsilon:
            return mac1 if abs(target_weight - w1) < epsilon else mac2
        
        # Linear interpolation: mac = mac1 + (mac2 - mac1) * (target_weight - w1) / (w2 - w1)
        interpolated_mac = mac1 + (mac2 - mac1) * (target_weight - w1) / (w2 - w1)
        return interpolated_mac
        
        

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

        first_env1_start, first_env1_end, _ = self.env1_segments[0]
        first_env2_start, first_env2_end, _ = self.env2_segments[0]

        last_env1_start, last_env1_end,_ = self.env1_segments[-1]
        last_env2_start, last_env2_end,_ = self.env2_segments[-1]
        
        for segments in self.env1_segments:
            start,end,_ = segments
            
            # Initialise connections for start and end points
            if start not in self.connections:
                self.connections[start] = set()
            if end not in self.connections:
                self.connections[end] = set()
            
            # Add the basic connection
            self.connections[start].add(end)

            # Handle interpolation if needed
            if start.weight < first_env2_start.weight and end.weight > first_env2_start.weight or (first_env1_start.weight != first_env2_start.weight and end.weight == first_env2_start.weight):
                print("start = ", start , " end = " ,end)
                print("first_env2_start = " , first_env2_start, " first_env2_end = " , first_env2_end) 
                interpolated_mac = self.interpolate_mac(first_env2_start.weight, start, end)
                interpolated_point = CGPoint(interpolated_mac, first_env2_start.weight, 0)

                # Initialise connections for interpolated point
                if interpolated_point not in self.connections:
                    self.connections[interpolated_point] = set()

                # Set up new connections with interpolation
                self.connections[interpolated_point].add(first_env2_start)
                self.connections[start].add(interpolated_point)
                self.connections[interpolated_point].add(end)

                # Remove the direct connection from start to end
                self.connections[start].discard(end)

            if start.weight < last_env2_end.weight and end.weight > last_env2_end.weight:
                print("start = ", start , " end = " ,end)
                print("last_env2_start = " , last_env2_start, " last_env2_end = " , last_env2_end)
                interpolated_mac = self.interpolate_mac(last_env2_end.weight, start, end)
                interpolated_point = CGPoint(interpolated_mac, last_env2_end.weight, 0)
                # Initialise connections for interpolated point
                if interpolated_point not in self.connections:
                    self.connections[interpolated_point] = set()

                # Set up new connections with interpolation
                self.connections[interpolated_point].add(last_env2_end)
                self.connections[start].add(interpolated_point)
                self.connections[interpolated_point].add(end)

                # Remove the direct connection from start to end
                self.connections[start].discard(end)
        

        for segments in self.env2_segments:
            start, end, _ = segments

            # Initialise connections for start and end points
            if start not in self.connections:
                self.connections[start] = set()
            if end not in self.connections:
                self.connections[end] = set()

            # Add the basic connection
            self.connections[start].add(end)

            # Handle interpolation if needed at min weight
            if start.weight < first_env1_start.weight and end.weight > first_env1_start.weight or (first_env1_start.weight != first_env2_start.weight and end.weight == first_env1_start.weight):
                print("start = ", start , " end = " ,end)
                print("first_env1_start = " , first_env1_start, " first_env1_end = " , first_env1_end) 
                interpolated_mac = self.interpolate_mac(first_env1_start.weight, start, end)
                interpolated_point = CGPoint(interpolated_mac, first_env1_start.weight, 0)

                # Initialise connections for interpolated point
                if interpolated_point not in self.connections:
                    self.connections[interpolated_point] = set()

                # Set up new connections with interpolation
                self.connections[interpolated_point].add(first_env1_start)
                self.connections[start].add(interpolated_point)
                self.connections[interpolated_point].add(end)

                # Remove the direct connection from start to end
                self.connections[start].discard(end)

            # interpolate at the max weight
            if start.weight < last_env1_end.weight and end.weight > last_env1_end.weight:
                print("start = ", start , " end = " ,end)
                print("last_env1_start = " , last_env1_start, " last_env2_end = " , last_env1_end)
                interpolated_mac = self.interpolate_mac(last_env1_end.weight, start, end)
                interpolated_point = CGPoint(interpolated_mac, last_env1_end.weight, 0)
                # Initialise connections for interpolated point
                if interpolated_point not in self.connections:
                    self.connections[interpolated_point] = set()

                # Set up new connections with interpolation
                self.connections[interpolated_point].add(last_env1_end)
                self.connections[start].add(interpolated_point)
                self.connections[interpolated_point].add(end)

                # Remove the direct connection from start to end
                self.connections[start].discard(end)



        # Convert sets to sorted lists
        for key in self.connections:
            self.connections[key] = sorted(list(self.connections[key]), 
                                         key=lambda p: (p.weight, -p.mac))
    
    def get_points_at_weight(self, target_weight: float, isFwd = True) -> List[CGPoint]:
        """Get all points at a specific weight, sorted by MAC descending."""
        points_at_weight = []
        
        for point in self.connections.keys():
            if abs(point.weight - target_weight) < 0.01:
                points_at_weight.append(point)
        
        points_at_weight.sort(key=lambda p: p.mac, reverse=isFwd)
        return points_at_weight
    
    def find_optimal_path(self, isFwd=False) -> List[CGPoint]:
        """
        Find the optimal path (highest MAC values) from start_point to end_point.
        """
        path = []
        print("LISTEN")
        current_point = None

        start_A,_,_ = self.env1_segments[0]
        start_B,_,_ = self.env2_segments[0]
        if start_A.weight == start_B.weight:
            if isFwd and start_A.mac > start_B.mac:
                current_point = start_A
                print("Current point is A")
            else:
                current_point = start_B
                print("Current point is B")
                
        elif start_A.weight > start_B.weight:
            current_point = start_B
            print("Current point is B")    
        else:
            current_point = start_A
            print("Current point is A")
    
        _,end_A,_ = self.env1_segments[-1]
        _,end_B,_ = self.env2_segments[-1]
        end_weight = max(end_A.weight,end_B.weight)
        
        
        
        # Find the closest actual point to our start point
        current_env = current_point.env
        path.append(current_point)
        
        while current_point.weight < end_weight:
            # Get outgoing connections from current point
            outgoing = self.connections.get(current_point, [])
            print()
            print("CURRENT WEIGHT",current_point.weight)
            print("outgoing", outgoing)
            print("current_point",current_point)
            if not outgoing:
                break
            
            # Find connections that lead to higher weight
            next_candidates = [p for p in outgoing if p.weight > current_point.weight]
            print(next_candidates)
            
            if not next_candidates:
                # Edge case where if we start with same macs or if we interpolated to this point
                # We check if the outgoing has another outgoing with a higher weight RARELY we should enter here
                for point in outgoing:
                    for pt in self.connections.get(point,[]):
                        if pt.weight > current_point.weight:
                            next_candidates.append(point)
                
                if not next_candidates:
                    break
            print(next_candidates)
            # Sort by MAC descending
            if isFwd:
                next_candidates.sort(key=lambda p: -p.mac)
            else:
                next_candidates.sort(key=lambda p: p.mac)
            next_point = next_candidates[0]
            current_env = next_point.env
            
            # Add the connected point to path
            path.append(next_point)
            current_point = next_point
            
            # Check for better MAC options at this weight level
            points_at_weight = self.get_points_at_weight(next_point.weight,isFwd=isFwd)

            # Look for higher MAC option at this weight
            for point in points_at_weight:
                #Only consider points of intersection or in the same envelope
                if current_env== 0 or point.env == current_env or point.env == 0:
                    # Check if this higher MAC point can progress further
                    higher_outgoing = self.connections.get(point, [])
                    print("higher_outgoing = ", higher_outgoing)
                    can_progress = any(p.weight > current_point.weight for p in higher_outgoing)
                    if current_point.weight >= end_weight or can_progress:
                        path.append(point)
                        current_point = point
                        break
            print("current_point", current_point)            
            #Check to see if the current weight is max. If not add mac and add outgoing again for the same env
            
            filtered_points = [point for point in points_at_weight
                               if current_env == 0 or point.env == current_env or point.env == 0]
            max_at_weight = max(filtered_points, key=lambda point: point.mac, default=current_point) if isFwd else min(filtered_points, key=lambda point: point.mac, default=current_point)
            if max_at_weight == current_point:
                continue
            else:
                #Add max point at that weight
                path.append(max_at_weight)
                # add the current_point back
                path.append(current_point)
            
            if current_point.weight >=end_weight:
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
                    markeredgewidth=3, label='Intersections',alpha=0.7)
        
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
    CGPoint(weight=100000, mac=21.00,env=1),
    CGPoint(weight=200000, mac=21.00,env=1),
    CGPoint(weight=200000, mac=16.00,env=1),
    CGPoint(weight=250000, mac=16.00,env=1),
    CGPoint(weight=250000, mac=21.00,env=1),
    CGPoint(weight=300000, mac=22.00,env=1),
    CGPoint(weight=300000, mac=17.00,env=1),
    CGPoint(weight=350000, mac=17.00,env=1),
]

env2 = [
    CGPoint(weight=150000, mac=15.00,env=2),
    CGPoint(weight=200000, mac=15.00,env=2),
    CGPoint(weight=330000, mac=20.00,env=2),
    CGPoint(weight=350000, mac=20.00,env=2),
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


optimal_path = analyzer.find_optimal_path()

# Print summary
analyzer.print_summary(optimal_path)

# Plot the analysis
analyzer.plot_analysis(optimal_path, "Aircraft CG Envelope Analysis")

print(f"\nScript completed successfully!")
print(f"Total intersections found: {len(analyzer.intersections)}")
print(f"Envelope 1 split into {len(analyzer.env1_segments)} segments")
print(f"Envelope 2 split into {len(analyzer.env2_segments)} segments")
print(f"Total unique points in network: {len(analyzer.connections)}")
