from typing import List, Optional, Set
from itertools import groupby

class CGPoint:
    def __init__(self, weight: float, mac: float, vectorPointId: Optional[int], vectorId: Optional[int]):
        self.weight = weight
        self.mac = mac
        self.vectorPointId = vectorPointId
        self.vectorId = vectorId
    
    def __repr__(self):
        return (f"CGPoint(weight={self.weight}, mac={self.mac}, "
                f"vectorPointId={self.vectorPointId}, vectorId={self.vectorId})")

def clean_duplicate_weights(points: List[CGPoint]) -> List[CGPoint]:
    """
    Clean up duplicate weights following the specified ordering logic.
    Assumes points are already sorted by weight.
    """
    # Pre-scan to identify duplicate weights in O(n) time
    duplicate_weights: Set[float] = set()
    prev_weight = None
    
    for point in points:
        if prev_weight is not None and point.weight == prev_weight:
            duplicate_weights.add(point.weight)
        prev_weight = point.weight
    
    # Process groups using itertools.groupby (leverages pre-sorted data)
    result = []
    for weight, group_iter in groupby(points, key=lambda p: p.weight):
        group = list(group_iter)
        
        if weight in duplicate_weights:
            # Complex ordering for duplicates
            ordered_group = order_duplicate_weight_group(group, points, weight)
            result.extend(ordered_group)
        else:
            # Single point, add directly
            result.extend(group)
    
    return result

def order_duplicate_weight_group(group: List[CGPoint], all_points: List[CGPoint], 
                               current_weight: float) -> List[CGPoint]:
    """Order points within a duplicate weight group according to the algorithm."""
    
    # Separate envelope and intersection points
    envelope_points = [p for p in group if p.vectorId is not None]
    intersection_points = [p for p in group if p.vectorId is None]
    
    # Simple case: no complex ordering needed
    if len(envelope_points) <= 1:
        return sorted(group, key=lambda p: (p.vectorId is None, -p.mac))
    
    # Step 1: Find the first element
    # Group envelope points by vectorId and find the smallest vectorPointId for each vectorId
    vector_groups = {}
    for point in envelope_points:
        if point.vectorId not in vector_groups:
            vector_groups[point.vectorId] = []
        vector_groups[point.vectorId].append(point)
    
    # For each vectorId, find the point with smallest vectorPointId
    vector_candidates = []
    for vector_id, points_in_vector in vector_groups.items():
        min_point = min(points_in_vector, key=lambda p: p.vectorPointId if p.vectorPointId else float('inf'))
        vector_candidates.append((vector_id, min_point.vectorPointId, min_point.mac, min_point))
    
    # Among these candidates (one per vectorId), pick the one with highest MAC
    first_point = max(vector_candidates, key=lambda x: x[2])[3]  # x[2] is MAC, x[3] is the point object
    
    remaining_points = [p for p in group if p != first_point]
    
    # Step 2: Find the last element (continuation point)
    continuation_vector_id = _find_continuation_vector_optimized(envelope_points, all_points, current_weight)
    
    continuation_point = None
    if continuation_vector_id:
        # Find all points with the continuation vectorId from remaining points
        continuation_candidates = [p for p in remaining_points if p.vectorId == continuation_vector_id]
        if continuation_candidates:
            # Choose the one with the highest vectorPointId
            continuation_point = max(continuation_candidates, key=lambda p: p.vectorPointId if p.vectorPointId else 0)
    
    # Step 3: Build ordered result
    ordered_points = [first_point]
    
    # Add middle points (everything except continuation point)
    # Sort: intersection points first, then other envelope points by MAC descending
    middle_points = [p for p in remaining_points if p != continuation_point]
    middle_points.sort(key=lambda p: (p.vectorId is not None, -p.mac))
    ordered_points.extend(middle_points)
    
    # Add continuation point last
    if continuation_point:
        ordered_points.append(continuation_point)
    
    return ordered_points

def _find_continuation_vector_optimized(envelope_points: List[CGPoint], all_points: List[CGPoint], 
                                      current_weight: float) -> Optional[int]:
    """
    Find which vectorId continues to the next weight.
    Optimized for pre-sorted data - can scan forward efficiently.
    """
    current_vector_ids = {p.vectorId for p in envelope_points}
    
    # Scan forward from current position to find next different weight
    found_current = False
    for point in all_points:
        if point.weight == current_weight:
            found_current = True
            continue
        elif found_current and point.weight > current_weight:
            # This is the next weight group
            if point.vectorId is not None and point.vectorId in current_vector_ids:
                return point.vectorId
            # Keep checking other points at this same next weight
            next_weight = point.weight
            for next_point in all_points:
                if next_point.weight == next_weight and next_point.vectorId is not None:
                    if next_point.vectorId in current_vector_ids:
                        return next_point.vectorId
                elif next_point.weight > next_weight:
                    break
            break
    
    return None

def analyze_duplicates(points: List[CGPoint]) -> dict:
    """Utility function to analyze duplicate patterns in the data."""
    duplicate_info = {}
    
    for weight, group_iter in groupby(points, key=lambda p: p.weight):
        group = list(group_iter)
        if len(group) > 1:
            envelope_count = sum(1 for p in group if p.vectorId is not None)
            intersection_count = sum(1 for p in group if p.vectorId is None)
            vector_ids = {p.vectorId for p in group if p.vectorId is not None}
            
            duplicate_info[weight] = {
                'total_points': len(group),
                'envelope_points': envelope_count,
                'intersection_points': intersection_count,
                'unique_vector_ids': len(vector_ids),
                'vector_ids': vector_ids
            }
    
    return duplicate_info

# Test with the data
def main():
    # Test with your corrected data where 21 has the smallest vectorPointId
    points = [
        CGPoint(weight=160000, mac=16, vectorPointId=500005641, vectorId=500005640),
        CGPoint(weight=188500, mac=16, vectorPointId=500005642, vectorId=500005640),
        CGPoint(weight=197800, mac=15, vectorPointId=500005643, vectorId=500005640),
        CGPoint(weight=233000, mac=15, vectorPointId=500005644, vectorId=500005640),
        CGPoint(weight=245000, mac=15, vectorPointId=500005645, vectorId=500005640),
        CGPoint(weight=270000, mac=15, vectorPointId=500005646, vectorId=500005640),
        CGPoint(weight=280000, mac=19, vectorPointId=500005647, vectorId=500005640),
        CGPoint(weight=300000, mac=18, vectorPointId=None, vectorId=None),
        CGPoint(weight=300000, mac=15, vectorPointId=500005648, vectorId=500005640),  # smallest vectorPointId 
        CGPoint(weight=300000, mac=21, vectorPointId=500005649, vectorId=500005640),  # highest vectorPointId for continuation
        CGPoint(weight=300000, mac=18, vectorPointId=500005613, vectorId=500005794), # second smallest vectorPointId
        CGPoint(weight=350000, mac=17, vectorPointId=500005650, vectorId=500005640),
        CGPoint(weight=354600, mac=18, vectorPointId=500005651, vectorId=500005640),
        CGPoint(weight=368000, mac=15, vectorPointId=500005652, vectorId=500005640),
    ]

    print("Duplicate analysis:")
    duplicates = analyze_duplicates(points)
    for weight, info in duplicates.items():
        print(f"Weight {weight}: {info}")

    print("\nOriginal points:")
    for point in points:
        print(point)

    print("\nCleaned points:")
    cleaned_points = clean_duplicate_weights(points)
    for point in cleaned_points:
        print(point)

    print("\nWeight 300000 group specifically:")
    weight_300k_points = [p for p in cleaned_points if p.weight == 300000]
    for i, point in enumerate(weight_300k_points, 1):
        print(f"{i}. {point}")
    
    print("\nExpected order for weight 300000:")
    print("1. First: Compare smallest vectorPointId from each vectorId group:")
    print("   - vectorId 500005640: smallest vectorPointId=500005648, mac=21")
    print("   - vectorId 500005794: smallest vectorPointId=500005613, mac=18") 
    print("   Choose highest MAC: mac=21, so first point is (500005648, mac=21)")
    print("2-3. Middle: Other points sorted by type and MAC")
    print("4. Last: Continuation point (highest vectorPointId for continuing vectorId)")

if __name__ == "__main__":
    main()