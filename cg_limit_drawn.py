import matplotlib.pyplot as plt
from typing import List, Optional, Tuple


class CGPoint:
    """
    Represents a Center of Gravity point with MAC position and weight.
    
    In aircraft weight and balance analysis, the CG envelope defines the allowable
    center of gravity positions (as % MAC - Mean Aerodynamic Chord) for different weights.
    """
    def __init__(self, mac: float, weight: float):
        self.mac = mac        # CG position as percentage of Mean Aerodynamic Chord
        self.weight = weight  # Aircraft weight in pounds
    
    def __repr__(self):
        return f"CGPoint(mac={self.mac:.2f}, weight={self.weight:.0f})"
    
    def __eq__(self, other):
        if not isinstance(other, CGPoint):
            return False
        return self.mac == other.mac and self.weight == other.weight


def segment_intersection(p1: CGPoint, p2: CGPoint, q1: CGPoint, q2: CGPoint) -> Optional[CGPoint]:
    """
    Find intersection point between two line segments if it exists.
    
    This uses determinant-based line intersection to find where two CG envelope
    segments cross. Critical for determining where to switch between envelopes.
    
    Args:
        p1, p2: Endpoints of first line segment
        q1, q2: Endpoints of second line segment
    
    Returns:
        CGPoint at intersection, or None if segments don't intersect
    """
    def det(a, b):
        """Calculate 2D determinant"""
        return a[0] * b[1] - a[1] * b[0]

    # Calculate direction vectors
    xdiff = (p1.mac - p2.mac, q1.mac - q2.mac)
    ydiff = (p1.weight - p2.weight, q1.weight - q2.weight)

    # Check if lines are parallel
    div = det(xdiff, ydiff)
    if div == 0:
        return None  # Lines are parallel or coincident

    # Calculate intersection point using determinant method
    d = (det((p1.mac, p1.weight), (p2.mac, p2.weight)),
         det((q1.mac, q1.weight), (q2.mac, q2.weight)))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    # Verify intersection point lies within both line segments
    # (not just on the extended lines)
    if (min(p1.mac, p2.mac) <= x <= max(p1.mac, p2.mac) and
        min(p1.weight, p2.weight) <= y <= max(p1.weight, p2.weight) and
        min(q1.mac, q2.mac) <= x <= max(q1.mac, q2.mac) and
        min(q1.weight, q2.weight) <= y <= max(q1.weight, q2.weight)):
        return CGPoint(x, y)
    return None


def find_all_intersections(env1: List[CGPoint], env2: List[CGPoint]) -> List[Tuple[CGPoint, int, int]]:
    """
    Find all intersection points between two CG envelopes.
    
    Intersections are critical points where the most limiting envelope may
    switch from one envelope to another.
    
    Args:
        env1, env2: Two CG envelopes as lists of CGPoints
    
    Returns:
        List of tuples: (intersection_point, env1_segment_index, env2_segment_index)
        Sorted by weight for processing from light to heavy
    """
    intersections = []
    
    # Check every segment of env1 against every segment of env2
    for i in range(len(env1) - 1):
        for j in range(len(env2) - 1):
            inter = segment_intersection(env1[i], env1[i + 1], env2[j], env2[j + 1])
            if inter:
                intersections.append((inter, i, j))
    
    # Sort by weight (process from light to heavy aircraft weights)
    intersections.sort(key=lambda x: x[0].weight)
    return intersections


def get_segment_direction_at_intersection(envelope: List[CGPoint], seg_idx: int, 
                                        intersection: CGPoint) -> Tuple[float, float]:
    """
    Determine the direction a CG envelope is heading after an intersection point.
    
    This is crucial for determining which envelope becomes more limiting after
    an intersection. We need to know how the MAC position changes as weight increases.
    
    Args:
        envelope: The CG envelope
        seg_idx: Index of the segment containing the intersection
        intersection: The intersection point
    
    Returns:
        Tuple of (mac_direction, weight_direction) - how MAC and weight change
    """
    if seg_idx >= len(envelope) - 1:
        return (0, 0)  # No segment after this point
    
    p1 = envelope[seg_idx]      # Start of current segment
    p2 = envelope[seg_idx + 1]  # End of current segment
    
    # Determine if intersection is at the end of current segment
    # If so, we need to look at the next segment's direction
    dist_to_p1 = abs(intersection.weight - p1.weight)
    dist_to_p2 = abs(intersection.weight - p2.weight)
    
    if dist_to_p2 == 0:  # Intersection is exactly at p2 (segment endpoint)
        # Look at the next segment's direction if it exists
        if seg_idx + 1 < len(envelope) - 1:
            next_p = envelope[seg_idx + 2]
            mac_dir = next_p.mac - p2.mac      # How MAC changes in next segment
            weight_dir = next_p.weight - p2.weight  # How weight changes
        else:
            # This is the last segment, use current segment direction
            mac_dir = p2.mac - p1.mac
            weight_dir = p2.weight - p1.weight
    else:
        # Intersection is along the current segment, use this segment's direction
        mac_dir = p2.mac - p1.mac
        weight_dir = p2.weight - p1.weight
    
    return (mac_dir, weight_dir)


def compare_envelopes_at_intersection(env1: List[CGPoint], env2: List[CGPoint], 
                                    intersection: CGPoint, seg1_idx: int, seg2_idx: int, 
                                    is_forward_limit: bool = True) -> int:
    """
    Determine which envelope is more limiting after an intersection point.
    
    This is the core logic for CG envelope analysis. At intersections, we must
    decide which envelope to follow based on which provides the more restrictive
    (limiting) CG constraint.
    
    For forward CG limits: Higher MAC values are MORE limiting (closer to forward limit)
    For aft CG limits: Lower MAC values are MORE limiting (closer to aft limit)
    
    Args:
        env1, env2: The two CG envelopes
        intersection: Point where envelopes cross
        seg1_idx, seg2_idx: Segment indices where intersection occurs
        is_forward_limit: True for forward CG analysis, False for aft CG analysis
    
    Returns:
        1 if env1 is more limiting after intersection, 2 if env2 is more limiting
    """
    # Get the directional trends of both envelopes after the intersection
    mac_dir1, weight_dir1 = get_segment_direction_at_intersection(env1, seg1_idx, intersection)
    mac_dir2, weight_dir2 = get_segment_direction_at_intersection(env2, seg2_idx, intersection)
    
    # Normal case: both envelopes continue to higher weights
    if weight_dir1 > 0 and weight_dir2 > 0:
        # Calculate slope (rate of MAC change per unit weight change)
        slope1 = mac_dir1 / weight_dir1 if weight_dir1 != 0 else 0
        slope2 = mac_dir2 / weight_dir2 if weight_dir2 != 0 else 0
        
        # Compare slopes based on limit type
        if is_forward_limit:
            # Forward limits: higher MAC = more limiting, so prefer higher slope
            # (envelope that moves MAC forward faster as weight increases)
            return 1 if slope1 > slope2 else 2
        else:
            # Aft limits: lower MAC = more limiting, so prefer lower slope
            # (envelope that moves MAC aft faster as weight increases)
            return 1 if slope1 < slope2 else 2
    
    # Handle edge cases where segments are horizontal or decreasing in weight
    elif weight_dir1 == 0 and weight_dir2 == 0:
        # Both envelopes are horizontal at this weight - compare MAC positions directly
        if is_forward_limit:
            return 1 if mac_dir1 > mac_dir2 else 2  # Higher MAC more limiting
        else:
            return 1 if mac_dir1 < mac_dir2 else 2  # Lower MAC more limiting
    elif weight_dir1 == 0:
        # env1 is horizontal, env2 is not - evaluate based on env2's trend
        if is_forward_limit:
            return 2 if weight_dir2 > 0 and mac_dir2 > 0 else 1
        else:
            return 2 if weight_dir2 > 0 and mac_dir2 < 0 else 1
    elif weight_dir2 == 0:
        # env2 is horizontal, env1 is not - evaluate based on env1's trend
        if is_forward_limit:
            return 1 if weight_dir1 > 0 and mac_dir1 > 0 else 2
        else:
            return 1 if weight_dir1 > 0 and mac_dir1 < 0 else 2
    else:
        # One envelope going forward, one backward in weight - prefer forward
        if weight_dir1 > 0 and weight_dir2 <= 0:
            return 1
        elif weight_dir2 > 0 and weight_dir1 <= 0:
            return 2
        else:
            # Both going same direction, compare slopes as in normal case
            slope1 = mac_dir1 / weight_dir1 if weight_dir1 != 0 else 0
            slope2 = mac_dir2 / weight_dir2 if weight_dir2 != 0 else 0
            if is_forward_limit:
                return 1 if slope1 > slope2 else 2
            else:
                return 1 if slope1 < slope2 else 2


def should_include_point(envelope: List[CGPoint], idx: int, is_forward_limit: bool = True) -> bool:
    """
    Determine if a point should be included in the limiting envelope analysis.
    
    This function filters out non-limiting intermediate points that create
    "backward" steps at the same weight, which don't contribute to the limiting envelope.
    
    For example, at weight 1500: points going 17→16→18 MAC, the 16 point should be
    excluded for forward limits since 16 < 17 is not more limiting (less forward).
    
    Args:
        envelope: The CG envelope being analyzed
        idx: Index of point to evaluate
        is_forward_limit: True for forward limits, False for aft limits
    
    Returns:
        True if point should be included in limiting analysis
    """
    # Always include first and last points (envelope boundaries)
    if idx == 0 or idx == len(envelope) - 1:
        return True
    
    current = envelope[idx]
    prev_point = envelope[idx - 1]
    next_point = envelope[idx + 1]
    
    # Check for points at same weight as both neighbors
    if (current.weight == prev_point.weight and 
        current.weight == next_point.weight):
        
        if is_forward_limit:
            # For forward limits: skip backward steps (less limiting)
            if current.mac < prev_point.mac and current.mac < next_point.mac:
                return False
            # Skip "notch" patterns like 17→16→18 (16 is not limiting)
            if prev_point.mac > current.mac < next_point.mac:
                return False
        else:
            # For aft limits: skip forward steps (less limiting)
            if current.mac > prev_point.mac and current.mac > next_point.mac:
                return False
            # Skip "notch" patterns like 16→17→15 (17 is not limiting for aft)
            if prev_point.mac < current.mac > next_point.mac:
                return False
    
    # Check points at same weight as previous point only
    elif current.weight == prev_point.weight:
        if is_forward_limit:
            # For forward limits: check if this point is backward (less limiting)
            if current.mac < prev_point.mac:
                # Look ahead to see if there's a more limiting point at same weight
                for j in range(idx + 1, len(envelope)):
                    if envelope[j].weight == current.weight:
                        if envelope[j].mac > current.mac:
                            return False  # Skip this less limiting point
                    else:
                        break  # Different weight, stop looking
        else:
            # For aft limits: check if this point is forward (less limiting)
            if current.mac > prev_point.mac:
                # Look ahead for more limiting point at same weight
                for j in range(idx + 1, len(envelope)):
                    if envelope[j].weight == current.weight:
                        if envelope[j].mac < current.mac:
                            return False  # Skip this less limiting point
                    else:
                        break
    
    return True


def clean_envelope_for_limiting(envelope: List[CGPoint], is_forward_limit: bool = True) -> List[CGPoint]:
    """
    Clean envelope by removing non-limiting intermediate points.
    
    This preprocessing step removes points that create "backward" movements
    in the limiting direction, which would never be part of the actual
    limiting envelope boundary.
    
    Args:
        envelope: Original CG envelope
        is_forward_limit: True for forward CG limits, False for aft limits
    
    Returns:
        Cleaned envelope with only potentially limiting points
    """
    if len(envelope) <= 2:
        return envelope[:]  # Can't clean very short envelopes
    
    cleaned = []
    
    # Filter points using limiting criteria
    for i in range(len(envelope)):
        if should_include_point(envelope, i, is_forward_limit):
            cleaned.append(envelope[i])
    
    return cleaned


def find_most_limiting_envelope(env1: List[CGPoint], env2: List[CGPoint], 
                               is_forward_limit: bool = True) -> List[CGPoint]:
    """
    Find the most limiting CG envelope by combining two envelopes optimally.
    
    This is the main algorithm that:
    1. Cleans both envelopes to remove non-limiting points
    2. Finds all intersection points where envelopes cross
    3. Determines which envelope to follow at each weight range
    4. Constructs the composite most-limiting envelope
    
    The algorithm processes from light to heavy weights, switching between
    envelopes at intersections when the other becomes more limiting.
    
    Args:
        env1, env2: The two CG envelopes to compare
        is_forward_limit: True for forward CG limits (higher MAC = more limiting)
                         False for aft CG limits (lower MAC = more limiting)
    
    Returns:
        List of CGPoints representing the most limiting envelope boundary
    """
    # Handle empty envelopes
    if not env1 or not env2:
        return env1 if env1 else env2
    
    # Step 1: Clean both envelopes to remove non-limiting intermediate points
    clean_env1 = clean_envelope_for_limiting(env1, is_forward_limit)
    clean_env2 = clean_envelope_for_limiting(env2, is_forward_limit)
    
    # Step 2: Find all intersection points between cleaned envelopes
    intersections = find_all_intersections(clean_env1, clean_env2)
    
    # Step 3: Create a master list of all significant points (envelope points + intersections)
    all_points = []
    
    # Add all points from envelope 1 with metadata
    for i, pt in enumerate(clean_env1):
        all_points.append((pt, pt.weight, 1, i, False, None))
    
    # Add all points from envelope 2 with metadata
    for i, pt in enumerate(clean_env2):
        all_points.append((pt, pt.weight, 2, i, False, None))
    
    # Add intersection points with segment information
    for inter, seg1_idx, seg2_idx in intersections:
        all_points.append((inter, inter.weight, 0, -1, True, (seg1_idx, seg2_idx)))
    
    # Step 4: Sort all points by weight (light to heavy processing order)
    all_points.sort(key=lambda x: x[1])
    
    # Step 5: Determine starting envelope (most limiting at minimum weight)
    min_weight = min(clean_env1[0].weight, clean_env2[0].weight)
    
    if clean_env1[0].weight == min_weight and clean_env2[0].weight == min_weight:
        # Both start at same weight - choose based on which is more limiting
        if is_forward_limit:
            # For forward limits, higher MAC is more limiting
            current_env_num = 1 if clean_env1[0].mac >= clean_env2[0].mac else 2
        else:
            # For aft limits, lower MAC is more limiting
            current_env_num = 1 if clean_env1[0].mac <= clean_env2[0].mac else 2
    elif clean_env1[0].weight == min_weight:
        current_env_num = 1  # Start with envelope 1
    else:
        current_env_num = 2  # Start with envelope 2
    
    # Step 6: Build the limiting envelope by processing points in weight order
    limiting_envelope = []
    current_weight = min_weight
    
    for point_info in all_points:
        pt, weight, env_num, pt_idx, is_intersection, inter_info = point_info
        
        # Skip points at weights we've already processed
        if weight < current_weight:
            continue
            
        # Skip points from non-current envelope unless it's an intersection
        if not is_intersection and env_num != current_env_num:
            continue
            
        if is_intersection:
            # At intersection: decide whether to switch envelopes
            seg1_idx, seg2_idx = inter_info
            
            # Analyze which envelope becomes more limiting after intersection
            more_limiting = compare_envelopes_at_intersection(
                clean_env1, clean_env2, pt, seg1_idx, seg2_idx, is_forward_limit)
            
            # Add intersection point (avoid duplicates)
            if not limiting_envelope or pt != limiting_envelope[-1]:
                limiting_envelope.append(pt)
                current_weight = weight
            
            # Switch to the more limiting envelope
            current_env_num = more_limiting
        else:
            # Regular envelope point from the current active envelope
            if env_num == current_env_num:
                # Add point if it's not a duplicate of the previous point
                if not limiting_envelope or pt != limiting_envelope[-1]:
                    limiting_envelope.append(pt)
                    current_weight = weight
    
    return limiting_envelope


def plot_cg_envelopes(env1: List[CGPoint], env2: List[CGPoint], limiting_env: List[CGPoint], 
                     is_forward_limit: bool = True):
    """
    Create a comprehensive plot showing both original envelopes and the limiting envelope.
    
    The plot visualizes:
    - Original CG envelopes with labeled points
    - Intersection points where envelopes cross
    - The computed most limiting envelope
    
    Args:
        env1, env2: Original CG envelopes
        limiting_env: Computed most limiting envelope
        is_forward_limit: True for forward limit analysis, False for aft
    
    Returns:
        matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot Original Envelope 1
    mac1 = [pt.mac for pt in env1]
    wt1 = [pt.weight for pt in env1]
    ax.plot(mac1, wt1, 'b-', linewidth=2, label='CG Envelope 1', alpha=0.7)
    ax.scatter(mac1, wt1, color='blue', s=50, zorder=5)

    # Add point labels for envelope 1
    for i, pt in enumerate(env1):
        ax.annotate(f'1-{i}', (pt.mac, pt.weight), xytext=(5, 5), 
                   textcoords='offset points', fontsize=8, color='blue')

    # Plot Original Envelope 2
    mac2 = [pt.mac for pt in env2]
    wt2 = [pt.weight for pt in env2]
    ax.plot(mac2, wt2, 'g-', linewidth=2, label='CG Envelope 2', alpha=0.7)
    ax.scatter(mac2, wt2, color='green', s=50, zorder=5)

    # Add point labels for envelope 2
    for i, pt in enumerate(env2):
        ax.annotate(f'2-{i}', (pt.mac, pt.weight), xytext=(-15, 5), 
                   textcoords='offset points', fontsize=8, color='green')

    # Plot and label intersection points
    intersections = find_all_intersections(env1, env2)
    for inter, _, _ in intersections:
        ax.scatter(inter.mac, inter.weight, color='black', s=100, marker='x', linewidth=3, zorder=10)
        ax.annotate(f'INT({inter.mac:.1f}, {inter.weight:.0f})', 
                   (inter.mac, inter.weight),
                   xytext=(10, -15), textcoords='offset points',
                   fontsize=9, ha='left', color='black', weight='bold')

    # Plot the Most Limiting Envelope (main result)
    if limiting_env:
        mac_lim = [pt.mac for pt in limiting_env]
        wt_lim = [pt.weight for pt in limiting_env]
        limit_type = "Forward" if is_forward_limit else "Aft"
        ax.plot(mac_lim, wt_lim, 'r-', linewidth=4, label=f'Most {limit_type}-Limiting', alpha=0.9)
        ax.scatter(mac_lim, wt_lim, color='red', s=80, zorder=8, marker='D')

    # Format the plot
    ax.set_xlabel('CG Position (% MAC)', fontsize=12)
    ax.set_ylabel('Weight (lbs)', fontsize=12)
    limit_type = "Forward" if is_forward_limit else "Aft"
    ax.set_title(f'Most {limit_type}-Limiting CG Envelope Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Set appropriate axis limits with padding
    all_macs = mac1 + mac2 + ([pt.mac for pt in limiting_env] if limiting_env else [])
    all_weights = wt1 + wt2 + ([pt.weight for pt in limiting_env] if limiting_env else [])
    
    if all_macs and all_weights:
        mac_range = max(all_macs) - min(all_macs)
        weight_range = max(all_weights) - min(all_weights)
        
        ax.set_xlim(min(all_macs) - 0.1 * mac_range, max(all_macs) + 0.1 * mac_range)
        ax.set_ylim(min(all_weights) - 0.1 * weight_range, max(all_weights) + 0.1 * weight_range)

    plt.tight_layout()
    return fig, ax


# ============================================================================
# EXAMPLE USAGE AND DEMONSTRATION
# ============================================================================

# Define example CG envelopes for demonstration
# These represent typical aircraft CG envelopes with different loading configurations

env1 = [
    CGPoint(15, 1200),    # 1-0: Entry point at light weight
    CGPoint(17, 1500),    # 1-1: Forward movement to weight 1500
    CGPoint(16, 1500),    # 1-2: Backward step at same weight (will be filtered for FWD limits)
    CGPoint(18, 1500),    # 1-3: Forward again at same weight  
    CGPoint(22, 2200),    # 1-4: Continued forward trend
    CGPoint(30, 2800),    # 1-5: Maximum weight, maximum forward CG
]

env2 = [
    CGPoint(16, 1200),    # 2-0: Entry point (slightly more forward than env1)
    CGPoint(15, 1500),    # 2-1: Backward movement (less limiting for forward)
    CGPoint(25, 2500),    # 2-2: Sharp forward movement
    CGPoint(31, 2800),    # 2-3: Most forward at maximum weight
]

# Analyze for forward CG limits (higher MAC = more limiting)
print("="*60)
print("FORWARD CG LIMIT ANALYSIS")
print("="*60)

limiting_envelope = find_most_limiting_envelope(env1, env2, is_forward_limit=True)

# Display detailed results
print("Envelope 1 points:")
for i, pt in enumerate(env1):
    print(f"  1-{i}: {pt}")

print("\nEnvelope 2 points:")
for i, pt in enumerate(env2):
    print(f"  2-{i}: {pt}")

print("\nIntersection points:")
intersections = find_all_intersections(env1, env2)
for inter, i, j in intersections:
    print(f"  {inter} (between env1[{i}]-env1[{i+1}] and env2[{j}]-env2[{j+1}])")

print("\nMost Forward-Limiting Envelope:")
for i, pt in enumerate(limiting_envelope):
    print(f"  {i}: {pt}")

print("\nCleaned Envelope 1 (non-limiting points removed):")
clean_env1 = clean_envelope_for_limiting(env1, is_forward_limit=True)
for i, pt in enumerate(clean_env1):
    print(f"  {i}: {pt}")

# Create and display the plot
fig, ax = plot_cg_envelopes(env1, env2, limiting_envelope, is_forward_limit=True)
plt.show()

# Optional: Analyze for aft CG limits (uncomment to run)

print("\n" + "="*60)
print("AFT CG LIMIT ANALYSIS")
print("="*60)

aft_limiting_envelope = find_most_limiting_envelope(env1, env2, is_forward_limit=False)
print("\nMost Aft-Limiting Envelope:")
for i, pt in enumerate(aft_limiting_envelope):
    print(f"  {i}: {pt}")

fig2, ax2 = plot_cg_envelopes(env1, env2, aft_limiting_envelope, is_forward_limit=False)
plt.show()
