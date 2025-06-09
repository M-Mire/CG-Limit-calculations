#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <iomanip>

struct CGPoint {
    double mac;
    double weight;
    int env;
    
    CGPoint(double m, double w, int e) : mac(m), weight(w), env(e) {}
    
    // Comparison operators for use in sets and maps
    bool operator<(const CGPoint& other) const {
        if (std::abs(weight - other.weight) > 0.01) return weight < other.weight;
        if (std::abs(mac - other.mac) > 0.01) return mac < other.mac;
        return env < other.env;
    }
    
    bool operator==(const CGPoint& other) const {
        return std::abs(mac - other.mac) < 0.01 && 
               std::abs(weight - other.weight) < 0.01;
    }
    
    friend std::ostream& operator<<(std::ostream& os, const CGPoint& point) {
        os << "CGPoint(weight=" << std::fixed << std::setprecision(0) << point.weight 
           << ", mac=" << std::setprecision(2) << point.mac 
           << ", env=" << std::setprecision(0) << point.env << ")";
        return os;
    }
};

using Segment = std::tuple<CGPoint, CGPoint, std::string>;
using Intersection = std::tuple<int, int, CGPoint>;

class EnvelopeIntersectionAnalyzer {
private:
    std::vector<CGPoint> env1_original;
    std::vector<CGPoint> env2_original;
    std::vector<Intersection> intersections;
    std::vector<Segment> env1_segments;
    std::vector<Segment> env2_segments;
    std::map<CGPoint, std::set<CGPoint>> connections;
    
    // Returns true if intersection exists, false otherwise
    // If true, intersection point is stored in result parameter
    bool lineIntersection(const CGPoint& p1, const CGPoint& p2, 
                         const CGPoint& p3, const CGPoint& p4, CGPoint& result) {
        double mac1 = p1.mac, weight1 = p1.weight;
        double mac2 = p2.mac, weight2 = p2.weight;
        double mac3 = p3.mac, weight3 = p3.weight;
        double mac4 = p4.mac, weight4 = p4.weight;
        
        double denom = (mac1 - mac2) * (weight3 - weight4) - (weight1 - weight2) * (mac3 - mac4);
        
        if (std::abs(denom) < 1e-10) return false;
        
        double t = ((mac1 - mac3) * (weight3 - weight4) - (weight1 - weight3) * (mac3 - mac4)) / denom;
        double u = -((mac1 - mac2) * (weight1 - weight3) - (weight1 - weight2) * (mac1 - mac3)) / denom;
        
        if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
            double i_mac = mac1 + t * (mac2 - mac1);
            double i_weight = weight1 + t * (weight2 - weight1);
            result = CGPoint(i_mac, i_weight, 0);
            return true;
        }
        
        return false;
    }
    
    std::vector<Intersection> findAllIntersections(const std::vector<CGPoint>& env1, 
                                                 const std::vector<CGPoint>& env2) {
        std::vector<Intersection> result;
        
        for (auto it1 = env1.begin(); it1 != env1.end() - 1; ++it1) {
            for (auto it2 = env2.begin(); it2 != env2.end() - 1; ++it2) {
                CGPoint intersection_point(0, 0, 0);
                if (lineIntersection(*it1, *(it1 + 1), *it2, *(it2 + 1), intersection_point)) {
                    int i = std::distance(env1.begin(), it1);
                    int j = std::distance(env2.begin(), it2);
                    result.emplace_back(i, j, intersection_point);
                }
            }
        }
        
        return result;
    }
    
    // Returns true if interpolation is possible, false otherwise
    // If true, interpolated MAC value is stored in result parameter
    bool interpolateMac(double target_weight, const CGPoint& point1, 
                       const CGPoint& point2, double& result, double epsilon = 1e-6) {
        double w1 = point1.weight, w2 = point2.weight;
        double mac1 = point1.mac, mac2 = point2.mac;
        
        if (std::abs(w2 - w1) < epsilon) {
            if (std::abs(target_weight - w1) < epsilon) {
                result = mac1;
                return true;
            } else if (std::abs(target_weight - w2) < epsilon) {
                result = mac2;
                return true;
            }
            return false;
        }
        
        result = mac1 + (mac2 - mac1) * (target_weight - w1) / (w2 - w1);
        return true;
    }
    
    double distanceAlongSegment(const CGPoint& start, const CGPoint& end, const CGPoint& point) {
        if (std::abs(end.mac - start.mac) > 1e-10) {
            return (point.mac - start.mac) / (end.mac - start.mac);
        } else {
            return (point.weight - start.weight) / (end.weight - start.weight);
        }
    }
    
    std::vector<Segment> splitEnvelopeAtIntersections(const std::vector<CGPoint>& envelope,
                                                    const std::vector<std::pair<int, CGPoint>>& intersections) {
        std::vector<Segment> segments;
        
        std::map<int, std::vector<CGPoint>> intersectionsBySegment;
        for (const auto& intersection : intersections) {
            int seg_idx = intersection.first;
            const CGPoint& point = intersection.second;
            intersectionsBySegment[seg_idx].push_back(point);
        }
        
        for (auto it = envelope.begin(); it != envelope.end() - 1; ++it) {
            int current_seg = std::distance(envelope.begin(), it);
            CGPoint start_point = *it;
            CGPoint end_point = *(it + 1);
            
            auto seg_it = intersectionsBySegment.find(current_seg);
            if (seg_it == intersectionsBySegment.end()) {
                segments.emplace_back(start_point, end_point, "direct");
            } else {
                auto& seg_intersections = seg_it->second;
                std::sort(seg_intersections.begin(), seg_intersections.end(),
                         [&](const CGPoint& a, const CGPoint& b) {
                             return distanceAlongSegment(start_point, end_point, a) < 
                                    distanceAlongSegment(start_point, end_point, b);
                         });
                
                CGPoint prev_point = start_point;
                for (const auto& intersection : seg_intersections) {
                    segments.emplace_back(prev_point, intersection, "to_intersection");
                    prev_point = intersection;
                }
                segments.emplace_back(prev_point, end_point, "from_intersection");
            }
        }
        
        return segments;
    }
    
    void analyzeIntersections() {
        intersections = findAllIntersections(env1_original, env2_original);
        
        std::vector<std::pair<int, CGPoint>> env1_intersections;
        std::vector<std::pair<int, CGPoint>> env2_intersections;
        
        for (const auto& intersection : intersections) {
            int seg1 = std::get<0>(intersection);
            int seg2 = std::get<1>(intersection);
            const CGPoint& point = std::get<2>(intersection);
            
            env1_intersections.emplace_back(seg1, point);
            env2_intersections.emplace_back(seg2, point);
        }
        
        env1_segments = splitEnvelopeAtIntersections(env1_original, env1_intersections);
        env2_segments = splitEnvelopeAtIntersections(env2_original, env2_intersections);
    }
    
    void createConnectionGraph() {
        connections.clear();
        
        const CGPoint& first_env1_start = std::get<0>(*env1_segments.begin());
        const CGPoint& first_env1_end = std::get<1>(*env1_segments.begin());
        const CGPoint& first_env2_start = std::get<0>(*env2_segments.begin());
        const CGPoint& first_env2_end = std::get<1>(*env2_segments.begin());
        const CGPoint& last_env1_start = std::get<0>(*env1_segments.rbegin());
        const CGPoint& last_env1_end = std::get<1>(*env1_segments.rbegin());
        const CGPoint& last_env2_start = std::get<0>(*env2_segments.rbegin());
        const CGPoint& last_env2_end = std::get<1>(*env2_segments.rbegin());
        
        // Process env1 segments
        for (const auto& segment : env1_segments) {
            const CGPoint& start = std::get<0>(segment);
            const CGPoint& end = std::get<1>(segment);
            const std::string& seg_type = std::get<2>(segment);
            
            connections[start].insert(end);
            
            // Handle interpolation at start weight
            if ((start.weight < first_env2_start.weight && end.weight > first_env2_start.weight) ||
                (first_env1_start.weight != first_env2_start.weight && end.weight == first_env2_start.weight)) {
                
                double interpolated_mac;
                if (interpolateMac(first_env2_start.weight, start, end, interpolated_mac)) {
                    CGPoint interpolated_point(interpolated_mac, first_env2_start.weight, 0);
                    connections[interpolated_point].insert(first_env2_start);
                    connections[start].insert(interpolated_point);
                    connections[interpolated_point].insert(end);
                    connections[start].erase(end);
                }
            }
            
            // Handle interpolation at end weight
            if (start.weight < last_env2_end.weight && end.weight > last_env2_end.weight) {
                double interpolated_mac;
                if (interpolateMac(last_env2_end.weight, start, end, interpolated_mac)) {
                    CGPoint interpolated_point(interpolated_mac, last_env2_end.weight, 0);
                    connections[interpolated_point].insert(last_env2_end);
                    connections[start].insert(interpolated_point);
                    connections[interpolated_point].insert(end);
                    connections[start].erase(end);
                }
            }
        }
        
        // Process env2 segments
        for (const auto& segment : env2_segments) {
            const CGPoint& start = std::get<0>(segment);
            const CGPoint& end = std::get<1>(segment);
            const std::string& seg_type = std::get<2>(segment);
            
            connections[start].insert(end);
            
            // Handle interpolation at start weight
            if ((start.weight < first_env1_start.weight && end.weight > first_env1_start.weight) ||
                (first_env1_start.weight != first_env2_start.weight && end.weight == first_env1_start.weight)) {
                
                double interpolated_mac;
                if (interpolateMac(first_env1_start.weight, start, end, interpolated_mac)) {
                    CGPoint interpolated_point(interpolated_mac, first_env1_start.weight, 0);
                    connections[interpolated_point].insert(first_env1_start);
                    connections[start].insert(interpolated_point);
                    connections[interpolated_point].insert(end);
                    connections[start].erase(end);
                }
            }
            
            // Handle interpolation at end weight
            if (start.weight < last_env1_end.weight && end.weight > last_env1_end.weight) {
                double interpolated_mac;
                if (interpolateMac(last_env1_end.weight, start, end, interpolated_mac)) {
                    CGPoint interpolated_point(interpolated_mac, last_env1_end.weight, 0);
                    connections[interpolated_point].insert(last_env1_end);
                    connections[start].insert(interpolated_point);
                    connections[interpolated_point].insert(end);
                    connections[start].erase(end);
                }
            }
        }
    }
    
    std::vector<CGPoint> getPointsAtWeight(double target_weight, bool isFwd = true) {
        std::vector<CGPoint> points_at_weight;
        
        for (const auto& connection : connections) {
            const CGPoint& point = connection.first;
            if (std::abs(point.weight - target_weight) < 0.01) {
                points_at_weight.push_back(point);
            }
        }
        
        std::sort(points_at_weight.begin(), points_at_weight.end(),
                 [isFwd](const CGPoint& a, const CGPoint& b) {
                     return isFwd ? a.mac > b.mac : a.mac < b.mac;
                 });
        
        return points_at_weight;
    }
    
public:
    EnvelopeIntersectionAnalyzer(const std::vector<CGPoint>& env1, const std::vector<CGPoint>& env2)
        : env1_original(env1), env2_original(env2) {
        analyzeIntersections();
        createConnectionGraph();
    }
    
    std::vector<CGPoint> findOptimalPath(bool isFwd = true) {
        std::vector<CGPoint> path;
        
        const CGPoint& start_A = std::get<0>(*env1_segments.begin());
        const CGPoint& start_B = std::get<0>(*env2_segments.begin());
        
        CGPoint current_point = start_A;
        if (start_A.weight == start_B.weight) {
            current_point = (isFwd && start_A.mac > start_B.mac) ? start_A : start_B;
        } else if (start_A.weight > start_B.weight) {
            current_point = start_B;
        } else {
            current_point = start_A;
        }
        
        const CGPoint& end_A = std::get<1>(*env1_segments.rbegin());
        const CGPoint& end_B = std::get<1>(*env2_segments.rbegin());
        double end_weight = std::max(end_A.weight, end_B.weight);
        
        int current_env = current_point.env;
        path.push_back(current_point);
        
        while (current_point.weight < end_weight) {
            auto conn_it = connections.find(current_point);
            if (conn_it == connections.end() || conn_it->second.empty()) {
                break;
            }
            
            const auto& outgoing = conn_it->second;
            std::vector<CGPoint> next_candidates;
            
            std::copy_if(outgoing.begin(), outgoing.end(), std::back_inserter(next_candidates),
                        [&current_point](const CGPoint& p) { return p.weight > current_point.weight; });
            
            if (next_candidates.empty()) {
                for (const auto& point : outgoing) {
                    auto sub_conn_it = connections.find(point);
                    if (sub_conn_it != connections.end()) {
                        for (const auto& pt : sub_conn_it->second) {
                            if (pt.weight > current_point.weight) {
                                next_candidates.push_back(point);
                                break;
                            }
                        }
                    }
                }
                if (next_candidates.empty()) break;
            }
            
            std::sort(next_candidates.begin(), next_candidates.end(),
                     [isFwd](const CGPoint& a, const CGPoint& b) {
                         return isFwd ? a.mac > b.mac : a.mac < b.mac;
                     });
            
            CGPoint next_point = *next_candidates.begin();
            current_env = next_point.env;
            path.push_back(next_point);
            current_point = next_point;
            
            // Check for better MAC options at this weight level
            auto points_at_weight = getPointsAtWeight(next_point.weight, isFwd);
            
            for (const auto& point : points_at_weight) {
                if (current_env == 0 || point.env == current_env || point.env == 0) {
                    auto higher_conn_it = connections.find(point);
                    if (higher_conn_it != connections.end()) {
                        bool can_progress = std::any_of(higher_conn_it->second.begin(), 
                                                       higher_conn_it->second.end(),
                                                       [&current_point](const CGPoint& p) {
                                                           return p.weight > current_point.weight;
                                                       });
                        if (current_point.weight >= end_weight || can_progress) {
                            path.push_back(point);
                            current_point = point;
                            break;
                        }
                    }
                }
            }
            
            // Check for max point at current weight
            std::vector<CGPoint> filtered_points;
            std::copy_if(points_at_weight.begin(), points_at_weight.end(), 
                        std::back_inserter(filtered_points),
                        [current_env](const CGPoint& point) {
                            return current_env == 0 || point.env == current_env || point.env == 0;
                        });
            
            if (!filtered_points.empty()) {
                auto max_it = isFwd ? 
                    std::max_element(filtered_points.begin(), filtered_points.end(),
                                   [](const CGPoint& a, const CGPoint& b) { return a.mac < b.mac; }) :
                    std::min_element(filtered_points.begin(), filtered_points.end(),
                                   [](const CGPoint& a, const CGPoint& b) { return a.mac < b.mac; });
                
                if (!(*max_it == current_point)) {
                    path.push_back(*max_it);
                    path.push_back(current_point);
                }
            }
            
            if (current_point.weight >= end_weight) break;
        }
        
        return path;
    }
    
    void printSummary(const std::vector<CGPoint>& optimal_path = {}) {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "ENVELOPE INTERSECTION ANALYSIS SUMMARY\n";
        std::cout << std::string(80, '=') << "\n";
        
        std::cout << "Envelope 1: " << env1_original.size() << " points\n";
        std::cout << "Envelope 2: " << env2_original.size() << " points\n";
        std::cout << "Intersections found: " << intersections.size() << "\n";
        std::cout << "Envelope 1 split into: " << env1_segments.size() << " segments\n";
        std::cout << "Envelope 2 split into: " << env2_segments.size() << " segments\n";
        std::cout << "Total unique points in graph: " << connections.size() << "\n";
        
        if (!intersections.empty()) {
            std::cout << "\nIntersection Points:\n";
            int i = 1;
            for (const auto& intersection : intersections) {
                int seg1 = std::get<0>(intersection);
                int seg2 = std::get<1>(intersection);
                const CGPoint& point = std::get<2>(intersection);
                
                std::cout << "  " << i++ << ": " << point 
                         << " (Env1 seg " << seg1 << " ↔ Env2 seg " << seg2 << ")\n";
            }
        }
        
        if (!optimal_path.empty()) {
            std::cout << "\nOptimal Path (" << optimal_path.size() << " points):\n";
            int i = 1;
            for (const auto& point : optimal_path) {
                std::cout << "  Step " << i++ << ": " << point 
                         << " (MAC: " << std::fixed << std::setprecision(2) << point.mac << ")\n";
            }
            
            double total_mac = 0.0;
            for (const auto& point : optimal_path) {
                total_mac += point.mac;
            }
            double avg_mac = total_mac / optimal_path.size();
            
            std::cout << "\nPath Statistics:\n";
            std::cout << "  Average MAC: " << std::fixed << std::setprecision(2) << avg_mac << "\n";
            std::cout << "  Weight range: " << std::setprecision(0) << optimal_path.begin()->weight 
                     << " → " << optimal_path.rbegin()->weight << "\n";
        }
    }
};

int main() {
    // Define the envelopes
    std::vector<CGPoint> env1 = {
        CGPoint(21.00, 100000, 1),
        CGPoint(21.00, 200000, 1),
        CGPoint(16.00, 200000, 1),
        CGPoint(16.00, 250000, 1),
        CGPoint(21.00, 250000, 1),
        CGPoint(22.00, 300000, 1),
        CGPoint(17.00, 300000, 1),
        CGPoint(17.00, 350000, 1)
    };
    
    std::vector<CGPoint> env2 = {
        CGPoint(15.00, 150000, 2),
        CGPoint(15.00, 200000, 2),
        CGPoint(20.00, 330000, 2),
        CGPoint(20.00, 350000, 2)
    };
    
    // Create analyzer instance
    EnvelopeIntersectionAnalyzer analyzer(env1, env2);
    
    // Find optimal path
    auto optimal_path = analyzer.findOptimalPath();
    
    // Print summary
    analyzer.printSummary(optimal_path);
    
    std::cout << "\nScript completed successfully!\n";
    
    return 0;
}
