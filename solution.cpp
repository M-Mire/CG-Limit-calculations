#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <climits>

struct CGPoint
{
    double weight;
    double mac;
    std::optional<int> vectorPointId;
    std::optional<int> vectorId;

    CGPoint(double w, double m, std::optional<int> vpId = std::nullopt, std::optional<int> vId = std::nullopt)
        : weight(w), mac(m), vectorPointId(vpId), vectorId(vId) {}

    // Add equality operator for comparison
    bool operator==(const CGPoint &other) const
    {
        return weight == other.weight && mac == other.mac &&
               vectorPointId == other.vectorPointId && vectorId == other.vectorId;
    }

    friend std::ostream &operator<<(std::ostream &os, const CGPoint &point)
    {
        return os << "CGPoint(weight=" << point.weight << ", mac=" << point.mac
                  << ", vectorPointId=" << (point.vectorPointId ? std::to_string(*point.vectorPointId) : "null")
                  << ", vectorId=" << (point.vectorId ? std::to_string(*point.vectorId) : "null") << ")";
    }
};

std::optional<int> findContinuationVector(const std::vector<CGPoint> &envelopePoints,
                                          const std::vector<CGPoint> &allPoints, double currentWeight)
{
    std::unordered_set<int> currentVectorIds;
    for (const auto &p : envelopePoints)
        if (p.vectorId)
            currentVectorIds.insert(*p.vectorId);

    // Find first point after current weight
    auto it = std::find_if(allPoints.begin(), allPoints.end(),
                           [currentWeight](const auto &p)
                           { return p.weight > currentWeight; });

    if (it != allPoints.end())
    {
        double nextWeight = it->weight;
        // Check all points at next weight for continuation
        for (; it != allPoints.end() && it->weight == nextWeight; ++it)
        {
            if (it->vectorId && currentVectorIds.count(*it->vectorId))
            {
                return *it->vectorId;
            }
        }
    }
    return std::nullopt;
}

std::vector<CGPoint> orderDuplicateGroup(const std::vector<CGPoint> &group,
                                         const std::vector<CGPoint> &allPoints, double weight, bool isAft = false)
{
    std::vector<CGPoint> envelope, intersection;
    for (const auto &p : group)
    {
        (!p.vectorId.has_value() ? intersection : envelope).push_back(p);
    }

    if (envelope.size() <= 1)
    {
        auto result = group;
        std::sort(result.begin(), result.end(), [isAft](const auto &a, const auto &b)
                  {
                      bool aIsIntersection = !a.vectorId.has_value();
                      bool bIsIntersection = !b.vectorId.has_value();
                      if (aIsIntersection != bIsIntersection)
                          return aIsIntersection;
                      return isAft ? a.mac < b.mac : a.mac > b.mac; // AFT: smallest MAC first, FWD: highest MAC first
                  });
        return result;
    }

    // Build vectorId map with ordered points
    std::unordered_map<int, std::vector<CGPoint>> vectorMap;
    for (const auto &p : envelope)
    {
        if (p.vectorId)
            vectorMap[*p.vectorId].push_back(p);
    }

    // Sort each vector's points by vectorPointId
    for (auto &[vid, points] : vectorMap)
    {
        std::sort(points.begin(), points.end(), [](const auto &a, const auto &b)
                  { return a.vectorPointId.value_or(INT_MAX) < b.vectorPointId.value_or(INT_MAX); });
    }

    // Find first point: smallest vectorPointId per vectorId, then best MAC
    CGPoint firstPoint = std::max_element(vectorMap.begin(), vectorMap.end(),
                                          [isAft](const auto &a, const auto &b)
                                          {
                                              double macA = a.second.front().mac;
                                              double macB = b.second.front().mac;
                                              return isAft ? macA > macB : macA < macB;
                                          })
                             ->second.front();

    // Find continuation point
    auto contVectorId = findContinuationVector(envelope, allPoints, weight);
    std::optional<CGPoint> contPoint;
    if (contVectorId && vectorMap.count(*contVectorId))
    {
        // Get the last point (highest vectorPointId) for continuation vectorId
        auto &contPoints = vectorMap[*contVectorId];
        for (const auto &p : contPoints)
        {
            if (!(p == firstPoint))
            {
                if (!contPoint || p.vectorPointId.value_or(0) > contPoint->vectorPointId.value_or(0))
                {
                    contPoint = p;
                }
            }
        }
    }

    // Build result: first + middle + continuation
    std::vector<CGPoint> result = {firstPoint};

    // Add middle points (sorted: intersections first, then by MAC)
    std::vector<CGPoint> middlePoints;
    for (const auto &p : group)
    {
        if (!(p == firstPoint) && (!contPoint || !(p == *contPoint)))
        {
            middlePoints.push_back(p);
        }
    }

    std::sort(middlePoints.begin(), middlePoints.end(), [isAft](const auto &a, const auto &b)
              {
                  bool aIsIntersection = !a.vectorId.has_value();
                  bool bIsIntersection = !b.vectorId.has_value();
                  if (aIsIntersection != bIsIntersection)
                      return aIsIntersection;
                  return isAft ? a.mac < b.mac : a.mac > b.mac; // AFT: smallest MAC first, FWD: highest MAC first
              });

    result.insert(result.end(), middlePoints.begin(), middlePoints.end());

    if (contPoint)
        result.push_back(*contPoint);
    return result;
}

std::vector<CGPoint> cleanDuplicateWeights(const std::vector<CGPoint> &points, bool isAft = false)
{
    // Find duplicate weights
    std::unordered_set<double> duplicates;
    for (size_t i = 1; i < points.size(); ++i)
    {
        if (points[i].weight == points[i - 1].weight)
        {
            duplicates.insert(points[i].weight);
        }
    }

    std::vector<CGPoint> result;
    for (size_t i = 0; i < points.size();)
    {
        double w = points[i].weight;
        std::vector<CGPoint> group;

        // Collect same-weight points
        for (; i < points.size() && points[i].weight == w; ++i)
        {
            group.push_back(points[i]);
        }

        if (duplicates.count(w))
        {
            auto ordered = orderDuplicateGroup(group, points, w, isAft);
            result.insert(result.end(), ordered.begin(), ordered.end());
        }
        else
        {
            result.insert(result.end(), group.begin(), group.end());
        }
    }
    return result;
}

std::map<double, std::map<std::string, int>> analyzeDuplicates(const std::vector<CGPoint> &points)
{
    std::map<double, std::map<std::string, int>> info;

    for (size_t i = 0; i < points.size();)
    {
        double w = points[i].weight;
        int count = 0, envelope = 0, intersection = 0;
        std::set<int> vectorIds;

        for (; i < points.size() && points[i].weight == w; ++i, ++count)
        {
            if (!points[i].vectorId.has_value())
            {
                intersection++;
            }
            else
            {
                envelope++;
                vectorIds.insert(*points[i].vectorId);
            }
        }

        if (count > 1)
        {
            info[w] = {{"total_points", count}, {"envelope_points", envelope}, {"intersection_points", intersection}, {"unique_vector_ids", (int)vectorIds.size()}};
        }
    }
    return info;
}

int main()
{
    std::vector<CGPoint> points = {
        {160000, 16, 500005641, 500005640},
        {188500, 16, 500005642, 500005640},
        {197800, 15, 500005643, 500005640},
        {233000, 15, 500005644, 500005640},
        {245000, 15, 500005645, 500005640},
        {270000, 15, 500005646, 500005640},
        {280000, 19, 500005647, 500005640},
        {300000, 18}, // intersection point
        {300000, 15, 500005648, 500005640},
        {300000, 21, 500005649, 500005640},
        {300000, 18, 500005613, 500005794},
        {350000, 17, 500005650, 500005640},
        {354600, 18, 500005651, 500005640},
        {368000, 15, 500005652, 500005640},
    };

    std::cout << "Duplicate analysis:\n";
    for (const auto &[weight, info] : analyzeDuplicates(points))
    {
        std::cout << "Weight " << weight << ": ";
        for (const auto &[key, value] : info)
            std::cout << key << "=" << value << " ";
        std::cout << "\n";
    }

    std::cout << "\nOriginal points:\n";
    for (const auto &p : points)
        std::cout << p << "\n";

    // Test FWD (default behavior)
    std::cout << "\n=== FWD PROCESSING ===\n";
    std::cout << "Cleaned points (FWD):\n";
    auto cleanedFwd = cleanDuplicateWeights(points, false);
    for (const auto &p : cleanedFwd)
        std::cout << p << "\n";

    std::cout << "\nWeight 300000 group (FWD):\n";
    int i = 1;
    for (const auto &p : cleanedFwd)
    {
        if (p.weight == 300000)
            std::cout << i++ << ". " << p << "\n";
    }

    // Test AFT (new behavior)
    std::cout << "\n=== AFT PROCESSING ===\n";
    std::cout << "Cleaned points (AFT):\n";
    auto cleanedAft = cleanDuplicateWeights(points, true);
    for (const auto &p : cleanedAft)
        std::cout << p << "\n";

    std::cout << "\nWeight 300000 group (AFT):\n";
    i = 1;
    for (const auto &p : cleanedAft)
    {
        if (p.weight == 300000)
            std::cout << i++ << ". " << p << "\n";
    }

    return 0;
}