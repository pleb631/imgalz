#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_set>
#include <vector>
#include <utility>
#include <cstdint>

namespace py = pybind11;

#ifdef _MSC_VER
    #include <intrin.h>
    #pragma intrinsic(__popcnt)
    #pragma intrinsic(__popcnt64)
#endif

int hamming64(uint64_t h1, uint64_t h2)
{
    uint64_t x = h1 ^ h2;

#ifdef _MSC_VER
    return static_cast<int>(__popcnt64(x));  // MSVC 64位环境
#else
    return __builtin_popcountll(x);          // GCC/Clang
#endif
}

std::pair<std::vector<std::string>, std::unordered_set<std::string>>
filter_similar_hashes(
    const std::vector<std::pair<std::string, uint64_t>>& image_hashes,
    int max_distance)
{
    std::unordered_set<std::string> removed;
    std::vector<std::string> keep;
    int n = image_hashes.size();

    for (int i = 0; i < n; ++i) {
        const auto& [p1, h1] = image_hashes[i];
        if (removed.count(p1)) continue;
        for (int j = i + 1; j < n; ++j) {
            const auto& [p2, h2] = image_hashes[j];
            if (removed.count(p2)) continue;
            if (hamming64(h1, h2) <= max_distance) {
                removed.insert(p2);
            }
        }
        keep.push_back(p1);
    }

    return {keep, removed};
}



int hamming_vector(const std::vector<uint64_t>& h1, const std::vector<uint64_t>& h2)
{
    if (h1.size() != h2.size())
        throw std::runtime_error("Hash lengths must match");

    int dist = 0;
    for (size_t i = 0; i < h1.size(); ++i)
        dist += hamming64(h1[i], h2[i]);
    return dist;
}

std::pair<std::vector<std::string>, std::unordered_set<std::string>>
filter_similar_hashes256(
    const std::vector<std::pair<std::string, std::vector<uint64_t>>>& image_hashes,
    int max_distance)
{
    std::unordered_set<std::string> removed;
    std::vector<std::string> keep;
    int n = image_hashes.size();

    for (int i = 0; i < n; ++i) {
        const auto& [p1, h1] = image_hashes[i];
        if (removed.count(p1)) continue;
        for (int j = i + 1; j < n; ++j) {
            const auto& [p2, h2] = image_hashes[j];
            if (removed.count(p2)) continue;
            if (hamming_vector(h1, h2) <= max_distance) {
                removed.insert(p2);
            }
        }
        keep.push_back(p1);
    }

    return {keep, removed};
}


PYBIND11_MODULE(_hashfilter, m) {
    m.doc() = "Fast image hash filtering implemented in C++";
    m.def("filter_similar_hashes", &filter_similar_hashes,
          "Filter similar image hashes using Hamming distance");
    m.def("filter_similar_hashes256", &filter_similar_hashes256,
          "Filter similar image hashes using Hamming256 distance");
}