#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#if defined(_MSC_VER)
#include <xmmintrin.h>
#endif

namespace py = pybind11;

namespace {

constexpr std::size_t kCells = 16;
constexpr std::size_t kPatterns = 8;
constexpr std::size_t kSymmetries = 8;
constexpr std::size_t kTupleLength = 6;
constexpr std::size_t kSlots = kPatterns * kSymmetries;
constexpr std::uint32_t kFeatureCount = 1u << 24;
constexpr std::uint64_t kWeightCount = static_cast<std::uint64_t>(kPatterns) * kFeatureCount;

constexpr std::uint8_t kSource[kPatterns][kSymmetries][kTupleLength] = {
    {{0,1,2,4,5,6},{12,8,4,13,9,5},{15,14,13,11,10,9},{3,7,11,2,6,10},
     {3,2,1,7,6,5},{15,11,7,14,10,6},{12,13,14,8,9,10},{0,4,8,1,5,9}},
    {{4,5,6,7,8,9},{13,9,5,1,14,10},{11,10,9,8,7,6},{2,6,10,14,1,5},
     {7,6,5,4,11,10},{14,10,6,2,13,9},{8,9,10,11,4,5},{1,5,9,13,2,6}},
    {{0,1,2,3,4,5},{12,8,4,0,13,9},{15,14,13,12,11,10},{3,7,11,15,2,6},
     {3,2,1,0,7,6},{15,11,7,3,14,10},{12,13,14,15,8,9},{0,4,8,12,1,5}},
    {{2,3,4,5,6,9},{4,0,13,9,5,10},{13,12,11,10,9,6},{11,15,2,6,10,5},
     {1,0,7,6,5,10},{7,3,14,10,6,9},{14,15,8,9,10,5},{8,12,1,5,9,6}},
    {{0,1,2,5,9,10},{12,8,4,9,10,6},{15,14,13,10,6,5},{3,7,11,6,5,9},
     {3,2,1,6,10,9},{15,11,7,10,9,5},{12,13,14,9,5,6},{0,4,8,5,6,10}},
    {{3,4,5,6,7,8},{0,13,9,5,1,14},{12,11,10,9,8,7},{15,2,6,10,14,1},
     {0,7,6,5,4,11},{3,14,10,6,2,13},{15,8,9,10,11,4},{12,1,5,9,13,2}},
    {{1,3,4,5,6,7},{8,0,13,9,5,1},{14,12,11,10,9,8},{7,15,2,6,10,14},
     {2,0,7,6,5,4},{11,3,14,10,6,2},{13,15,8,9,10,11},{4,12,1,5,9,13}},
    {{0,1,4,8,9,10},{12,8,13,14,10,6},{15,14,11,7,6,5},{3,7,2,1,5,9},
     {3,2,7,11,10,9},{15,11,14,13,9,5},{12,13,8,4,5,6},{0,4,1,2,6,10}}
};

inline void prefetch_read(const float* ptr) {
#if defined(_MSC_VER)
    _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(ptr, 0, 1);
#else
    (void)ptr;
#endif
}


std::uint8_t threshold_exponent(std::uint64_t value) {
    if (value == 0) return 0;
    if ((value & (value - 1)) != 0) {
        throw std::invalid_argument("stage thresholds must be powers of two");
    }
    std::uint8_t exponent = 0;
    while (value > 1) {
        value >>= 1;
        ++exponent;
    }
    return exponent;
}

std::uint32_t stage_for(
    const std::uint8_t* board,
    const std::uint64_t* thresholds,
    std::size_t stage_count) {
    std::uint8_t max_exp = 0;
    for (std::size_t i = 0; i < kCells; ++i) {
        if (board[i] > max_exp) max_exp = board[i];
    }
    std::uint32_t stage = 0;
    for (std::size_t k = 1; k < stage_count; ++k) {
        if (max_exp >= threshold_exponent(thresholds[k])) {
            stage = static_cast<std::uint32_t>(k);
        }
    }
    return stage;
}

void pack_features(const std::uint8_t* board, std::uint32_t* out) {
    for (std::size_t pattern = 0; pattern < kPatterns; ++pattern) {
        for (std::size_t symmetry = 0; symmetry < kSymmetries; ++symmetry) {
            std::uint32_t index = 0;
            for (std::size_t cell = 0; cell < kTupleLength; ++cell) {
                std::uint32_t exponent = board[kSource[pattern][symmetry][cell]];
                if (exponent > 15u) exponent = 15u;
                index |= exponent << (4u * static_cast<std::uint32_t>(cell));
            }
            out[pattern * kSymmetries + symmetry] = index;
        }
    }
}

void validate_boards(const py::array_t<std::uint8_t, py::array::c_style>& boards) {
    if (boards.ndim() != 2 || boards.shape(1) != static_cast<py::ssize_t>(kCells)) {
        throw std::invalid_argument("boards must have shape (N, 16)");
    }
}


py::array_t<std::uint32_t> feature_indices(
    const py::array_t<std::uint8_t, py::array::c_style>& boards) {
    validate_boards(boards);
    const auto n = boards.shape(0);
    py::array_t<std::uint32_t> result({n, static_cast<py::ssize_t>(kSlots)});
    const auto* src = boards.data();
    auto* dst = result.mutable_data();
    {
        py::gil_scoped_release release;
        for (py::ssize_t i = 0; i < n; ++i) {
            pack_features(src + i * kCells, dst + i * kSlots);
        }
    }
    return result;
}

py::array_t<std::uint32_t> stage_indices(
    const py::array_t<std::uint8_t, py::array::c_style>& boards,
    const py::array_t<std::uint64_t, py::array::c_style>& thresholds) {
    validate_boards(boards);
    if (thresholds.ndim() != 1 || thresholds.shape(0) < 1 || thresholds.shape(0) > 4) {
        throw std::invalid_argument("stage_thresholds must have shape (1..4,)");
    }
    const auto n = boards.shape(0);
    const auto stages = static_cast<std::size_t>(thresholds.shape(0));
    py::array_t<std::uint32_t> result({n});
    const auto* src = boards.data();
    const auto* thr = thresholds.data();
    auto* dst = result.mutable_data();
    {
        py::gil_scoped_release release;
        for (py::ssize_t i = 0; i < n; ++i) {
            dst[i] = stage_for(src + i * kCells, thr, stages);
        }
    }
    return result;
}


py::array_t<float> afterstate_values(
    const py::array_t<std::uint8_t, py::array::c_style>& boards,
    const py::array_t<float, py::array::c_style>& weights,
    const py::array_t<std::uint64_t, py::array::c_style>& thresholds,
    bool use_prefetch) {
    validate_boards(boards);
    if (weights.ndim() != 2 || weights.shape(1) != static_cast<py::ssize_t>(kWeightCount)) {
        throw std::invalid_argument("weights must have shape (stage_count, 8*2^24)");
    }
    if (weights.shape(0) < 1 || weights.shape(0) > 4) {
        throw std::invalid_argument("weights stage_count must be 1..4");
    }
    if (thresholds.ndim() != 1 || thresholds.shape(0) != weights.shape(0)) {
        throw std::invalid_argument("stage_thresholds must match weights stage_count");
    }

    const auto n = boards.shape(0);
    const auto stage_count = static_cast<std::size_t>(weights.shape(0));
    py::array_t<float> result({n});
    const auto* src = boards.data();
    const auto* weight_data = weights.data();
    const auto* thr = thresholds.data();
    auto* dst = result.mutable_data();

    {
        py::gil_scoped_release release;
        for (py::ssize_t i = 0; i < n; ++i) {
            const auto* board = src + i * kCells;
            const auto stage = stage_for(board, thr, stage_count);
            const auto* stage_weights = weight_data + static_cast<std::uint64_t>(stage) * kWeightCount;
            std::uint32_t features[kSlots];
            const float* cells[kSlots];
            pack_features(board, features);

            for (std::size_t pattern = 0; pattern < kPatterns; ++pattern) {
                const auto* base = stage_weights + static_cast<std::uint64_t>(pattern) * kFeatureCount;
                for (std::size_t symmetry = 0; symmetry < kSymmetries; ++symmetry) {
                    const auto slot = pattern * kSymmetries + symmetry;
                    cells[slot] = base + features[slot];
                    if (use_prefetch) prefetch_read(cells[slot]);
                }
            }
            double total = 0.0;
            for (std::size_t slot = 0; slot < kSlots; ++slot) {
                total += static_cast<double>(*cells[slot]);
            }
            if (!std::isfinite(total) ||
                total > static_cast<double>(std::numeric_limits<float>::max()) ||
                total < -static_cast<double>(std::numeric_limits<float>::max())) {
                throw std::overflow_error("M3 tuple value became non-finite");
            }
            dst[i] = static_cast<float>(total);
        }
    }
    return result;
}

}  // namespace

PYBIND11_MODULE(_m3_tuple_backend, m) {
    m.doc() = "M3 exact U2048NT6 batch tuple evaluator";
    m.def("feature_indices", &feature_indices);
    m.def("stage_indices", &stage_indices);
    m.def("afterstate_values", &afterstate_values,
          py::arg("boards"), py::arg("weights"), py::arg("stage_thresholds"),
          py::arg("prefetch") = true);
    m.attr("FEATURE_COUNT_PER_PATTERN") = py::int_(kFeatureCount);
    m.attr("WEIGHT_COUNT_PER_STAGE") = py::int_(kWeightCount);
}
