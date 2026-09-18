#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>

#include "movement_core.h"

namespace py = pybind11;

namespace {

void validate_boards(const py::array_t<std::uint8_t, py::array::c_style>& boards) {
    if (boards.ndim() != 2 || boards.shape(1) != m2fast::kCells) {
        throw std::invalid_argument("boards must have shape (N, 16)");
    }
    if (boards.shape(0) <= 0) {
        throw std::invalid_argument("boards must contain at least one board");
    }
}

py::array_t<bool> legal_mask_batch(
    const py::array_t<std::uint8_t, py::array::c_style>& boards) {
    validate_boards(boards);
    const py::ssize_t n = boards.shape(0);
    py::array_t<bool> out({n, static_cast<py::ssize_t>(m2fast::kActions)});
    const auto* src = boards.data();
    auto* dst = out.mutable_data();

    {
        py::gil_scoped_release release;
        for (py::ssize_t i = 0; i < n; ++i) {
            m2fast::legal_mask_board(
                src + i * m2fast::kCells,
                dst + i * m2fast::kActions);
        }
    }
    return out;
}

py::tuple move_selected_batch(
    const py::array_t<std::uint8_t, py::array::c_style>& boards,
    const py::array_t<std::uint8_t, py::array::c_style>& actions) {
    validate_boards(boards);
    const py::ssize_t n = boards.shape(0);
    if (actions.ndim() != 1 || actions.shape(0) != n) {
        throw std::invalid_argument("actions must have shape (N,)");
    }

    py::array_t<std::uint8_t> afterstates(
        {n, static_cast<py::ssize_t>(m2fast::kCells)});
    py::array_t<std::int64_t> rewards({n});
    py::array_t<bool> moved({n});

    const auto* src = boards.data();
    const auto* act = actions.data();
    auto* out_board = afterstates.mutable_data();
    auto* out_reward = rewards.mutable_data();
    auto* out_moved = moved.mutable_data();

    {
        py::gil_scoped_release release;
        for (py::ssize_t i = 0; i < n; ++i) {
            const auto result = m2fast::move_board(
                src + i * m2fast::kCells, act[i], true);
            for (int cell = 0; cell < m2fast::kCells; ++cell) {
                out_board[i * m2fast::kCells + cell] = result.afterstate[cell];
            }
            out_reward[i] = result.reward;
            out_moved[i] = result.moved;
        }
    }
    return py::make_tuple(afterstates, rewards, moved);
}

}  // namespace

PYBIND11_MODULE(_m2_fast_backend, m) {
    m.doc() = "M2 C++ movement/legal backend with exact high-exponent fallback";
    m.def("legal_mask_batch", &legal_mask_batch);
    m.def("move_selected_batch", &move_selected_batch);
    m.attr("BACKEND_KIND") = "scalar+row-lut";
    m.attr("SIMD_USED") = false;
    m.attr("LUT_USED") = true;
}
