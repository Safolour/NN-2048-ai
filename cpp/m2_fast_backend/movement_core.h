#pragma once

#include <array>
#include <cstdint>

namespace m2fast {

constexpr int kCells = 16;
constexpr int kActions = 4;

struct MoveResult {
    std::array<std::uint8_t, kCells> afterstate{};
    std::int64_t reward = 0;
    bool moved = false;
};

MoveResult move_board(const std::uint8_t* board, std::uint8_t action, bool compute_reward);
void legal_mask_board(const std::uint8_t* board, bool* out4);

}  // namespace m2fast
