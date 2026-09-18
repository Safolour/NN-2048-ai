#include "movement_core.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace m2fast {
namespace {

constexpr std::array<std::array<int, kCells>, kActions> kLineOrder = {{
    {{0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15}},
    {{12,8,4,0, 13,9,5,1, 14,10,6,2, 15,11,7,3}},
    {{0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15}},
    {{3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12}},
}};

constexpr std::uint8_t kMaxExponent = 255;
constexpr std::uint8_t kFirstUnrepresentableRewardExponent = 62;
constexpr std::size_t kRowLutSize = 1u << 16;

struct RowResult {
    std::array<std::uint8_t, 4> values{};
    std::int64_t reward = 0;
};

void checked_add_reward(std::int64_t& total, std::int64_t value) {
    if (value > std::numeric_limits<std::int64_t>::max() - total) {
        throw std::overflow_error(
            "board merge reward aggregation does not fit in numpy.int64");
    }
    total += value;
}

RowResult move_row_generic(
    const std::array<std::uint8_t, 4>& input,
    bool compute_reward) {
    std::array<std::uint8_t, 4> compact{};
    int count = 0;
    for (const auto value : input) {
        if (value != 0) {
            compact[count++] = value;
        }
    }

    RowResult result;
    int out = 0;
    for (int i = 0; i < count;) {
        if (i + 1 < count && compact[i] == compact[i + 1]) {
            const std::uint8_t exponent = compact[i];
            if (exponent == kMaxExponent) {
                throw std::overflow_error(
                    "tile exponent overflow: merging exponent 255 would produce 256");
            }
            if (compute_reward) {
                if (exponent >= kFirstUnrepresentableRewardExponent) {
                    throw std::overflow_error(
                        "merge reward does not fit in numpy.int64: exponent >= 62");
                }
                const std::int64_t value =
                    (std::int64_t{1} << (static_cast<int>(exponent) + 1));
                checked_add_reward(result.reward, value);
            }
            result.values[out++] = static_cast<std::uint8_t>(exponent + 1);
            i += 2;
        } else {
            result.values[out++] = compact[i];
            ++i;
        }
    }
    return result;
}

struct RowLutEntry {
    std::array<std::uint8_t, 4> values{};
    std::int64_t reward = 0;
};

std::vector<RowLutEntry> build_row_lut() {
    std::vector<RowLutEntry> table(kRowLutSize);
    for (std::uint32_t key = 0; key < kRowLutSize; ++key) {
        std::array<std::uint8_t, 4> row{{
            static_cast<std::uint8_t>(key & 0xF),
            static_cast<std::uint8_t>((key >> 4) & 0xF),
            static_cast<std::uint8_t>((key >> 8) & 0xF),
            static_cast<std::uint8_t>((key >> 12) & 0xF),
        }};
        const auto moved = move_row_generic(row, true);
        table[key].values = moved.values;
        table[key].reward = moved.reward;
    }
    return table;
}

const std::vector<RowLutEntry>& row_lut() {
    static const auto table = build_row_lut();
    return table;
}

bool row_is_lut_eligible(const std::array<std::uint8_t, 4>& row) {
    return row[0] <= 15 && row[1] <= 15 && row[2] <= 15 && row[3] <= 15;
}

std::uint16_t row_key(const std::array<std::uint8_t, 4>& row) {
    return static_cast<std::uint16_t>(
        static_cast<std::uint16_t>(row[0]) |
        (static_cast<std::uint16_t>(row[1]) << 4) |
        (static_cast<std::uint16_t>(row[2]) << 8) |
        (static_cast<std::uint16_t>(row[3]) << 12));
}

}  // namespace

MoveResult move_board(const std::uint8_t* board, std::uint8_t action, bool compute_reward) {
    if (action >= kActions) {
        throw std::invalid_argument("action must lie in 0..3");
    }

    MoveResult result;
    std::copy(board, board + kCells, result.afterstate.begin());

    const auto& order = kLineOrder[action];
    std::int64_t reward = 0;
    const auto& lut = row_lut();

    for (int line = 0; line < 4; ++line) {
        std::array<std::uint8_t, 4> row{};
        for (int pos = 0; pos < 4; ++pos) {
            row[pos] = board[order[line * 4 + pos]];
        }

        RowResult moved;
        if (row_is_lut_eligible(row)) {
            const auto& entry = lut[row_key(row)];
            moved.values = entry.values;
            moved.reward = compute_reward ? entry.reward : 0;
        } else {
            moved = move_row_generic(row, compute_reward);
        }

        if (compute_reward) {
            checked_add_reward(reward, moved.reward);
        }
        for (int pos = 0; pos < 4; ++pos) {
            result.afterstate[order[line * 4 + pos]] = moved.values[pos];
        }
    }

    result.reward = compute_reward ? reward : 0;
    result.moved = !std::equal(
        result.afterstate.begin(), result.afterstate.end(), board);
    return result;
}

void legal_mask_board(const std::uint8_t* board, bool* out4) {
    for (std::uint8_t action = 0; action < kActions; ++action) {
        out4[action] = move_board(board, action, false).moved;
    }
}

}  // namespace m2fast
