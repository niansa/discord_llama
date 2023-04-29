#ifndef UTILS_HPP
#define UTILS_HPP
#include <string>
#include <string_view>
#include <initializer_list>
#include <vector>
#include <chrono>


namespace utils {
class Timer {
    std::chrono::time_point<std::chrono::high_resolution_clock> value;

public:
    Timer() {
        reset();
    }

    void reset() {
        value = std::chrono::high_resolution_clock::now();
    }

    template<typename Unit = std::chrono::milliseconds>
    auto get() {
        auto duration = std::chrono::duration_cast<Unit>(std::chrono::high_resolution_clock::now() - value);
        return duration.count();
    }
};


std::vector<std::string_view> str_split(std::string_view s, char delimiter, size_t times = -1);

void str_replace_in_place(std::string& subject, std::string_view search, const std::string& replace);

void clean_for_command_name(std::string& value);

std::string_view max_words(std::string_view text, unsigned count);

inline
uint32_t get_unique_color(const auto& input) {
    auto i = std::hash<typename std::remove_const<typename std::remove_reference<decltype(input)>::type>::type>{}(input);
    const std::initializer_list<uint32_t> colors = {
        0xf44336,
        0xe91e63,
        0x9c27b0,
        0x673ab7,
        0x3f51b5,
        0x2196f3,
        0x03a9f4,
        0x00bcd4,
        0x009688,
        0x4caf50,
        0x8bc34a,
        0xcddc39,
        0xffeb3b,
        0xffc107,
        0xff9800,
        0xff5722,
        0x795548,
        0xcfd8dc
    };
    return *(colors.begin()+(i%colors.size()));
}
}
#endif // UTILS_HPP
