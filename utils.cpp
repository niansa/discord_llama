#include "utils.hpp"



namespace utils {
std::vector<std::string_view> str_split(std::string_view s, char delimiter, size_t times) {
    std::vector<std::string_view> to_return;
    decltype(s.size()) start = 0, finish = 0;
    while ((finish = s.find_first_of(delimiter, start)) != std::string_view::npos) {
        to_return.emplace_back(s.substr(start, finish - start));
        start = finish + 1;
        if (to_return.size() == times) { break; }
    }
    to_return.emplace_back(s.substr(start));
    return to_return;
}

void str_replace_in_place(std::string& subject, std::string_view search,
                         const std::string& replace) {
    if (search.empty()) return;
    size_t pos = 0;
    while ((pos = subject.find(search, pos)) != std::string::npos) {
         subject.replace(pos, search.length(), replace);
         pos += replace.length();
    }
}

void clean_for_command_name(std::string& value) {
    for (auto& c : value) {
        if (c == '.') c = '_';
        if (isalpha(c)) c = tolower(c);
    }
}

std::string_view max_words(std::string_view text, unsigned count) {
    unsigned word_len = 0,
             word_count = 0,
             idx;
    // Get idx after last word
    for (idx = 0; idx != text.size() && word_count != count; idx++) {
        char c = text[idx];
        if (c == ' ' || word_len == 8) {
            if (word_len != 0) {
                word_count++;
                word_len = 0;
            }
        } else {
            word_len++;
        }
    }
    // Return resulting string
    return {text.data(), idx};
}
}
