#include "ProcPipe.hpp"
#include "Random.hpp"
#include "Timer.hpp"

#include <string>
#include <string_view>
#include <sstream>
#include <stdexcept>
#include <fstream>
#include <thread>
#include <chrono>
#include <functional>
#include <mutex>
#include <memory>
#include <dpp/dpp.h>

#ifndef _POSIX_VERSION
#   error "Not compatible with non-POSIX systems"
#endif



static
std::vector<std::string_view> str_split(std::string_view s, char delimiter, size_t times = -1) {
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

static
void str_replace_in_place(std::string& subject, std::string_view search,
                         const std::string& replace) {
    size_t pos = 0;
    while ((pos = subject.find(search, pos)) != std::string::npos) {
         subject.replace(pos, search.length(), replace);
         pos += replace.length();
    }
}


class LLM {
    struct Exception : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    ProcPipe<false, true, false> llama;
    struct {
        std::string model = "7B-ggml-model-quant.bin";

        int32_t seed; // RNG seed
        int32_t n_threads = static_cast<int32_t>(std::thread::hardware_concurrency()) / 2;
        int32_t n_predict = 20000; // new tokens to predict
        int32_t repeat_last_n = 256;  // last n tokens to penalize
        int32_t n_ctx = 2024; //context size

        int32_t top_k = 40;
        float   top_p = 0.5f;
        float   temp  = 0.81f;
        float   repeat_penalty  = 1.17647f;
    } params;

    std::string get_temp_file_path() {
        return "/tmp/discord_llama_"+std::to_string(getpid())+".txt";
    }

    void start() {
        // Start process
        const auto exe_path = "./llama.cpp/llama";
        llama.start(exe_path, "-m", params.model.c_str(), "-s", std::to_string(params.seed).c_str(),
                              "-t", std::to_string(params.n_threads).c_str(),
                              "-f", get_temp_file_path().c_str(), "-n", std::to_string(params.n_predict).c_str(),
                              "--top_k", std::to_string(params.top_k).c_str(), "--top_p", std::to_string(params.top_p).c_str(),
                              "--repeat_last_n", std::to_string(params.repeat_last_n).c_str(), "--repeat_penalty", std::to_string(params.repeat_penalty).c_str(),
                              "-c", std::to_string(params.n_ctx).c_str(), "--temp", std::to_string(params.temp).c_str());
    }

public:
    LLM(int32_t seed = 0) {
        // Set random seed
        params.seed = seed?seed:time(NULL);
    }

    std::string run(std::string_view prompt, const char *end = nullptr, const std::function<void ()>& on_tick = nullptr) {
        std::string fres;

        // Write prompt into file
        if (!(std::ofstream(get_temp_file_path()) << prompt)) {
            throw Exception("Failed to write out initial prompt");
        }

        // Start AI
        const char prompt_based_end[2] = {prompt[0], '\0'};
        auto end_length = end?strlen(end):sizeof(prompt_based_end);
        end = end?end:prompt_based_end;
        start();

        // Wait for a bit
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Make sure everything is alright
        if (!llama.isRunning()) {
            throw Exception("Llama didn't start. Read stderr for more info.");
        }

        // Read until done
        do {
            // Receive a byte
            const auto text = llama.recvStd<1>();
            // Break on EOF
            if (text.empty()) break;
            // Debug
            putchar(text[0]);
            fflush(stdout);
            // Append byte to fres
            fres.append(std::string_view{text.data(), text.size()});
            // Check if end is reached
            auto res = fres.rfind(end);
            if (res != fres.npos && res > prompt.size()) {
                break;
            }
            // Tick
            if (on_tick) on_tick();
        } while (llama.isRunning());

        // Erase end
        fres.erase(fres.size()-end_length, end_length);

        // Kill llama
        llama.kill();

        // Return final result
        std::cout << fres << std::endl;
        return fres.substr(prompt.size()+1);
    }
};


class Bot {
    RandomGenerator rng;
    Timer last_message_timer;
    std::shared_ptr<bool> stopping;
    std::mutex llm_lock;

    dpp::cluster bot;
    dpp::channel channel;
    dpp::snowflake channel_id;
    std::vector<dpp::message> history;
    std::vector<dpp::snowflake> my_messages;

    void reply() {
        puts("LLLLLLLLLLLLLLL");
        // Generate prompt
        std::string prompt;
        {
            std::ostringstream prompts;
            // Append channel name
            prompts << "Log des #chat Kanals.\nNotiz: "+bot.me.username+" ist ein freundlicher Chatbot, der gerne mitredet.\n\n";
            // Append each message to stream
            for (const auto& msg : history) {
                if (msg.author.id == bot.me.id && !msg.edited) continue;
                for (const auto line : str_split(msg.content, '\n')) {
                    prompts << msg.author.username << ": " << line << '\n';
                }
            }
            // Make LLM respond
            prompts << bot.me.username << ':';
            // Keep resulting string
            prompt = prompts.str();
        }
        // Make sure prompt isn't to long; if so, erase a message and retry
        if (prompt.size() > 400) {
            history.erase(history.begin());
            puts("WWWWWWWWWWW");
            return reply();
        }
        puts("MMMMMMMMMMM");
        // Start new thread
        std::thread([this, prompt = std::move(prompt)] () {
            // Create placeholder message
            auto msg = bot.message_create_sync(dpp::message(channel_id, "Bitte warte... :thinking:"));
            // Run model
            std::scoped_lock L(llm_lock);
            std::string output;
            Timer typingIndicatorTimer;
            puts("000000000000");
            try {
                output = LLM().run(prompt, "\n");
            } catch (...) {
                std::rethrow_exception(std::current_exception());
            }
            // Send resulting message
            msg.content = output;
            bot.message_edit(msg);
            // Add message to list of my messages
            my_messages.push_back(msg.id); // Unsafe!!
        }).detach();
    }

    void idle_auto_reply() {
        auto s = stopping;
        do {
            // Wait for a bit
            std::this_thread::sleep_for(std::chrono::minutes(5));
            // Check if last message was more than 20 minutes ago
            if (last_message_timer.get<std::chrono::minutes>() > 20) {
                // Force reply
                reply();
            }
        } while (!*s);
    }

    void attempt_reply(const dpp::message& msg) {
        auto content = msg.content;
        // Always reply to 5th message
        if (history.size() == 5) {
            puts("33333333333");
            return reply();
        }
        // Do not reply before 5th message
        if (history.size() > 5) {
            // Decide randomly
            /*if (rng.getBool(0.075f)) {
                return reply();
            }*/
            // Reply if message contains username, mention or ID
            str_replace_in_place(content, "<@"+std::to_string(bot.me.id)+'>', bot.me.username);
            if (content.find(bot.me.username) != std::string::npos) {
                puts("1111111111");
                return reply();
            }
            // Reply if message references user
            for (const auto msg_id : my_messages) {
                if (msg.message_reference.message_id == msg_id) {
                    puts("222222222222");
                    return reply();
                }
            }
        }
    }

public:
    Bot(const char *token, dpp::snowflake channel_id) : bot(token), channel_id(channel_id) {
        bot.on_log(dpp::utility::cout_logger());
        bot.intents = dpp::i_guild_messages | dpp::i_message_content;

        // Set callbacks
        bot.on_ready([=] (const dpp::ready_t&) {
            // Get channel
            bot.channel_get(channel_id, [=] (const dpp::confirmation_callback_t& cbt) {
                if (cbt.is_error()) {
                    throw std::runtime_error("Failed to get channel: "+cbt.get_error().message);
                }
                channel = cbt.get<dpp::channel>();
            });
            // Initialize random generator
            rng.seed(bot.me.id);
            // Start idle auto reply thread
            std::thread([this] () {
                idle_auto_reply();
            }).detach();
        });
        bot.on_message_create([=] (const dpp::message_create_t& event) {
            // Make sure message source is correct
            if (event.msg.channel_id != channel_id) return;
            // Make sure message has content
            if (event.msg.content.empty()) return;
            // Append message to history
            history.push_back(event.msg);
            // Attempt to send a reply
            attempt_reply(event.msg);
            // Reset last message timer
            last_message_timer.reset();
        });
    }

    void start() {
        stopping = std::make_shared<bool>(false);
        bot.start(dpp::st_wait);
        *stopping = true;
    }
};


int main(int argc, char **argv) {
    // Check arguments
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <token> <channel>" << std::endl;
        return -1;
    }

    // Construct and configure bot
    Bot bot(argv[1], std::stoull(argv[2]));

    // Start bot
    bot.start();
}
