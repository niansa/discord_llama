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
#include <vector>
#include <map>
#include <mutex>
#include <memory>
#include <dpp/dpp.h>
#include <ggml.h>
#include <llama.h>

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

    struct {
        std::string model = "7B-ggml-model-quant.bin";

        int32_t seed; // RNG seed
        int32_t n_threads = static_cast<int32_t>(std::thread::hardware_concurrency()) / 4;
        int32_t n_ctx = 2024; // Context size
        int32_t n_batch = 8; // Batch size

        int32_t top_k = 40;
        float   top_p = 0.5f;
        float   temp  = 0.81f;
    } params;

    struct State {
        std::string prompt;
        std::vector<llama_token> embd;
        int n_ctx;
    } state;

    llama_context *ctx;
    std::mutex lock;

    void init() {
        // Get llama parameters
        puts("30");
        auto lparams = llama_context_default_params();
        lparams.seed = params.seed;
        lparams.n_ctx = 2024;

        // Create context
        puts("31");
        ctx = llama_init_from_file(params.model.c_str(), lparams);
        puts("32");

        // Initialize some variables
        state.n_ctx = llama_n_ctx(ctx);
    }

public:
    LLM(int32_t seed = 0) {
        // Set random seed
        params.seed = seed?seed:time(NULL);

        // Initialize llama
        init();
    }

    void append(const std::string& prompt) {
        std::scoped_lock L(lock);

        // Check if prompt was empty
        const bool was_empty = state.prompt.empty();

        // Append to current prompt
        printf("ddd %s\n", prompt.c_str());
        state.prompt.append(prompt);

        // Resize buffer for tokens
        puts("cccc");
        const auto old_token_count = state.embd.size();
        state.embd.resize(old_token_count+state.prompt.size()+1);

        // Run tokenizer
        puts("bbbb");
        const auto token_count = llama_tokenize(ctx, prompt.data(), state.embd.data()+old_token_count, state.embd.size()-old_token_count, was_empty);
        state.embd.resize(old_token_count+token_count);

        // Evaluate new tokens
        // TODO: Larger batch size
        printf("aaa %lu+%d=%lu\n", old_token_count, token_count, old_token_count+token_count);
        for (int it = old_token_count; it != old_token_count+token_count; it++) {
            printf("aaa %i %s\n", it, llama_token_to_str(ctx, state.embd.data()[it]));
            llama_eval(ctx, state.embd.data()+it, 1, it, params.n_threads);
        }
    }

    std::string run(std::string_view end, const std::function<bool ()>& on_tick = nullptr) {
        std::scoped_lock L(lock);
        std::string fres;

        // Loop until done
        puts("6");
        bool abort = false;
        while (!abort && !fres.ends_with(end)) {

            // Sample top p and top k
            const auto id = llama_sample_top_p_top_k(ctx, nullptr, 0, params.top_k, params.top_p, params.temp, 1.0f);

            // Add token
            state.embd.push_back(id);

            // Get token as string
            const auto str = llama_token_to_str(ctx, id);

            // Debug
            std::cout << str << std::flush;

            // Append string to function result
            fres.append(str);

            // Evaluate token
            // TODO: Larger batch size
            llama_eval(ctx, state.embd.data()+state.embd.size()-1, 1, state.embd.size()-1, params.n_threads);

            // Tick
            if (on_tick && !on_tick()) abort = true;
        }

        // Return final string
        puts("23");
        state.prompt.append(fres);
        return std::string(fres.data(), fres.size()-end.size());
    }
};


class Bot {
    RandomGenerator rng;
    Timer last_message_timer;
    std::shared_ptr<bool> stopping;
    LLM llm;
    std::vector<dpp::snowflake> my_messages;

    dpp::cluster bot;
    dpp::channel channel;
    dpp::snowflake channel_id;

    void prompt_init() {
        llm.append("Verlauf des #chat Kanals.\nNotiz: "+bot.me.username+" ist ein freundlicher Chatbot, der immer gerne auf deutsch mitredet.\n\n");
    }
    void prompt_add_msg(const dpp::message& msg) {
        // Ignore own messages
        if (msg.author.id == bot.me.id) {
            // Add message to list of own messages
            my_messages.push_back(msg.id);
            return;
        }
        // Format and append line
        for (const auto line : str_split(msg.content, '\n')) {
            llm.append(msg.author.username+": ");
            llm.append(std::string(line));
            llm.append("\n");
        }
    }
    void prompt_add_trigger() {
        llm.append(bot.me.username+": ");
    }

    void reply() {
        // Start new thread
        std::thread([this] () {
            // Create placeholder message
            auto msg = bot.message_create_sync(dpp::message(channel_id, "Bitte warte... :thinking:"));
            // Trigger LLM  correctly
            prompt_add_trigger();
            // Run model
            Timer timeout;
            bool timed_out = false;
            auto output = llm.run("\n", [&] () {
                if (timeout.get<std::chrono::minutes>() > 4) {
                    timed_out = true;
                    return false;
                }
                return true;
            });
            if (timed_out) output = "Fehler: Zeitüberschreitung";
            // Send resulting message
            msg.content = output;
            bot.message_edit(msg);
        }).detach();
    }

    void idle_auto_reply() {
        auto s = stopping;
        do {
            // Wait for a bit
            std::this_thread::sleep_for(std::chrono::minutes(5));
            // Check if last message was more than 20 minutes ago
            if (last_message_timer.get<std::chrono::hours>() > 3) {
                // Force reply
                reply();
            }
        } while (!*s);
    }

    void attempt_reply(const dpp::message& msg) {
        // Decide randomly
        /*if (rng.getBool(0.075f)) {
            return reply();
        }*/
        // Reply if message contains username, mention or ID
        if (msg.content.find(bot.me.username) != std::string::npos) {
            return reply();
        }
        // Reply if message references user
        for (const auto msg_id : my_messages) {
            if (msg.message_reference.message_id == msg_id) {
                return reply();
            }
        }
    }

public:
    Bot(const char *token, dpp::snowflake channel_id) : bot(token), channel_id(channel_id) {
        // Configure bot
        bot.on_log(dpp::utility::cout_logger());
        bot.intents = dpp::i_guild_messages | dpp::i_message_content;

        // Set callbacks
        bot.on_ready([=, this] (const dpp::ready_t&) {
            // Get channel
            bot.channel_get(channel_id, [=, this] (const dpp::confirmation_callback_t& cbt) {
                if (cbt.is_error()) {
                    throw std::runtime_error("Failed to get channel: "+cbt.get_error().message);
                }
                channel = cbt.get<dpp::channel>();
                // Initialize random generator
                rng.seed(bot.me.id);
                // Append initial prompt
                prompt_init();
                // Start idle auto reply thread
                std::thread([this] () {
                    idle_auto_reply();
                }).detach();
            });
        });
        bot.on_message_create([=, this] (const dpp::message_create_t& event) {
            // Make sure message source is correct
            if (event.msg.channel_id != channel_id) return;
            // Make sure message has content
            if (event.msg.content.empty()) return;
            // Append message to history
            auto msg = event.msg;
            str_replace_in_place(msg.content, "<@"+std::to_string(bot.me.id)+'>', bot.me.username);
            prompt_add_msg(msg);
            // Attempt to send a reply
            attempt_reply(msg);
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
    // Init GGML
    ggml_time_init();

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
