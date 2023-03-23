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


class OwningStringView : public std::string_view {
    std::string owned;

public:
    OwningStringView(std::string&& str, std::string_view view)
        : owned(std::move(str)), std::string_view(view) {}
    OwningStringView(const char *str) : std::string_view(str) {}
    OwningStringView() : std::string_view("") {}

    using std::string_view::operator[];
};


class LLM {
    struct Exception : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    llama_context *ctx;

    struct {
        std::string model = "7B-ggml-model-quant.bin";

        int32_t seed; // RNG seed
        int32_t n_threads = static_cast<int32_t>(std::thread::hardware_concurrency()) / 4;
        int32_t repeat_last_n = 256;  // Last n tokens to penalize
        int32_t n_ctx = 2024; // Context size
        int32_t n_batch = 8; // Batch size

        int32_t top_k = 40;
        float   top_p = 0.5f;
        float   temp  = 0.81f;
        float   repeat_penalty  = 1.17647f;
    } params;

    std::string get_temp_file_path() {
        return "/tmp/discord_llama_"+std::to_string(getpid())+".txt";
    }

    void init() {
        // Get llama parameters
        puts("30");
        auto lparams = llama_context_default_params();
        lparams.seed = params.seed;
        lparams.n_ctx = 2024;

        // Create context
        puts("31");
        ctx = llama_init_from_file(params.model.c_str(), lparams);

        // Determine the required inference memory per token
        puts("32");
        const std::vector<llama_token> tmp = { 0, 1, 2, 3 };
        llama_eval(ctx, tmp.data(), tmp.size(), 0, params.n_threads);
        puts("33");
    }

public:
    LLM(int32_t seed = 0) {
        // Set random seed
        params.seed = seed?seed:time(NULL);

        // Initialize llama
        init();
    }

    struct State {
        std::string fres;
        std::string end;
        std::vector<llama_token> embd_inp;
        std::vector<llama_token> embd;
        int token_count;
        int n_ctx;
        int n_predict;
        int remaining_tokens;
        int input_consumed;
        int n_past;
        std::vector<llama_token> last_n_tokens;
    };

    /**
     * @brief run Runs the inference
     * @param prompt String to start generation with. Must be null-terminated and should start with a space
     * @param end String to end generation at
     * @param on_tick Function to execute every now and then
     * @return
     */
    OwningStringView run(std::string_view prompt, std::string_view end = {nullptr, 0}, const std::function<bool ()>& on_tick = nullptr) {
        std::string fres;

        // Set end if nullptr
        puts("0");
        end = end.data()?end.data():std::string_view{prompt.data(), 1};

        // Create buffers for tokens
        puts("1");
        std::vector<llama_token> embd_inp(prompt.size());
        std::vector<llama_token> embd;

        // Run tokenizer
        puts("2");
        auto token_count = llama_tokenize(ctx, prompt.data(), embd_inp.data(), embd_inp.size(), true);
        embd_inp.resize(token_count);

        // Do some other preparations
        puts("3");
        const auto n_ctx = llama_n_ctx(ctx);
        const auto n_predict = n_ctx - (int) embd_inp.size();

        // Evaluate
        puts("4");
        llama_eval(ctx, embd_inp.data(), embd_inp.size(), 0, params.n_threads);

        // Prepare some other variables
        puts("5");
        int remaining_tokens = n_predict;
        int input_consumed = 0;
        int n_past = 0;
        std::vector<llama_token> last_n_tokens(16);
        std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

        // Loop until done
        puts("6");
        bool abort = false;
        while (!abort && !fres.ends_with(end)) {
            // Predict
            if (embd.size() > 0) {
                if (llama_eval(ctx, embd.data(), embd.size(), n_past, params.n_threads)) {
                    throw Exception("Failed to eval");
                }
            }

            n_past += embd.size();
            embd.clear();

            if ((int) embd_inp.size() <= input_consumed) {
                // Out of user input, sample next token
                const float top_k          = params.top_k;
                const float top_p          = params.top_p;
                const float temp           = params.temp;
                const float repeat_penalty = params.repeat_penalty;

                llama_token id = 0;

                // Get logits and sample
                {
                    auto logits = llama_get_logits(ctx);

                    logits[llama_token_eos()] = 0;

                    id = llama_sample_top_p_top_k(ctx, last_n_tokens.data(), last_n_tokens.size(), top_k, top_p, temp, repeat_penalty);

                    last_n_tokens.erase(last_n_tokens.begin());
                    last_n_tokens.push_back(id);
                }

                // Add it to the context
                embd.push_back(id);

                // Decrement remaining sampling budget
                --remaining_tokens;

                // Tick
                if (on_tick && !on_tick()) abort = true;
            } else {
                // Some user input remains from prompt or interaction, forward it to processing
                while ((int) embd_inp.size() > input_consumed) {
                    embd.push_back(embd_inp[input_consumed]);
                    last_n_tokens.erase(last_n_tokens.begin());
                    last_n_tokens.push_back(embd_inp[input_consumed]);
                    ++input_consumed;
                    if ((int) embd.size() >= params.n_batch) {
                        break;
                    }
                }
            }

            // Append text to result
            for (auto id : embd) {
                // Get token as string
                const auto str = llama_token_to_str(ctx, id);

                // Append string to result
                fres.append(str);

                // Debug
                std::cout << str << std::flush;

                // Tick
                if (on_tick && !on_tick()) abort = true;
            }
        }

        // Check final string
        if (fres.size() < prompt.size()) {
            throw Exception("Unknown error: result seems truncated");
        }

        // Return final string
        puts("23");
        return {std::move(fres), std::string_view{fres.data()+prompt.size(), fres.size()-prompt.size()-end.size()}};
    }
};


class Bot {
    RandomGenerator rng;
    Timer last_message_timer;
    std::shared_ptr<bool> stopping;
    std::mutex reply_lock;
    LLM llm;

    dpp::cluster bot;
    dpp::channel channel;
    dpp::snowflake channel_id;
    std::map<dpp::snowflake, dpp::message> history;
    std::vector<dpp::snowflake> my_messages;

    void reply() {
        // Generate prompt
        std::string prompt;
        {
            std::ostringstream prompts;
            // Append channel name
            prompts << "Verlauf des #chat Kanals.\nNotiz: "+bot.me.username+" ist ein freundlicher Chatbot, der immer gerne auf deutsch mitredet.\n\n";
            // Append each message to stream
            for (const auto& [id, msg] : history) {
                bool hide = msg.author.id == bot.me.id;
                if (hide) {
                    for (const auto msg_id : my_messages) {
                        if (id == msg_id) hide = false;
                    }
                }
                if (hide) continue;
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
        if (prompt.size() > 250) {
            history.erase(history.begin());
            return reply();
        }
        // Start new thread
        std::thread([this, prompt = std::move(prompt)] () {
            std::scoped_lock L(reply_lock);
            // Create placeholder message
            auto msg = bot.message_create_sync(dpp::message(channel_id, "Bitte warte... :thinking:"));
            // Run model
            Timer timeout;
            bool timed_out = false;
            OwningStringView output = llm.run(prompt, "\n", [&] () {
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
            });
            // Initialize random generator
            rng.seed(bot.me.id);
            // Start idle auto reply thread
            std::thread([this] () {
                idle_auto_reply();
            }).detach();
        });
        bot.on_message_create([=, this] (const dpp::message_create_t& event) {
            // Make sure message source is correct
            if (event.msg.channel_id != channel_id) return;
            // Make sure message has content
            if (event.msg.content.empty()) return;
            // Append message to history
            auto msg = event.msg;
            str_replace_in_place(msg.content, "<@"+std::to_string(bot.me.id)+'>', bot.me.username);
            history[msg.id] = msg;
            // Attempt to send a reply
            attempt_reply(msg);
            // Reset last message timer
            last_message_timer.reset();
        });
        bot.on_message_update([=, this] (const dpp::message_update_t& event) {
            // Make sure message source is correct
            if (event.msg.channel_id != channel_id) return;
            // Make sure message has content
            if (event.msg.content.empty()) return;
            // Update message in history
            auto msg = event.msg;
            str_replace_in_place(msg.content, "<@"+std::to_string(bot.me.id)+'>', bot.me.username);
            history[msg.id] = msg;
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
