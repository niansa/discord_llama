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
    struct {
        std::string model = "7B-ggml-model-quant.bin";

        int32_t seed; // RNG seed
        int32_t n_threads = static_cast<int32_t>(std::thread::hardware_concurrency()) / 4;
        int32_t n_ctx = 2024; // Context size
        int32_t n_batch = 8; // Batch size

        int32_t top_k = 40;
        float   top_p = 0.5f;
        float   temp  = 0.83f;
    } params;

    struct State {
        std::string prompt;
        std::vector<llama_token> embd;
        int n_ctx;
    } state;

    llama_context *ctx = nullptr;
    std::mutex lock;

    void init() {
        // Get llama parameters
        auto lparams = llama_context_default_params();
        lparams.seed = params.seed;
        lparams.n_ctx = 2024;

        // Create context
        ctx = llama_init_from_file(params.model.c_str(), lparams);
        if (!ctx) {
            throw Exception("Failed to initialize llama from file");
        }

        // Initialize some variables
        state.n_ctx = llama_n_ctx(ctx);
    }

public:
    struct Exception : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };
    struct ContextLengthException : public Exception {
        ContextLengthException() : Exception("Max. context length exceeded") {}
    };


    LLM(int32_t seed = 0) {
        // Set random seed
        params.seed = seed?seed:time(NULL);

        // Initialize llama
        init();
    }
    ~LLM() {
        if (ctx) llama_free(ctx);
    }

    void append(const std::string& prompt, const std::function<bool (float progress)>& on_tick = nullptr) {
        std::scoped_lock L(lock);

        // Check if prompt was empty
        const bool was_empty = state.prompt.empty();

        // Append to current prompt
        state.prompt.append(prompt);

        // Resize buffer for tokens
        const auto old_token_count = state.embd.size();
        state.embd.resize(old_token_count+state.prompt.size()+1);

        // Run tokenizer
        const auto token_count = llama_tokenize(ctx, prompt.data(), state.embd.data()+old_token_count, state.embd.size()-old_token_count, was_empty);
        state.embd.resize(old_token_count+token_count);

        // Make sure limit is far from being hit
        if (token_count > state.n_ctx-6) {
            // Yup. *this MUST be decomposed now.
            throw ContextLengthException();
        }

        // Evaluate new tokens
        // TODO: Larger batch size
        std::cout << "Context size: " << old_token_count << '+' << token_count << '=' << old_token_count+token_count << std::endl;
        for (int it = old_token_count; it != old_token_count+token_count; it++) {
            std::cout << llama_token_to_str(ctx, state.embd.data()[it]) << std::flush;
            llama_eval(ctx, state.embd.data()+it, 1, it, params.n_threads);

            // Tick
            if (on_tick) {
                // Calculate progress
                auto progress = float(it) / (state.embd.size()) * 100.f;
                // Run callback
                if (!on_tick(progress)) break;
            }
        }
        std::cout << std::endl;
    }

    std::string run(std::string_view end, const std::function<bool ()>& on_tick = nullptr) {
        std::scoped_lock L(lock);
        std::string fres;

        // Loop until done
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
        state.prompt.append(fres);
        return std::string(fres.data(), fres.size()-end.size());
    }
};


class Bot {
    RandomGenerator rng;
    Timer last_message_timer;
    std::shared_ptr<bool> stopping;
    std::unique_ptr<LLM> llm;
    std::vector<dpp::snowflake> my_messages;
    std::mutex llm_init_lock;

    dpp::cluster bot;
    dpp::channel channel;
    dpp::snowflake channel_id;

    inline static
    std::string create_text_progress_indicator(uint8_t percentage) {
        constexpr uint8_t divisor = 3,
                          width = 100 / divisor;
        // Initialize string
        std::string fres = "`[";
        fres.reserve(width+4);
        // Append progress
        uint8_t bars = percentage / divisor;
        for (uint8_t it = 0; it != width; it++) {
            if (it <= bars) fres.push_back('#');
            else fres.push_back(' ');
        }
        // Finalize and return string
        fres.append("]`");
        return fres;
    }

    void llm_init() {
        if (!llm) {
            // Make sure llm is initialized
            {
                std::unique_lock L(llm_init_lock);
                llm = std::make_unique<LLM>();
            }
            // Create message for reporting progress
            dpp::message msg(channel_id, "Wird geladen...");
            bot.message_create(msg, [this] (const dpp::confirmation_callback_t& cbt) {
                // Error check
                if (cbt.is_error()) {
                    throw std::runtime_error("Failed to send message to channel: "+cbt.get_error().message);
                }
                // Callback for reporting progress
                Timer timer;
                auto msg = cbt.get<dpp::message>();
                auto cb = [&, this] (float progress) mutable {
                    if (timer.get<std::chrono::seconds>() > 5) {
                        msg.content = "Wird geladen... "+create_text_progress_indicator(progress);
                        bot.message_edit(msg);
                        timer.reset();
                    }
                    return true;
                };
                // Add initial context
                llm->append("Verlauf des #"+channel.name+" Kanals.\nNotiz: "+bot.me.username+" ist ein freundlicher Chatbot, der immer gerne auf deutsch mitredet. Er ist freundlich, hilfsbereit und antwortet immer praezise und genau in einer Nachricht. Er macht gerne Witze und mag jeden. Sein Alter ist 16, das aktuelle Jahr ist 2023 und er wurde 2007 geboren.\n\n", cb);
                // Delete progress message
                bot.message_delete(msg.id, msg.channel_id);
            });
        }
    }
    void prompt_add_msg(const dpp::message& msg) {
        try {
            // Format and append line
            for (const auto line : str_split(msg.content, '\n')) {
                Timer timeout;
                llm->append(msg.author.username+": "+std::string(line)+'\n', [&] (float) {
                    if (timeout.get<std::chrono::minutes>() > 1) {
                        std::cerr << "\nWarning: Timeout reached processing message" << std::endl;
                        return false;
                    }
                    return true;
                });
            }
        } catch (const LLM::ContextLengthException&) {
            llm.reset();
            llm_init();
        }
    }
    void prompt_add_trigger() {
        try {
            llm->append(bot.me.username+':');
        } catch (const LLM::ContextLengthException&) {
            llm.reset();
            llm_init();
        }
    }

    void reply() {
        // Start new thread
        std::thread([this] () {
            try {
                // Create placeholder message
                auto msg = bot.message_create_sync(dpp::message(channel_id, "Bitte warte... :thinking:"));
                // Trigger LLM  correctly
                prompt_add_trigger();
                // Run model
                Timer timeout;
                bool timed_out = false;
                auto output = llm->run("\n", [&] () {
                    if (timeout.get<std::chrono::minutes>() > 2) {
                        timed_out = true;
                        std::cerr << "\nWarning: Timeout reached generating message" << std::endl;
                        return false;
                    }
                    return true;
                });
                if (timed_out) output = "Fehler: Zeitüberschreitung";
                // Send resulting message
                msg.content = output;
                bot.message_edit(msg);
            } catch (const std::exception& e) {
                std::cerr << "Warning: " << e.what() << std::endl;
            }
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
                llm_init();
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
            // Reset last message timer
            last_message_timer.reset();
            // Ignore own messages
            if (event.msg.author.id == bot.me.id) {
                // Add message to list of own messages
                my_messages.push_back(event.msg.id);
                return;
            }
            // Move on in another thread
            std::thread([this, msg = event.msg] () mutable {
                try {
                    // Replace bot mentions with bot username
                    str_replace_in_place(msg.content, "<@"+std::to_string(bot.me.id)+'>', bot.me.username);
                    // Attempt to send a reply
                    attempt_reply(msg);
                    // Append message to history
                    prompt_add_msg(msg);
                } catch (const std::exception& e) {
                    std::cerr << "Warning: " << e.what() << std::endl;
                }
            }).detach();
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
