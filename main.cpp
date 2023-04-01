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
#include <array>
#include <vector>
#include <map>
#include <mutex>
#include <memory>
#include <dpp/dpp.h>
#include <justlm.hpp>
#include <ThreadPool.h>



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

static inline
std::string clean_string(std::string_view str) {
    std::string fres;
    for (const auto c : str) {
        if ((c >= 0x20 && c <= 0x7E)
         || c == '\n'
         || c == "ä"[0] || c == "ä"[1] || c == "ä"[2]
         || c == "ö"[0] || c == "ö"[1] || c == "ö"[2]
         || c == "ü"[0] || c == "ü"[1] || c == "ü"[2]
         || c == "Ä"[0] || c == "Ä"[1] || c == "Ä"[2]
         || c == "Ö"[0] || c == "Ö"[1] || c == "Ö"[2]
         || c == "Ü"[0] || c == "Ü"[1] || c == "Ü"[2]
         || c == "ß"[0] || c == "ß"[1] || c == "ß"[2]) {
            fres.push_back(c);
        }
    }
    return fres;
}


class Bot {
    RandomGenerator rng;
    ThreadPool tPool{1};
    Timer last_message_timer;
    std::shared_ptr<bool> stopping;
    std::unique_ptr<LM::Inference> llm;
    std::vector<dpp::snowflake> my_messages;
    std::mutex llm_lock;
    std::thread::id llm_tid;

    dpp::cluster bot;
    dpp::channel channel;
    dpp::snowflake channel_id;

    inline static
    std::string create_text_progress_indicator(uint8_t percentage) {
        static constexpr uint8_t divisor = 3,
                                 width = 100 / divisor;
        // Progress bar percentage lookup
        const static auto indicator_lookup = [] () consteval {
            std::array<uint8_t, 101> fres;
            for (uint8_t it = 0; it != 101; it++) {
                fres[it] = it / divisor;
            }
            return fres;
        }();
        // Initialize string
        std::string fres;
        fres.resize(width+4);
        fres[0] = '`';
        fres[1] = '[';
        // Append progress
        const uint8_t bars = indicator_lookup[percentage];
        for (uint8_t it = 0; it != width; it++) {
            if (it < bars) fres[it+2] = '#';
            else fres[it+2] = ' ';
        }
        // Finalize and return string
        fres[width+2] = ']';
        fres[width+3] = '`';
        return fres;
    }

    // Must run in llama thread
#   define ENSURE_LLM_THREAD() if (std::this_thread::get_id() != llm_tid) {throw std::runtime_error("LLM execution of '"+std::string(__PRETTY_FUNCTION__)+"' on wrong thread detected");} 0

    // Must run in llama thread
    void llm_init() {
        if (!llm) {
            // Create params
            LM::Inference::Params params;
            params.use_mlock = false;
            // Make sure llm is initialized
            {
                std::unique_lock L(llm_lock);
                llm = std::make_unique<LM::Inference>("7B-ggml-model-quant.bin", params);
            }
            // Set LLM thread
            llm_tid = std::this_thread::get_id();
            // Create message for reporting progress
            dpp::message msg(channel_id, "Wird initialisiert...");
            bot.message_create(msg, [this] (const dpp::confirmation_callback_t& cbt) {
                // Error check
                if (cbt.is_error()) {
                    throw std::runtime_error("Failed to send message to channel: "+cbt.get_error().message);
                }
                // Callback for reporting progress
                Timer timer;
                auto msg = cbt.get<dpp::message>();
                uint8_t last_progress = 0;
                auto cb = [&, this] (float progress) mutable {
                    uint8_t progress_i = progress;
                    if (timer.get<std::chrono::seconds>() > 5) {
                        msg.content = "Wird geladen... "+create_text_progress_indicator(progress_i)+
                                      " **"+std::to_string(progress_i)+"**% (**"+std::to_string((progress_i-last_progress)/5)+"**%/s)";
                        last_progress = progress_i;
                        bot.message_edit(msg);
                        timer.reset();
                    }
                    return true;
                };
                // Add initial context
                std::unique_lock L(llm_lock);
                llm->append("Verlauf des #"+channel.name+" Kanals.\n"
                            "Notiz 1: "+bot.me.username+" ist ein freundlicher Chatbot, der immer gerne auf deutsch mitredet. Er ist freundlich und hilfsbereit und antwortet immer sofort. Er hat guten Humor und mag jeden. Sein Alter ist 16 und er wurde 2007 geboren.\n"
                            "Notiz 2: Ecki heisst in Wirklichkeit Eckhard Kohlhuber und kommt aus Bayern.\n"
                            "\n"
                            "Bob: "+bot.me.username+" warst du schonmal in Frankreich und wenn ja wo?\n"
                            +bot.me.username+": Ich war in Paris, in den Museen!\n"
                            "Bob: "+bot.me.username+" was bist du eigentlich genau?\n"
                            +bot.me.username+": Ich bin "+bot.me.username+", ein Chatbot der auf einem neuronalen Netzwerk basiert! Ich kann Fragen beantworten, mich in Diskussionen einmischen und die Aktivitaet des Servers steigen. Ich denke mir gerne Sachen aus wie zum Beispiel dass ich in Frankreich war, um menschlicher zu wirken.\n", cb);
                // Delete progress message
                bot.message_delete(msg.id, msg.channel_id);
            });
        }
    }
    // Must run in llama thread
    void prompt_add_msg(const dpp::message& msg) {
        ENSURE_LLM_THREAD();
        try {
            // Make sure message isn't too long
            if (msg.content.size() > 512) {
                return;
            }
            // Format and append line
            std::unique_lock L(llm_lock);
            for (const auto line : str_split(msg.content, '\n')) {
                Timer timeout;
                llm->append(msg.author.username+": "+clean_string(line)+'\n', [&] (float) {
                    if (timeout.get<std::chrono::minutes>() > 1) {
                        std::cerr << "\nWarning: Timeout reached processing message" << std::endl;
                        return false;
                    }
                    return true;
                });
            }
        } catch (const LM::Inference::ContextLengthException&) {
            llm.reset();
            llm_init();
        }
    }
    // Must run in llama thread
    void prompt_add_trigger() {
        ENSURE_LLM_THREAD();
        try {
            std::unique_lock L(llm_lock);
            llm->append(bot.me.username+':');
        } catch (const LM::Inference::ContextLengthException&) {
            llm.reset();
            llm_init();
        }
    }

    // Must run in llama thread
    void reply(const std::function<void ()>& after_placeholder_creation = nullptr) {
        ENSURE_LLM_THREAD();
        try {
            // Create placeholder message
            auto msg = bot.message_create_sync(dpp::message(channel_id, "Bitte warte... :thinking:"));
            // Call after_placeholder_creation callback
            if (after_placeholder_creation) after_placeholder_creation();
            // Trigger LLM  correctly
            prompt_add_trigger();
            // Run model
            Timer timeout;
            bool timed_out = false;
            auto output = llm->run("\n", [&] (std::string_view str) {
                std::cout << str << std::flush;
                if (timeout.get<std::chrono::minutes>() > 2) {
                    timed_out = true;
                    std::cerr << "\nWarning: Timeout reached generating message";
                    return false;
                }
                return true;
            });
            std::cout << std::endl;
            if (timed_out) output = "Fehler: Zeitüberschreitung";
            // Send resulting message
            msg.content = output;
            bot.message_edit(msg);
        } catch (const std::exception& e) {
            std::cerr << "Warning: " << e.what() << std::endl;
        }
    }

    // Must run in llama thread
    void attempt_reply(const dpp::message& msg, const std::function<void ()>& after_placeholder_creation = nullptr) {
        ENSURE_LLM_THREAD();
        // Decide randomly
        /*if (rng.getBool(0.075f)) {
            return reply();
        }*/
        // Reply if message contains username, mention or ID
        if (msg.content.find(bot.me.username) != std::string::npos) {
            return reply(after_placeholder_creation);
        }
        // Reply if message references user
        for (const auto msg_id : my_messages) {
            if (msg.message_reference.message_id == msg_id) {
                return reply(after_placeholder_creation);
            }
        }
    }

    void enqueue_reply() {
        tPool.submit(std::bind(&Bot::reply, this, nullptr));
    }

    void idle_auto_reply() {
        auto s = stopping;
        do {
            // Wait for a bit
            std::this_thread::sleep_for(std::chrono::minutes(5));
            // Check if last message was more than 20 minutes ago
            if (last_message_timer.get<std::chrono::hours>() > 3) {
                // Force reply
                enqueue_reply();
            }
        } while (!*s);
    }

public:
    Bot(const char *token, dpp::snowflake channel_id) : bot(token), channel_id(channel_id) {
        // Initialize thread pool
        tPool.init();

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
                tPool.submit(std::bind(&Bot::llm_init, this));
                // Start idle auto reply thread
                std::thread(std::bind(&Bot::idle_auto_reply, this)).detach();
            });
        });
        bot.on_message_create([=, this] (const dpp::message_create_t& event) {
            // Ignore messages before full startup
            if (!llm) return;
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
            tPool.submit([this, msg = event.msg] () mutable {
                try {
                    // Replace bot mentions with bot username
                    str_replace_in_place(msg.content, "<@"+std::to_string(bot.me.id)+'>', bot.me.username);
                    if (msg.content == "!trigger") {
                        // Delete message
                        bot.message_delete(msg.id, msg.channel_id);
                        // Send a reply
                        reply();
                    } else {
                        tPool.submit([=, this] () {
                            // Attempt to send a reply
                            attempt_reply(msg, [=, this] () {
                                // Append message to history
                                prompt_add_msg(msg);
                            });
                        });
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Warning: " << e.what() << std::endl;
                }
            });
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
