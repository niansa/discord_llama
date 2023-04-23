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
#include <unordered_map>
#include <sstream>
#include <mutex>
#include <memory>
#include <dpp/dpp.h>
#include <justlm.hpp>
#include <justlm_pool.hpp>
#include <anyproc.hpp>
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


class Bot {
    ThreadPool tPool{1};
    Timer last_message_timer;
    std::shared_ptr<bool> stopping;
    LM::InferencePool llm_pool;
    Translator translator;
    std::vector<dpp::snowflake> my_messages;
    std::unordered_map<dpp::snowflake, dpp::user> users;
    std::thread::id llm_tid;

    std::string_view language;
    dpp::cluster bot;

    struct Texts {
        std::string please_wait = "Please wait...",
                    loading = "Loading...",
                    initializing = "Initializing...",
                    timeout = "Error: Timeout";
        bool translated = false;
    } texts;

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

    inline static
    bool show_console_progress(float progress) {
        std::cout << ' ' << unsigned(progress) << "% \r" << std::flush;
        return true;
    }

    // Must run in llama thread
#   define ENSURE_LLM_THREAD() if (std::this_thread::get_id() != llm_tid) {throw std::runtime_error("LLM execution of '"+std::string(__PRETTY_FUNCTION__)+"' on wrong thread detected");} 0

    // Must run in llama thread
    std::string_view llm_translate_to_en(std::string_view text) {
        ENSURE_LLM_THREAD();
        // No need for translation if language is english already
        if (language == "EN") {
            std::cout << "(" << language << ") " << text << std::endl;
            return text;
        }
        // I am optimizing heavily for the above case. This function always returns a reference so a trick is needed here
        static std::string fres;
        fres = text;
        // Replace bot username with [43]
        str_replace_in_place(fres, bot.me.username, "[43]");
        // Run translation
        try {
            fres = translator.translate(fres, "EN", show_console_progress);
        } catch (const LM::Inference::ContextLengthException&) {
            // Handle potential context overflow error
            return "(Translation impossible)";
        }
        // Replace [43] back with bot username
        str_replace_in_place(fres, "[43]", bot.me.username);
        std::cout << text << " --> (EN) " << fres << std::endl;
        return fres;
    }

    // Must run in llama thread
    std::string_view llm_translate_from_en(std::string_view text) {
        ENSURE_LLM_THREAD();
        // No need for translation if language is english already
        if (language == "EN") {
            std::cout << "(" << language << ") " << text << std::endl;
            return text;
        }
        // I am optimizing heavily for the above case. This function always returns a reference so a trick is needed here
        static std::string fres;
        fres = text;
        // Replace bot username with [43]
        str_replace_in_place(fres, bot.me.username, "[43]");
        // Run translation
        try {
            fres = translator.translate(fres, language, show_console_progress);
        } catch (const LM::Inference::ContextLengthException&) {
            // Handle potential context overflow error
            return "(Translation impossible)";
        }
        // Replace [43] back with bot username
        str_replace_in_place(fres, "[43]", bot.me.username);
        std::cout << text << " --> (" << language << ") " << fres << std::endl;
        return fres;
    }

    LM::Inference::Params llm_get_translation_params() const {
        auto fres = translator.get_params();
        fres.n_threads = config.threads;
        fres.use_mlock = config.mlock;
        return fres;
    }
    LM::Inference::Params llm_get_params() const {
        return {
            .n_threads = int(config.threads),
            .n_ctx = 1012,
            .n_repeat_last = 256,
            .temp = 0.3f,
            .repeat_penalty = 1.372222224f,
            .use_mlock = config.mlock
        };
    }

    // Must run in llama thread
    void llm_restart(LM::Inference& inference) {
        // Deserialize init cache
        std::ifstream f("init_cache", std::ios::binary);
        inference.deserialize(f);
    }
    // Must run in llama thread
    LM::Inference &llm_restart(dpp::snowflake id) {
        ENSURE_LLM_THREAD();
        // Get or create inference
        auto& inference = llm_pool.get_or_create_inference(id, config.inference_model, llm_get_params());
        llm_restart(inference);
        return inference;
    }

    // Must run in llama thread
    LM::Inference &llm_get_inference(dpp::snowflake id) {
        ENSURE_LLM_THREAD();
        auto inference_opt = llm_pool.get_inference(id);
        if (!inference_opt.has_value()) {
            // Start new inference
            inference_opt = llm_restart(id);
        }
        return inference_opt.value();
    }

    // Must run in llama thread
    void llm_init() {
        // Set LLM thread
        llm_tid = std::this_thread::get_id();
        // Translate texts
        if (!texts.translated) {
            texts.please_wait = llm_translate_from_en(texts.please_wait);
            texts.initializing = llm_translate_from_en(texts.initializing);
            texts.loading = llm_translate_from_en(texts.loading);
            texts.timeout = llm_translate_from_en(texts.timeout);
            texts.translated = true;
        }
        // Inference for init cache TODO: Don't recreate on each startup
        LM::Inference llm(config.inference_model, llm_get_params());
        std::ofstream f("init_cache", std::ios::binary);
        // Add initial context
        llm.append("History of the discord server.\n"
                   "Note 1: "+bot.me.username+" is a friendly chatbot that is always happy to talk. He is friendly and helpful and always answers immediately. He has a good sense of humor and likes everyone. His age is unknown.\n"
                   "Note 2: Ecki's real name is Eckhard Kohlhuber and he comes from Bavaria.\n" // Little easter egg
                   "\n"
                   "This is the #meta channel.\n"
                   "Bob: "+bot.me.username+" have you ever been to France and if yes where?\n"
                   +bot.me.username+": I was in Paris, in the museums!\n"
                   "Bob: "+bot.me.username+" what are you exactly?\n"
                   +bot.me.username+": I am "+bot.me.username+", your chatbot! I can answer questions and increase the activity of the server.\n"
                   "Bob: Shall we talk about sex? "+bot.me.username+"?\n"
                   +bot.me.username+": No! I will **not** talk about any NSFW topics.\n"
                   "Bob: "+bot.me.username+" How are you?\n"
                   +bot.me.username+": I am quite well! :-)\n"
                   "Ecki: Hey "+bot.me.username+", what is 10 times 90??\n"
                   +bot.me.username+": that is 900!\n", show_console_progress);
        // Serialize end result
        llm.serialize(f);
    }
    // Must run in llama thread
    void prompt_add_msg(const dpp::message& msg) {
        ENSURE_LLM_THREAD();
        // Make sure message isn't too long
        if (msg.content.size() > 512) {
            return;
        }
        // Get inference
        auto& inference = llm_get_inference(msg.channel_id);
        try {
            // Format and append lines
            for (const auto line : str_split(msg.content, '\n')) {
                Timer timeout;
                bool timeout_exceeded = false;
                inference.append(msg.author.username+": "+std::string(llm_translate_to_en(line))+'\n', [&] (float progress) {
                    if (timeout.get<std::chrono::minutes>() > 1) {
                        std::cerr << "\nWarning: Timeout exceeded processing message" << std::endl;
                        timeout_exceeded = true;
                        return false;
                    }
                    return show_console_progress(progress);
                });
                if (timeout_exceeded) inference.append("\n");
            }
        } catch (const LM::Inference::ContextLengthException&) {
            llm_restart(inference);
            prompt_add_msg(msg);
        }
    }
    // Must run in llama thread
    void prompt_add_trigger(dpp::snowflake id) {
        ENSURE_LLM_THREAD();
        auto& inference = llm_get_inference(id);
        try {
            inference.append(bot.me.username+':', show_console_progress);
        } catch (const LM::Inference::ContextLengthException&) {
            llm_restart(inference);
        }
    }

    // Must run in llama thread
    void reply(dpp::snowflake id, const std::function<void ()>& after_placeholder_creation = nullptr) {
        ENSURE_LLM_THREAD();
        try {
            // Create placeholder message
            auto msg = bot.message_create_sync(dpp::message(id, texts.please_wait+" :thinking:"));
            // Call after_placeholder_creation callback
            if (after_placeholder_creation) after_placeholder_creation();
            // Trigger LLM  correctly
            prompt_add_trigger(id);
            // Get inference
            auto& inference = llm_get_inference(id);
            // Run model
            Timer timeout;
            bool timeout_exceeded = false;
            auto output = inference.run("\n", [&] (std::string_view str) {
                std::cout << str << std::flush;
                if (timeout.get<std::chrono::minutes>() > 2) {
                    timeout_exceeded = true;
                    std::cerr << "\nWarning: Timeout exceeded generating message";
                    return false;
                }
                return true;
            });
            std::cout << std::endl;
            if (timeout_exceeded) {
                inference.append("\n");
                output = texts.timeout;
            }
            // Send resulting message
            msg.content = llm_translate_from_en(output);
            bot.message_edit(msg);
        } catch (const std::exception& e) {
            std::cerr << "Warning: " << e.what() << std::endl;
        }
    }

    // Must run in llama thread
    bool attempt_reply(const dpp::message& msg, const std::function<void ()>& after_placeholder_creation = nullptr) {
        ENSURE_LLM_THREAD();
        // Reply if message contains username, mention or ID
        if (msg.content.find(bot.me.username) != std::string::npos) {
            reply(msg.channel_id, after_placeholder_creation);
            return true;
        }
        // Reply if message references user
        for (const auto msg_id : my_messages) {
            if (msg.message_reference.message_id == msg_id) {
                reply(msg.channel_id, after_placeholder_creation);
                return true;
            }
        }
        // Don't reply otherwise
        return false;
    }

    void enqueue_reply(dpp::snowflake id) {
        tPool.submit(std::bind(&Bot::reply, this, id, nullptr));
    }

public:
    struct Configuration {
        std::string token,
                    language = "EN",
                    inference_model = "13B-ggml-model-quant.bin",
                    translation_model = "13B-ggml-model-quant.bin";
        unsigned pool_size = 2,
                 threads = 4,
                 persistance = true;
        bool mlock = false;
    } config;

    Bot(const Configuration& cfg) : config(cfg), bot(cfg.token), language(cfg.language),
                                    llm_pool(cfg.pool_size, "discord_llama", !cfg.persistance), translator(cfg.translation_model, llm_get_translation_params()) {
        // Configure llm_pool
        llm_pool.set_store_on_destruct(cfg.persistance);

        // Initialize thread pool
        tPool.init();

        // Prepare llm
        tPool.submit(std::bind(&Bot::llm_init, this));

        // Configure bot
        bot.on_log(dpp::utility::cout_logger());
        bot.intents = dpp::i_guild_messages | dpp::i_message_content;

        // Set callbacks
        bot.on_ready([=, this] (const dpp::ready_t&) { //TODO: Consider removal
            std::cout << "Connected to Discord." << std::endl;
        });
        bot.on_message_create([=, this] (const dpp::message_create_t& event) {
            // Update user cache
            users[event.msg.author.id] = event.msg.author;
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
                    // Replace all other known users
                    for (const auto& [user_id, user] : users) {
                        str_replace_in_place(msg.content, "<@"+std::to_string(user_id)+'>', user.username);
                    }
                    // Handle message somehow...
                    if (msg.content == "!trigger") {
                        // Delete message
                        bot.message_delete(msg.id, msg.channel_id);
                        // Send a reply
                        reply(msg.channel_id);
                    } else {
                        tPool.submit([=, this] () {
                            // Attempt to send a reply
                            bool replied = attempt_reply(msg, [=, this] () {
                                // Append message to history
                                prompt_add_msg(msg);
                            });
                            // If none was send, still append message to history.
                            if (!replied) {
                                // Append message to history
                                prompt_add_msg(msg);
                            }
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
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <config file location>" << std::endl;
        return -1;
    }

    // Parse configuration
    Bot::Configuration cfg;
    std::ifstream cfgf(argv[1]);
    if (!cfgf) {
        std::cerr << "Failed to open configuration file: " << argv[1] << std::endl;
        return -1;
    }
    for (std::string key; cfgf >> key;) {
        // Read value
        std::string value;
        std::getline(cfgf, value);
        // Erase all leading spaces
        while (!value.empty() && (value[0] == ' ' || value[0] == '\t')) value.erase(0, 1);
        // Check key and ignore comment lines
        if (key == "token") {
            cfg.token = std::move(value);
        } else if (key == "language") {
            cfg.language = std::move(value);
        } else if (key == "inference_model") {
            cfg.inference_model = std::move(value);
        } else if (key == "translation_model") {
            cfg.translation_model = std::move(value);
        } else if (key == "pool_size") {
            cfg.pool_size = std::stoi(value);
        } else if (key == "threads") {
            cfg.threads = std::stoi(value);
        } else if (key == "mlock") {
            cfg.mlock = (value=="true")?true:false;
        } else if (key == "persistance") {
            cfg.persistance = (value=="true")?true:false;
        } else if (!key.empty() && key[0] != '#') {
            std::cerr << "Failed to parse configuration file: Unknown key: " << key << std::endl;
            return -2;
        }
    }

    // Construct and configure bot
    Bot bot(cfg);

    // Start bot
    bot.start();
}
