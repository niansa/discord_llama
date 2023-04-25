#include "Timer.hpp"
#include "sqlite_modern_cpp/sqlite_modern_cpp.h"

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
#include <filesystem>
#include <sstream>
#include <mutex>
#include <memory>
#include <dpp/dpp.h>
#include <fmt/format.h>
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
    if (search.empty()) return;
    size_t pos = 0;
    while ((pos = subject.find(search, pos)) != std::string::npos) {
         subject.replace(pos, search.length(), replace);
         pos += replace.length();
    }
}

static
void clean_for_command_name(std::string& value) {
    for (auto& c : value) {
        if (c == '.') c = '_';
        if (isalpha(c)) c = tolower(c);
    }
}
[[nodiscard]] static
std::string clean_command_name(std::string_view input) {
    std::string fres(input);
    clean_for_command_name(fres);
    return fres;
}


class Bot {
    ThreadPool thread_pool{1};
    Timer last_message_timer;
    std::shared_ptr<bool> stopping;
    LM::InferencePool llm_pool;
    std::unique_ptr<Translator> translator;
    std::vector<dpp::snowflake> my_messages;
    std::unordered_map<dpp::snowflake, dpp::user> users;
    std::thread::id llm_tid;
    sqlite::database db;

    std::string_view language;
    dpp::cluster bot;

public:
    struct ModelConfig {
        std::string weight_path,
                    user_prompt,
                    bot_prompt;
        bool emits_eos = false;
        enum class InstructModePolicy {
            Allow = 0b11,
            Force = 0b10,
            Forbid = 0b01
        } instruct_mode_policy = InstructModePolicy::Allow;

        bool is_instruct_mode_allowed() const {
            return static_cast<unsigned>(instruct_mode_policy) & 0b10;
        }
        bool is_non_instruct_mode_allowed() const {
            return static_cast<unsigned>(instruct_mode_policy) & 0b01;
        }
    };
    struct BotChannelConfig {
        const std::string *model_name;
        const ModelConfig *model_config;
        bool instruct_mode = false;
    };
    struct Configuration {
        std::string token,
                    language = "EN",
                    default_inference_model = "13B-vanilla",
                    translation_model = "none",
                    prompt_file = "none",
                    instruct_prompt_file = "none",
                    models_dir = "models";
        unsigned ctx_size = 1012,
                 pool_size = 2,
                 threads = 4,
                 persistance = true;
        bool mlock = false,
             live_edit = false,
             threads_only = true;
        const ModelConfig *default_inference_model_cfg = nullptr,
                          *translation_model_cfg = nullptr;
    };

private:
    const Configuration& config;
    const std::unordered_map<std::string, ModelConfig>& model_configs;

    struct Texts {
        std::string please_wait = "Please wait...",
                    thread_create_fail = "Error: I couldn't create a thread here. Do I have enough permissions?",
                    model_missing = "Error: The model that was used in this thread could no longer be found.",
                    timeout = "Error: Timeout";
        bool translated = false;
    } texts;

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
        // Skip if there is no translator
        if (translator == nullptr) {
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
            fres = translator->translate(fres, "EN", show_console_progress);
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
        // Skip if there is no translator
        if (translator == nullptr) {
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
            fres = translator->translate(fres, language, show_console_progress);
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
        auto fres = translator->get_params();
        fres.n_threads = config.threads;
        fres.use_mlock = config.mlock;
        return fres;
    }
    LM::Inference::Params llm_get_params(bool instruct_mode = false) const {
        return {
            .n_threads = int(config.threads),
            .n_ctx = int(config.ctx_size),
            .n_repeat_last = instruct_mode?0:256,
            .temp = 0.3f,
            .repeat_penalty = instruct_mode?1.0f:1.372222224f,
            .use_mlock = config.mlock
        };
    }

    // Must run in llama thread
    void llm_restart(LM::Inference& inference, const BotChannelConfig& channel_cfg) {
        ENSURE_LLM_THREAD();
        // Deserialize init cache if not instruct mode without prompt file
        if (channel_cfg.instruct_mode && config.instruct_prompt_file == "none") return;
        std::ifstream f((*channel_cfg.model_name)+(channel_cfg.instruct_mode?"_instruct_init_cache":"_init_cache"), std::ios::binary);
        inference.deserialize(f);
    }
    // Must run in llama thread
    LM::Inference &llm_start(dpp::snowflake id, const BotChannelConfig& channel_cfg) {
        ENSURE_LLM_THREAD();
        // Get or create inference
        auto& inference = llm_pool.create_inference(id, channel_cfg.model_config->weight_path, llm_get_params(channel_cfg.instruct_mode));
        llm_restart(inference, channel_cfg);
        return inference;
    }

    // Must run in llama thread
    LM::Inference &llm_get_inference(dpp::snowflake id, const BotChannelConfig& channel_cfg) {
        ENSURE_LLM_THREAD();
        auto inference_opt = llm_pool.get_inference(id);
        if (!inference_opt.has_value()) {
            // Start new inference
            inference_opt = llm_start(id, channel_cfg);
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
            texts.model_missing = llm_translate_from_en(texts.model_missing);
            texts.thread_create_fail = llm_translate_from_en(texts.thread_create_fail);
            texts.timeout = llm_translate_from_en(texts.timeout);
            texts.translated = true;
        }
        // Build init caches
        std::string filename;
        for (const auto& [model_name, model_config] : model_configs) {
            //TODO: Add hashes to regenerate these as needed
            // Standard prompt
            filename = model_name+"_init_cache";
            if (model_config.is_non_instruct_mode_allowed() &&
                    !std::filesystem::exists(filename) && config.prompt_file != "none") {
                std::cout << "Building init_cache for "+model_name+"..." << std::endl;
                LM::Inference llm(model_config.weight_path, llm_get_params());
                // Add initial context
                std::string prompt;
                {
                    // Read whole file
                    std::ifstream f(config.prompt_file);
                    if (!f) {
                        // Clean up and abort on error
                        std::cerr << "Error: Failed to open prompt file." << std::endl;
                        abort();
                    }
                    std::ostringstream sstr;
                    sstr << f.rdbuf();
                    prompt = sstr.str();
                }
                // Append
                using namespace fmt::literals;
                if (prompt.back() != '\n') prompt.push_back('\n');
                llm.append(fmt::format(fmt::runtime(prompt), "bot_name"_a=bot.me.username), show_console_progress);
                // Serialize end result
                std::ofstream f(filename, std::ios::binary);
                llm.serialize(f);
            }
            // Instruct prompt
            filename = model_name+"_instruct_init_cache";
            if (model_config.is_instruct_mode_allowed() &&
                    !std::filesystem::exists(filename) && config.instruct_prompt_file != "none") {
                std::cout << "Building instruct_init_cache for "+model_name+"..." << std::endl;
                LM::Inference llm(model_config.weight_path, llm_get_params());
                // Add initial context
                std::string prompt;
                {
                    // Read whole file
                    std::ifstream f(config.instruct_prompt_file);
                    if (!f) {
                        // Clean up and abort on error
                        std::cerr << "Error: Failed to open prompt file." << std::endl;
                        abort();
                    }
                    std::ostringstream sstr;
                    sstr << f.rdbuf();
                    prompt = sstr.str();
                }
                // Append
                using namespace fmt::literals;
                if (prompt.back() != '\n') prompt.push_back('\n');
                llm.append(fmt::format(fmt::runtime(prompt), "bot_name"_a=bot.me.username)+"\n\n"+model_config.user_prompt, show_console_progress);
                // Serialize end result
                std::ofstream f(filename, std::ios::binary);
                llm.serialize(f);
            }
        }
        // Report complete init
        std::cout << "Init done!" << std::endl;
    }
    // Must run in llama thread
    void prompt_add_msg(const dpp::message& msg, const BotChannelConfig& channel_cfg) {
        ENSURE_LLM_THREAD();
        // Make sure message isn't too long
        if (msg.content.size() > 512) {
            return;
        }
        // Get inference
        auto& inference = llm_get_inference(msg.channel_id, channel_cfg);
        try {
            std::string prefix;
            // Instruct mode user prompt
            if (channel_cfg.instruct_mode) {
                inference.append("\n\n");
            } else {
                prefix = msg.author.username+": ";
            }
            // Format and append lines
            for (const auto line : str_split(msg.content, '\n')) {
                Timer timeout;
                bool timeout_exceeded = false;
                inference.append(prefix+std::string(llm_translate_to_en(line))+'\n', [&] (float progress) {
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
            llm_restart(inference, channel_cfg);
            prompt_add_msg(msg, channel_cfg);
        }
    }
    // Must run in llama thread
    void prompt_add_trigger(LM::Inference& inference, const BotChannelConfig& channel_cfg) {
        ENSURE_LLM_THREAD();
        try {
            if (channel_cfg.instruct_mode) {
                inference.append('\n'+channel_cfg.model_config->bot_prompt+"\n\n");
            } else {
                inference.append(bot.me.username+':', show_console_progress);
            }
        } catch (const LM::Inference::ContextLengthException&) {
            llm_restart(inference, channel_cfg);
        }
    }

    // Must run in llama thread
    void reply(dpp::snowflake id, dpp::message msg, const BotChannelConfig& channel_cfg) {
        ENSURE_LLM_THREAD();
        try {
            // Get inference
            auto& inference = llm_get_inference(id, channel_cfg);
            // Trigger LLM  correctly
            prompt_add_trigger(inference, channel_cfg);
            // Run model
            Timer timeout;
            Timer edit_timer;
            bool timeout_exceeded = false;
            msg.content.clear();
            auto output = inference.run(channel_cfg.instruct_mode?channel_cfg.model_config->user_prompt:"\n", [&] (std::string_view token) {
                std::cout << token << std::flush;
                if (timeout.get<std::chrono::minutes>() > 2) {
                    timeout_exceeded = true;
                    std::cerr << "\nWarning: Timeout exceeded generating message";
                    return false;
                }
                if (config.live_edit) {
                    msg.content += token;
                    if (edit_timer.get<std::chrono::seconds>() > 5) {
                        bot.message_edit(msg);
                        edit_timer.reset();
                    }
                }
                return true;
            });
            std::cout << std::endl;
            // Handle timeout
            if (timeout_exceeded) {
                inference.append("\n");
                output = texts.timeout;
            }
            // Send resulting message
            msg.content = llm_translate_from_en(output);
            bot.message_edit(msg);
            // Prepare for next message
            if (channel_cfg.model_config->emits_eos) {
                inference.append("\n\n"+channel_cfg.model_config->user_prompt);
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: " << e.what() << std::endl;
        }
    }

    bool attempt_reply(const dpp::message& msg, const BotChannelConfig& channel_cfg) {
        // Reply if message contains username, mention or ID
        if (msg.content.find(bot.me.username) != std::string::npos) {
            enqueue_reply(msg.channel_id, channel_cfg);
            return true;
        }
        // Reply if message references user
        for (const auto msg_id : my_messages) {
            if (msg.message_reference.message_id == msg_id) {
                enqueue_reply(msg.channel_id, channel_cfg);
                return true;
            }
        }
        // Don't reply otherwise
        return false;
    }

    void enqueue_reply(dpp::snowflake id, const BotChannelConfig& channel_cfg) {
        bot.message_create(dpp::message(id, texts.please_wait+" :thinking:"), [=, this] (const dpp::confirmation_callback_t& ccb) {
            if (ccb.is_error()) return;
            thread_pool.submit(std::bind(&Bot::reply, this, id, ccb.get<dpp::message>(), channel_cfg));
        });
    }

public:
    Bot(decltype(config) cfg, decltype(model_configs) model_configs)
                : config(cfg), model_configs(model_configs), bot(cfg.token),
                  language(cfg.language), db("database.sqlite3"),
                  llm_pool(cfg.pool_size, "discord_llama", !cfg.persistance) {
        // Initialize database
        db << "CREATE TABLE IF NOT EXISTS threads ("
              "    id TEXT PRIMARY KEY NOT NULL,"
              "    model TEXT,"
              "    instruct_mode INTEGER,"
              "    UNIQUE(id)"
              ");";

        // Configure llm_pool
        llm_pool.set_store_on_destruct(cfg.persistance);

        // Initialize thread pool
        thread_pool.init();

        // Prepare translator
        if (language != "EN") {
            thread_pool.submit([this] () {
                std::cout << "Preparing translator..." << std::endl;
                translator = std::make_unique<Translator>(config.translation_model, llm_get_translation_params());
            });
        }

        // Configure bot
        bot.on_log(dpp::utility::cout_logger());
        bot.intents = dpp::i_guild_messages | dpp::i_message_content;

        // Set callbacks
        bot.on_ready([=, this] (const dpp::ready_t&) { //TODO: Consider removal
            std::cout << "Connected to Discord." << std::endl;
            // Register chat command once
            if (dpp::run_once<struct register_bot_commands>()) {
                for (const auto& [name, model] : model_configs) {
                    // Create command
                    dpp::slashcommand command(name, "Start a chat with me", bot.me.id);
                    // Add instruct mode option
                    if (model.instruct_mode_policy == ModelConfig::InstructModePolicy::Allow) {
                        command.add_option(dpp::command_option(dpp::co_boolean, "instruct_mode", "Weather to enable instruct mode", true));
                    }
                    // Register command
                    bot.global_command_edit(command, [this, command] (const dpp::confirmation_callback_t& ccb) {
                        if (ccb.is_error()) bot.global_command_create(command);
                    });
                }
            }
            if (dpp::run_once<struct LM::Inference>()) {
                // Prepare llm
                thread_pool.submit(std::bind(&Bot::llm_init, this));
            }
        });
        bot.on_slashcommand([=, this](const dpp::slashcommand_t& event) {
            // Get model by name
            auto res = model_configs.find(event.command.get_command_name());
            if (res == model_configs.end()) {
                // Model does not exit, delete corresponding command
                bot.global_command_delete(event.command.id);
                return;
            }
            const auto& [model_name, model_config] = *res;
            // Get weather to enable instruct mode
            bool instruct_mode;
            if (model_config.instruct_mode_policy == ModelConfig::InstructModePolicy::Allow) {
                instruct_mode = std::get<bool>(event.get_parameter("instruct_mode"));
            } else {
                instruct_mode = model_config.instruct_mode_policy == ModelConfig::InstructModePolicy::Force;
            }
            // Create thread
            bot.thread_create("Chat with "+model_name, event.command.channel_id, 1440, dpp::CHANNEL_PUBLIC_THREAD, true, 15,
                              [this, event, instruct_mode, model_name = res->first] (const dpp::confirmation_callback_t& ccb) {
                // Check for error
                if (ccb.is_error()) {
                    std::cout << "Thread creation failed: " << ccb.get_error().message << std::endl;
                    event.reply(dpp::message(texts.thread_create_fail).set_flags(dpp::message_flags::m_ephemeral));
                    return;
                }
                // Get thread
                const auto& thread = ccb.get<dpp::thread>();
                // Add thread to database
                db << "INSERT INTO threads (id, model, instruct_mode) VALUES (?, ?, ?);"
                   << std::to_string(thread.id) << model_name << instruct_mode;
                // Report success
                event.reply(dpp::message("Okay!").set_flags(dpp::message_flags::m_ephemeral));
            });
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
            try {
                dpp::message msg = event.msg;
                // Replace bot mentions with bot username
                str_replace_in_place(msg.content, "<@"+std::to_string(bot.me.id)+'>', bot.me.username);
                // Replace all other known users
                for (const auto& [user_id, user] : users) {
                    str_replace_in_place(msg.content, "<@"+std::to_string(user_id)+'>', user.username);
                }
                // Get channel config
                BotChannelConfig channel_cfg;
                // Attempt to find thread first...
                bool in_bot_thread = false;
                db << "SELECT model, instruct_mode FROM threads "
                      "WHERE id = ?;"
                        << std::to_string(msg.channel_id)
                        >> [&](const std::string& model_name, int instruct_mode) {
                    in_bot_thread = true;
                    channel_cfg.instruct_mode = instruct_mode;
                    // Find model
                    auto res = model_configs.find(model_name);
                    if (res == model_configs.end()) {
                        bot.message_create(dpp::message(msg.channel_id, texts.model_missing));
                        return;
                    }
                    channel_cfg.model_name = &res->first;
                    channel_cfg.model_config = &res->second;
                };
                // Otherwise just fall back to the default model config if allowed
                if (!in_bot_thread) {
                    if (config.threads_only) return;
                    channel_cfg.model_name = &config.default_inference_model;
                    channel_cfg.model_config = config.default_inference_model_cfg;
                }
                // Debug store command
                if (msg.content == "!store") {
                    llm_pool.store_all(); //DEBUG
#                   warning DEBUG CODE!!!
                    return;
                }
                // Append message
                thread_pool.submit([=, this] () {
                    prompt_add_msg(msg, channel_cfg);
                });
                // Handle message somehow...
                if (in_bot_thread) {
                    // Send a reply
                    enqueue_reply(msg.channel_id, channel_cfg);
                } else if (msg.content == "!trigger") {
                    // Delete message
                    bot.message_delete(msg.id, msg.channel_id);
                    // Send a reply
                    enqueue_reply(msg.channel_id, channel_cfg);
                } else {
                    attempt_reply(msg, channel_cfg);
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: " << e.what() << std::endl;
            }
        });
    }

    void start() {
        stopping = std::make_shared<bool>(false);
        bot.start(dpp::st_wait);
        *stopping = true;
    }
};


bool parse_bool(const std::string& value) {
    if (value == "true")
        return true;
    if (value == "false")
        return false;
    std::cerr << "Error: Failed to parse configuration file: Unknown bool (true/false): " << value << std::endl;
    exit(-4);
}
Bot::ModelConfig::InstructModePolicy parse_instruct_mode_policy(const std::string& value) {
    if (value == "allow")
        return Bot::ModelConfig::InstructModePolicy::Allow;
    if (value == "force")
        return Bot::ModelConfig::InstructModePolicy::Force;
    if (value == "forbid")
        return Bot::ModelConfig::InstructModePolicy::Forbid;
    std::cerr << "Error: Failed to parse model configuration file: Unknown instruct mode policy (allow/force/forbid): " << value << std::endl;
    exit(-4);
}

bool file_exists(const auto& p) {
    // Make sure we don't respond to some file that is actually called "none"...
    if (p == "none") return false;
    return std::filesystem::exists(p);
}

int main(int argc, char **argv) {
    // Check arguments
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <config file location>" << std::endl;
        return -1;
    }

    // Parse main configuration
    Bot::Configuration cfg;
    std::ifstream cfgf(argv[1]);
    if (!cfgf) {
        std::cerr << "Error: Failed to open configuration file: " << argv[1] << std::endl;
        exit(-1);
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
        } else if (key == "default_inference_model") {
            cfg.default_inference_model = std::move(value);
        } else if (key == "translation_model") {
            cfg.translation_model = std::move(value);
        } else if (key == "prompt_file") {
            cfg.prompt_file = std::move(value);
        } else if (key == "instruct_prompt_file") {
            cfg.instruct_prompt_file = std::move(value);
        } else if (key == "models_dir") {
            cfg.models_dir = std::move(value);
        } else if (key == "pool_size") {
            cfg.pool_size = std::stoi(value);
        } else if (key == "threads") {
            cfg.threads = std::stoi(value);
        } else if (key == "ctx_size") {
            cfg.ctx_size = std::stoi(value);
        } else if (key == "mlock") {
            cfg.mlock = parse_bool(value);
        } else if (key == "live_edit") {
            cfg.live_edit = parse_bool(value);
        } else if (key == "threads_only") {
            cfg.threads_only = parse_bool(value);
        } else if (key == "persistance") {
            cfg.persistance = parse_bool(value);
        } else if (!key.empty() && key[0] != '#') {
            std::cerr << "Error: Failed to parse configuration file: Unknown key: " << key << std::endl;
            exit(-3);
        }
    }

    // Parse model configurations
    std::unordered_map<std::string, Bot::ModelConfig> models;
    std::filesystem::path models_dir(cfg.models_dir);
    bool allow_non_instruct = false;
    for (const auto& file : std::filesystem::directory_iterator(models_dir)) {
        // Check that file is model config
        if (file.is_directory() ||
            file.path().filename().extension() != ".txt") continue;
        // Get model name
        auto model_name = file.path().filename().string();
        model_name.erase(model_name.size()-4, 4);
        clean_for_command_name(model_name);
        // Parse model config
        Bot::ModelConfig model_cfg;
        std::ifstream cfgf(file.path());
        if (!cfgf) {
            std::cerr << "Error: Failed to open model configuration file: " << file << std::endl;
            exit(-2);
        }
        std::string filename;
        for (std::string key; cfgf >> key;) {
            // Read value
            std::string value;
            std::getline(cfgf, value);
            // Erase all leading spaces
            while (!value.empty() && (value[0] == ' ' || value[0] == '\t')) value.erase(0, 1);
            // Check key and ignore comment lines
            if (key == "filename") {
                filename = std::move(value);
            } else if (key == "user_prompt") {
                model_cfg.user_prompt = std::move(value);
            } else if (key == "bot_prompt") {
                model_cfg.bot_prompt = std::move(value);
            } else if (key == "instruct_mode_policy") {
                model_cfg.instruct_mode_policy = parse_instruct_mode_policy(value);
            } else if (key == "emits_eos") {
                model_cfg.emits_eos = parse_bool(value);
            } else if (!key.empty() && key[0] != '#') {
                std::cerr << "Error: Failed to parse model configuration file: Unknown key: " << key << std::endl;
                exit(-3);
            }
        }
        // Get full path
        model_cfg.weight_path = file.path().parent_path()/filename;
        // Safety checks
        if (filename.empty() || !file_exists(model_cfg.weight_path)) {
            std::cerr << "Error: Failed to parse model configuration file: Invalid weight filename: " << model_name << std::endl;
            exit(-8);
        }
        if (model_cfg.instruct_mode_policy != Bot::ModelConfig::InstructModePolicy::Forbid &&
            (model_cfg.user_prompt.empty() || model_cfg.bot_prompt.empty())) {
            std::cerr << "Error: Failed to parse model configuration file: Instruct mode allowed but user prompt and bot prompt not given: " << model_name << std::endl;
            exit(-9);
        }
        if (model_cfg.instruct_mode_policy != Bot::ModelConfig::InstructModePolicy::Force) {
            allow_non_instruct = true;
        }
        // Add model to list
        const auto& [stored_model_name, stored_model_cfg] = *models.emplace(std::move(model_name), std::move(model_cfg)).first;
        // Set model pointer in config
        if (stored_model_name == cfg.default_inference_model)
            cfg.default_inference_model_cfg = &stored_model_cfg;
        if (stored_model_name == cfg.translation_model)
            cfg.translation_model_cfg = &stored_model_cfg;
    }

    // Safety checks
    if (cfg.language != "EN") {
        if (cfg.translation_model_cfg == nullptr) {
            std::cerr << "Error: Translation model required for non-english language, but is invalid" << std::endl;
            exit(-5);
        }
        if (cfg.translation_model_cfg->instruct_mode_policy == Bot::ModelConfig::InstructModePolicy::Force) {
            std::cerr << "Error: Translation model is required to not have instruct mode forced" << std::endl;
            exit(-10);
        }
        if (cfg.translation_model_cfg->instruct_mode_policy == Bot::ModelConfig::InstructModePolicy::Force) {
            std::cerr << "Warning: Live edit should not be enabled for non-english language" << std::endl;
        }
    }
    if (allow_non_instruct && !file_exists(cfg.prompt_file)) {
        std::cerr << "Error: Prompt file required when allowing non-instruct-mode use, but is invalid" << std::endl;
        exit(-11);
    }
    if (!cfg.threads_only) {
        if (cfg.default_inference_model_cfg == nullptr) {
            std::cerr << "Error: Default model required if not threads only, but is invalid" << std::endl;
            exit(-6);
        }
        if (cfg.default_inference_model_cfg->instruct_mode_policy == Bot::ModelConfig::InstructModePolicy::Force) {
            std::cerr << "Error: Default model must not have instruct mode forced if not threads only" << std::endl;
            exit(-7);
        }
    }

    // Clean model names in config
    clean_for_command_name(cfg.default_inference_model);
    clean_for_command_name(cfg.translation_model);

    // Construct and configure bot
    Bot bot(cfg, models);

    // Start bot
    bot.start();
}
