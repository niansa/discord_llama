#include "utils.hpp"
#include "config.hpp"
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
#include <utility>
#include <dpp/dpp.h>
#include <fmt/format.h>
#include <justlm.hpp>
#include <justlm_pool.hpp>
#include <anyproc.hpp>
#include <scheduled_thread.hpp>



class Bot {
    CoSched::ScheduledThread sched_thread;
    LM::InferencePool llm_pool;
    std::unique_ptr<Translator> translator;
    std::vector<dpp::snowflake> my_messages;
    std::unordered_map<dpp::snowflake, dpp::user> users;
    std::thread::id llm_tid;
    utils::Timer cleanup_timer;
    sqlite::database db;

    std::mutex command_completion_buffer_mutex;
    std::unordered_map<dpp::snowflake, dpp::slashcommand_t> command_completion_buffer;

    std::mutex thread_embeds_mutex;
    std::unordered_map<dpp::snowflake, dpp::message> thread_embeds;

    dpp::cluster bot;

public:    
    struct BotChannelConfig {
        const std::string *model_name;
        const Configuration::Model *model;
        bool instruct_mode = false;
    };

private:
    Configuration& config;

    inline static
    bool show_console_progress(float progress) {
        std::cout << ' ' << unsigned(progress) << "% \r" << std::flush;
        return true;
    }

    // Must run in llama thread
#   define ENSURE_LLM_THREAD() if (std::this_thread::get_id() != llm_tid) {throw std::runtime_error("LLM execution of '"+std::string(__PRETTY_FUNCTION__)+"' on wrong thread detected");} 0

    // Must run in llama thread
    async::result<std::string_view> llm_translate_to_en(std::string_view text, bool skip = false) {
        ENSURE_LLM_THREAD();
        // Skip if there is no translator
        if (translator == nullptr || skip) {
            std::cout << "(" << config.language << ") " << text << std::endl;
            co_return text;
        }
        // I am optimizing heavily for the above case. This function always returns a reference so a trick is needed here
        static std::string fres;
        fres = text;
        // Replace bot username with [43]
        utils::str_replace_in_place(fres, bot.me.username, "[43]");
        // Run translation
        fres = co_await translator->translate(fres, "EN", show_console_progress);
        // Replace [43] back with bot username
        utils::str_replace_in_place(fres, "[43]", bot.me.username);
        std::cout << text << " --> (EN) " << fres << std::endl;
        co_return fres;
    }

    // Must run in llama thread
    async::result<std::string_view> llm_translate_from_en(std::string_view text, bool skip = false) {
        ENSURE_LLM_THREAD();
        // Skip if there is no translator
        if (translator == nullptr || skip) {
            std::cout << "(" << config.language << ") " << text << std::endl;
            co_return text;
        }
        // I am optimizing heavily for the above case. This function always returns a reference so a trick is needed here
        static std::string fres;
        fres = text;
        // Replace bot username with [43]
        utils::str_replace_in_place(fres, bot.me.username, "[43]");
        // Run translation
        fres = co_await translator->translate(fres, config.language, show_console_progress);
        // Replace [43] back with bot username
        utils::str_replace_in_place(fres, "[43]", bot.me.username);
        std::cout << text << " --> (" << config.language << ") " << fres << std::endl;
        co_return fres;
    }

    LM::Inference::Params llm_get_translation_params() const {
        auto fres = translator->get_params();
        fres.n_threads = config.threads;
        fres.use_mlock = config.mlock;
        return fres;
    }
    LM::Inference::Params llm_get_params(bool instruct_mode = false) const {
        return {
            .n_threads = config.threads,
            .n_ctx = config.ctx_size,
            .n_repeat_last = unsigned(instruct_mode?0:256),
            .temp = 0.3f,
            .repeat_penalty = instruct_mode?1.0f:1.372222224f,
            .use_mlock = config.mlock
        };
    }

    // Must run in llama thread
    async::result<void> llm_restart(const std::shared_ptr<LM::Inference>& inference, const BotChannelConfig& channel_cfg) {
        ENSURE_LLM_THREAD();
        // Deserialize init cache if not instruct mode without prompt file
        if (channel_cfg.instruct_mode && config.instruct_prompt_file == "none") co_return;
        std::ifstream f((*channel_cfg.model_name)+(channel_cfg.instruct_mode?"_instruct_init_cache":"_init_cache"), std::ios::binary);
        co_await inference->deserialize(f);
        // Set params
        inference->params.n_ctx_window_top_bar = inference->get_context_size();
        inference->params.scroll_keep = float(config.scroll_keep) * 0.01f;
    }
    // Must run in llama thread
    async::result<std::shared_ptr<LM::Inference>> llm_start(dpp::snowflake id, const BotChannelConfig& channel_cfg) {
        ENSURE_LLM_THREAD();
        // Get or create inference
        auto inference = co_await llm_pool.create_inference(id, channel_cfg.model->weights_path, llm_get_params(channel_cfg.instruct_mode));
        llm_restart(inference, channel_cfg);
        co_return inference;
    }

    // Must run in llama thread
    async::result<std::shared_ptr<LM::Inference>> llm_get_inference(dpp::snowflake id, const BotChannelConfig& channel_cfg) {
        ENSURE_LLM_THREAD();
        // Get inference
        auto fres = co_await llm_pool.get_inference(id);
        if (!fres) {
            // Start new inference
            fres = co_await llm_start(id, channel_cfg);
        }
        // Set scroll callback
        fres->set_scroll_callback([msg = dpp::message(), channel_id = id] (float progress) {
            std::cout << "WARNING: " << channel_id << " is scrolling! " << progress << "% \r" << std::flush;
            return true;
        });
        // Return inference
        co_return fres;
    }

    // Must run in llama thread
    async::result<void> llm_init() {
        // Run at realtime priority
        CoSched::Task::get_current().set_priority(CoSched::PRIO_REALTIME);
        // Set LLM thread
        llm_tid = std::this_thread::get_id();
        // Translate texts
        if (!config.texts.translated) {
            config.texts.please_wait = co_await llm_translate_from_en(config.texts.please_wait);
            config.texts.model_missing = co_await llm_translate_from_en(config.texts.model_missing);
            config.texts.thread_create_fail = co_await llm_translate_from_en(config.texts.thread_create_fail);
            config.texts.timeout = co_await llm_translate_from_en(config.texts.timeout);
            config.texts.translated = true;
        }
        // Set scroll callback
        auto scroll_cb = [] (float) {
            std::cerr << "Error: Prompt doesn't fit into max. context size!" << std::endl;
            abort();
            return false;
        };
        // Build init caches
        std::string filename;
        for (const auto& [model_name, model_config] : config.models) {
            //TODO: Add hashes to regenerate these as needed
            // Standard prompt
            filename = model_name+"_init_cache";
            if (model_config.is_non_instruct_mode_allowed() &&
                    !std::filesystem::exists(filename) && config.prompt_file != "none") {
                std::cout << "Building init_cache for "+model_name+"..." << std::endl;
                auto llm = LM::Inference::construct(model_config.weights_path, llm_get_params());
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
                llm->set_scroll_callback(scroll_cb);
                llm->append(fmt::format(fmt::runtime(prompt), "bot_name"_a=bot.me.username), show_console_progress);
                // Serialize end result
                std::ofstream f(filename, std::ios::binary);
                llm->serialize(f);
            }
            // Instruct prompt
            filename = model_name+"_instruct_init_cache";
            if (model_config.is_instruct_mode_allowed() &&
                    !std::filesystem::exists(filename) && config.instruct_prompt_file != "none") {
                std::cout << "Building instruct_init_cache for "+model_name+"..." << std::endl;
                auto llm = LM::Inference::construct(model_config.weights_path, llm_get_params());
                // Add initial context
                std::string prompt;
                {
                    // Read whole file
                    std::ifstream f(config.instruct_prompt_file);
                    if (!f) {
                        // Clean up and abort on error
                        std::cerr << "Error: Failed to open instruct prompt file." << std::endl;
                        abort();
                    }
                    std::ostringstream sstr;
                    sstr << f.rdbuf();
                    prompt = sstr.str();
                }
                // Append
                using namespace fmt::literals;
                if (prompt.back() != '\n') prompt.push_back('\n');
                llm->set_scroll_callback(scroll_cb);
                llm->append(fmt::format(fmt::runtime(prompt), "bot_name"_a=bot.me.username)+"\n\n"+model_config.user_prompt, show_console_progress);
                // Serialize end result
                std::ofstream f(filename, std::ios::binary);
                llm->serialize(f);
            }
        }
        // Report complete init
        std::cout << "Init done!" << std::endl;
    }

    // Must run in llama thread
    async::result<void> prompt_add_msg(const dpp::message& msg, const BotChannelConfig& channel_cfg) {
        ENSURE_LLM_THREAD();
        // Get inference
        auto inference = co_await llm_get_inference(msg.channel_id, channel_cfg);
        std::string prefix;
        // Define callback for console progress and timeout
        utils::Timer timeout;
        bool timeout_exceeded = false;
        const auto cb = [&] (float progress) {
            if (timeout.get<std::chrono::minutes>() > 1) {
                std::cerr << "\nWarning: Timeout exceeded processing message" << std::endl;
                timeout_exceeded = true;
                return false;
            }
            return show_console_progress(progress);
        };
        // Instruct mode user prompt
        if (channel_cfg.instruct_mode) {
            // Append line as-is
            inference->append("\n\n"+std::string(co_await llm_translate_to_en(msg.content, channel_cfg.model->no_translate))+'\n', cb);
        } else {
            // Format and append lines
            for (const auto line : utils::str_split(msg.content, '\n')) {
                inference->append(msg.author.username+": "+std::string(co_await llm_translate_to_en(line, channel_cfg.model->no_translate))+'\n', cb);
            }
        }
        // Append line break on timeout
        if (timeout_exceeded) inference->append("\n");
    }
    // Must run in llama thread
    async::result<void> prompt_add_trigger(const std::shared_ptr<LM::Inference>& inference, const BotChannelConfig& channel_cfg) {
        ENSURE_LLM_THREAD();
        if (channel_cfg.instruct_mode) {
            inference->append('\n'+channel_cfg.model->bot_prompt+"\n\n");
        } else {
            inference->append(bot.me.username+':', show_console_progress);
        }
        co_return;
    }

    // Must run in llama thread
    async::result<void> reply(dpp::snowflake id, dpp::message msg, const BotChannelConfig& channel_cfg) {
        ENSURE_LLM_THREAD();
        try {
            // Get inference
            auto inference = co_await llm_get_inference(id, channel_cfg);
            // Trigger LLM  correctly
            co_await prompt_add_trigger(inference, channel_cfg);
            // Run model
            utils::Timer timeout;
            utils::Timer edit_timer;
            bool timeout_exceeded = false;
            msg.content.clear();
            auto output = co_await inference->run(channel_cfg.instruct_mode?channel_cfg.model->user_prompt:"\n", [&] (std::string_view token) {
                std::cout << token << std::flush;
                if (timeout.get<std::chrono::seconds>() > config.timeout) {
                    timeout_exceeded = true;
                    std::cerr << "\nWarning: Timeout exceeded generating message";
                    return false;
                }
                if (config.live_edit) {
                    msg.content += token;
                    if (edit_timer.get<std::chrono::seconds>() > 3) {
                        try {
                            bot.message_edit(msg);
                        } catch (...) {}
                        edit_timer.reset();
                    }
                }
                return true;
            });
            std::cout << std::endl;
            // Handle timeout
            if (timeout_exceeded) {
                if (config.live_edit) {
                    output += "...\n"+config.texts.timeout;
                } else {
                    output = config.texts.timeout;
                }
            }
            // Send resulting message
            msg.content = co_await llm_translate_from_en(output, channel_cfg.model->no_translate);
            bot.message_edit(msg);
            // Prepare for next message
            co_await inference->append("\n");
            if (channel_cfg.model->emits_eos) {
                co_await inference->append("\n"+channel_cfg.model->user_prompt);
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: " << e.what() << std::endl;
        }
    }

    bool attempt_reply(const dpp::message& msg, const BotChannelConfig& channel_cfg) {
        // Reply if message contains username, mention or ID
        if (msg.content.find(bot.me.username) != std::string::npos) {
            create_task_reply(msg.channel_id, channel_cfg);
            return true;
        }
        // Reply if message references user
        for (const auto msg_id : my_messages) {
            if (msg.message_reference.message_id == msg_id) {
                create_task_reply(msg.channel_id, channel_cfg);
                return true;
            }
        }
        // Don't reply otherwise
        return false;
    }

    void create_task_reply(dpp::snowflake id, const BotChannelConfig& channel_cfg) {
        bot.message_create(dpp::message(id, config.texts.please_wait+" :thinking:"), [=, this] (const dpp::confirmation_callback_t& ccb) {
            if (ccb.is_error()) return;
            sched_thread.create_task("Language Model Shutdown", [=, this] () -> async::result<void> {
                co_await reply(id, ccb.get<dpp::message>(), channel_cfg);
            });
        });
    }

    bool is_on_own_shard(dpp::snowflake id) const {
        return (unsigned(id.get_creation_time()) % config.shard_count) == config.shard_id;
    }

    void cleanup() {
        // Clean up InferencePool
        llm_pool.cleanup(config.max_context_age);
        // Reset timer
        cleanup_timer.reset();
    }
    void attempt_cleanup() {
        // Run cleanup if enough time has passed
        if (cleanup_timer.get<std::chrono::seconds>() > config.max_context_age / 4) {
            cleanup();
        }
    }

    std::string create_thread_name(const std::string& model_name, bool instruct_mode) const {
        return "Chat with "+model_name+" " // Model name
                +(instruct_mode?"":"(Non Instruct mode) ") // Instruct mode
                +'#'+(config.shard_count!=1?std::to_string(config.shard_id):""); // Shard ID
    }

    dpp::embed create_chat_embed(dpp::snowflake guild_id, dpp::snowflake thread_id, const std::string& model_name, bool instruct_mode, const dpp::user& author, std::string_view first_message = "") const {
        dpp::embed embed;
        // Create embed
        embed.set_title(create_thread_name(model_name, instruct_mode))
             .set_description("[Open the chat](https://discord.com/channels/"+std::to_string(guild_id)+'/'+std::to_string(thread_id)+')')
             .set_footer(dpp::embed_footer().set_text("Started by "+author.format_username()))
             .set_color(utils::get_unique_color(model_name));
        // Add first message if any
        if (!first_message.empty()) {
            // Make sure it's not too long
            std::string shorted(utils::max_words(first_message, 12));
            if (shorted.size() != first_message.size()) {
                shorted += "...";
            }
            embed.description += "\n\n> "+shorted;
        }
        // Return final result
        return embed;
    }

    // This function is responsible for sharding thread creation
    // A bit ugly but a nice way to avoid having to communicate over any other means than just the Discord API
    bool command_completion_handler(dpp::slashcommand_t&& event, dpp::channel *thread = nullptr) {
        // Stop if this is not the correct shard for thread creation
        if (thread == nullptr) {
            // But register this command first
            std::scoped_lock L(command_completion_buffer_mutex);
            command_completion_buffer.emplace(event.command.id, std::move(event));
            // And then actually stop
            if (!is_on_own_shard(event.command.channel_id)) return false;
        }
        // Get model by name
        auto res = config.models.find(event.command.get_command_name());
        if (res == config.models.end()) {
            // Model does not exit, delete corresponding command
            bot.global_command_delete(event.command.get_command_interaction().id);
            return false;
        }
        const auto& [model_name, model_config] = *res;
        // Get weather to enable instruct mode
        bool instruct_mode;
        {
            const auto& instruct_mode_param = event.get_parameter("instruct_mode");
            if (instruct_mode_param.index()) {
                instruct_mode = std::get<bool>(instruct_mode_param);
            } else {
                instruct_mode = true;
            }
        }
        // Create thread if it doesn't exist or update it if it does
        if (thread == nullptr) {
            bot.thread_create(std::to_string(event.command.id), event.command.channel_id, 1440, dpp::CHANNEL_PUBLIC_THREAD, true, 15,
                              [this, event, instruct_mode, model_name = res->first] (const dpp::confirmation_callback_t& ccb) {
                // Check for error
                if (ccb.is_error()) {
                    std::cout << "Thread creation failed: " << ccb.get_error().message << std::endl;
                    event.reply(dpp::message(config.texts.thread_create_fail).set_flags(dpp::message_flags::m_ephemeral));
                    return;
                }
                std::cout << "Responsible for creating thread: " << ccb.get<dpp::thread>().id << std::endl;
                // Report success
                event.reply(dpp::message("Okay!").set_flags(dpp::message_flags::m_ephemeral));
            });
        } else {
            bool this_shard = is_on_own_shard(thread->id);
            // Add thread to database
            db << "INSERT INTO threads (id, model, instruct_mode, this_shard) VALUES (?, ?, ?, ?);"
               << std::to_string(thread->id) << model_name << instruct_mode << this_shard;
            // Stop if this is not the correct shard for thread finalization
            if (!this_shard) return false;
            // Set name
            std::cout << "Responsible for finalizing thread: " << thread->id << std::endl;
            thread->name = create_thread_name(model_name, instruct_mode);
            bot.channel_edit(*thread);
            // Send embed
            const auto embed = create_chat_embed(event.command.guild_id, thread->id, model_name, instruct_mode, event.command.usr);
            bot.message_create(dpp::message(event.command.channel_id, embed),
                               [this, thread_id = thread->id] (const dpp::confirmation_callback_t& ccb) {
                // Check for error
                if (ccb.is_error()) {
                    std::cerr << "Warning: Failed to create embed: " << ccb.get_error().message << std::endl;
                    return;
                }
                // Get message
                const auto& msg = ccb.get<dpp::message>();
                // Add to embed list
                thread_embeds[thread_id] = msg;
            });
        }
        return true;
    }

public:
    Bot(decltype(config) cfg)
                : config(cfg), bot(cfg.token), db("database.sqlite3"),
                  llm_pool(cfg.pool_size, "discord_llama", !cfg.persistance) {
        // Initialize database
        db << "CREATE TABLE IF NOT EXISTS threads ("
              "    id TEXT PRIMARY KEY NOT NULL,"
              "    model TEXT,"
              "    instruct_mode INTEGER,"
              "    this_shard INTEGER,"
              "    UNIQUE(id)"
              ");";

        // Start Scheduled Thread
        sched_thread.start();

        // Prepare translator
        if (cfg.language != "EN") {
            sched_thread.create_task("Translator", [this] () -> async::result<void> {
                                     std::cout << "Preparing translator..." << std::endl;
                                     translator = std::make_unique<Translator>(config.translation_model_cfg->weights_path, llm_get_translation_params());
                                     co_return;
                                 });
        }

        // Configure bot
        bot.on_log(dpp::utility::cout_logger());
        bot.intents = dpp::i_guild_messages | dpp::i_message_content | dpp::i_message_content;

        // Set callbacks
        bot.on_ready([=, this] (const dpp::ready_t&) { //TODO: Consider removal
            std::cout << "Connected to Discord." << std::endl;
            // Register chat command once
            if (dpp::run_once<struct register_bot_commands>()) {
                for (const auto& [name, model] : config.models) {
                    // Create command
                    dpp::slashcommand command(name, "Start a chat with me", bot.me.id);
                    // Add instruct mode option
                    if (model.instruct_mode_policy == Configuration::Model::InstructModePolicy::Allow) {
                        command.add_option(dpp::command_option(dpp::co_boolean, "instruct_mode", "Weather to enable instruct mode", false));
                    }
                    // Register command
                    bot.global_command_edit(command, [this, command] (const dpp::confirmation_callback_t& ccb) {
                        if (ccb.is_error()) bot.global_command_create(command);
                    });
                }
            }
            if (dpp::run_once<struct LM::Inference>()) {
                // Prepare llm
                sched_thread.create_task("Language Model Initialization", [this] () -> async::result<void> {
                                         co_await llm_init();
                                     });
            }
        });
        bot.on_slashcommand([=, this](dpp::slashcommand_t event) {
            command_completion_handler(std::move(event));
        });
        bot.on_message_create([=, this](...) {
            // Attempt cleanup
            attempt_cleanup();
        });
        bot.on_message_create([=, this](const dpp::message_create_t& event) {
            // Check that this is for thread creation
            if (event.msg.type != dpp::mt_thread_created) return;
            // Get thread that was created
            bot.channel_get(event.msg.id, [this, msg_id = event.msg.id, channel_id = event.msg.channel_id] (const dpp::confirmation_callback_t& ccb) {
                // Stop on error
                if (ccb.is_error()) return;
                // Get thread
                auto thread = ccb.get<dpp::channel>();
                // Attempt to get command ID
                dpp::snowflake command_id;
                try {
                    command_id = thread.name;
                } catch (...) {
                    return;
                }
                // Find command
                std::scoped_lock L(command_completion_buffer_mutex);
                auto res = command_completion_buffer.find(command_id);
                if (res == command_completion_buffer.end()) {
                    return;
                }
                // Complete command
                auto handled = command_completion_handler(std::move(res->second), &thread);
                // Remove command from buffer
                command_completion_buffer.erase(res);
                // Delete this message if we handled it
                if (handled) bot.message_delete(msg_id, channel_id);
            });
        });
        bot.on_message_create([=, this] (const dpp::message_create_t& event) {
            // Update user cache
            users[event.msg.author.id] = event.msg.author;
            // Make sure message has content
            if (event.msg.content.empty()) return;
            // Ignore messges from channel on another shard
            bool this_shard = false;
            db << "SELECT this_shard FROM threads "
                  "WHERE id = ?;"
                    << std::to_string(event.msg.channel_id)
                    >> [&](int _this_shard) {
                this_shard = _this_shard;
            };
            if (!this_shard) return;
            // Ignore own messages
            if (event.msg.author.id == bot.me.id) {
                // Add message to list of own messages
                my_messages.push_back(event.msg.id);
                return;
            }
            // Process message
            try {
                dpp::message msg = event.msg;
                // Check for reset command
                if (msg.content == "!reset") {
                    // Delete inference from pool
                    sched_thread.create_task("Language Model Inference Pool", [=, this] () -> async::result<void> {
                                             co_await llm_pool.delete_inference(msg.channel_id);
                                         });
                    // Delete message
                    bot.message_delete(msg.id, msg.channel_id);
                    return;
                }
                // Replace bot mentions with bot username
                utils::str_replace_in_place(msg.content, "<@"+std::to_string(bot.me.id)+'>', bot.me.username);
                // Replace all other known users
                for (const auto& [user_id, user] : users) {
                    utils::str_replace_in_place(msg.content, "<@"+std::to_string(user_id)+'>', user.username);
                }
                // Get channel config
                BotChannelConfig channel_cfg;
                // Attempt to find thread first...
                bool in_bot_thread = false,
                     model_missing = false;
                db << "SELECT model, instruct_mode, this_shard FROM threads "
                      "WHERE id = ?;"
                        << std::to_string(msg.channel_id)
                        >> [&](const std::string& model_name, int instruct_mode) {
                    in_bot_thread = true;
                    channel_cfg.instruct_mode = instruct_mode;
                    // Find model
                    auto res = config.models.find(model_name);
                    if (res == config.models.end()) {
                        bot.message_create(dpp::message(msg.channel_id, config.texts.model_missing));
                        model_missing = true;
                        return;
                    }
                    channel_cfg.model_name = &res->first;
                    channel_cfg.model = &res->second;
                };
                if (model_missing) return;
                // Otherwise just fall back to the default model config if allowed
                if (!in_bot_thread) {
                    if (config.threads_only) return;
                    channel_cfg.model_name = &config.default_inference_model;
                    channel_cfg.model = config.default_inference_model_cfg;
                }
                // Append message
                sched_thread.create_task("Language Model Inference ("+*channel_cfg.model_name+')', [=, this] () -> async::result<void> {
                                         co_await prompt_add_msg(msg, channel_cfg);
                                     });
                // Handle message somehow...
                if (in_bot_thread) {
                    // Send a reply
                    create_task_reply(msg.channel_id, channel_cfg);
                } else if (msg.content == "!trigger") {
                    // Delete message
                    bot.message_delete(msg.id, msg.channel_id);
                    // Send a reply
                    create_task_reply(msg.channel_id, channel_cfg);
                } else {
                    attempt_reply(msg, channel_cfg);
                }
                // Find thread embed
                std::scoped_lock L(thread_embeds_mutex);
                auto res = thread_embeds.find(msg.channel_id);
                if (res == thread_embeds.end()) {
                    return;
                }
                // Update that embed
                auto embed_msg = res->second;
                embed_msg.embeds[0] = create_chat_embed(msg.guild_id, msg.channel_id, *channel_cfg.model_name, channel_cfg.instruct_mode, msg.author, msg.content);
                bot.message_edit(embed_msg);
                // Remove thread embed linkage from vector
                thread_embeds.erase(res);
            } catch (const std::exception& e) {
                std::cerr << "Warning: " << e.what() << std::endl;
            }
        });
    }

    void start() {
        cleanup();
        bot.start(dpp::st_wait);
    }
    void stop_prepare() {
        if (config.persistance) {
            sched_thread.create_task("Language Model Shutdown", [=, this] () -> async::result<void> {
                                     co_await llm_pool.store_all();
                                 });
        }
        sched_thread.shutdown();
    }
};


int main(int argc, char **argv) {
    // Parse configuration
    Configuration cfg;
    cfg.parse_configs(argc<2?"":argv[1]);

    // Construct and configure bot
    Bot bot(cfg);

    // Set signal handlers if available
#   ifdef sa_sigaction
    struct sigaction sigact;
    static Bot& bot_st = bot;
    static const auto main_thread = std::this_thread::get_id();
    sigact.sa_handler = [] (int) {
        if (std::this_thread::get_id() == main_thread) {
            bot_st.stop_prepare();
            exit(0);
        }
    };
    sigemptyset(&sigact.sa_mask);
    sigact.sa_flags = 0;
    sigaction(SIGTERM, &sigact, nullptr);
    sigaction(SIGINT, &sigact, nullptr);
    sigaction(SIGHUP, &sigact, nullptr);
#   endif

    // Start bot
    bot.start();
}
