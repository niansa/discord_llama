#include "utils.hpp"
#include "config.hpp"
#include "sqlite_modern_cpp/sqlite_modern_cpp.h"

#include <string>
#include <string_view>
#include <stdexcept>
#include <fstream>
#include <thread>
#include <chrono>
#include <functional>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <optional>
#include <mutex>
#include <memory>
#include <utility>
#include <dpp/dpp.h>
#include <fmt/format.h>
#include <justlm.hpp>
#include <justlm_pool.hpp>
#include <cosched2/scheduled_thread.hpp>
#include <cosched2/scheduler_mutex.hpp>



class Bot {
    CoSched::ScheduledThread sched_thread;
    LM::InferencePool llm_pool;
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

    bool check_timeout(utils::Timer& timer, const dpp::message& msg, uint8_t& slow) {
        auto passed = timer.get<std::chrono::seconds>();
        if (passed > config.timeout) {
            auto& task = CoSched::Task::get_current();
            // Calculate new priority
            const CoSched::Priority prio = task.get_priority()-5;
            // Make sure it's above minimum
            if (prio < CoSched::PRIO_LOWEST) {
                // Stop
                slow = 2;
                return false;
            }
            // Decrease priority
            task.set_priority(prio);
            // Add snail reaction
            if (!slow) {
                slow = 1;
                bot.message_add_reaction(msg, "🐌");
            }
            // Reset timeout timer
            timer.reset();
        }
        // No need to stop
        return true;
    }

    // Must run in llama thread
    bool llm_restart(const std::shared_ptr<LM::Inference>& inference, const BotChannelConfig& channel_cfg) {
        ENSURE_LLM_THREAD();
        // Deserialize init cache if not instruct mode without prompt file
        if (channel_cfg.instruct_mode && config.instruct_prompt_file == "none") return true;
        const auto path = (*channel_cfg.model_name)+(channel_cfg.instruct_mode?"_instruct_init_cache":"_init_cache");
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            std::cerr << "Warning: Failed to init cache open file, consider regeneration: " << path << std::endl;
            return false;
        }
        if (!inference->deserialize(f)) {
            return false;
        }
        // Set params
        inference->params.n_ctx_window_top_bar = inference->get_context_size();
        inference->params.scroll_keep = float(config.scroll_keep) * 0.01f;
        return true;
    }
    // Must run in llama thread
    std::shared_ptr<LM::Inference> llm_start(dpp::snowflake id, const BotChannelConfig& channel_cfg) {
        ENSURE_LLM_THREAD();
        // Get or create inference
        auto inference = llm_pool.create_inference(id, channel_cfg.model->weights_path, llm_get_params(channel_cfg.instruct_mode));
        if (!llm_restart(inference, channel_cfg)) {
            std::cerr << "Warning: Failed to deserialize cache: " << inference->get_last_error() << std::endl;
            return nullptr;
        }
        return inference;
    }

    // Must run in llama thread
    std::shared_ptr<LM::Inference> llm_get_inference(dpp::snowflake id, const BotChannelConfig& channel_cfg) {
        ENSURE_LLM_THREAD();
        // Get inference
        auto fres = llm_pool.get_inference(id);
        if (!fres) {
            // Start new inference
            fres = llm_start(id, channel_cfg);
            // Check for error
            if (!fres) {
                return nullptr;
            }
        }
        // Set scroll callback
        fres->set_scroll_callback([msg = dpp::message(), channel_id = id] (float progress) {
            std::cout << "WARNING: " << channel_id << " is scrolling! " << progress << "% \r" << std::flush;
            return true;
        });
        // Return inference
        return fres;
    }

    // Must run in llama thread
    void llm_init() {
        // Run at high priority
        CoSched::Task::get_current().set_priority(CoSched::PRIO_HIGHER);
        // Set LLM thread
        llm_tid = std::this_thread::get_id();
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
                    !std::filesystem::exists(filename)) {
                std::cout << "Building instruct_init_cache for "+model_name+"..." << std::endl;
                auto llm = LM::Inference::construct(model_config.weights_path, llm_get_params());
                // Add initial context
                std::string prompt;
                if (config.instruct_prompt_file != "none" && !model_config.no_instruct_prompt) {
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
                    // Append instruct prompt
                    using namespace fmt::literals;
                    if (prompt.back() != '\n' && !model_config.no_extra_linebreaks) prompt.push_back('\n');
                    llm->set_scroll_callback(scroll_cb);
                    llm->append(fmt::format(fmt::runtime(prompt), "bot_name"_a=bot.me.username, "bot_prompt"_a=model_config.bot_prompt, "user_prompt"_a=model_config.user_prompt)+(model_config.no_extra_linebreaks?"":"\n\n")+model_config.user_prompt, show_console_progress);
                }
                // Append user prompt
                llm->append(model_config.user_prompt);
                // Serialize end result
                std::ofstream f(filename, std::ios::binary);
                llm->serialize(f);
            }
        }
        // Report complete init
        std::cout << "Init done!" << std::endl;
    }

    // Must run in llama thread
    bool prompt_add_msg(const dpp::message& msg, const BotChannelConfig& channel_cfg) {
        ENSURE_LLM_THREAD();
        // Get inference
        auto inference = llm_get_inference(msg.channel_id, channel_cfg);
        if (!inference) {
            std::cerr << "Warning: Failed to get inference" << std::endl;
            return false;
        }
        std::string prefix;
        // Define callback for console progress and timeout
        utils::Timer timeout;
        bool timeout_exceeded = false;
        uint8_t slow = 0;
        const auto cb = [&] (float progress) {
            // Check for timeout
            if (!check_timeout(timeout, msg, slow)) return false;
            // Show progress in console
            return show_console_progress(progress);
        };
        // Instruct mode user prompt
        if (channel_cfg.instruct_mode) {
            // Append line as-is
            if (!inference->append((channel_cfg.model->no_extra_linebreaks?"\n":"\n\n")
                                                +msg.content
                                                +(channel_cfg.model->no_extra_linebreaks?"":"\n"), cb)) {
                std::cerr << "Warning: Failed to append user prompt: " << inference->get_last_error() << std::endl;
                return false;
            }
        } else {
            // Format and append lines
            for (const auto line : utils::str_split(msg.content, '\n')) {
                if (!inference->append(msg.author.username+": "+std::string(line)+'\n', cb)) {
                    std::cerr << "Warning: Failed to append user prompt (single line): " << inference->get_last_error() << std::endl;
                    return false;
                }
            }
        }
        // Append line break on timeout
        if (timeout_exceeded) return inference->append("\n");
        return true;
    }
    // Must run in llama thread
    bool prompt_add_trigger(const std::shared_ptr<LM::Inference>& inference, const BotChannelConfig& channel_cfg) {
        ENSURE_LLM_THREAD();
        if (channel_cfg.instruct_mode) {
            return inference->append((channel_cfg.model->no_extra_linebreaks?"":"\n")
                                                 +channel_cfg.model->bot_prompt
                                                 +(channel_cfg.model->no_extra_linebreaks?"\n":"\n\n"));
        } else {
            return inference->append(bot.me.username+':', show_console_progress);
        }
    }

    // Must run in llama thread
    void reply(dpp::snowflake id, dpp::message& new_msg, const BotChannelConfig& channel_cfg) {
        ENSURE_LLM_THREAD();
        // Get inference
        auto inference = llm_get_inference(id, channel_cfg);
        if (!inference) {
            std::cerr << "Warning: Failed to get inference" << std::endl;
            return;
        }
        // Trigger LLM correctly
        if (!prompt_add_trigger(inference, channel_cfg)) {
            std::cerr << "Warning: Failed to add trigger to prompt: " << inference->get_last_error() << std::endl;
            return;
        }
        if (CoSched::Task::get_current().is_dead()) {
            return;
        }
        // Run model
        utils::Timer timeout;
        utils::Timer edit_timer;
        new_msg.content.clear();
        const std::string reverse_prompt = channel_cfg.instruct_mode?channel_cfg.model->user_prompt:"\n";
        uint8_t slow = 0;
        bool response_too_long = false;
        auto output = inference->run(reverse_prompt, [&] (std::string_view token) {
            std::cout << token << std::flush;
            // Check for timeout
            if (!check_timeout(timeout, new_msg, slow)) return false;
            // Make sure message isn't too long
            if (new_msg.content.size() > 1995-config.texts.length_error.size()) {
                response_too_long = true;
                return false;
            }
            // Edit live
            if (config.live_edit) {
                new_msg.content += token;
                if (edit_timer.get<std::chrono::seconds>() > 3) {
                    try {
                        bot.message_edit(new_msg);
                    } catch (...) {}
                    edit_timer.reset();
                }
            }
            return true;
        });
        if (output.empty()) {
            std::cerr << "Warning: Failed to generate message: " << inference->get_last_error() << std::endl;
            output = '<'+config.texts.empty_response+'>';
        }
        std::cout << std::endl;
        // Handle message length error
        if (response_too_long) {
            output += "...\n"+config.texts.length_error;
        }
        // Handle timeout
        else if (slow == 2) {
            output += "...\n"+config.texts.timeout;
        }
        // Handle termination
        else if (CoSched::Task::get_current().is_dead()) {
            output += "...\n"+config.texts.terminated;
        }
        // Send resulting message
        new_msg.content = std::move(output);
        try {
            bot.message_edit(new_msg);
        } catch (...) {}
        // Tell model about length error
        if (response_too_long) {
            inference->append("... Response interrupted due to length error");
        }
        // Prepare for next message
        if (!channel_cfg.instruct_mode || !channel_cfg.model->no_extra_linebreaks) {
            inference->append("\n");
        }
        if (channel_cfg.instruct_mode && channel_cfg.model->emits_eos) {
            inference->append("\n"+channel_cfg.model->user_prompt);
        }
    }

    bool check_should_reply(const dpp::message& msg) {
        // Reply if message contains username, mention or ID
        if (msg.content.find(bot.me.username) != std::string::npos) {
            return true;
        }
        // Reply if message references user
        for (const auto msg_id : my_messages) {
            if (msg.message_reference.message_id == msg_id) {
                return true;
            }
        }
        // Reply at random
        if (config.random_response_chance) {
            if (!(unsigned(msg.id.get_creation_time()) % config.random_response_chance)) {
                return true;
            }
        }
        // Don't reply otherwise
        return false;
    }

    bool is_on_own_shard(dpp::snowflake id) const {
        return (unsigned(id.get_creation_time()) % config.shard_count) == config.shard_id;
    }

    void cleanup() {
        // Clean up InferencePool
        if (config.max_context_age) llm_pool.cleanup(config.max_context_age);
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
                +(instruct_mode?"":"(Non Instruct mode)") // Instruct mode
                +(config.shard_count!=1?(" #"+std::to_string(config.shard_id)):""); // Shard ID
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
        // Warn about non-instruct mode
        if (instruct_mode == false) {
            embed.description += "\n\n**In the selected mode, the quality is highly degraded**, but the conversation more humorous. Please avoid this if you want helpful responses or want to evaluate the models quality.";
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
                instruct_mode = model_config.instruct_mode_policy != Configuration::Model::InstructModePolicy::Forbid;
            }
        }
        // Create thread if it doesn't exist or update it if it does
        if (thread == nullptr) {
            bot.thread_create(std::to_string(event.command.id), event.command.channel_id, 1440, dpp::CHANNEL_PUBLIC_THREAD, true, 15,
                              [this, event, model_name = res->first] (const dpp::confirmation_callback_t& ccb) {
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
            : llm_pool(cfg.pool_size, "discord_llama", !cfg.persistance),
              db("database.sqlite3"), bot(cfg.token), config(cfg) {
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

        // Configure bot
        bot.on_log(dpp::utility::cout_logger());
        bot.intents = dpp::i_guild_messages | dpp::i_message_content | dpp::i_message_content;

        // Set callbacks
        bot.on_ready([=, this] (const dpp::ready_t&) { //TODO: Consider removal
            std::cout << "Connected to Discord." << std::endl;
            // Register chat command once
            if (dpp::run_once<struct register_bot_commands>()) {
                auto register_command = [this] (dpp::slashcommand&& c) {
                    bot.global_command_edit(c, [this, c] (const dpp::confirmation_callback_t& ccb) {
                        if (ccb.is_error()) bot.global_command_create(c);
                    });
                };
                // Register model commands
                for (const auto& [name, model] : config.models) {
                    // Create command
                    dpp::slashcommand command(name, "Start a chat with me", bot.me.id);
                    // Add instruct mode option
                    if (model.instruct_mode_policy == Configuration::Model::InstructModePolicy::Allow) {
                        command.add_option(dpp::command_option(dpp::co_boolean, "instruct_mode", "Defaults to \"True\" for best output quality. Weather to enable instruct mode", false));
                    }
                    // Register command
                    register_command(std::move(command));
                }
                // Register other commands
                register_command(dpp::slashcommand("ping", "Check my status", bot.me.id));
                register_command(dpp::slashcommand("reset", "Reset this conversation", bot.me.id));
                register_command(dpp::slashcommand("tasklist", "Get list of tasks", bot.me.id));
                //register_command(dpp::slashcommand("taskkill", "Kill a task", bot.me.id)); TODO
            }
            if (dpp::run_once<class LM::Inference>()) {
                // Prepare llm
                sched_thread.create_task("Language Model Initialization", [this] () -> void {
                                         llm_init();
                                     });
            }
        });
        bot.on_slashcommand([=, this](dpp::slashcommand_t event) {
            const auto invalidate_event = [this] (const dpp::slashcommand_t& event) {
                if (is_on_own_shard(event.command.channel_id)) {
                    event.thinking(true, [this, event] (const dpp::confirmation_callback_t& ccb) {
                        event.delete_original_response();
                    });
                }
            };
            // Process basic commands
            const auto& command_name = event.command.get_command_name();
            if (command_name == "ping") {
                // Sender message
                if (is_on_own_shard(event.command.channel_id)) {
                    bot.message_create(dpp::message(event.command.channel_id, "Ping from user "+event.command.usr.format_username()+'!'));
                }
                // Recipient message
                bot.message_create(dpp::message(event.command.channel_id, "Pong from shard "+std::to_string(config.shard_id+1)+'/'+std::to_string(config.shard_count)+'!'));
                // Finalize
                invalidate_event(event);
                return;
            } else if (command_name == "reset") {
                // Delete inference from pool
                sched_thread.create_task("Language Model Inference Pool", [this, id = event.command.channel_id, user = event.command.usr] () -> void {
                    CoSched::Task::get_current().user_data = std::move(user);
                    llm_pool.delete_inference(id);
                });
                // Sender message
                if (is_on_own_shard(event.command.channel_id)) {
                    bot.message_create(dpp::message(event.command.channel_id, "Conversation was reset by "+event.command.usr.format_username()+'!'));
                }
                // Finalize
                invalidate_event(event);
                return;
            } else if (command_name == "tasklist") {
                // Build task list
                sched_thread.create_task("tasklist", [this, event, id = event.command.channel_id, user = event.command.usr] () -> void {
                    auto& task = CoSched::Task::get_current();
                    task.user_data = std::move(user);
                    // Set priority to max
                    task.set_priority(CoSched::PRIO_REALTIME);
                    // Header
                    std::string str = "**__Task List on Shard "+std::to_string(config.shard_id)+"__**\n";
                    // Produce list
                    for (const auto& task : task.get_scheduler().get_tasks()) {
                        // Get user
                        const dpp::user *user = nullptr;
                        {
                            if (task->user_data.has_value()) {
                                user = &std::any_cast<const dpp::user&>(task->user_data);
                            }
                        }
                        // Append line
                        str += fmt::format("- `{}` (State: **{}**, Priority: **{}**, User: **{}**)\n", task->get_name(), task->is_suspended()?"suspended":task->get_state_string(), task->get_priority(), user?user->format_username():bot.me.format_username());
                    }
                    // Delete original thinking response
                    if (is_on_own_shard(event.command.channel_id)) {
                        event.delete_original_response();
                    }
                    // Send list
                    bot.message_create(dpp::message(id, str));
                    return;
                });
                // Finalize
                if (is_on_own_shard(event.command.channel_id)) {
                    event.thinking(false);
                }
                return;
            }
            // Run command completion handler
            command_completion_handler(std::move(event));
        });
        bot.on_message_create([=, this](const dpp::message_create_t&) {
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
            bool this_shard = is_on_own_shard(event.msg.channel_id);
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
                // Copy message
                dpp::message msg = event.msg;
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
                sched_thread.create_task("Language Model Inference ("+*channel_cfg.model_name+" at "+std::to_string(msg.channel_id)+")", [=, this] () -> void {
                    CoSched::Task::get_current().user_data = msg.author;
                    // Create initial message
                    dpp::message placeholder_msg(msg.channel_id, config.texts.please_wait+" :thinking:");
                    // Get task
                    auto &task = CoSched::Task::get_current();
                    // Await previous completion
                    while (true) {
                        // Check that there are no other tasks with the same name
                        bool is_unique = true;
                        bool any_non_suspended = false;
                        for (const auto& other_task : task.get_scheduler().get_tasks()) {
                            if (&task == other_task.get()) continue;
                            if (task.get_name() == other_task->get_name()) {
                                is_unique = false;
                            } else if (!other_task->is_suspended()) {
                                any_non_suspended = true;
                            }
                        }
                        // Stop looking if there is no task that isn't suspended
                        if (!any_non_suspended) break;
                        // Stop looking if task is unique
                        if (is_unique) break;
                        // Suspend, we'll be woken up by that other task
                        task.set_suspended(true);
                        if (!task.yield()) return;
                    }
                    // Check if message should reply
                    bool should_reply = false;
                    if (in_bot_thread) {
                        should_reply = true;
                    } else if (msg.content == "!trigger") {
                        bot.message_delete(msg.id, msg.channel_id);
                        should_reply = true;
                    } else {
                        should_reply = check_should_reply(msg);
                    }
                    if (should_reply) {
                        // Send placeholder
                        placeholder_msg = bot.message_create_sync(placeholder_msg);
                        // Add user message
                        if (!prompt_add_msg(msg, channel_cfg)) {
                            std::cerr << "Warning: Failed to add user message, not going to reply" << std::endl;
                            return;
                        }
                        // Send a reply
                        reply(msg.channel_id, placeholder_msg, channel_cfg);
                    } else {
                        // Add user message
                        if (!prompt_add_msg(msg, channel_cfg)) {
                            std::cerr << "Warning: Failed to add user message" << std::endl;
                            return;
                        }
                    }
                    // Unsuspend other tasks with same name
                    for (const auto& other_task : task.get_scheduler().get_tasks()) {
                        if (task.get_name() == other_task->get_name()) {
                            other_task->set_suspended(false);
                        }
                    }
                });
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
            sched_thread.create_task("Language Model Shutdown", [=, this] () -> void {
                                     llm_pool.store_all();
                                 });
        }
        sched_thread.wait();
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
