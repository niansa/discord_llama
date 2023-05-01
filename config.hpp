#ifndef CONFIGURATION_H
#define CONFIGURATION_H
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <filesystem>


class Configuration {
    struct Exception : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    inline static
    bool parse_bool(const std::string& value) {
        if (value == "true")
            return true;
        if (value == "false")
            return false;
        throw Exception("Error: Failed to parse configuration file: Unknown bool (true/false): "+value);
    }

    inline static
    bool file_exists(const auto& p) {
        // Make sure we don't respond to some file that is actually called "none"...
        if (p == "none") return false;
        return std::filesystem::exists(p);
    }

    static
    std::unordered_map<std::string, std::string> file_parser(const std::string& path);
    static
    std::unordered_map<std::string, std::string> environment_parser();

    void fill(std::unordered_map<std::string, std::string>&&, bool ignore_extra = false);
    void check(bool allow_non_instruct) const;

public:
    struct Model {
        std::string weight_filename,
                    weight_path,
                    user_prompt,
                    bot_prompt;
        bool emits_eos = false,
             no_translate = false;
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

        InstructModePolicy parse_instruct_mode_policy(const std::string& value) {
            if (value == "allow")
                return Model::InstructModePolicy::Allow;
            if (value == "force")
                return Model::InstructModePolicy::Force;
            if (value == "forbid")
                return Model::InstructModePolicy::Forbid;
            throw Exception("Error: Failed to parse model configuration file: Unknown instruct mode policy (allow/force/forbid): "+value);
            exit(-4);
        }

        void fill(const Configuration&, std::unordered_map<std::string, std::string>&&, bool ignore_extra = false);
        void check(std::string& model_name, bool& allow_non_instruct) const;
    };
    struct Texts {
        std::string please_wait = "Please wait...",
                    thread_create_fail = "Error: I couldn't create a thread here. Do I have enough permissions?",
                    model_missing = "Error: The model that was used in this thread could no longer be found.",
                    timeout = "Error: Timeout";
        bool translated = false;

        void fill(std::unordered_map<std::string, std::string>&&, bool ignore_extra = false);
        void check() const;
    };

    std::string token,
                language = "EN",
                default_inference_model = "13B-vanilla",
                translation_model = "none",
                prompt_file = "none",
                instruct_prompt_file = "none",
                models_dir = "models",
                texts_file = "none";
    unsigned ctx_size = 1012,
             pool_size = 2,
             timeout = 120,
             threads = 4,
             scroll_keep = 20,
             shard_count = 1,
             shard_id = 0,
             max_context_age = 0;
    bool persistance = true,
         mlock = false,
         live_edit = false,
         threads_only = true;
    const Model *default_inference_model_cfg = nullptr,
                      *translation_model_cfg = nullptr;

    std::unordered_map<std::string, Model> models;
    Texts texts;

    void parse_configs(const std::string& main_file = "");

    Configuration() {}
    Configuration(Configuration&) = delete;
    Configuration(const Configuration&) = delete;
    Configuration(Configuration&&) = delete;
    Configuration(const Configuration&&) = delete;
};
#endif // CONFIGURATION_H
