#include "config.hpp"
#include "utils.hpp"

#include <string>
#include <fstream>
#include <unordered_map>
#include <filesystem>



std::unordered_map<std::string, std::string> Configuration::file_parser(const std::string& path) {
    std::unordered_map<std::string, std::string> fres;
    // Open file
    std::ifstream cfgf(path);
    if (!cfgf) {
        throw Exception("Failed to open configuration file: "+path);
    }
    // Read each entry
    for (std::string key; cfgf >> key;) {
        // Read value
        std::string value;
        std::getline(cfgf, value);
        // Ignore comment and empty lines
        if (key.empty() || key[0] == '#') continue;
        // Erase all leading spaces in value
        while (!value.empty() && (value[0] == ' ' || value[0] == '\t')) value.erase(0, 1);
        // Add to function result
        fres.emplace(std::move(key), std::move(value));
    }
    // Return final result
    return fres;
}


extern "C" char **environ;
std::unordered_map<std::string, std::string> Configuration::environment_parser() {
    std::unordered_map<std::string, std::string> fres;
    for (char **s = environ; *s; s++) {
        const auto pair = utils::str_split(*s, '=', 1);
        fres.emplace(pair[0], pair[1]);
    }
    return fres;
}


void Configuration::Model::fill(const Configuration& cfg, std::unordered_map<std::string, std::string>&& map, bool ignore_extra) {
    for (auto& [key, value] : map) {
        if (key == "filename") {
            weights_filename = std::move(value);
        } else if (key == "user_prompt") {
            user_prompt = std::move(value);
        } else if (key == "bot_prompt") {
            bot_prompt = std::move(value);
        } else if (key == "instruct_mode_policy") {
            instruct_mode_policy = parse_instruct_mode_policy(value);
        } else if (key == "emits_eos") {
            emits_eos = parse_bool(value);
        } else if (key == "no_translate") {
            no_translate = parse_bool(value);
        } else if (!ignore_extra) {
            throw Exception("Error: Failed to parse model configuration file: Unknown key: "+key);
        }
    }
    // Get full path
    weights_path = std::filesystem::path(cfg.models_dir)/weights_filename;
}

void Configuration::Model::check(std::string& model_name, bool& allow_non_instruct) const {
    utils::clean_for_command_name(model_name);
    // Checks
    if (weights_filename.empty() || !file_exists(weights_path)) {
        throw Exception("Error: Failed to parse model configuration file: Invalid weight filename: "+model_name);
    }
    if (instruct_mode_policy != InstructModePolicy::Forbid &&
        (user_prompt.empty() || bot_prompt.empty())) {
        throw Exception("Error: Failed to parse model configuration file: Instruct mode allowed but user prompt and bot prompt not given: "+model_name);
    }
    // Set allow_non_instruct
    if (instruct_mode_policy != InstructModePolicy::Force) {
        allow_non_instruct = true;
    }
}


void Configuration::Texts::fill(std::unordered_map<std::string, std::string>&& map, bool ignore_extra) {
    for (auto& [key, value] : map) {
        if (key == "model_missing") {
            model_missing = std::move(value);
        } else if (key == "please_wait") {
            please_wait = std::move(value);
        } else if (key == "thread_create_fail") {
            thread_create_fail = std::move(value);
        } else if (key == "timeout") {
            timeout = std::move(value);
        } else if (key == "translated") {
            translated = parse_bool(value);
        } else if (!ignore_extra) {
            throw Exception("Error: Failed to parse texts file: Unknown key: "+key);
        }
    }
}

void Configuration::Texts::check() const {
    // Nothing, for now
}


void Configuration::fill(std::unordered_map<std::string, std::string>&& map, bool ignore_extra) {
    for (auto& [key, value] : map) {
        if (key == "token") {
            token = std::move(value);
        } else if (key == "language") {
            language = std::move(value);
        } else if (key == "default_inference_model") {
            default_inference_model = std::move(value);
            utils::clean_for_command_name(default_inference_model);
        } else if (key == "translation_model") {
            translation_model = std::move(value);
            utils::clean_for_command_name(translation_model);
        } else if (key == "prompt_file") {
            prompt_file = std::move(value);
        } else if (key == "instruct_prompt_file") {
            instruct_prompt_file = std::move(value);
        } else if (key == "models_dir") {
            models_dir = std::move(value);
        } else if (key == "texts_file") {
            texts_file = std::move(value);
        } else if (key == "pool_size") {
            pool_size = std::stoi(value);
        } else if (key == "threads") {
            threads = std::stoi(value);
        } else if (key == "scroll_keep") {
            scroll_keep = std::stoi(value);
        } else if (key == "shard_count") {
            shard_count = std::stoi(value);
        } else if (key == "shard_id") {
            shard_id = std::stoi(value);
        } else if (key == "timeout") {
            timeout = std::stoi(value);
        } else if (key == "ctx_size") {
            ctx_size = std::stoi(value);
        } else if (key == "max_context_age") {
            max_context_age = std::stoi(value);
        } else if (key == "mlock") {
            mlock = parse_bool(value);
        } else if (key == "live_edit") {
            live_edit = parse_bool(value);
        } else if (key == "threads_only") {
            threads_only = parse_bool(value);
        } else if (key == "persistance") {
            persistance = parse_bool(value);
        } else if (!ignore_extra) {
            throw Exception("Error: Failed to parse configuration file: Unknown key: "+key);
        }
    }
}

void Configuration::check(bool allow_non_instruct) const {
    if (language != "EN") {
        if (translation_model_cfg == nullptr) {
            throw Exception("Error: Translation model required for non-english language, but is invalid");
        }
        if (translation_model_cfg->instruct_mode_policy == Model::InstructModePolicy::Force) {
            throw Exception("Error: Translation model is required to not have instruct mode forced");
        }
        if (live_edit) {
            throw Exception("Warning: Live edit should not be enabled for non-english language");
        }
    }
    if (allow_non_instruct && !file_exists(prompt_file)) {
        throw Exception("Error: Prompt file required when allowing non-instruct-mode use, but is invalid");
    }
    if (!threads_only) {
        if (default_inference_model_cfg == nullptr) {
            throw Exception("Error: Default model required if not threads only, but is invalid");
        }
        if (default_inference_model_cfg->instruct_mode_policy == Model::InstructModePolicy::Force) {
            throw Exception("Error: Default model must not have instruct mode forced if not threads only");
        }
    }
    if (scroll_keep >= 99) {
        throw Exception("Error: Scroll_keep must be a non-float percentage and in a range of 0-99.");
    }
    if (shard_count == 0) {
        throw Exception("Error: Shard count must be above zero.");
        exit(-13);
    }
    if (shard_id >= shard_count) {
        throw Exception("Error: Not enough shards for this ID to exist.");
    }
}

#include <iostream>
void Configuration::parse_configs(const std::string &main_file) {
    const auto file_location = main_file.empty()?
                std::filesystem::current_path():
                std::filesystem::path(main_file).parent_path();

    std::cout << main_file << std::endl;

    // Parse main configuration
    fill(environment_parser(), true);
    if (!main_file.empty()) fill(file_parser(main_file));

    // Parse and check texts configuration
    if (texts_file != "none") {
        texts.fill(environment_parser(), true);
        if (std::filesystem::path(texts_file).is_absolute()) {
            texts.fill(file_parser(texts_file));
        } else {
            texts.fill(file_parser(file_location/texts_file));
        }
        texts.check();
    }

    // Parse model configurations
    std::filesystem::path models_dir;
    if (std::filesystem::path(models_dir).is_absolute()) {
        models_dir = this->models_dir;
    } else {
        models_dir = file_location/this->models_dir;
    }
    bool allow_non_instruct = false;
    for (const auto& file : std::filesystem::directory_iterator(models_dir)) {
        // Check that file is model config
        if (file.is_directory() ||
            file.path().filename().extension() != ".txt") continue;
        // Get model name
        auto model_name = file.path().filename().string();
        model_name.erase(model_name.size()-4, 4);
        utils::clean_for_command_name(model_name);
        // Parse and check model config
        Model model;
        model.fill(*this, file_parser(file.path()));
        model.check(model_name, allow_non_instruct);
        // Add model to list
        const auto& [stored_model_name, stored_model_cfg] = *models.emplace(std::move(model_name), std::move(model)).first;
        // Set model pointer in config
        if (stored_model_name == default_inference_model)
            default_inference_model_cfg = &stored_model_cfg;
        if (stored_model_name == translation_model)
            translation_model_cfg = &stored_model_cfg;
    }

    // Check main configuration
    check(allow_non_instruct);
}
