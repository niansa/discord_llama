# Discord LLaMa

## Overview
This Discord bot lets you use GPT-J and LLaMa models to both participate in your conversations and answer your questions in thread channels, in a ChatGPT-like fashion!

## Setup
On a fresh Debian 11 install, you'll need the following packages: `git` `g++` `make` `cmake` `zlib1g-dev` `libssl-dev` `libsqlite3-dev`. For better performance with large batch sizes, `libopenblas-dev` is recommended.

Then, you can clone this repository including all its submodules:

    git clone https://gitlab.com/niansa/discord_llama.git --recursive

Once the command has completed, create a build directory, configure the project and compile it:

    cd discord_llama
    mkdir build
    cd build
    cmake ..
    make -j$(nproc)

We should now download a model. Let's start with `gpt4all-j-v1.3-groovy`! I've already written an example model config for it, so let's just grab that and download the model:

    mkdir models
    cd models
    cp ../../example_models/gpt4all-j-v1.3-groovy.txt .
    wget https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin
    cd ..

Next, we need a configuration file. Let's just grab the example config:

    cp ../example_config.txt ./config.txt

and edit it. For reference, you can read [explained_config.txt](https://gitlab.com/niansa/discord_llama/-/blob/master/explained_config.txt).

Please note that `gpt4all-j-v1.3-groovy` is only capable of instruct mode, so you'll have to set `threads_only` to `true`, if you've downloaded just that model.

Finally, start the Discord Bot by passing the config file to the executable:

    ./discord_llama config.txt

And that's it! Feel free to play around, try different models, tweak the config file, etc... It's really easy! If you still have any questions, please write an issue or contact me on Discord: *Tuxifan#0981*.

## Credits
A huge thank you to *Nomic AI* for helping me get this project to where it is now!
Also, this project wouldn't have been possible without the effort of *Georgi Gerganov (ggerganov)*, who has writte the core of it all, the inference code itself.
