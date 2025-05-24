#!/bin/bash

docker run \
       -it --rm \
       --memory=12g \
       --memory-swap=12g \
       --add-host=host.docker.internal:host-gateway \
       -v `pwd`:/aider \
       -v `pwd`/byterover.benchmarks/.:/benchmarks \
       -e OPENAI_API_KEY=$OPENAI_API_KEY \
       -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
       -e BYTEROVER_API_KEY=$BYTEROVER_API_KEY \
       -e BYTEROVER_USER_ID=$BYTEROVER_USER_ID \
       -e HISTFILE=/aider/.bash_history \
       -e PROMPT_COMMAND='history -a' \
       -e HISTCONTROL=ignoredups \
       -e HISTSIZE=10000 \
       -e HISTFILESIZE=20000 \
       -e AIDER_DOCKER=1 \
       -e AIDER_BENCHMARK_DIR=/benchmarks \
       aider-byterover-benchmark \
       bash
