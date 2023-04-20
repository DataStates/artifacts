#!/usr/bin/bash
backend=$1
source $HOME/.bashrc
source init-dh-environment2.sh
#source ./combined.env
case "$backend" in
  redis)
    SERVER_CMD="${EXP_DIR}/servers/redis.py"
    ;;
  datastates)
    SERVER_CMD="./cpp-store/build/server --thallium_connection_string $THALLIUM_NETWORK --num_threads 2"
    ;;
esac

$SERVER_CMD 
