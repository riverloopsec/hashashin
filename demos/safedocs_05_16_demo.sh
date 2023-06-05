#!/bin/bash
# Usage:
#   safedocs_05_16_demo.sh [-x] <step_num>
# Args:
#   -x: Execute commands


# if no arguments are supplied throw an error
if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    echo "Usage: safedocs_05_16_demo.sh [-x] <step_num>"
    exit 1
fi

# get step_num
step_num=
echo $step_num

# switch statement on step_num
case $step_num in
    1)
    echo "hashashin -db"
    # if -x flag is supplied, execute command
    if [ "$1" == "-x" ]; then
        hashashin -db
    fi
    ;;
    2)
    echo "hashashin -db -p"
    # if -x flag is supplied, execute command
    if [ "$1" == "-x" ]; then
        hashashin -db -p
    fi
    ;;

esac