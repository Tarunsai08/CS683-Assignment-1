#!/bin/bash
set -e
CORE=0
MSR=0x1a4

ORIG=$(sudo rdmsr -p $CORE $MSR)
echo "Original prefetcher MSR value: $ORIG"

echo "Disabling all 4 HW prefetchers..."
sudo wrmsr -p $CORE $MSR 0xf
sudo rdmsr -p $CORE $MSR

echo "Running benchmark (pinned to core $CORE)..."
taskset -c $CORE ./bin/matmul "$@"

echo "Restoring original MSR value ($ORIG)..."
sudo wrmsr -p $CORE $MSR $ORIG
sudo rdmsr -p $CORE $MSR

echo "Done."
