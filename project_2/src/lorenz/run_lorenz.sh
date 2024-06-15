#!/bin/zsh

# Define the number of iterations
ITERATIONS=10

# Loop for the specified number of iterations
for ((i=1; i<=$ITERATIONS; i++)); do
    echo "Running iteration $i"

    # Run PINN_points.py
    python3 lorenzTraining.py $i
done