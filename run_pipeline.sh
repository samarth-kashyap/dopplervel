#!/bin/bash
# Simple DopplerVel Pipeline
# Run the complete processing pipeline for HMI Doppler data

set -e  # Exit on error

# Default parameters
YEAR=${1:-2018}
START_DAY=${2:-0}
END_DAY=${3:-365}
NCORES=${4:-48}
LMIN=${5:-0}
LMAX=${6:-1535}

echo "Running DopplerVel pipeline for year $YEAR, days $START_DAY-$END_DAY, l=$LMIN-$LMAX"

echo "Step 1: Preprocessing data..."
seq $START_DAY $END_DAY | parallel -j $NCORES "python hathaway.py --gnup {}"

echo "Step 2: Computing spherical harmonics..."
seq $START_DAY $END_DAY | parallel -j $NCORES "python data_analysis.py --gnup {}"

echo "Step 3: Generating leakage matrices..."
mkdir -p "${SCRATCH_DIR:-/scratch/g.samarth}/matrixA/lmax${LMAX}"
seq $LMIN $LMAX | parallel -j $NCORES "python generate_matrix.py --gnup {}"

echo "Step 4: Running inversion ..."
for day in $(seq $START_DAY $END_DAY); do
    echo "Processing day $day"
    seq $LMIN $LMAX | parallel -j $NCORES "python inversion.py --gnup {}"
done

# Step 5: Coordinate rotation
echo "Step 5: Rotating coordinates..."
seq $START_DAY $END_DAY | parallel -j $NCORES "python rotation.py --gnup {}"

# Step 6: Time series analysis
echo "Step 6: Analyzing time series..."
python time_series.py --datatype doppler

echo "Pipeline completed successfully!"
