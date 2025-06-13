#!/bin/bash
DATA_DIR=$1
DATE=$2

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data dir '$DATA_DIR' does not exist."
    echo "Usage: $0 DATA_DIR [DATE]"
    exit 1
fi

docker run --rm -v $(pwd)/$DATA_DIR:/data taxi-rides-outlier-detection /data $DATE


#docker run -v ./work:/contdata taxi-rides-outlier-detection /contdata 2025-01-01

#docker run --rm -v ./$DATA_DIR:/contdata taxi-rides-outlier-detection /contdata $DATE