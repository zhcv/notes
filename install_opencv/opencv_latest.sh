#!/bin/bash

. `dirname $0`/download_latest_version.sh

if [ $? -ne 0 ]; then
    exit $?;
fi

# . `dirname $0`/opencv_install.sh
