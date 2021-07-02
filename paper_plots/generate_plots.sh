#!/bin/zsh

for folder in */; do
    cd $folder
    for pyfile in *.py; do
        echo Generating $pyfile
        python $pyfile
    done
    cd ..
done
