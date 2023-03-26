#!/bin/bash

root="/usr/local/nginx/html"
if [ -n "$1" ];then
    root="$1";
fi

cp -r "/home/dbcloud/caffeebar/3DCafe" $root