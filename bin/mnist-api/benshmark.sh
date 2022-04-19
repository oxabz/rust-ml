#!/bin/sh
for i in $(seq 1 800);  
do
    curl --location --request GET 'http://localhost:3030/' \
    --header 'Content-Type: multipart/form-data' \
    --form 'file=@./1.png' &
done