#!/bin/bash
echo "UP!"
file='Version.txt'
( [ -e "$file" ] || touch "$file" ) && ( [ ! -w "$file" ] || echo "0" > ${file} ) && echo cannot write to $file
cd ~/website/uebusaito
date=$(date +%Y%m%d)
preMSG="SHup"
mesg=${preMSG}${date}
git init
git add -A
git commit -m "${mesg}"
git push -u origin gh-pages
exit 0
