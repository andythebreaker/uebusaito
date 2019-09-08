#!/bin/bash
echo "UP!"
cd ~/website/uebusaito
date=$(date +%Y%m%d)
preMSG="SHup"
mesg=${preMSG}${date}
git init
git add -A
git commit -m "${mesg}"
git push -u origin gh-pages
exit 0
