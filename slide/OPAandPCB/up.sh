#!/bin/bash
echo "UP!"
git init
git add -A
git commit -m "UPSH"
cd ~/website/uebusaito
git push -u origin gh-pages
exit 0
