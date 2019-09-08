#!/bin/bash
echo "UP!"
cd ~/website/uebusaito
git init
git add -A
git commit -m "UPSH"
git push -u origin gh-pages
exit 0
