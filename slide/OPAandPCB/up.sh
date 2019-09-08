#!/bin/bash
echo "UP!"
if [[ ! -e /Version.txt ]]; then
	        touch /Version.txt
	fi
cd ~/website/uebusaito
date=$(date +%Y%m%d)
preMSG="SHup"
mesg=${preMSG}${date}
git init
git add -A
git commit -m "${mesg}"
git push -u origin gh-pages
exit 0
