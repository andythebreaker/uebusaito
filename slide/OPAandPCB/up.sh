#!/bin/bash
echo "UP!"
file='Version.txt'
( [ -e "$file" ] || touch "$file" ) && ( ( ( [ ! -w "$file" ] ) && ( echo cannot write to $file ) ) || ( ( echo "0" > ${file} ) && ( echo writed $file ) ) )
ver=0
cat $file > $ver
verP=`expr $ver + 1`
echo ${verP} > ${file}
echo 'Version change from' $ver 'to' $verP
cd ~/website/uebusaito
date=$(date +%Y%m%d)
preMSG="SHup"
strVER=' Version = '
mesg=${preMSG}${date}${strVER}${verP}
git init
git add -A
git commit -m "${mesg}"
git push -u origin gh-pages
exit 0
