#!/bin/bash
echo "UP!"
file='Version.txt'
( ( [ -e "$file" ] ) && ( ( [ ! -w "$file" ] && echo Ver~.txt EXIT but cannot write ) || echo file Ver~.txt checked ! ) ) || ( ( touch "$file" ) && ( ( ( [ ! -w "$file" ] ) && ( echo cannot write to $file ) ) || ( ( echo "0" > ${file} ) && ( echo writed $file ) ) ) )
ver=`cat $file`
file2='msg.txt'
( ( [ -e "$file2" ] ) && ( ( [ ! -w "$file2" ] && echo msg.txt EXIT but cannot write ) || echo file msg.txt checked ! ) ) || ( ( touch "$file2" ) && ( ( ( [ ! -w "$file2" ] ) && ( echo cannot write to $file2 ) ) || ( ( echo "no msg!" > ${file2} ) && ( echo writed $file2 ) ) ) )
vers=`cat $file2`
echo previous ver. = ${ver}
ver=`expr $ver + 1`
echo current ver. = $ver
echo `expr $(cat $file) + 1` > ${file}
cd ~/uebusaito
date=$(date +%Y%m%d)
preMSG="SHup"
strVER=' Version = '
mesg=${preMSG}${date}${strVER}${ver}${vers}
git init
git add -A
git commit -m "${mesg}"
git push
exit 0        
