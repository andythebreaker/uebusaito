#!/bin/bash
echo "UP!"
file='Version.txt'
( ( [ -e "$file" ] ) && ( ( [ ! -w "$file" ] && echo Ver~.txt EXIT but cannot write ) || echo file Ver~.txt checked ! ) ) || ( ( touch "$file" ) && ( ( ( [ ! -w "$file" ] ) && ( echo cannot write to $file ) ) || ( ( echo "0" > ${file} ) && ( echo writed $file ) ) ) )
ver=`cat $file`
echo ver= ${ver}
ver=`expr $ver + 1`
echo ver+1= $ver
echo `expr $(cat $file) + 1` > ${file}
