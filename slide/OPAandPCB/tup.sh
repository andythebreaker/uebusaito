#!/bin/bash
echo "UP!"
file='Version.txt'
( [ -e "$file" ] || touch "$file" ) && ( ( ( [ ! -w "$file" ] ) && ( echo cannot write to $file ) ) || ( ( echo "0" > ${file} ) && ( echo writed $file ) ) )
ver=0
cat $file > $ver 
echo ver= ${ver}
ver=`expr $ver + 1`
echo ver+1= $ver

