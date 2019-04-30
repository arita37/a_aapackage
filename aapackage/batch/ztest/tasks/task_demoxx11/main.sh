#!/usr/bin/env bash
a=0
while [ $a -lt 10 ]
do
   echo $a
   sleep 5
   a=`expr $a + 1`
done

