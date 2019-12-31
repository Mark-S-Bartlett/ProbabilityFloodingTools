#!/bin/sh

cmd1="python ./hdf_generation.py"


$cmd1 && \

RAS_BIN_PATH=../bin_ras
echo $RAS_BIN_PATH 

export LD_LIBRARY_PATH=$RAS_BIN_PATH:$LD_LIBRARY_PATH
date

cmd1="../bin_ras/rasUnsteady64 AnacostiaRiver.c01 b38"

for d in */ ; do
	echo "Starting Rainfall Event $d"	
	cp $d/AnacostiaRiver.p38.tmp.hdf ./AnacostiaRiver.p38.tmp.hdf
	$cmd1
	mv AnacostiaRiver.p38.tmp.hdf $d/AnacostiaRiver.p38.hdf
	echo "Done Rainfall Event $d"
done