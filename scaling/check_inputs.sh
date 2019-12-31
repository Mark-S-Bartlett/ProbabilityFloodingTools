#!/bin/bash
# Verify inputs on S3 prior to testing production runs
# TODO: Update with appropriate model files & add python script for setup testing

# slawler@dewberry.com

# Arguments
REGION=$1   # Name of Project Area 
RUN=$2      # Run name
MODEL=$3      # Model name
GLOBAL=global # global input directory

# AWS Directories
ROOT=s3://probmodelingrepository/${REGION}/ProductionRuns/inputs/

# Make list of files to check for (e.g. RUN.dss, RUN_mannings.csv)
declare -a GLOBALFILES=(${REGION}.dss ${REGION}_bc_data.csv); globalfiles_length=${#GLOBALFILES[@]}
declare -a RUNFILES=(${RUN}.prj); runfiles_length=${#RUNFILES[@]}

# Write Test to log with date executed appended to log filename
logfile="$1_$(date +"%m_%d_%Y_%HM").log"
echo Writing output to ${logfile}

# Common Messages saved as Variables
WIKIPAGE=https://github.com/Dewberry/probmod-tools/wiki/Production-Protocols
DSUCCESS="Directory found: "
FSUCCESS="File found: "
DERROR="!ERROR! Verify setup is correct ($WIKIPAGE) Directory not found: "
FERROR="!ERROR! Verify setup is correct ($WIKIPAGE) File not found: "
RUN_PYTHON="Starting Python job"

#------------------------------------Perform checks-----------------------------#
# Check for directory existence usin ls
grep -q $RUN <<< `aws s3 ls ${ROOT}` && echo "$DSUCCESS $RUN/"  && RUN_EXISTS=true  ||  echo "$DERROR $RUN/"
grep -q $GLOBAL <<< `aws s3 ls ${ROOT}` && echo "$DSUCCESS $GLOBAL/"  && GLOBAL_EXISTS=true  ||  echo "$DERROR $GLOBAL/"

# Search Global Directory for required files
if ($GLOBAL_EXISTS ); then 
    GLOBALDIR=${ROOT}${GLOBAL}"/"

    for (( i=1; i<${globalfiles_length}+1; i++ ));
        do
           GLOBALFILE=${GLOBALFILES[$i-1]}
           grep -q $GLOBALFILE <<< `aws s3 ls ${GLOBALDIR}` && echo "$FSUCCESS $GLOBALFILE"  && FILE_EXISTS=true  ||  echo "$FERROR $GLOBALDIR$GLOBALFILE"

           if ($FILE_EXISTS); then
                FILE_EXISTS=false
           else
               echo  "File Not Found: $GLOBALFILE" > $logfile
           fi
        done
else
    echo  "Please verify input formatting from $WIKIPAGE" > $logfile
fi


# Search RUN Directory for required files
if ($RUN_EXISTS ); then 
    MODELDIR=${ROOT}${RUN}"/"

    for (( i=1; i<${runfiles_length}+1; i++ ));
        do
           RUNFILE=${RUNFILES[$i-1]}
           grep -q $MODELDIR$RUNFILE <<< `aws s3 ls ${MODELDIR}` && echo "$FSUCCESS $MODELDIR$RUNFILE"  && FILE_EXISTS=true  ||  echo "$FERROR $MODELDIR$RUNFILE"

           if ($FILE_EXISTS); then
                FILE_EXISTS=false
           else
               echo  "File Not Found: $RUNFILE" > $logfile
           fi

        done
else
    echo  "Please verify input formatting from $WIKIPAGE" > $logfile
fi

#echo `cat $logfile`
