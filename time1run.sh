START_TIME=`date`
echo "Enter script to be timed:"
FULLSCRIPT="python inversion.py --cchpc --read --gnup 2"
echo $FULLSCRIPT
$FULLSCRIPT
END_TIME=`date`
echo "Start time: $START_TIME"
echo "End time: $END_TIME"
