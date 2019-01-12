cd /d d:\_devs\Python01\project27\aapackage


rem Activate python 3.6  ###############################################################
call  activate tf_gpu_12
python  util.py         --do test   >> ztest_log_all2.txt 2>>&1
python  datanalysis.py  --do test   >> ztest_log_all2.txt 2>>&1
python  util_ml.py      --do test   >> ztest_log_all2.txt 2>>&1




rem Activate python 2.7 ###############################################################
call  activate python2
python  util.py         --do test  >> ztest_log_all2.txt 2>>&1
python  datanalysis.py  --do test  >> ztest_log_all2.txt 2>>&1



rem pause


rem Comments #########################################################################
goto comment
# python  util_ml.py      --do test






:comment








