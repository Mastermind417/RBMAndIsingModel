#!/bin/bash

start=`date`
echo ${start}

# mkdir ./data
mkdir ./mc_data
for i in $(seq 1 0.01 1.02)
do
echo Running temperature ${i}..
./rbm_pickle_t.sh ${i}
done

rm -r ./mc_data

end=`date`
echo Start: ${start}
echo End: ${end}
