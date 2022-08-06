#!/bin/bash

temp=$1
index=$2

nconf=100000
lsize=10
autocorrelation_spin=158

reading_dir=mc_data3
writing_dir=data[1Jan2]
file_to_read=${reading_dir}/statesl=${lsize}${index}.txt
file_to_write=${writing_dir}/states_ind_l=${lsize}_T=${temp}.txt

echo ${temp} > ${file_to_write} 

a=`echo ${lsize}+1 | bc`
K=`echo 1 - ${lsize}| bc`
for i in `seq 1 ${autocorrelation_spin} ${nconf}`
do
start=`echo ${a}*${i} + ${K} | bc`
end=`echo ${start}+${lsize}-1 | bc`
sed -n ${start},${end}p ${file_to_read} >> ${file_to_write}
echo >> ${file_to_write}
done

#cat ${file_to_write}
