#!/bin/bash

temp=$1
nstates=10000 #$2
size=8 #$3
path_to_mc_data="./mc_data"
path_to_mc_ising="../../../magneto"
path_to_pickle="data"

states_dir="${path_to_mc_data}/temp-${temp}_nstates-${nstates}"
mkdir ${states_dir}

for i in $(seq 1 1 ${nstates})
do 
  ${path_to_mc_ising}/magneto.exe -TMin=${temp} -L=${size} -TSteps=1 -N1=1000 -N2=500 -states=${i}states

done

mv *states* ${states_dir}

python3 create_pickle2.py ${states_dir} ${nstates} ${size} ${path_to_pickle} ${temp}

rm -r ${states_dir}
