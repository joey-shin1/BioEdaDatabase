#!/bin/bash
#SBATCH -J EDA
#SBATCH -A m2834
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 128
#SBATCH --constraint cpu
#SBATCH --qos regular
#SBATCH --time 24:0:0
#SBATCH --mem=64G

start=$(date +%s)

module load qchem/6.2.1-cpu

#scriptdir="$(pwd)"
#topdir=${scriptdir%/*}

for dir in opt_*/
do
  cd $dir
  echo Running $dir
  if [[ -f done.tag || -f error.tag || -f running.tag ]]; then
    echo $dir Skipped.
  else
    touch running.tag
    qchem -nt 128 opt.in opt.out > log
    if [ $(grep E_vct opt.out | wc -l) -eq 1 ]; then
      echo $dir Success.
      mv running.tag done.tag
    else
      echo $dir Error.
      mv running.tag error.tag
    fi
  fi
  cd -
done
