#!/bin/bash
i=0;

lrs=(0.0001);
size1=(3 5 7);
size2=(1 3 5);
size3=(1 3 5);
filters=(64);

for lr in "${lrs[@]}";
do for s1 in "${size1[@]}";
  do for s2 in "${size2[@]}";
    do for s3 in "${size3[@]}";
      do for f in "${filters[@]}";
        do
          dirname="$WORK/enhance_parallel_$i";
          export dirname;
          export lr;
          export s1;
          export s2;
          export s3;
          export f;
          echo $dirname, $i, $lr, $s1, $s2, $s3, $f;
          sbatch tf.job;
          i=$((i+1));
        done;        
      done;
    done;
  done;
done;
