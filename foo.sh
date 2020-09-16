#!/usr/bin/bash -x

export start_epoch=100
export final_epoch=200

((i=${start_epoch}))
((j=${final_epoch}))

echo i $i
echo j $j

until [ $i -gt $j  ]
do
  export n_epochs=$i
  echo ${n_epochs}
  ((i=i+10))
done

