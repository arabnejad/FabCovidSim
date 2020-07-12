 
#!/bin/bash
allR="2.0 2.2"
allx="60"
allz="0.25"
alli="PC_CI_HQ_SD"

for R in $allR
do
    rs=$(echo $R | awk '{print $1/2}')
    echo CovidSim /NR:10 /c:8 /PP:preGB_R0=2.0.txt /P:p_NoInt.txt /CLP1:100000 /CLP2:0 /O:meanT8_NR10/NoInt_R0=${R} /D:../population/GB_pop2018.bin /L:../population/NetworkGB_8T.bin /R:${rs} 98798150 729101 17389101 4797132
    for x in $allx
    do
	for z in $allz
	do
	    q=1000
	    y=$(echo $x  $z | awk '{print $1 * $2}')
	    #echo $y
	    for i in $alli
	    do
		echo CovidSim /NR:10 /c:8 /PP:preGB_R0=2.0.txt /P:p_${i}.txt /CLP1:${x} /CLP2:${q} /CLP3:${q} /CLP4:${q} /CLP5:${y} /O:meanT8_NR10/${i}_${x}_${y}_R0=${R} /D:../population/GB_pop2018.bin /L:../population/NetworkGB_8T.bin /R:${rs} 98798150 729101 17389101 4797132
	    done
	done
    done
done
