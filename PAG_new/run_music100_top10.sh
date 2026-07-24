# ------------------------------------------------------------------------------
#  Parameters  (top-10 variant, auto-generated)
# ------------------------------------------------------------------------------
name=music100
n=1000000  #data size
d=100     #dimension
qn=10000    #query size
k=10      #topk

efc=1000   #HNSW parameter
M=64       #HNSW parameter
L=32       #level

dPath=../data/${name}/${name}_base.bin   #data path
qPath=../data/${name}/${name}_query.bin  #query path
tPath=../data/${name}/${name}_truth10.bin        #groundtruth path
iPath=./${name}/index_top10/             #index path

#----Indexing for the first execution and searching for the following executions---------

./build/PEOs ${dPath} ${qPath} ${tPath} ${iPath} ${n} ${qn} ${d} ${k} ${efc} ${M} ${L}
