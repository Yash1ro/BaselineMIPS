# ------------------------------------------------------------------------------
#  Parameters
# ------------------------------------------------------------------------------
name=text2img10m
prefix=text2img
n=10000000  #data size
d=200       #dimension
qn=100000   #query size
k=100       #topk

efc=1000    #HNSW parameter
M=64        #HNSW parameter
L=32        #level

dPath=../data/${name}/${prefix}_base.bin  #data path
qPath=../data/${name}/${prefix}_query.bin #query path
tPath=../data/${name}/${name}_truth.bin   #groundtruth path
iPath=./${name}/index_top100/             #index path
#----Indexing for the first execution and searching for the following executions---------

./build/PEOs ${dPath} ${qPath} ${tPath} ${iPath} ${n} ${qn} ${d} ${k} ${efc} ${M} ${L}
