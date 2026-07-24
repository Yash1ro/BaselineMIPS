# ------------------------------------------------------------------------------
#  Parameters  (top-10 variant)
# ------------------------------------------------------------------------------
name=openai1536
n=1000000  #data size
d=1536     #dimension
qn=1000    #query size
k=10      #topk

efc=500   #HNSW parameter
M=32       #HNSW parameter
L=16       #level

dPath=../data/${name}/${name}_base.bin   #data path
qPath=../data/${name}/${name}_query.bin  #query path
tPath=../data/${name}/${name}_truth10.bin        #groundtruth path
iPath=./${name}/index_top10/             #index path

#----Indexing for the first execution and searching for the following executions---------

./build/PEOs ${dPath} ${qPath} ${tPath} ${iPath} ${n} ${qn} ${d} ${k} ${efc} ${M} ${L}
