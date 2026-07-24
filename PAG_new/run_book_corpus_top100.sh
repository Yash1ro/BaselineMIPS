# ------------------------------------------------------------------------------
#  Parameters  (top-100 variant, auto-generated)
# ------------------------------------------------------------------------------
name=book_corpus
n=9250529  #data size
d=1024     #dimension
qn=10000    #query size
k=100      #topk

efc=1000   #HNSW parameter
M=64       #HNSW parameter
L=16       #level

dPath=../data/${name}/${name}_base.bin   #data path
qPath=../data/${name}/${name}_query.bin  #query path
tPath=../data/${name}/${name}_truth.bin        #groundtruth path
iPath=./${name}/index_top100/             #index path

#----Indexing for the first execution and searching for the following executions---------

./build/PEOs ${dPath} ${qPath} ${tPath} ${iPath} ${n} ${qn} ${d} ${k} ${efc} ${M} ${L}
