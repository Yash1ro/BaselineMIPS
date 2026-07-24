#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include <time.h>
#include "hnswlib/hnswlib.h"
#include <omp.h>
#include <unordered_set>
#include <filesystem>
#include <stdexcept>
#define ALIGNMENT 64

using namespace std;
using namespace hnswlib;
namespace fs = std::filesystem;

static thread_local std::mt19937 rng(std::random_device{}());
static thread_local std::normal_distribution<double> nd(0.0, 1.0);

class StopW {
    std::chrono::steady_clock::time_point time_begin;
public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }

    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }

    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }

};

void random_orthogonal_matrix(int subdim, std::mt19937 &rng, std::vector<std::vector<float>> &R) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    R.assign(subdim, std::vector<float>(subdim, 0.0f));

    for (int i = 0; i < subdim; i++)
        for (int j = 0; j < subdim; j++)
            R[i][j] = nd(rng);

    for (int i = 0; i < subdim; i++) {
        for (int j = 0; j < i; j++) {
            float dot = 0.0f;
            for (int k = 0; k < subdim; k++)
                dot += R[i][k] * R[j][k];
            for (int k = 0; k < subdim; k++)
                R[i][k] -= dot * R[j][k];
        }
        float norm = 0.0f;
        for (int k = 0; k < subdim; k++)
            norm += R[i][k] * R[i][k];
        norm = std::sqrt(norm);
        for (int k = 0; k < subdim; k++)
            R[i][k] /= norm;
    }
}

void shuffle_for_equal_norm(std::vector<float>& dim_norm, int vecsize, int vecdim, int level,
                            std::vector<int>& permutation, std::vector<int>& zero_positions) {
    permutation.resize(vecdim);
	
	zero_positions.resize(level); 
	
    std::vector<int> idx(vecdim);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&dim_norm](int a, int b) {
        return dim_norm[a] > dim_norm[b];
    });

    std::vector<float> seg_norm(level, 0.0f);
	std::vector<int> seg_size(level, 0);
    std::vector<std::vector<int>> segments(level);

    int K = vecdim / level;
    for (int k = 0; k < vecdim; k++) {
        int dim_id = idx[k];
    
        int best_seg = -1;
        float best_norm = std::numeric_limits<float>::max();
    
        for (int s = 0; s < level; s++) {
            if (seg_size[s] < K && seg_norm[s] < best_norm) {
                best_norm = seg_norm[s];
                best_seg = s;
            }
        }

        segments[best_seg].push_back(dim_id);
        seg_norm[best_seg] += dim_norm[dim_id];
        seg_size[best_seg]++;
    }

    for (int l = 0; l < level; l++) {
        int subdim = segments[l].size();
        int subdim0 = 0;

        for (int k = subdim - 1; k >= 0; k--) {
            int dim_id = segments[l][k];
            if (dim_norm[dim_id] > 0.0f) {
                subdim0 = k + 1;
                break;
            }
        }
        zero_positions[l] = subdim0;
    }

    int pos = 0;
    for (int l = 0; l < level; l++)
        for (int d : segments[l])
            permutation[pos++] = d;
}

void generate_subspace_vectors_projvec(std::vector<std::vector<std::vector<float>>> &projVec,
                                       int level, int subdim, int m, std::vector<int>& zero_positions) {
    std::normal_distribution<float> nd(0.0f, 1.0f);

#pragma omp parallel for
for(int l = 0; l < level; l++){
    std::random_device rd;
    std::mt19937 rng_thread(rd() + l);

    int subdim0 = zero_positions[l];   
    if (subdim0 <= 0) subdim0 = subdim;
	
    std::vector<std::vector<float>> vectors(m, std::vector<float>(subdim0, 0.0f));
    if (m <= subdim0) {    
        for (int i = 0; i < m; i++)
            vectors[i][i] = 1.0f;

        std::vector<std::vector<float>> R;
        random_orthogonal_matrix(subdim0, rng_thread, R);
        for (int i = 0; i < m; i++) {
            std::vector<float> tmp(subdim0, 0.0f);
            for (int r = 0; r < subdim0; r++)
                for (int c = 0; c < subdim0; c++)
                    tmp[r] += R[r][c] * vectors[i][c];
            vectors[i] = tmp;
        }
    } else {
        int n_poly = m / subdim0;
        int remainder = m % subdim0;
        int idx = 0;

        for (int p = 0; p < n_poly; p++) {
            for (int i = 0; i < subdim0; i++) {
                std::fill(vectors[idx+i].begin(), vectors[idx+i].end(), 0.0f);
                vectors[idx+i][i] = 1.0f;
            }

            std::vector<std::vector<float>> R;
            random_orthogonal_matrix(subdim0, rng_thread, R);
            for (int i = 0; i < subdim0; i++) {
                std::vector<float> tmp(subdim0, 0.0f);
                for (int r = 0; r < subdim0; r++)
                    for (int c = 0; c < subdim0; c++)
                        tmp[r] += R[r][c] * vectors[idx+i][c];
                vectors[idx+i] = tmp;
            }
            idx += subdim0;
        }
		
        if (remainder > 0) {
            for (int i = 0; i < remainder; i++) {
                std::fill(vectors[idx+i].begin(), vectors[idx+i].end(), 0.0f);
                vectors[idx+i][i] = 1.0f;
            }
			
            std::vector<std::vector<float>> R;
            random_orthogonal_matrix(subdim0, rng_thread, R);
            for (int i = idx; i < idx + remainder; i++) {
                std::vector<float> tmp(subdim0, 0.0f);
                for (int r = 0; r < subdim0; r++)
                    for (int c = 0; c < subdim0; c++)
                        tmp[r] += R[r][c] * vectors[i][c];
            
			    vectors[i] = tmp;
            }
        }	
    }

    float scale = 1.0f / std::sqrt((float)level);
        for (int i = 0; i < m; i++){
            for (int j = 0; j < subdim0; j++)
                projVec[l][i][j] = vectors[i][j] * scale;

            for (int j = subdim0; j < subdim; j++)
                projVec[l][i][j] = 0;
	    }
    }
}

static void
get_gt(unsigned int *massQA, size_t qsize, vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, size_t maxk) {
    (vector<std::priority_queue<std::pair<float, labeltype >>>(qsize)).swap(answers);
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < k; j++) {
            answers[i].emplace(0.0f, massQA[maxk * i + j]);
        }
    }
}

static void
test_vs_recall(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg,
               vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, float* table, size_t vecdim, size_t extended_dim, std::vector<Node>& InitialTable, char* path_index) {
    vector<size_t> efs;

	size_t padded_dim = (vecdim + 15) & ~0xF;
    std::vector<int> permutation(extended_dim);
    std::vector<int> permutation0(extended_dim);

    std::string folderPath(path_index);
    std::string fullPath;
    if (!folderPath.empty() && (folderPath.back() == '/' || folderPath.back() == '\\')) {
        fullPath = folderPath + "permutation.bin";
    } else {
        fullPath = folderPath + "/permutation.bin";
    }
	
    std::string tablePath;
    if (!folderPath.empty() && (folderPath.back() == '/' || folderPath.back() == '\\')) {
        tablePath = folderPath + "table.bin";
    } else {
        tablePath = folderPath + "/table.bin";
    }	

    std::ifstream fin(fullPath, std::ios::binary);
    if (!fin) {
        throw std::runtime_error("Failed to open permutation file: " + fullPath);
    }
	fin.read((char*)permutation.data(), permutation.size() * sizeof(int));
	fin.read((char*)permutation0.data(), permutation0.size() * sizeof(int));

    std::ifstream fin0(tablePath, std::ios::binary);
    if (!fin0) {
        throw std::runtime_error("Failed to open permutation file: " + tablePath);
    }
	fin0.read((char*)InitialTable.data(), InitialTable.size() * sizeof(Node));

    float* permutedQ = new (std::align_val_t{ALIGNMENT}) float[qsize * extended_dim]; 
    float* permutedQ0 = new (std::align_val_t{ALIGNMENT}) float[qsize * extended_dim]; 

	int step;
	
	if(k <= 10)
		step = 10;
	else if(k <= 100)
		step = 100;
	else
		step = k;

    std::vector<std::vector<Neighbor>> result; 
    result.resize(qsize);
    for (size_t i = 0; i < qsize; ++i) {
        result[i].resize(step);
    }
	
    const size_t ef_points = (k <= 10) ? 20 : 99;
    for (size_t i = 1; i <= ef_points; ++i) {
        efs.push_back(i * step);
    }
	
    for (size_t ef : efs) {
        appr_alg.setEf(ef);
        StopW stopw = StopW();

        for (int q = 0; q < qsize; q++) {
            float* curQ = massQ + q * padded_dim;
            float* curP = permutedQ + q * extended_dim;
            float* curP0 = permutedQ0 + q * extended_dim;

            for (int i = 0; i < extended_dim; i++){
				int new_pos = permutation[i];
                if(new_pos < padded_dim)
				    curP[i] = curQ[new_pos];
				else
					curP[i] = 0.0f;
			}
			
            for (int i = 0; i < extended_dim; i++){
				int new_pos = permutation0[i];
                if(new_pos < padded_dim)
				    curP0[i] = curQ[new_pos];
				else
					curP0[i] = 0.0f;
			}			
        }

        for (int i = 0; i < qsize; i++) {
            float* query_org = massQ + i * padded_dim;		
		    float* query_extended = permutedQ + extended_dim * i;
			float* query_extended0 = permutedQ0 + extended_dim * i;
            appr_alg.searchKnn(query_org, query_extended, query_extended0, k, result[i], table, step, InitialTable);
        }
	    float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;	

        size_t correct = 0;
        size_t total = 0;		
		for (int i = 0; i < qsize; i++) { 
		    std::priority_queue<std::pair<float, labeltype >> gt(answers[i]);
            unordered_set<labeltype> g;
            total += gt.size();

            while (gt.size()) {

                g.insert(gt.top().second);
                gt.pop();
            }

            for(int j = 0; j < k; j++) {
                if (g.find(result[i][j].id) != g.end()) {
                    correct++;
                } 
            }
		}
		
        float recall = 1.0f * correct / total;
        cout << ef << "\t" << recall << "\t" << 1e6 / time_us_per_query << " QPS\n";
        if (recall > 1.0) {
            cout << recall << "\t" << time_us_per_query << " us\n";
            break;
        }
    }
		
    ::operator delete[](permutedQ, std::align_val_t{ALIGNMENT});
    ::operator delete[](permutedQ0, std::align_val_t{ALIGNMENT});	
}

void FastGraph(int efc_, int M_, int data_size_, int query_size_, int dim_,
               char* path_q_, char* path_data_, char* truth_data_,
               char* path_index_, int L_, int topk_) {

    int efConstruction = efc_;
    int M = M_;
    int step;
    if (efConstruction <= 2 * M) {
        efConstruction = 2 * M;
        step = 2 * M;
    } else if (efConstruction < 100 && efConstruction % (2 * M) == 0) {
        step = 2 * M;
    } else {
        efConstruction = (efConstruction + 99) / 100 * 100;
        step = 100;
    }

    if (L_ % 8 != 0 || M_ % 8 != 0) {
        throw std::invalid_argument("The value of L_ or M_ is invalid");
    }

    size_t maxk = topk_;
    size_t gt_maxk = 100; 
	size_t tablesize = std::max(topk_,100);
    size_t vecsize = data_size_;
    size_t qsize = query_size_;
    size_t vecdim = dim_;
    size_t padded_dim = (vecdim + 15) & ~0xF;
    size_t extended_dim = ((padded_dim + L_ - 1) / L_) * L_;

    if ((size_t)step > vecsize) step = (int)vecsize;

    char* path_q = path_q_;
    char* path_data = path_data_;
    char* path_index = path_index_;

    int m = 8;
    int level = L_;
    int subdim = extended_dim / level;

    std::vector<std::vector<std::vector<float>>> projVec(
        level,
        std::vector<std::vector<float>>(m, std::vector<float>(subdim, 0.0f)));

    int m0 = 8;

    std::vector<std::vector<std::vector<float>>> projVec0(
        4,
        std::vector<std::vector<float>>(m0, std::vector<float>(extended_dim / 4, 0.0f)));

    std::vector<Node> InitialTable;
    InitialTable.resize(static_cast<size_t>(16 * m0 * m0 * m0 * m0) * tablesize);

    

    L2Space l2space(padded_dim);
    InnerProductSpace ipsubspace0(extended_dim / 4);
    InnerProductSpace ipsubspace(subdim);
    InnerProductSpace ipspace(padded_dim);
    HierarchicalNSW<float>* appr_alg;

    std::vector<float> buf(vecdim);
    fs::path dir(path_index_);

    if (fs::exists(dir)) {
        cout << "Loading index from " << path_index << ":\n";
        appr_alg = new HierarchicalNSW<float>(&l2space, &ipspace, &ipsubspace, &ipsubspace0, path_index, false);

        cout << "Loading GT:\n";
        ifstream inputGT(truth_data_, ios::binary);
        unsigned int rows;
        unsigned int cols;
        
        

        if (rows != (unsigned int)qsize) {
            printf("Warning: Ground truth dimensions mismatch!\n");
        }

        unsigned int* massQA = new unsigned int[qsize * gt_maxk];
        inputGT.read((char*)massQA, (size_t)qsize * gt_maxk * 4);
        inputGT.close();

        cout << "Loading queries:\n";

        float* massQ = (float*)std::aligned_alloc(ALIGNMENT, qsize * padded_dim * sizeof(float));
        ifstream inputQ(path_q, ios::binary);

        
        

        for (int i = 0; i < qsize; i++) {
            inputQ.read((char*)buf.data(), 4 * vecdim);
            float* dst = &massQ[i * padded_dim];

            float norm = 0.0f;
            for (int j = 0; j < vecdim; j++)
                norm += buf[j] * buf[j];

            norm = std::sqrt(norm);
            if (norm == 0.0f) norm = 1.0f;

            for (int j = 0; j < vecdim; j++)
                dst[j] = buf[j] / norm;

            for (int j = vecdim; j < padded_dim; j++)
                dst[j] = 0.0f;
        }

        inputQ.close();

        vector<std::priority_queue<std::pair<float, labeltype>>> answers;
        size_t k = topk_;
        get_gt(massQA, qsize, answers, k, gt_maxk);

        const int TOTAL_ELEMENTS = level * 2 * m;
        float* table = (float*)std::aligned_alloc(ALIGNMENT, TOTAL_ELEMENTS * sizeof(float));
        test_vs_recall(massQ, vecsize, qsize, *appr_alg, answers, k, table, vecdim, extended_dim, InitialTable, path_index);

    } 
    else {

        fs::create_directories(dir);
        cout << "Building FastGraph index:\n";

        StopW stopw = StopW();
        StopW stopw_full = StopW();

    ifstream input(path_data, ios::binary);
    if (!input) {
        throw std::runtime_error(std::string("Failed to open data file: ") + path_data);
    }

    std::vector<float> norm(vecsize, 0.0f);
    std::vector<float> true_norm(vecsize, 0.0f);
    std::vector<float> dim_norm(extended_dim, 0.0f);
    std::vector<double> dim_norm2(extended_dim, 0.0);

    
    
    
    
    float* raw_data = (float*)std::aligned_alloc(
        ALIGNMENT, vecsize * extended_dim * sizeof(float));
    if (!raw_data) {
        throw std::bad_alloc();
    }

    
    for (size_t i = 0; i < vecsize; i++) {
        input.read((char*)buf.data(), sizeof(float) * vecdim);
        if (!input) {
            ::free(raw_data);
            throw std::runtime_error("Failed while reading raw data.");
        }

        float* dst = raw_data + i * extended_dim;
        std::memcpy(dst, buf.data(), sizeof(float) * vecdim);
        std::memset(dst + vecdim, 0, sizeof(float) * (extended_dim - vecdim));

        double sum = 0.0;
        for (size_t j = 0; j < vecdim; j++) {
            const double v = dst[j];
            sum += v * v;
            dim_norm2[j] += v * v;
        }

        norm[i] = (float)sum;
        true_norm[i] = std::sqrt(norm[i]);
    }

    input.close();

    for (size_t d = 0; d < vecdim; d++) {
        dim_norm[d] = static_cast<float>(dim_norm2[d] / (double)vecsize);
    }
    for (size_t d = vecdim; d < extended_dim; d++) {
        dim_norm[d] = 0.0f;
    }

    
    
    
    
    std::vector<std::pair<float, int>> array(vecsize);
    for (int i = 0; i < (int)vecsize; i++) {
        array[i].first = -norm[i];   
        array[i].second = i;         
    }

    std::sort(array.begin(), array.end(),
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                  if (a.first != b.first) return a.first < b.first;
                  return a.second < b.second;
              });

    
    
    
    
    std::vector<std::vector<float>> result_vectors(tablesize, std::vector<float>(vecdim, 0.0f));
    for (int i = 0; i < step; i++) {
        const int target_id = array[i].second;
        const float* src = raw_data + (size_t)target_id * extended_dim;
        std::memcpy(result_vectors[i].data(), src, sizeof(float) * vecdim);
    }

    std::vector<int> permutation0;
    std::vector<int> zero_positions0;

    std::vector<std::vector<float>> projVal(step, std::vector<float>(8 * m0, 0.0f));

    shuffle_for_equal_norm(dim_norm, vecsize, extended_dim, 4, permutation0, zero_positions0);
    generate_subspace_vectors_projvec(projVec0, 4, extended_dim / 4, m0, zero_positions0);

    std::mt19937 rng(12345);
    std::vector<int> permutation;
    std::vector<int> zero_positions;

    shuffle_for_equal_norm(dim_norm, vecsize, extended_dim, level, permutation, zero_positions);
    generate_subspace_vectors_projvec(projVec, level, subdim, m, zero_positions);

    std::string folderPath(path_index);
    std::string fullPath;
    if (!folderPath.empty() && (folderPath.back() == '/' || folderPath.back() == '\\')) {
        fullPath = folderPath + "permutation.bin";
    } else {
        fullPath = folderPath + "/permutation.bin";
    }

    {
        std::ofstream fout(fullPath, std::ios::binary);
        if (!fout) {
            ::free(raw_data);
            throw std::runtime_error("Failed to open permutation.bin for writing.");
        }
        fout.write((char*)permutation.data(), permutation.size() * sizeof(int));
        fout.write((char*)permutation0.data(), permutation0.size() * sizeof(int));
    }

    appr_alg = new HierarchicalNSW<float>(
        maxk, step, m, vecdim, projVec, level, subdim, permutation,
        &l2space, &ipspace, &ipsubspace, &ipsubspace0, vecsize,
        path_index, m0, projVec0, permutation0, tablesize, M, efConstruction);
         
    
    
    
    float** init_points = new float*[tablesize];
    for (int i = 0; i < tablesize; i++) {
        init_points[i] = new float[extended_dim];
        std::memset(init_points[i], 0, extended_dim * sizeof(float));
        std::memcpy(init_points[i], result_vectors[i].data(), vecdim * sizeof(float));
    }
	
    
    
    

    for (int i = 0; i < tablesize; i++) {
        const int cur_id = array[i].second;
        appr_alg->addPoint((void*)init_points[i], (size_t)cur_id, &(true_norm[cur_id]), InitialTable);
    }

    appr_alg->buildTable(InitialTable, init_points);

    
    
    
    
    size_t report_every = 10000;
    int progress_count = 0;

#pragma omp parallel for schedule(dynamic)
    for (int ord = tablesize; ord < (int)vecsize; ord++) {
        const int cur_id = array[ord].second;

        float* mass = new float[extended_dim];
        const float* src = raw_data + (size_t)cur_id * extended_dim;
        std::memcpy(mass, src, extended_dim * sizeof(float));

        appr_alg->addPoint((void*)mass, (size_t)cur_id, &(true_norm[cur_id]), InitialTable);

#pragma omp critical(progress_print)
        {
            progress_count++;
            if (progress_count % report_every == 0) {
                cout << (tablesize + progress_count) / (0.01 * vecsize) << " %, "
                     << report_every / (1000.0 * 1e-6 * stopw.getElapsedTimeMicro()) << " kips\n";
                stopw.reset();
            }
        }

        delete[] mass;
    }

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)vecsize; i++) {
        appr_alg->completeEdge(i);
    }

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)vecsize; i++) {
        appr_alg->expandSpace(i);
    }

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)vecsize; i++) {
        appr_alg->addEdgeProj(i);
    }

    appr_alg->compression(vecsize, InitialTable);
    cout << "FastGraph build time:" << 1e-6 * stopw_full.getElapsedTimeMicro() << "  seconds\n";
    appr_alg->saveIndex();

    
    
    
    if (!folderPath.empty() && (folderPath.back() == '/' || folderPath.back() == '\\')) {
        fullPath = folderPath + "table.bin";
    } else {
        fullPath = folderPath + "/table.bin";
    }

    {
        std::ofstream fout0(fullPath, std::ios::binary);
        if (!fout0) {
            for (int i = 0; i < tablesize; i++) delete[] init_points[i];
            delete[] init_points;
            ::free(raw_data);
            throw std::runtime_error("Failed to open table.bin for writing.");
        }
        fout0.write((char*)InitialTable.data(), InitialTable.size() * sizeof(Node));
    }

    
    
    
    for (int i = 0; i < tablesize; i++) {
        delete[] init_points[i];
    }
    delete[] init_points;

    ::free(raw_data);
}	

    return;
}
