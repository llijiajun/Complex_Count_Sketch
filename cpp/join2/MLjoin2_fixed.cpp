/*
join2/MLjoin2_fixed.cpp — MovieLens 2-join, JoinSketch driver with a corrected
key interface, plus a budget sweep.
================================================================================
[Correction note] The original driver (MLjoin2_matched.cpp) passed the data-row string
directly to Classifier::Insert(const char*), but that interface takes the key as
((unsigned*)str)[0], i.e. it only reads the first 4 bytes. MovieLens userId/movieId are
as long as 6–7 decimal digits, so the prefix beyond 4 bytes is truncated: e.g. "123456"
and "123499" both collapse to "1234". Both sides truncate by the same rule, so the join
estimate sees the sum of products of merged-prefix frequencies Σ_P F1(P)F2(P) ≥ the true
value Σ_u f1(u)f2(u), systematically overestimating (userId rows measured ~10x overestimate,
the source of the 9.7E+00 in old Table 7 of the paper; Airlines keys are ≤4 chars,
unaffected, so the old numbers remain valid).

This driver instead generates a 32-bit key with MurmurHash32(the whole row string, a fixed
seed) and calls Classifier::Insert(unsigned), eliminating the truncation; the Heavy layer
does exact matching on this 32-bit key, and the collision probability of 1.6e5 distinct
keys in the 2^32 space is negligible.

It also sweeps the same budget coefficient over W ∈ {21845, 43690, 87380} (i.e. 1x/2x/4x of
the paper's matched budget of 131,072 counters), with 3 seeds each, reporting estimates and
ground truth.
*/

#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include "countsketch.hpp"

int Heavy_Thes=500;
int BILI=4;

using namespace std;

static inline unsigned key_id(const string& s){
    return (unsigned)MurmurHash32(s.data(), (int)s.size(), 20260926u);
}

int main(int argc,char *argv[]){
    // three budget tiers: 1x / 2x / 4x of the matched budget
    int budgets[3] = {21845, 43690, 87380};

    std::ofstream outfile("ML2join_fixed.txt");
    outfile << "# truth userId=1243199075 movieId=10883253790 (full scans)" << endl;
    outfile << "# key interface: MurmurHash32(full string); budget sweep 1x/2x/4x" << endl;

    vector<string> list ={"userId","movieId"};
    double truth[2] = {1243199075.0, 10883253790.0};

    for(int j=0;j<(int)list.size();j++){
        ifstream if1("../datasets/ML/tags/"+list[j]);
        ifstream if2("../datasets/ML/ratings/"+list[j]);
        vector<string> data1, data2;
        if(if1.is_open()){ string index; while(getline(if1,index)) data1.push_back(index); }
        if(if2.is_open()){ string index; while(getline(if2,index)) data2.push_back(index); }
        cout << "loaded " << list[j] << ": " << data1.size() << " x " << data2.size() << endl;

        for(int b=0;b<3;b++){
            int W = budgets[b];
            for(int seed=0; seed<5; seed++){
                sketch::Classifier js1 = sketch::Classifier(W, 6, seed);
                sketch::Classifier js2 = sketch::Classifier(W, 6, seed);
                for(size_t ind=0; ind<data1.size(); ind++)
                    js1.Insert( key_id(data1[ind]) );
                for(size_t ind=0; ind<data2.size(); ind++)
                    js2.Insert( key_id(data2[ind]) );
                double est = (double)js1.Join(&js2);
                double rel = fabs(est - truth[j]) / truth[j];
                cout << list[j] << " W=" << W << " seed=" << seed
                     << " est=" << est << " rel=" << rel << endl;
                outfile << list[j] << ",W" << W << ",seed" << seed << ","
                        << est << "," << rel << endl;
            }
        }
    }
    outfile.close();
    return 0;
}
