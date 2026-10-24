/*
lp_norm/MG4CCS.cpp — MGCCS (Misra-Gries two-level CCS) experiment
================================================================================
Paper correspondence: none. The MG-family methods have been completely removed from the
paper; this file is archived only and must NOT be used for reconciliation.
*/

#include <iostream>
#include <vector>
#include <fstream>
#include <numeric>
#include <iomanip>
#include "countsketch.hpp"

using namespace std;


int main(){
	std::ofstream outfile("result_MG4CCS.txt");
	vector<double> tru={10029895125034,10059832803193072,1.009984190e20};

	vector<double> param={0.1,0.05,0.025,0.0125,0.00625,0.003125,0.0015625,0.00078125,0.000390625};
	for( int p = 3 ; p<6 ; p++){
		double ccs_res=0.0;
		double mgccs_res=0.0;

		for(int i=0;i<param.size(); i++){
			ifstream ifs("zipf1.1.dat");
			
			sketch::ComplexCountSketch ccs = sketch::ComplexCountSketch(p,0.1,0.01,666);
			
			sketch::MGCCS mgccs = sketch::MGCCS(p,param[i],0.1,0.01,666);
			

			if(ifs.is_open()){
				string index;
				while(getline(ifs,index)){
					ccs.Insert(index.c_str(),index.size());
					mgccs.Insert(index.c_str(),index.size());
				}
				ifs.close();
			}
			ccs_res = ccs.estimate_lp();
			mgccs_res = mgccs.estimate_lp();
			cout<<param[i]<<","<<ccs_res<<","<<mgccs_res<<","<<p<<endl;
			outfile<<param[i]<<","<<mgccs_res<<","<<p<<endl;
		}
		
		
	}

	outfile.close();

}