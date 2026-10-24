/*
lp_norm/ML.cpp — MovieLens l_p norm experiment
================================================================================
Paper correspondence: Table 2 (table:realpnorm), MovieLens row
(Current paper mapping: this is the real-data portion of Table 2 (label table:lpnorm);
 the 2024-era label table:realpnorm no longer exists in the submitted paper.)
Data: experiment/Distributed JoinSize/datasets/ratings.csv (678MB) and tags.csv
Output/reconciliation: the mean/median/max aggregates in the paper table come from
post-processing of multiple runs (see the original directory's xlsx); this driver
outputs the raw per-column results.
(Current paper mapping: this is the real-data portion of Table 2 (label table:lpnorm).)
*/

#include <iostream>
#include <vector>
#include <fstream>
#include <numeric>
#include <string>
#include <iomanip>
#include "countsketch.hpp"

using namespace std;


int main(int argc, char *argv[]){
	//sketch::CountSketch CS = sketch::CountSketch(0.001);

	int round = 5;
	
	if(argc<2){
		cout<<"lack of parameters"<<endl;
	}
	int p=stoi(argv[1]);


	std::ofstream outfile("ML"+to_string(p)+".txt");

	vector<string> list={"userID","movieId","rating"};
//vector<string> list={"utcdate"};
	

	for(int j=0; j<list.size();j++){
		double ams_res=0.0;
		double ccs_res=0.0;
		double mgccs_res=0.0;
		for(int i=0;i<round; i++){
			ifstream ifs("../datasets/ML/"+list[j]);
			sketch::AMS ams = sketch::AMS(0.05,25000096,0.01,i);
			sketch::ComplexCountSketch ccs = sketch::ComplexCountSketch(p,0.05,0.01,i);
			sketch::MGCCS mgccs = sketch::MGCCS(p,0.01,0.05,0.01,i);
			if(ifs.is_open()){
				string index;
				while(getline(ifs,index)){
					ams.Insert(index.c_str(),index.size());
					ccs.Insert(index.c_str(),index.size());
					mgccs.Insert(index.c_str(),index.size());
				}
				ifs.close();
			}

			ams_res = (ams_res*i+ams.estimate(p))/(i+1);
			ccs_res = (ccs_res*i+ccs.estimate_lp())/(i+1);
			mgccs_res = (mgccs_res*i+mgccs.estimate_lp())/(i+1);
			cout<<list[j]<<": "<<ams.estimate(p)<<","<<ccs.estimate_lp()<<","<<mgccs.estimate_lp()<<endl;
		}

		//cout<<list[j]<<": "<<ams_res<<","<<ccs_res<<","<<mgccs_res<<endl;
		outfile<<list[j]<<","<<ams_res<<","<<ccs_res<<","<<mgccs_res<<endl;

	}


	outfile.close();

}