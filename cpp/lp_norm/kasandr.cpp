/*
lp_norm/kasandr.cpp — Kasandr l_p norm experiment
================================================================================
Paper correspondence: Table 2 (table:realpnorm), Kasandr row
Data: experiment/Distributed JoinSize/datasets/kasandr.zip (944MB, use after unzipping)
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


	std::ofstream outfile("kasandr"+to_string(p)+".txt");

	vector<string> list={"rating","utcdate","merchant","category","countrycode","offerid","userid"};
//vector<string> list={"utcdate"};
	

	for(int j=0; j<list.size();j++){
		double ams_res=0.0;
		double ccs_res=0.0;
		double mgccs_res=0.0;
		for(int i=0;i<round; i++){
			ifstream ifs("../datasets/kasandr/merge/"+list[j]);
			sketch::AMS ams = sketch::AMS(0.05,17764278,0.01,i);
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