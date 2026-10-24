/*
lp_norm/SSB.cpp — l_p norm experiment on the SSB dataset
================================================================================
Paper correspondence: none (historical experiment; the final version of the paper does
not use SSB). Kept for completeness.
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

	std::ofstream outfile("SSB"+to_string(p)+".txt");

	//vector<string> list={"DepDelay","Month","DayofMonth","DayOfWeek","CRSDepTime","CRSArrTime","UniqueCarrier","Origin","Dest","Distance"};

	vector<string> list={"9","13"};

	for(int j=0; j<2;j++){

		double ams_res=0.0;
		double ccs_res=0.0;
		double mgccs_res=0.0;
		for(int i=0;i<round; i++){
			cout<<"round: "<<round<<", file:"<<list[j]<<endl;
			ifstream ifs("../datasets/SSB/"+list[j]);
			sketch::AMS ams = sketch::AMS(0.01,11997996,0.01,i);
			sketch::ComplexCountSketch ccs = sketch::ComplexCountSketch(p,0.01,0.01,i);
			sketch::MGCCS mgccs = sketch::MGCCS(p,0.01,0.01,0.01,i);
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
		}

		cout<<list[j]<<": "<<ams_res<<","<<ccs_res<<","<<mgccs_res<<endl;
		outfile<<list[j]<<","<<ams_res<<","<<ccs_res<<","<<mgccs_res<<endl;

	}


	outfile.close();

}