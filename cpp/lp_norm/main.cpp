/*
lp_norm/main.cpp — synthetic-data l_p norm experiment (AMS vs CCS)
================================================================================
Paper correspondence: Table 3 (table:sylpnorm) "The pth power of l_p norm estimation
          relative errors of AMS and CCS on different synthetic data distributions"
(Current paper mapping: the synthetic table and the real-data table were merged, so
 this driver now feeds the SYNTHETIC upper block of Table 2 (label table:lpnorm).
 The 2024-era label table:sylpnorm no longer exists in the submitted paper.)
Data: uniform.dat / zipf1.1.dat / zipf1.5.dat / uniform+zipf1.1.dat
      (10M rows, located in the original repo experiment/Distributed JoinSize/Lp/,
      can also be regenerated with the datasets_tools/ scripts)
Run: g++ -O2 -std=c++11 -I ../common main.cpp -o main && ./main
      (execute in the directory containing the .dat files; int p=5 selects the column,
      change it to 3/4 to obtain the other two value groups in the table)
Output: result{p}.txt, each line "distribution, AMS, CCS, MGCCS"
Note: the MGCCS column is a historical method already removed from the paper; only the
      CCS column is used for reconciliation; this file's logic is unchanged.
(Current paper mapping: this is the synthetic portion of Table 2 (label table:lpnorm).)
*/

#include <iostream>
#include <vector>
#include <fstream>
#include <numeric>
#include <iomanip>
#include "countsketch.hpp"

using namespace std;


int main(){
	//sketch::CountSketch CS = sketch::CountSketch(0.001);

	int round = 5;
	int p=5;

	std::ofstream outfile("result"+to_string(p)+".txt");

	double ams_res=0.0;
	double ccs_res=0.0;
	double mgccs_res=0.0;

	for(int i=0;i<round; i++){
		ifstream ifs("uniform.dat");
		sketch::AMS ams = sketch::AMS(0.1,10000000,0.01,i);
		
		sketch::ComplexCountSketch ccs = sketch::ComplexCountSketch(p,0.1,0.01,i);
		
		sketch::MGCCS mgccs = sketch::MGCCS(p,0.01,0.1,0.01,i);
		

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
	cout<<"final: "<<ams_res<<","<<ccs_res<<","<<mgccs_res<<endl;
	outfile<<"uniform,"<<ams_res<<","<<ccs_res<<","<<mgccs_res<<endl;

	ams_res=0.0;
	ccs_res=0.0;
	mgccs_res=0.0;

	for(int i=0;i<round; i++){
		ifstream ifs("zipf1.1.dat");
		sketch::AMS ams = sketch::AMS(0.1,10000000,0.01,i);
		
		sketch::ComplexCountSketch ccs = sketch::ComplexCountSketch(p,0.1,0.01,i);
		
		sketch::MGCCS mgccs = sketch::MGCCS(p,0.01,0.1,0.01,i);
		

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
	cout<<"final: "<<ams_res<<","<<ccs_res<<","<<mgccs_res<<endl;
	outfile<<"zipf 1.1,"<<ams_res<<","<<ccs_res<<","<<mgccs_res<<endl;

	ams_res=0.0;
	ccs_res=0.0;
	mgccs_res=0.0;

	for(int i=0;i<round; i++){
		ifstream ifs("zipf1.5.dat");
		sketch::AMS ams = sketch::AMS(0.1,10000000,0.01,i);
		
		sketch::ComplexCountSketch ccs = sketch::ComplexCountSketch(p,0.1,0.01,i);
		
		sketch::MGCCS mgccs = sketch::MGCCS(p,0.01,0.1,0.01,i);
		

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
	cout<<"final: "<<ams_res<<","<<ccs_res<<","<<mgccs_res<<endl;
	outfile<<"zipf 1.5,"<<ams_res<<","<<ccs_res<<","<<mgccs_res<<endl;

	for(int i=0;i<round; i++){
		ifstream ifs("uniform+zipf1.1.dat");
		sketch::AMS ams = sketch::AMS(0.1,10000000,0.01,i);
		
		sketch::ComplexCountSketch ccs = sketch::ComplexCountSketch(p,0.1,0.01,i);
		
		sketch::MGCCS mgccs = sketch::MGCCS(p,0.01,0.1,0.01,i);
		

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
	cout<<"final: "<<ams_res<<","<<ccs_res<<","<<mgccs_res<<endl;
	outfile<<"uniform+zipf1.1,"<<ams_res<<","<<ccs_res<<","<<mgccs_res<<endl;

	outfile.close();

}