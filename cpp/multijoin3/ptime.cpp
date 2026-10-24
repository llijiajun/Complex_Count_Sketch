/*
multijoin3/ptime.cpp — update/query time measurement for multi-way joins
================================================================================
Paper correspondence: no direct table (historical timing measurement).
*/

#include <iostream>
#include <vector>
#include <fstream>
#include <numeric>
#include <iomanip>
#include "countsketch.hpp"
#include <ctime>

int Heavy_Thes=500;
int BILI=4;

using namespace std;

int main(int argc,char *argv[]){
	int round = 5;

	std::ofstream outfile("join3.txt");
	
	double ccs_res = 0.0;
	double sp_res = 0.0;
	double sm_res = 0.0;
	double ccs_time=0,cs_time=0;
	double eps_js=0.01,eps=0.015,gam=0.01;
	//double mg_cs_res = 0.0;

	vector<string> list ={"uniform.dat","zipf1.1.dat","zipf1.5.dat"};
int i=0;
	for(int p=3; p < 11; p++){


	sketch::ComplexCountSketch ccs1 = sketch::ComplexCountSketch(p,eps,gam,i);
	vector< sketch::ComplexCountSketch> ccs_list = vector< sketch::ComplexCountSketch>(p-1,
			sketch::ComplexCountSketch(p,eps,gam,i));

	vector<int> seed1;
	for(int i=0; i<p-1;i++){
		seed1.push_back(i);
	}
	sketch::CountSketch2 cs1 = sketch::CountSketch2(eps,seed1,gam,666);
	vector< sketch::CountSketch2> cs_list;

		ifstream if1("../LP/"+list[0]);
		vector<string>data1;
		if(if1.is_open()){
			string index;
			while(getline(if1,index)){
				data1.push_back(index);
			}
		}
		clock_t start = clock();
		for(int ind = 0; ind< data1.size(); ind++){
				ccs1.Insert( data1[ind].c_str() ,data1[ind].size() );
		}
		clock_t end = clock();
		ccs_time+= static_cast<double>(end - start) / CLOCKS_PER_SEC ;
		clock_t start2 = clock();
		for(int ind = 0; ind< data1.size(); ind++){
				cs1.Insert(data1[ind].c_str(),data1[ind].size() );
		}
		clock_t end2 = clock();
		cs_time+= static_cast<double>(end2 - start2) / CLOCKS_PER_SEC ;

		for(int j = 0; j <p-1; j++){
			vector<int> seed2;
			seed2.push_back(j);
			sketch::CountSketch2 cs2 = sketch::CountSketch2(eps,seed2,gam,666);
			ifstream if1("../LP/"+list[1]);
			vector<string>data1;
			if(if1.is_open()){
				string index;
				while(getline(if1,index)){
					data1.push_back(index);
				}
			}
			clock_t start = clock();
			for(int ind = 0; ind< data1.size(); ind++){
				ccs_list[0].Insert( data1[ind].c_str() ,data1[ind].size() );
			}
			clock_t end = clock();
			//ccs_time+= static_cast<double>(end - start) / CLOCKS_PER_SEC ;
			clock_t start2 = clock();
			for(int ind = 0; ind< data1.size(); ind++){
				cs2.Insert(data1[ind].c_str(),data1[ind].size() );
			}
			cs_list.push_back(cs2);
			clock_t end2 = clock();
			//cs_time+= static_cast<double>(end2 - start2) / CLOCKS_PER_SEC ;
		}
		cout<<ccs1.Join(ccs_list)<<","<<ccs1.Join(ccs_list)<<endl;

		cout<<ccs_time<<","<<cs_time<<endl;
	
	}
	
	

	return 0;
}