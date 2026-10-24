/*
join2/MLjoin2.cpp — MovieLens real-data 2-join
================================================================================
Paper correspondence: same as Airlinesjoin2.cpp (table deleted, kept for reproducibility).
*/

#include <iostream>
#include <vector>
#include <fstream>
#include <numeric>
#include <iomanip>
#include "countsketch.hpp"

int Heavy_Thes=500;
int BILI=4;

using namespace std;
// vector<string> items;
// unordered_map<string, int>freq1,freq0;
// long double Join_Ground_Truth;

// int memaccess_i = 0, memaccess_q = 0, memaccess_d = 0;

// int thres_mid = 16, thres_large = 0;

// int item_num = 0, flow_num = 0;
// vector<double> experiment_MEM,experiment_MAX,experiment_MIN,experiment_ARE, experiment_AAE,experiment_CON,experiment_YangBen,experiment_INT,experiment_JOT;
// vector<vector<double> >experiment_all;
// int data_delimiter;


int main(int argc,char *argv[]){
	int round = 5;

	std::ofstream outfile("ML2join.txt");
	
	double agms_res = 0.0;
	double cs_res = 0.0;
	double js_res = 0.0;
	double mg_cs_res = 0.0;
	vector<string> list ={"userId","movieId"};
for(int j=0;j<list.size();j++){
	ifstream if1("../datasets/ML/tags/"+list[j]);
	ifstream if2("../datasets/ML/ratings/"+list[j]);
	vector<string>data1;
	vector<string>data2;
	if(if1.is_open()){
		string index;
		while(getline(if1,index)){
			data1.push_back(index);
		}
	}
	if(if2.is_open()){
		string index;
		while(getline(if2,index)){
			data2.push_back(index);
		}
	}


	for(int i=0;i<round;i++){
		double eps_js=0.01,eps=0.015,gam=0.01;
		// cout<<"begin agms"<<endl;
		// sketch::AGMS agms1 = sketch::AGMS(eps,gam,i);
		// sketch::AGMS agms2 = sketch::AGMS(eps,gam,i);
		cout<<"begin cs"<<endl;
		sketch::CountSketch cs1 = sketch::CountSketch(eps,gam,i);
		sketch::CountSketch cs2 = sketch::CountSketch(eps,gam,i);
		cout<<"begin js"<<endl;
		sketch::Classifier js1 = sketch::Classifier(ceil( 1/ eps_js/eps_js),ceil(log(4 / gam)),i);
		sketch::Classifier js2 = sketch::Classifier(ceil( 1/ eps_js/eps_js),ceil(log(4 / gam)),i);
		cout<<"begin mgcs"<<endl;
		sketch::MGCS mgcs1 = sketch::MGCS(0.002,eps,gam,i);
		sketch::MGCS mgcs2 = sketch::MGCS(0.002,eps,gam,i);
		cout<<"begin scan"<<endl;
		for(int ind = 0; ind< data1.size(); ind++){
				//agms1.Insert( data1[ind].c_str(),data1[ind].size() );
				js1.Insert( data1[ind].c_str()  );
				cs1.Insert( data1[ind].c_str(),data1[ind].size() );
				mgcs1.Insert( data1[ind].c_str(),data1[ind].size() );
		}
		cout<<"begin scan2"<<endl;
		for(int ind = 0; ind< data2.size(); ind++){
				//agms2.Insert( data2[ind].c_str(),data2[ind].size() );
				js2.Insert( data2[ind].c_str()  );
				cs2.Insert( data2[ind].c_str(),data2[ind].size() );
				mgcs2.Insert( data2[ind].c_str(),data2[ind].size() );
			
		}
		cout<<"begin esti"<<endl;
		//agms_res = (agms_res*i+agms1.Join(agms2) )/(i+1);
		cs_res = (cs_res*i+cs1.Join(&cs2) /* [paper_code patch] Join takes a pointer parameter, added &; semantics unchanged */ )/(i+1);
		js_res = (js_res*i+js1.Join(&js2) )/(i+1);
		mg_cs_res = (mg_cs_res*i+mgcs1.Join(&mgcs2) /* [paper_code patch] Join takes a pointer parameter, added &; semantics unchanged */ )/(i+1);
		cout<<"result: "<<cs_res<<","<<js_res<<","<<mg_cs_res<<endl;
	}
	outfile<<list[j]<<","<<cs_res<<","<<js_res<<","<<mg_cs_res<<endl;
	
}

	
	outfile.close();

	return 0;
}