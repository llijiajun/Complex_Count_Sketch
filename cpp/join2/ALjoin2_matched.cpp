/*
join2/Airlinesjoin2.cpp — Airlines real-data 2-join
================================================================================
Paper correspondence: none (the real-data 2-join table table:realjs from the old version
of the paper was deleted due to incomplete data/logs; see review report §7.6). Kept for
reproducibility.
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

	std::ofstream outfile("Airlines2join_matched.txt");
	
	double agms_res = 0.0;
	double cs_res = 0.0;
	double js_res = 0.0;
	double mg_cs_res = 0.0;


	for(int i=0;i<round;i++){
		ifstream if1("../datasets/Airlines/Origin");
		ifstream if2("../datasets/Airlines/Dest");
		double eps_js=0.01,eps=0.015,gam=0.01;
		//cout<<"begin agms"<<endl;
		//sketch::AGMS agms1 = sketch::AGMS(eps,gam,i);
		//sketch::AGMS agms2 = sketch::AGMS(eps,gam,i);
		cout<<"begin cs"<<endl;
		sketch::CountSketch cs1 = sketch::CountSketch(eps,gam,i);
		sketch::CountSketch cs2 = sketch::CountSketch(eps,gam,i);
		cout<<"begin js"<<endl;
		sketch::Classifier js1 = sketch::Classifier(21845, 6, i);
		sketch::Classifier js2 = sketch::Classifier(21845, 6, i);
		cout<<"begin mgcs"<<endl;
		sketch::MGCS mgcs1 = sketch::MGCS(0.002,eps,gam,i);
		sketch::MGCS mgcs2 = sketch::MGCS(0.002,eps,gam,i);
		cout<<"begin scan"<<endl;
		if(if1.is_open()){
			string index;
			while(getline(if1,index)){
				//agms1.Insert( index.c_str(),index.size() );
				//cout<<58<<endl;
				js1.Insert( index.c_str());
				//cout<<60<<endl;
				cs1.Insert( index.c_str(),index.size() );
				//cout<<62<<endl;
				mgcs1.Insert( index.c_str(),index.size() );
				//cout<<64<<endl;
			}
			if1.close();
		}
		cout<<"begin scan2"<<endl;
		if(if2.is_open()){
			string index;
			while(getline(if2,index)){
				//agms2.Insert( index.c_str(),index.size() );
				js2.Insert( index.c_str() );
				cs2.Insert( index.c_str(),index.size() );
				mgcs2.Insert( index.c_str(),index.size() );
			}
			if2.close();
		}
		cout<<"begin esti"<<endl;
		
		cs_res = (cs_res*i+cs1.Join(&cs2) /* [paper_code patch] Join takes a pointer parameter, added &; semantics unchanged */ )/(i+1);
		js_res = (js_res*i+js1.Join(&js2) )/(i+1);
		mg_cs_res = (mg_cs_res*i+mgcs1.Join(&mgcs2) /* [paper_code patch] Join takes a pointer parameter, added &; semantics unchanged */ )/(i+1);
	}
	cout<<"result: "<<cs_res<<","<<js_res<<","<<mg_cs_res<<endl;
	outfile<<"result: "<<cs_res<<","<<js_res<<","<<mg_cs_res<<endl;
	
	for(int i=0;i<round;i++){
		ifstream if1("../datasets/Airlines/CRSArrTime");
		ifstream if2("../datasets/Airlines/CRSDepTime");
		double eps_js=0.01,eps=0.015,gam=0.01;
		//cout<<"begin agms"<<endl;
		//sketch::AGMS agms1 = sketch::AGMS(eps,gam,i);
		//sketch::AGMS agms2 = sketch::AGMS(eps,gam,i);
		cout<<"begin cs"<<endl;
		sketch::CountSketch cs1 = sketch::CountSketch(eps,gam,i);
		sketch::CountSketch cs2 = sketch::CountSketch(eps,gam,i);
		cout<<"begin js"<<endl;
		sketch::Classifier js1 = sketch::Classifier(21845, 6, i);
		sketch::Classifier js2 = sketch::Classifier(21845, 6, i);
		cout<<"begin mgcs"<<endl;
		sketch::MGCS mgcs1 = sketch::MGCS(0.002,eps,gam,i);
		sketch::MGCS mgcs2 = sketch::MGCS(0.002,eps,gam,i);
		cout<<"begin scan"<<endl;
		if(if1.is_open()){
			string index;
			while(getline(if1,index)){
				//agms1.Insert( index.c_str(),index.size() );
				//cout<<58<<endl;
				js1.Insert( index.c_str());
				//cout<<60<<endl;
				cs1.Insert( index.c_str(),index.size() );
				//cout<<62<<endl;
				mgcs1.Insert( index.c_str(),index.size() );
				//cout<<64<<endl;
			}
			if1.close();
		}
		cout<<"begin scan2"<<endl;
		if(if2.is_open()){
			string index;
			while(getline(if2,index)){
				//agms2.Insert( index.c_str(),index.size() );
				js2.Insert( index.c_str() );
				cs2.Insert( index.c_str(),index.size() );
				mgcs2.Insert( index.c_str(),index.size() );
			}
			if2.close();
		}
		cout<<"begin esti"<<endl;
		
		cs_res = (cs_res*i+cs1.Join(&cs2) /* [paper_code patch] Join takes a pointer parameter, added &; semantics unchanged */ )/(i+1);
		js_res = (js_res*i+js1.Join(&js2) )/(i+1);
		mg_cs_res = (mg_cs_res*i+mgcs1.Join(&mgcs2) /* [paper_code patch] Join takes a pointer parameter, added &; semantics unchanged */ )/(i+1);
	}
	cout<<"result: "<<cs_res<<","<<js_res<<","<<mg_cs_res<<endl;
	outfile<<"result: "<<cs_res<<","<<js_res<<","<<mg_cs_res<<endl;
	
	outfile.close();

	return 0;
}