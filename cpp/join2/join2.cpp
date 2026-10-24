/*
join2/join2.cpp — 2-way join driver (CountSketch / JoinSketch / MGCS)
================================================================================
Paper correspondence: Table 6 (table:synjs) "Relative errors of 2-join size estimation on
          synthetic datasets" driver program.
(Current paper mapping: the submitted paper has NO table for this experiment -- the
 synthetic 2-join comparison is discussed in the text (Section 5.3) instead. The
 2024-era label table:synjs no longer exists.)
Data path: hardcoded in source, e.g. ifstream("../datasets/Airlines/Origin"); to run the
synthetic-data (Table 6) experiment, change the two ifstream lines to uniform.dat / zipf*.dat
(10M-row .dat files as above).
Ground truth: synthetic -> Join/JoinSythetciTrue.csv (6 combinations, in the same row order
as the table); real -> Join/JoinTrue.csv.
Sketch correspondence: CountSketch -> the paper's "Count Sketch (Fast-AGMS)" column;
             Classifier   -> JoinSketch column (include/ also has JoinSketch_ver3/4.h);
             MGCS         -> historical method, not used in the paper.
Output: Airlines2join.txt (filename not changed with the data, historical naming).
(Current paper mapping: Table 4 (label table:pstar).)
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

	std::ofstream outfile("Airlines2join.txt");
	
	double agms_res = 0.0;
	double cs_res = 0.0;
	double js_res = 0.0;
	double mg_cs_res = 0.0;


	for(int i=0;i<round;i++){
		ifstream if1("../datasets/Airlines/Origin");
		ifstream if2("../datasets/Airlines/Dest");
		double eps=0.01,gam=0.01;
		cout<<"begin agms"<<endl;
		//sketch::AGMS agms1 = sketch::AGMS(eps,gam,i);
		//sketch::AGMS agms2 = sketch::AGMS(eps,gam,i);
		cout<<"begin cs"<<endl;
		sketch::CountSketch cs1 = sketch::CountSketch(eps,gam,i);
		sketch::CountSketch cs2 = sketch::CountSketch(eps,gam,i);
		cout<<"begin js"<<endl;
		sketch::Classifier js1 = sketch::Classifier(ceil( 1/ eps/eps),ceil(log(4 / gam)),i);
		sketch::Classifier js2 = sketch::Classifier(ceil( 1/ eps/eps),ceil(log(4 / gam)),i);
		cout<<"begin mgcs"<<endl;
		sketch::MGCS mgcs1 = sketch::MGCS(0.002,eps,gam,i);
		sketch::MGCS mgcs2 = sketch::MGCS(0.002,eps,gam,i);
		cout<<"begin scan"<<endl;
		if(if1.is_open()){
			string index;
			while(getline(if1,index)){
				//agms1.Insert( index.c_str(),index.size() );
				//cout<<58<<endl;
				/* [paper_code patch] */ js1.Insert( index.c_str() ); /* Classifier::Insert takes only 1 argument (from header line 1072), the original 2-argument call would not compile */
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
				/* [paper_code patch] */ js2.Insert( index.c_str() );
				cs2.Insert( index.c_str(),index.size() );
				mgcs2.Insert( index.c_str(),index.size() );
			}
			if2.close();
		}
		cout<<"begin esti"<<endl;
		/* [paper_code patch] */ //agms_res = (agms_res*i+agms1.Join(agms2) )/(i+1); /* AGMS declaration is commented out and the paper excludes AGMS (see exp.tex Setup), so this line is also disabled */
		/* [paper_code patch] */ cs_res = (cs_res*i+cs1.Join(&cs2) )/(i+1); /* CountSketch::Join takes a pointer parameter (header line 220) */
		js_res = (js_res*i+js1.Join(&js2) )/(i+1);
		/* [paper_code patch] */ mg_cs_res = (mg_cs_res*i+mgcs1.Join(&mgcs2) )/(i+1); /* same as above, MGCS::Join takes a pointer parameter (line 551) */
	}
	cout<<"result: "<<agms_res<<","<<cs_res<<","<<js_res<<","<<mg_cs_res<<endl;
	outfile<<"result: "<<agms_res<<","<<cs_res<<","<<js_res<<","<<mg_cs_res<<endl;
	

	
	outfile.close();

	return 0;
}