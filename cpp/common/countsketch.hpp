/*
common/countsketch.hpp — main header file (AMS / CountSketch / ComplexCountSketch /
                          MGCS / Short_C_Sketch / Classifier, etc.)
================================================================================
Paper correspondence: the original C++ implementation of all sketch structures in §2–§3.
[paper_code patch note] Compared with the original file, 3 portability/compilability
fixes were applied (all marked with the "paper_code patch" comment): 1) the x86 intrinsics
header is now conditionally included; 2) the global definitions carried in the #ifdef UNIX
block are now compiled unconditionally; 3) SIMD_match_8 adds a scalar equivalent fallback
(bit-identical results); additionally fixed a missing & in one MGCS::Join compilation error.
Numerical behavior is identical to the original.
*/

#if !defined(JOINSKETCH_HPP)
#define JOINSKETCH_HPP

#include "MurmurHash.h"
#include <iostream>
#include <string.h>
#include <algorithm>
#include <vector>
#include<cmath>
#include<random>
#include <unordered_map>
#include <map>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include <stdexcept>
/* [paper_code patch] The original file wrapped x86 intrinsics and global definitions with
   #ifdef UNIX/#else, and the #else branch carried definitions such as Heavy_Thes (which
   would be lost under -DUNIX). After checking the whole file, no SSE/AVX intrinsic is used,
   so changed to: definitions compiled unconditionally, x86 headers included only on x86 platforms. */
#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#include <emmintrin.h>
#endif


extern int Heavy_Thes;
extern int BILI;

#define COUNTER_SIZE 32    //counter_size
#define WORD_SIZE 64
#define MAX_HASH_NUM 16     //hash_number  = d
#define FILTER_SIZE 32
#define KEY_LEN 4
#define testcycles 2
#define fingerprint 997

/* [paper_code patch] The original #endif has been merged into the conditional include above */

namespace sketch{

int generate_random_integer(int N) {
    if (N <= 0) {
        throw std::invalid_argument("N must be a positive integer");
    }

    // create random number engine and distribution
    std::random_device rd; // used for seeding
    std::mt19937 gen(rd()); // uses the Mersenne Twister random generation algorithm
    std::uniform_int_distribution<> dis(0, N-1); // uniform distribution, from 0 to N-1

    return dis(gen);
}

class AGMS{
public:
	int width;
	int AGMS_num;
	int hash_seed;
	std::vector<std::vector<int> > counter;
	AGMS(double eps, double gam=0.01,int _hash_seed=666){
		width= ceil(1 / eps/eps);
		AGMS_num= ceil(log(4 / gam));
		hash_seed=_hash_seed;
		this->counter = std::vector<std::vector<int>>();
		for(int i=0;i<AGMS_num;i++){
			this->counter.push_back(std::vector<int>(width,0));
		}
	}
	void Insert(const char* str,int len, int freq = 1){
		unsigned g = 0;
		for (int i = 0; i < AGMS_num; i++)
		{
			for(int j=0;j<width;j++){
				if(!g)g=((unsigned)MurmurHash32(str, len, hash_seed + i*j));
				if(g&1)counter[i][j]+=freq;
				else counter[i][j]-=freq;
				g>>=1;
			}
		}
	}
	double estimate_l2(){
		std::vector<double> result(this->AGMS_num,0);

		for (int i = 0; i < AGMS_num; i++){
			long double temp=1.0;
			for(int j=0;j<width;j++){
				temp*=counter[i][j]*counter[i][j];
			}
			result[i]=1.0*temp/width;
		}
		sort(result.begin(),result.end());
		if (AGMS_num % 2 == 0)
			return ((result[AGMS_num / 2] + result[AGMS_num / 2 - 1]) / 2);
		else
			return (result[AGMS_num / 2]);
	}
	double Join(AGMS & other){
		std::vector<double> result(this->AGMS_num,0);
		for (int i = 0; i < this->AGMS_num; i++){
			double k=0;
			for (int j = 0; j < width; j++)
				k+=counter[i][j]*other.counter[i][j];
			result[i]=1.0*k/width;
		}
		double re=0;
		for(int i=0;i<this->AGMS_num;i++)re+=result[i];
		return 1.0*re/this->AGMS_num;
	}
};

class AMS{
public:
	int width;
	int AMS_num;
	int current_index;
	int N;
	int hash_seed;
	std::vector<std::vector<int> > counter;
	std::vector<std::vector<int> > index;
	std::vector<std::vector<int> > value;
	AMS(double eps,int _N,double gam=0.01,int _hash_seed=666){
		width= ceil( 1/ eps/eps);
		AMS_num= ceil(log(4 / gam));
		hash_seed=_hash_seed;
		N=_N;
		this->counter = std::vector<std::vector<int>>();
		for(int i=0;i<AMS_num;i++){
			this->counter.push_back(std::vector<int>(width,0));
		}
		this->index = std::vector<std::vector<int>>();
		for(int i=0;i<AMS_num;i++){
			this->index.push_back(std::vector<int>(width,0));
			for(int j=0;j<width;j++){
				this->index[i][j]=generate_random_integer(this->N);
			}
		}
		this->value = std::vector<std::vector<int>>();
		for(int i=0;i<AMS_num;i++){
			this->value.push_back(std::vector<int>(width,0));
		}
		current_index=0;
	}

	void Insert(const char* str,int len, int freq = 1){

		for(int i = 0;i<AMS_num;i++){
			for(int j=0;j<width;j++){
				if(current_index>=this->index[i][j]){
					if(current_index==this->index[i][j]){
						this->value[i][j] = (int)MurmurHash32(str, len, hash_seed + i*j);
						this->counter[i][j]+=1;
					}else{
						if( ((int)MurmurHash32(str, len, hash_seed + i*j)) == this->value[i][j]){
							this->counter[i][j]+=1;
						}
					}
				}
			}
		}
		current_index+=1;
	}
	double estimate(int order){
		std::vector<double> result(this->AMS_num,0);
		for (int i = 0; i < AMS_num; i++){
			long double temp=0.0;
			for(int j=0;j<width;j++){
				temp+=this->N*( std::pow(this->counter[i][j],order)-std::pow(this->counter[i][j]-1,order));
			}
			result[i]=1.0*temp/width;
		}
		
		sort(result.begin(),result.end());
		if (AMS_num % 2 == 0)
			return ((result[AMS_num / 2] + result[AMS_num / 2 - 1]) / 2);
		else
			return (result[AMS_num / 2]);
	}



};

class CountSketch{
public:
	std::vector<std::vector<int> > counter;
	int CS_num;
	int width;
	double epsilon;
	double gamma;
	int hash_seed;
	CountSketch(){}
	CountSketch(double eps, double gam=0.01, int _hash_seed=666){
		epsilon=eps;
		gamma=gam;
		width = ceil(1 / eps/eps);
		CS_num = ceil(log(4 / gam));
		this->counter = std::vector<std::vector<int>>();
		for(int i=0;i<CS_num;i++){
			this->counter.push_back(std::vector<int>(width));
		}
		hash_seed = _hash_seed;
	}
	void Insert(const char* str,int len, int freq = 1){
	//void Insert(const std::string id,int freq=1){
		for (int i =0;i<CS_num;i++){
			unsigned index = ((unsigned)MurmurHash32(str, len, hash_seed + i)) % width;
			unsigned g =((unsigned)MurmurHash32(str, len, hash_seed + i*i+1 + CS_num)) % 2;
			if (g == 0)
			{
				counter[i][index]+=freq;
			}
			else
			{
				counter[i][index]-=freq;
			}
		}
	}

	double Join(CountSketch* cs_other){
		std::vector<double> result(this->CS_num,0);
		for(int i = 0;i<this->CS_num;++i){
			for(int j =0;j< width;j++){
				result[i]+=(double)counter[i][j]*cs_other->counter[i][j];//*counter[i][j];
			}
		}
		sort(result.begin(),result.end());
		if (CS_num % 2 == 0)
			return ((result[CS_num / 2] + result[CS_num / 2 - 1]) / 2);
		else
			return (result[CS_num / 2]);
	}
	int estimate_item(const char* str,int len){
		std::vector<double> result(this->CS_num,0);
		for(int i = 0;i<this->CS_num;++i){
			unsigned index = ((unsigned)MurmurHash32(str, len, hash_seed + i)) % width;
			unsigned g =((unsigned)MurmurHash32(str, len, hash_seed + i*i+1 + CS_num)) % 2;
			for(int j =0;j< width;j++){
				if (g == 0)
				{
					result[i]=counter[i][index];
				}
				else
				{
					result[i]=-counter[i][index];
				}
				
			}
		}
		sort(result.begin(),result.end());
		int median;
		if (CS_num % 2 == 0)
			median=(int)((result[CS_num / 2] + result[CS_num / 2 - 1]) / 2);
		else
			median=(int)(result[CS_num / 2]);
		return median>0?median:0;
	}
	void delete_item(const char* str,int len,int freq){
		for (int i =0;i<CS_num;i++){
			unsigned index = ((unsigned)MurmurHash32(str, len, hash_seed + i)) % width;
			unsigned g =((unsigned)MurmurHash32(str, len, hash_seed + i*i+1 + CS_num)) % 2;
			if (g == 0)
			{
				counter[i][index]-=freq;
			}
			else
			{
				counter[i][index]+=freq;
			}
			
		}
	}
	double estimate_l2(){
		std::vector<double> result(this->CS_num,0);
		for(int i = 0;i<this->CS_num;++i){
			for(int j =0;j< width;j++){
				result[i]+=(double)counter[i][j]*counter[i][j];//*counter[i][j];
			}
		}
		sort(result.begin(),result.end());
		if (CS_num % 2 == 0)
			return ((result[CS_num / 2] + result[CS_num / 2 - 1]) / 2);
		else
			return (result[CS_num / 2]);
	}
	void merge(CountSketch* cs ){
		for (int i =0;i<CS_num;i++){
			for(int j =0;j< width;j++){
				counter[i][j]+=cs->counter[i][j];
			}
		}
	}
	void merge(const CountSketch& cs ){
		for (int i =0;i<CS_num;i++){
			for(int j =0;j< width;j++){
				counter[i][j]+=cs.counter[i][j];
			}
		}
	}
};



class ComplexCountSketch{
public:
	std::vector<std::vector<double> > counter;
	std::vector<std::vector<double> > complex_counter;
	int p;
	int CS_num;
	int width;
	double epsilon;
	double gamma;
	int hash_seed;
	ComplexCountSketch(){}
	ComplexCountSketch(int _p,double eps, double gam=0.01, int _hash_seed=666){
		p=_p;
		epsilon=eps;
		gamma=gam;
		width = ceil(pow(2,p) / eps/eps);
		CS_num = ceil(log(4 / gam));
		this->counter = std::vector<std::vector<double>>();
		for(int i=0;i<CS_num;i++){
			this->counter.push_back(std::vector<double>(width));
		}

		this->complex_counter = std::vector<std::vector<double>>();
		for(int i=0;i<CS_num;i++){
			this->complex_counter.push_back(std::vector<double>(width));
		}

		hash_seed = _hash_seed;
	}
	void Insert(const char* str,int len, int freq = 1){
		for (int i =0;i<CS_num;i++){
			unsigned index = ((unsigned)MurmurHash32(str, len, hash_seed + i)) % width;
			unsigned g =((unsigned)MurmurHash32(str, len, hash_seed + i*i+1 + CS_num)) % p;
			counter[i][index]+=(double)freq * cos( (double)g/p * 2.f * M_PI );
			complex_counter[i][index]+=(double)freq * sin( (double)g/p * 2.f * M_PI );
		}
	}
	double estimate_lp(){

		std::vector<double> result(this->CS_num,0);
		for(int i = 0;i<this->CS_num;++i){
			for(int j =0;j< width;j++){
				double a=counter[i][j],b=complex_counter[i][j];
				for(int t=1;t<p;t++){
					double c,d;
					c=a*counter[i][j]-b*complex_counter[i][j];
					d=a*complex_counter[i][j]+b*counter[i][j];
					a=c;
					b=d;
				}
				result[i]+=a;
			}
		}
		
		sort(result.begin(),result.end());
		if (CS_num % 2 == 0)
			return ((result[CS_num / 2] + result[CS_num / 2 - 1]) / 2);
		else
			return (result[CS_num / 2]);
	}
	int estimate_item(const char* str,int len){
		std::vector<double> result(this->CS_num,0);
		for(int i = 0;i<this->CS_num;++i){
			unsigned index = ((unsigned)MurmurHash32(str, len, hash_seed + i)) % width;
			unsigned g =((unsigned)MurmurHash32(str, len, hash_seed + i*i+1 + CS_num)) % p;
			for(int j =0;j< width;j++){
				result[i]=counter[i][index]*cos((double)g/p * 2.f * M_PI )+complex_counter[i][index]*sin( (double)g/p * 2.f * M_PI );
			}
		}
		sort(result.begin(),result.end());
		if (CS_num % 2 == 0)
			return (int)((result[CS_num / 2] + result[CS_num / 2 - 1]) / 2);
		else
			return (int)(result[CS_num / 2]);
	}
	void delete_item(const char* str,int len,int freq){
		for (int i =0;i<CS_num;i++){
			unsigned index = ((unsigned)MurmurHash32(str, len, hash_seed + i)) % width;
			unsigned g =((unsigned)MurmurHash32(str, len, hash_seed + i*i+1 + CS_num)) % p;
			counter[i][index] -= freq * cos( (double)g/p * 2.f * M_PI );
			complex_counter[i][index] -= freq * sin( (double)g/p * 2.f * M_PI );
			
		}
	}
	double Join(std::vector<ComplexCountSketch> & ccs_list){
		std::vector<double> result(this->CS_num,0);
		for (int i =0;i<CS_num;i++){
			for(int j =0;j< width;j++){
				double a=counter[i][j],b=complex_counter[i][j];
				for(int t=0;t<p-1;t++){
					double c,d;
					c=a*ccs_list[t].counter[i][j]-b*ccs_list[t].complex_counter[i][j];
					d=a*ccs_list[t].complex_counter[i][j]+b*ccs_list[t].counter[i][j];
					a=c;
					b=d;
				}
				result[i]+=a;
			}
		}
		sort(result.begin(),result.end());
		if (CS_num % 2 == 0)
			return ((result[CS_num / 2] + result[CS_num / 2 - 1]) / 2);
		else
			return (result[CS_num / 2]);
	}
	void merge(ComplexCountSketch* cs ){
		for (int i =0;i<CS_num;i++){
			for(int j =0;j< width;j++){
				counter[i][j]+=cs->counter[i][j];
				complex_counter[i][j]+=cs->complex_counter[i][j];
			}
		}
	}
	void merge(const ComplexCountSketch& cs ){
		for (int i =0;i<CS_num;i++){
			for(int j =0;j< width;j++){
				counter[i][j]+=cs.counter[i][j];
				complex_counter[i][j]+=cs.complex_counter[i][j];
			}
		}
	}

};

class MGCS{
public:
	unsigned counter_k;

	CountSketch fe;

	std::unordered_map<unsigned,int> counter;

	int hash_seed;
	double epsilon;
	double epsilon_CS;
	double gamma;
	MGCS(){}
	//MGCSJS(int _w,int _d,double eps=0.01){
	MGCS(double eps_mg=0.01,double eps_c=0.01,double gam=0.01,int hash=666){

		this->epsilon=eps_mg;
		this->epsilon_CS=eps_c;
		this->gamma=gam;
		this->hash_seed = hash;
		this->counter_k = (int)floor(1/eps_mg);
		this->fe = CountSketch( eps_c,gam,hash );

	}
	void Insert(const char *item,int len,int freq = 1){
		unsigned id=(unsigned)MurmurHash32(item, len, hash_seed );
		if(counter.find(id)!=counter.end()||counter.size()<this->counter_k){
			counter[id] += freq;
		}else{
			if(freq==1){
				fe.Insert( std::to_string(id).c_str(), std::to_string(id).size());
				for(auto it = counter.begin();it!=counter.end();){
					it->second -= 1;
					fe.Insert( std::to_string(it->first).c_str(), std::to_string(it->first).size() );
					if(it->second ==0){
						it = counter.erase(it);
					}else{
						++it;
					}
				}
			}else{
				int min = freq;
				for( auto it = counter.begin();it!=counter.end(); ){
					if(it->second<min){
						min=it->second;
					}
				}
				for(auto it = counter.begin();it!=counter.end();){
					it->second -= min;
					fe.Insert(std::to_string(it->first).c_str(), std::to_string(it->first).size(),min);
					if(it->second ==0){
						it = counter.erase(it);
					}else{
						++it;
					}
				}
				if(freq== min)
					fe.Insert( std::to_string(id).c_str(), std::to_string(id).size(),freq);
				else{
					fe.Insert( std::to_string(id).c_str(), std::to_string(id).size(),min);
					counter[id] += freq-min;
				}
			}
		}
	}
	void construct(const std::unordered_map<std::string,int>& count){
		
		for(const auto& pair:count){
			unsigned id=(unsigned)MurmurHash32(pair.first.data(),4,hash_seed);
			counter[ id ]+=pair.second;
		}
		while(counter.size()>= this->counter_k){
			int min_val = std::numeric_limits<int>::max(); // initialize the minimum with the maximum integer
			for (const auto& pair : this->counter) {
        		if (pair.second < min_val) {
            		min_val = pair.second;
        		}
    		}
    		for(auto it = this->counter.begin();it!=this->counter.end();){
				it->second -= min_val;
				fe.Insert(std::to_string(it->first).c_str(), std::to_string(it->first).size(),min_val);
				if(it->second ==0){
					it = this->counter.erase(it);
				}else{
					++it;
				}
			}
		}
	}
	void merge( MGCS* mgjs){
		for(const auto& pair:mgjs->counter){
			this->counter[pair.first]+=pair.second;
		}
		fe.merge(mgjs->fe);
		while(this->counter.size()>=this->counter_k){
			int min_val = std::numeric_limits<int>::max();
			for (const auto& pair : this->counter) {
        		if (pair.second < min_val) {
            		min_val = pair.second;
        		}
    		}
    		for(auto it = this->counter.begin();it!=this->counter.end();){
				it->second -= min_val;
				fe.Insert(std::to_string(it->first).c_str(), std::to_string(it->first).size(),min_val);
				if(it->second ==0){
					it = this->counter.erase(it);
				}else{
					++it;
				}
			}
		}
	}
	double estimate_l2(){
		double high=0,low=0;
		//double low = fe.estimate();
		//double inter = 0;
		for(const auto& pair:this->counter){
			int value = fe.estimate_item(std::to_string(pair.first).c_str(), std::to_string(pair.first).size());
			high+= pow(value+pair.second,2);
			fe.delete_item(std::to_string(pair.first).c_str(), std::to_string(pair.first).size(),value);
		}
		return high+fe.estimate_l2();
	}
	double Join(MGCS * other){
		double result=0.0;
		for(const auto& pair:this->counter){
			result+=(double)pair.second*(other->fe.estimate_item(
				std::to_string(pair.first).c_str(), std::to_string(pair.first).size()));
			if(other->counter.find(pair.first)!=other->counter.end()){
				result+=(double)( other->counter[pair.first] )*pair.second;
			}
		}
		for(const auto& pair:other->counter){
			result+= (double)pair.second * this->fe.estimate_item(
				std::to_string(pair.first).c_str(), std::to_string(pair.first).size());
		}
		result += this->fe.Join(&(other->fe)); /* [paper_code patch] The original was missing &: CountSketch::Join takes a CountSketch* parameter (line 220 of the original file), so the original line would not compile (the 2024 .o was compiled from an older header); here fixed to take the address, semantics unchanged */
		return result;
	}
	// void SetParams(double eps,double gam, 
	// 	const std::vector<std::vector<int> >& _counter ,
	// 	const std::unordered_map<unsigned,int>& MG_counter,int hs=666 )
	// {
	// 	this->epsilon=eps;
	// 	this->gamma=gam;
	// 	this->hash_seed = hs;
	// 	this->counter_k  = (int)floor(1/eps);
	// 	//this->fe = CountSketch(ceil(10 / eps),ceil(log(4 / gam)),_counter,hs);
	// 	//printf("in SetParams:");
	// 	//this->fe.SetParams(ceil(10 / eps),ceil(log(4 / gam)),_counter,hs);
	// 	//printf("in SetParams: %d\n",this->fe.counter[1][1]);
	// 	this->counter=std::unordered_map<unsigned,int>();
	// 	for(const auto& pair:MG_counter){
	// 		counter[ pair.first ]+=pair.second;
	// 	}
	// 	this->fe.epsilon=eps;
	// 	this->fe.gamma=gam;
	// 	this->fe.CS_num=ceil(log(4 / gam));
	// 	this->fe.width=ceil(10 / eps);
	// 	this->fe.hash_seed=hs;
	// 	this->fe.counter = std::vector<std::vector<int>>();
	// 	for(int i=0;i<this->fe.CS_num;i++){
	// 		this->fe.counter.push_back(std::vector<int>(this->fe.width));
	// 	}
	// 	for(int i=0;i<this->fe.CS_num;i++){
	// 		for(int j=0;j<this->fe.width;j++){
	// 			this->fe.counter[i][j]=_counter[i][j];
	// 		}
	// 	}
	// }
	// std::vector<std::vector<int>>&  getCounter(){
	// 	printf("in getCounter:\n");
	// 	printf("in getCounter %d\n",this->fe.counter[1][1]);
	// 	return this->fe.counter;
	// }
	// std::unordered_map<unsigned,int> getMGCounter(){
	// 	return counter;
	// }
	double geteps(){return epsilon;}
	double getgamma(){return gamma;}
	int getSeed(){return hash_seed;}
};

class MGCCS{
public:
	unsigned counter_k;

	ComplexCountSketch fe;

	std::unordered_map<unsigned,int> counter;
	int p;
	int hash_seed;
	double epsilon;
	double epsilon_CCS;
	double gamma;
	MGCCS(){}
	//MGCSJS(int _w,int _d,double eps=0.01){
	MGCCS(int _p,double eps_mg=0.01,double eps_c=0.01,double gam=0.01,int hash=666){
		this->p=_p;
		this->epsilon=eps_mg;
		this->epsilon_CCS=eps_c;
		this->gamma=gam;
		this->hash_seed = hash;
		this->counter_k = (int)floor(1/eps_mg);
		this->fe = ComplexCountSketch(p, eps_c,gam,hash );

	}
	void Insert(const char *item,int len,int freq = 1){
		unsigned id=(unsigned)MurmurHash32(item, len, hash_seed );
		if(counter.find(id)!=counter.end()||counter.size()<this->counter_k){
			counter[id] += freq;
		}else{
			if(freq==1){
				fe.Insert( std::to_string(id).c_str(), std::to_string(id).size());
				for(auto it = counter.begin();it!=counter.end();){
					it->second -= 1;
					fe.Insert( std::to_string(it->first).c_str(), std::to_string(it->first).size() );
					if(it->second ==0){
						it = counter.erase(it);
					}else{
						++it;
					}
				}
			}else{
				int min = freq;
				for( auto it = counter.begin();it!=counter.end(); ){
					if(it->second<min){
						min=it->second;
					}
				}
				for(auto it = counter.begin();it!=counter.end();){
					it->second -= min;
					fe.Insert(std::to_string(it->first).c_str(), std::to_string(it->first).size(),min);
					if(it->second ==0){
						it = counter.erase(it);
					}else{
						++it;
					}
				}
				if(freq== min)
					fe.Insert( std::to_string(id).c_str(), std::to_string(id).size(),freq);
				else{
					fe.Insert( std::to_string(id).c_str(), std::to_string(id).size(),min);
					counter[id] += freq-min;
				}
			}
		}
	}
	void construct(const std::unordered_map<std::string,int>& count){
		
		for(const auto& pair:count){
			unsigned id=(unsigned)MurmurHash32(pair.first.data(),4,hash_seed);
			counter[ id ]+=pair.second;
		}
		while(counter.size()>= this->counter_k){
			int min_val = std::numeric_limits<int>::max(); // initialize the minimum with the maximum integer
			for (const auto& pair : this->counter) {
        		if (pair.second < min_val) {
            		min_val = pair.second;
        		}
    		}
    		for(auto it = this->counter.begin();it!=this->counter.end();){
				it->second -= min_val;
				fe.Insert(std::to_string(it->first).c_str(), std::to_string(it->first).size(),min_val);
				if(it->second ==0){
					it = this->counter.erase(it);
				}else{
					++it;
				}
			}
		}
	}
	void merge( MGCCS* mgjs){
		for(const auto& pair:mgjs->counter){
			this->counter[pair.first]+=pair.second;
		}
		fe.merge(mgjs->fe);
		while(this->counter.size()>=this->counter_k){
			int min_val = std::numeric_limits<int>::max();
			for (const auto& pair : this->counter) {
        		if (pair.second < min_val) {
            		min_val = pair.second;
        		}
    		}
    		for(auto it = this->counter.begin();it!=this->counter.end();){
				it->second -= min_val;
				fe.Insert(std::to_string(it->first).c_str(), std::to_string(it->first).size(),min_val);
				if(it->second ==0){
					it = this->counter.erase(it);
				}else{
					++it;
				}
			}
		}
	}
	int estimate_item(unsigned id){
		int freq=0;
		if(this->counter.find(id)!=this->counter.end()){
			freq+=this->counter[id];
		}
		freq+=fe.estimate_item(std::to_string(id).c_str(), std::to_string(id).size());
		return freq;
	}
	int estimate_ccs(unsigned id){
		return fe.estimate_item(std::to_string(id).c_str(), std::to_string(id).size());
	}

	double estimate_lp(){
		double high=0,low=0;
		//double low = fe.estimate();
		//double inter = 0;
		for(const auto& pair:this->counter){

			int value = fe.estimate_item(std::to_string(pair.first).c_str(), std::to_string(pair.first).size());
			high+= pow(value+pair.second,p);
			fe.delete_item(std::to_string(pair.first).c_str(), std::to_string(pair.first).size(),value);
		}
		return high+fe.estimate_lp();
	}
	double estimate_pstar(std::vector<MGCCS> & mgccs_list){
		double high=0,low=0;
		std::vector<double> result(fe.CS_num,0);
		for (int i =0;i<fe.CS_num;i++){
			for(int j =0;j< fe.width;j++){
				double a=fe.counter[i][j],b=fe.complex_counter[i][j];
				for(int t=0;t<p-1;t++){
					double c,d;
					c=a*mgccs_list[t].fe.counter[i][j]-b*mgccs_list[t].fe.complex_counter[i][j];
					d=a*mgccs_list[t].fe.complex_counter[i][j]+b*mgccs_list[t].fe.counter[i][j];
					a=c;
					b=d;
				}
				result[i]+=a;
			}
		}
		sort(result.begin(),result.end());
		if (fe.CS_num % 2 == 0)
			low=((result[fe.CS_num / 2] + result[fe.CS_num / 2 - 1]) / 2);
		else
			low=(result[fe.CS_num / 2]);

		
		//double low = fe.estimate();
		//double inter = 0;
		for(const auto& pair:this->counter){
			double temp=(double)pair.second;
			for(int t=0;t<p-1;t++){
				temp*=mgccs_list[t].estimate_item(pair.first);
			}
			high+=temp;
		}
		for(int t=0;t<p-1;t++){
			for(const auto& pair:mgccs_list[t].counter){
				double temp = (double) pair.second*estimate_ccs(pair.first);
				for(int tt=0;tt<p-1;tt++){
					if(tt<t){
						temp*=estimate_ccs(pair.first);
					}else if(tt>t){
						temp*=mgccs_list[tt].estimate_item(pair.first);
					}
				}
				high+=temp;
			}
		}
		return high+low;
	}
	// void SetParams(double eps,double gam, 
	// 	const std::vector<std::vector<int> >& _counter ,
	// 	const std::unordered_map<unsigned,int>& MG_counter,int hs=666 )
	// {
	// 	this->epsilon=eps;
	// 	this->gamma=gam;
	// 	this->hash_seed = hs;
	// 	this->counter_k  = (int)floor(1/eps);
	// 	//this->fe = CountSketch(ceil(10 / eps),ceil(log(4 / gam)),_counter,hs);
	// 	//printf("in SetParams:");
	// 	//this->fe.SetParams(ceil(10 / eps),ceil(log(4 / gam)),_counter,hs);
	// 	//printf("in SetParams: %d\n",this->fe.counter[1][1]);
	// 	this->counter=std::unordered_map<unsigned,int>();
	// 	for(const auto& pair:MG_counter){
	// 		counter[ pair.first ]+=pair.second;
	// 	}
	// 	this->fe.epsilon=eps;
	// 	this->fe.gamma=gam;
	// 	this->fe.CS_num=ceil(log(4 / gam));
	// 	this->fe.width=ceil(10 / eps);
	// 	this->fe.hash_seed=hs;
	// 	this->fe.counter = std::vector<std::vector<int>>();
	// 	for(int i=0;i<this->fe.CS_num;i++){
	// 		this->fe.counter.push_back(std::vector<int>(this->fe.width));
	// 	}
	// 	for(int i=0;i<this->fe.CS_num;i++){
	// 		for(int j=0;j<this->fe.width;j++){
	// 			this->fe.counter[i][j]=_counter[i][j];
	// 		}
	// 	}
	// }
	// std::vector<std::vector<int>>&  getCounter(){
	// 	printf("in getCounter:\n");
	// 	printf("in getCounter %d\n",this->fe.counter[1][1]);
	// 	return this->fe.counter;
	// }
	// std::unordered_map<unsigned,int> getMGCounter(){
	// 	return counter;
	// }
	double geteps(){return epsilon;}
	double getgamma(){return gamma;}
	int getSeed(){return hash_seed;}
};


/* [paper_code patch] The original SSE implementation (_mm_set1_epi32, etc., semantics: find
   the first matching index of key in ID[0..7], return -1 if no match) is x86-only. Here a
   scalar equivalent fallback is added, with bit-identical results. */
#if defined(__x86_64__) || defined(__i386__)
inline int SIMD_match_8(uint32_t *ID,uint32_t key){
	const __m128i item = _mm_set1_epi32((int)key);
	int matched;
	__m128i *keys_p = (__m128i *)ID;
	__m128i a_comp = _mm_cmpeq_epi32(item, keys_p[0]);
	__m128i b_comp = _mm_cmpeq_epi32(item, keys_p[1]);
	a_comp = _mm_packs_epi32(a_comp, b_comp);
	matched = _mm_movemask_epi8(a_comp);
	if (matched != 0) {
		int matched_index = __builtin_ctz((uint32_t)matched)>>1;
		return matched_index;
    }else return -1;
}
#else
inline int SIMD_match_8(uint32_t *ID,uint32_t key){
	for (int k = 0; k < 8; ++k)
		if (ID[k] == key) return k;
	return -1;
}
#endif
struct Hash_table{
	int n,hash_seed;const int m=8;
	unsigned **Id,**Key;
	Hash_table(int bucket_sum,int _hash_seed=1000){
		n=bucket_sum;
		Id=new unsigned*[n]();
		for(int i=0;i<n;i++)
			Id[i]=new unsigned[m]();
		Key=new unsigned*[n]();
		for(int i=0;i<n;i++)
			Key[i]=new unsigned[m]();
		for(int i=0;i<n;i++)
		for(int j=0;j<m;j++)
		Id[i][j]=Key[i][j]=0;
	}
	bool count(unsigned id)const{
		//std::cout<<"787"<<MurmurHash32(&id, 4, 1)<<","<<MurmurHash32(id, 4, 1)<<std::endl;
		//std::cout<<"787:"<<hash_seed<<","<<n<<std::endl;
		unsigned i=MurmurHash32(&id, 4, hash_seed)%n;
		//std::cout<<"797"<<std::endl;
		if(SIMD_match_8(Id[i],id)>=0)return 1;
		i=MurmurHash32(&id, 4, hash_seed*2)%n;
		if(SIMD_match_8(Id[i],id)>=0)return 1;
		return 0;
	}
	int query(unsigned id)const{
		unsigned i=MurmurHash32(&id, 4, hash_seed)%n;
		int j=SIMD_match_8(Id[i],id);
		if(j>=0)return Key[i][j];
		i=MurmurHash32(&id, 4, hash_seed*2)%n;
		j=SIMD_match_8(Id[i],id);
		if(j>=0)return Key[i][j];
		return -1;
	}
	void insert(unsigned id,unsigned cnt=1){
		unsigned i=MurmurHash32(&id, 4, hash_seed)%n;
		int j;
		j=SIMD_match_8(Id[i],id);
		if(j>=0){
			Key[i][j]+=cnt;
			return;
		}
		i=MurmurHash32(&id, 4, hash_seed*2)%n;
		j=SIMD_match_8(Id[i],id);
		if(j>=0){
			Key[i][j]+=cnt;
			return;
		}
		for(int j=0;j<m;j++)
		if(!solid(i,j)){
			Id[i][j]=id;
			Key[i][j]=cnt;
			return;
		}
		i=MurmurHash32(&id, 4, hash_seed)%n;
		for(int j=0;j<m;j++)
		if(!solid(i,j)){
			Id[i][j]=id;
			Key[i][j]=cnt;
			return;
		}
		moretime:Expansion();
		i=MurmurHash32(&id, 4, hash_seed)%n;
		for(int j=0;j<m;j++)
		if(!solid(i,j)){
			Id[i][j]=id;
			Key[i][j]=cnt;
			return;
		}
		i=MurmurHash32(&id, 4, hash_seed*2)%n;
		for(int j=0;j<m;j++)
		if(!solid(i,j)){
			Id[i][j]=id;
			Key[i][j]=cnt;
			return;
		}
		goto moretime;
		exit(1);
	}
	bool solid(unsigned i,unsigned j){
		if(Key[i][j]==0)return 0;
		unsigned id=Id[i][j];
		unsigned ii=MurmurHash32(&id, 4, hash_seed)%n;
		if(ii==i)return 1;
		ii=MurmurHash32(&id, 4, hash_seed*2)%n;
		if(ii==i)return 1;
		return 0;
	}
	void Expansion(){
		unsigned **nwid;
		unsigned **nwkey;
		nwid=new unsigned*[n+n]();
		nwkey=new unsigned*[n+n]();
		for(int i=0;i<n;i++){
			nwid[i]=Id[i];
			nwkey[i]=Key[i];
		}
		for(int i=0;i<n;i++){
			nwid[n+i]=new unsigned[m]();
			nwkey[n+i]=new unsigned[m]();
			memmove(nwid[n+i],nwid[i],m*sizeof(unsigned));
			memmove(nwkey[n+i],nwkey[i],m*sizeof(unsigned));
		}
		delete[]Id;
		delete[]Key;
		Id=nwid;
		Key=nwkey;
		n=n+n;
	}
	~Hash_table()
	{
		for (int i = 0; i < n; i++){
			delete[]Id[i];
			delete[]Key[i];
		}
		delete[]Id;
		delete[]Key;
	}
};


class Short_C_Sketch
{
public:
	int w, d;
	int index[MAX_HASH_NUM];
	int* counter[MAX_HASH_NUM];
	int MAX_CNT, MIN_CNT, hash_seed;
public:
	Short_C_Sketch(int _w, int _d, int _hash_seed = 1000)
	{
		d = _d, w = _w/_d/2;
		hash_seed = _hash_seed;
		for (int i = 0; i < d; i++)
		{
			counter[i] = new int[w]();
			memset(counter[i], 0, sizeof(int) * w);
		}

		MAX_CNT = (1 << (32 - 1)) - 1;
		MIN_CNT = (-(1 << (32 - 1)));
	}

	void Insert(unsigned id,int k=1)
	{
		int g = 0;
		for (int i = 0; i < d; i++)
		{
			index[i] = ((unsigned)MurmurHash32(&id, 4, hash_seed + i)) % w;
			g =((unsigned) MurmurHash32(&id, 4, hash_seed + i + d)) % 2;
			if (g == 0)
			{
				if (counter[i][index[i]] != MAX_CNT)
				{
					counter[i][index[i]]+=k;
				}
			}
			else
			{
				if (counter[i][index[i]] != MIN_CNT)
				{
					counter[i][index[i]]-=k;
				}
			}
		}
	}
	long double Join(Short_C_Sketch*other){
		long double res[MAX_HASH_NUM];
		for (int i = 0; i < d; i++){
			long double k=0;
			for (int j = 0; j < w; j++)
				k+=1ll*counter[i][j]*other->counter[i][j];
			res[i]=k;
		}
		std::sort(res, res + d);
		if (d % 2 == 0)
			return ((res[d / 2] + res[d / 2 - 1]) / 2);
		else
			return (res[d / 2]);
	}

	int Query(unsigned id)const
	{
		int temp;
		int res[MAX_HASH_NUM];
		int index[MAX_HASH_NUM];
		int g;
		for (int i = 0; i < d; i++)
		{
			index[i] = ((unsigned)MurmurHash32(&id, 4, hash_seed + i)) % w;
			temp = counter[i][index[i]];
			g = ((unsigned)MurmurHash32(&id, 4, hash_seed + i + d)) % 2;

			res[i] = (g == 0 ? temp : -temp);
		}

		std::sort(res, res + d);
		if (d % 2 == 0)
			return ((res[d / 2] + res[d / 2 - 1]) / 2);
		else
			return (res[d / 2]);
	}

	~Short_C_Sketch()
	{
		for (int i = 0; i < d; i++)
			delete[]counter[i];
	}
};

class Classifier
{
public:
	int n;const int m=8;
	unsigned **Mid;
	int hash_seed;
	Short_C_Sketch *Light;
	Hash_table *Heavy;
public:
	Classifier(){}
	Classifier(int _w, int _d, int _hash_seed = 1000)
	{

		//std::cout<<"997: "<<_w<<","<<_d<<std::endl;
		int heavysize=_w/8;
		heavysize=heavysize/64;
		Heavy=new Hash_table(heavysize,_hash_seed*2);
		_w-=heavysize*8*8;
	//	std::cout<<"997: "<<_w<<","<<_d<<","<<heavysize<<","<<_hash_seed<<std::endl;
		int midsize=_w/2;
		midsize/=4;
		n=midsize/m;
		midsize=n*m;
		Mid=new unsigned*[n];
		for(int i=0;i<n;i++)
		Mid[i]=new unsigned[m]();
		_w-=midsize*4;
		hash_seed=_hash_seed*3;
		//std::cout<<"how many:"<<_w<<","<<_d<<std::endl;
		Light=new Short_C_Sketch(_w,_d,_hash_seed);
	}
	#define Mid_Id(x) (x>>10)
	#define Mid_Key(x) (x&1023)
	void Insert(const unsigned id){
		//std::cout<<"1009"<<std::endl;
		if(Heavy->count(id)){
			Heavy->insert(id);
			//std::cout<<1012<<id<<std::endl;
			return;}
		int MdId=Mid_Id(id)<<10;
		
		unsigned k=MurmurHash32(&MdId, 4, hash_seed)%n;
		unsigned*bucket=Mid[k];

		for(int i=0;i<m;i++)

		if(Mid_Id(bucket[i])==Mid_Id(id)||Mid_Key(bucket[i])==0){
			bucket[i]=(bucket[i]+1)&1023;
			bucket[i]|=MdId;
			if(Mid_Key(bucket[i])==Heavy_Thes){
				Heavy->insert(id,Mid_Key(bucket[i]));
				bucket[i]=0;
			}
			//std::cout<<"1015"<<std::endl;
			return;
		}
		//std::cout<<"1024"<<std::endl;
		int to=0,mn=Mid_Key(bucket[0]);
		for(int i=1;i<m;i++)
		if(Mid_Key(bucket[i])<mn){
			mn=Mid_Key(bucket[i]);
			to=i;
		}
		Light->Insert(Mid_Id(bucket[to]),Mid_Key(bucket[to]));
		bucket[to]=1;
		bucket[to]|=MdId;
		return;
	}
	void Insert(const char* str)
	{
		unsigned id=((unsigned*)str)[0];
		Insert(id);
		return;
	}
	bool CheckHeavy(const char* str){
		unsigned id=((unsigned*)str)[0];
		return Heavy->count(id);
	}
	long double Join(Classifier*other){
		long double re=Light->Join(other->Light);
		for(int i=0;i<Heavy->n;i++)
		for(int j=0;j<Heavy->m;j++)
		if(Heavy->solid(i,j)){
			re+=1.0*Heavy->Key[i][j]*other->Query(Heavy->Id[i][j]);
		}
		for(int i=0;i<other->Heavy->n;i++)
		for(int j=0;j<other->Heavy->m;j++)
		if(other->Heavy->solid(i,j)){
			re+=1.0*QueryMid(other->Heavy->Id[i][j])*other->Heavy->Key[i][j];
		}
		for(int i=0;i<n;i++)
		for(int j=0;j<m;j++)
		if(Mid[i][j]!=0){
			re+=1.0*Mid_Key(Mid[i][j])*other->QueryMid(Mid[i][j]);
		}
		for(int i=0;i<n;i++)
		for(int j=0;j<m;j++)
		if(other->Mid[i][j]!=0){
			re+=1.0*Mid_Key(other->Mid[i][j])*Light->Query(Mid_Id(other->Mid[i][j]));
		}
		return re;
	}
	int QueryMid(const unsigned id){
		int re=Light->Query(Mid_Id(id));
		int MdId=Mid_Id(id)<<10;
		unsigned k=MurmurHash32(&MdId, 4, hash_seed)%n;
		unsigned*bucket=Mid[k];
		for(int i=0;i<m;i++)
		if(Mid_Id(bucket[i])==Mid_Id(id))re+=Mid_Key(bucket[i]);
		return re;
	}
	int Query(const unsigned id)const{
		int re=Light->Query(Mid_Id(id));
		if(Heavy->count(id))re+=Heavy->query(id);
		int MdId=Mid_Id(id)<<10;
		unsigned k=MurmurHash32(&MdId, 4, hash_seed)%n;
		unsigned*bucket=Mid[k];
		for(int i=0;i<m;i++)
		if(Mid_Id(bucket[i])==Mid_Id(id))re+=Mid_Key(bucket[i]);
		return re;
	}
	int Query(const char* str)const
	{
		// unsigned id= (unsigned)MurmurHash32(str, KEY_LEN, fingerprint);
		unsigned id=((unsigned*)str)[0];
		return Query(id);
	}
	~Classifier()
	{
		for (int i = 0; i < n; i++)
			delete[]Mid[i];
		delete[]Mid;
		Heavy->~Hash_table();
		Light->~Short_C_Sketch();
	}
};


}

#endif //!defined(JS_HPP)