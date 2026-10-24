/*
multijoin3/countsketch.hpp — trimmed complex-sketch header file
================================================================================
Paper correspondence: the C++ reference implementation of the CCS structure in §2–§3
(AMS/CountSketch/ComplexCountSketch/CountSketch2). Unlike common/countsketch.hpp, it does
not include the MG/JoinSketch classes.
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

    // generate random number
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
	std::vector<double> unit_roots_real;
	std::vector<double> unit_roots_imag;
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
		for(int i =0;i<p;i++){
			unit_roots_real.push_back( cos( (double)i/p * 2.f * M_PI ) );
			unit_roots_imag.push_back( sin( (double)i/p * 2.f * M_PI ) );
		}

		hash_seed = _hash_seed;
	}
	void Insert(const char* str,int len, int freq = 1){
		for (int i =0;i<CS_num;i++){
			unsigned index = ((unsigned)MurmurHash32(str, len, hash_seed + i)) % width;
			unsigned g =((unsigned)MurmurHash32(str, len, hash_seed + i*i+1 + CS_num)) % p;
			counter[i][index]+=(double)freq * unit_roots_real[g];//cos( (double)g/p * 2.f * M_PI );
			complex_counter[i][index]+=(double)freq * unit_roots_imag[g]; //sin( (double)g/p * 2.f * M_PI );
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

class SP{
public:
	std::vector<std::vector< std::vector<int> > > counter;
	int CS_num;
	int width;
	double epsilon;
	double gamma;
	int hash_seed1;
	int hash_seed2;
	SP(){}
	SP(double eps, double gam=0.01, int _hash_seed1=666, int _hash_seed2 = 888){
		epsilon=eps;
		gamma=gam;
		width = ceil(1 / eps/eps);
		CS_num = ceil(log(4 / gam));

		this->counter = std::vector< std::vector< std::vector<int> > >(CS_num,
			std::vector<std::vector<int>>(width,std::vector<int>(width,0)) );
		//for(int i=0;i<CS_num;i++){
		//	this->counter.push_back(std::vector<int>(width));
		//}
		hash_seed1 = _hash_seed1;
		hash_seed2 = _hash_seed2;
	}
	void Insert(const char* str,int len, int freq = 1){
	//void Insert(const std::string id,int freq=1){
		for (int i =0;i<CS_num;i++){
			unsigned index1 = ((unsigned)MurmurHash32(str, len, hash_seed1 +  i)) % width;
			unsigned g1 =((unsigned)MurmurHash32(str, len, hash_seed1 + i*i+1 + CS_num)) % 2;
			unsigned index2 = ((unsigned)MurmurHash32(str, len, hash_seed2 + i )) % width;
			unsigned g2 =((unsigned)MurmurHash32(str, len, hash_seed2 + i*i+1 + CS_num)) % 2;
			counter[i][index1][index2]+=(g1*2-1)*(g2*2-1)*freq;
		}
	}

	double Join(CountSketch* cs_left, CountSketch* cs_right){
		std::vector<double> result(this->CS_num,0);
		for(int i = 0;i<this->CS_num;++i){
			for(int l = 0; l<width;++l){
				for(int r =0;r< width;r++){
					result[i]+=(double)cs_left->counter[i][l]*counter[i][l][r]*cs_right->counter[i][r];//*counter[i][j];
				}
			}
		}
		sort(result.begin(),result.end());
		if (CS_num % 2 == 0)
			return ((result[CS_num / 2] + result[CS_num / 2 - 1]) / 2);
		else
			return (result[CS_num / 2]);
	}
	
};

class SM{
public:
	CountSketch CS1;
	CountSketch CS2;
	int CS_num;
	int width;
	SM(){}
	SM(double eps, double gam=0.01, int _hash_seed1=666, int _hash_seed2 = 888){
		width = ceil(1 / eps/eps);
		CS_num = ceil(log(4 / gam));
		CS1=CountSketch(eps,gam,_hash_seed1);
		CS2=CountSketch(eps,gam,_hash_seed2);
	}
	void Insert(const char* str,int len, int freq = 1){
	//void Insert(const std::string id,int freq=1){
		CS1.Insert(str,len,freq);
		CS2.Insert(str,len,freq);
	}

	double Join(CountSketch* cs_left, CountSketch* cs_right){
		std::vector<double> result(this->CS_num,0);
		for(int i = 0;i<this->CS_num;++i){
			for(int l = 0; l<width;++l){
				for(int r =0;r< width;r++){
					if(abs(CS1.counter[i][l])<abs(CS2.counter[i][r])){
						result[i]+=(double)cs_left->counter[i][l]*CS1.counter[i][l]*cs_right->counter[i][r];//*counter[i][j];
					}else{
						result[i]+=(double)cs_left->counter[i][l]*CS1.counter[i][r]*cs_right->counter[i][r];
					}
					
				}
			}
		}
		sort(result.begin(),result.end());
		if (CS_num % 2 == 0)
			return ((result[CS_num / 2] + result[CS_num / 2 - 1]) / 2);
		else
			return (result[CS_num / 2]);
	}
	
};

class CountSketch2{
public:
	std::vector<std::vector<int> > counter;
	int CS_num;
	int width;
	double epsilon;
	double gamma;
	int hash_seed;
	std::vector<int> sign_seed;
	CountSketch2(){}
	CountSketch2(double eps, std::vector<int> &sign, double gam=0.01, int _hash_seed=666 ){
		epsilon=eps;
		gamma=gam;
		width = ceil(1 / eps/eps);
		CS_num = ceil(log(4 / gam));
		this->counter = std::vector<std::vector<int>>();
		for(int i=0;i<CS_num;i++){
			this->counter.push_back(std::vector<int>(width));
		}
		hash_seed = _hash_seed;
		for(int i=0;i<sign.size();i++){
			this->sign_seed.push_back( sign[i]);
			//std::cout<<sign[i]<<std::endl;
		}
		//std::cout<<hash_seed;

	}
	void Insert(const char* str,int len, int freq = 1){
	//void Insert(const std::string id,int freq=1){
		for (int i =0;i<CS_num;i++){
			int sign = 1;
			unsigned index = ((unsigned)MurmurHash32(str, len, hash_seed + i)) % width;
			for (int j = 0; j< sign_seed.size();j++){
				unsigned g =((unsigned)MurmurHash32(str, len, sign_seed[j] + i*i+1 + CS_num)) % 2;
				if (g == 0)
				{
					sign*=1;
				}
				else
				{
					sign*=-1;
				}
			}
			counter[i][index]+=sign;
			
		}
	}


	double Join(std::vector<CountSketch2> & cs_list){
		std::vector<double> result(this->CS_num,0);
		for (int i =0;i<CS_num;i++){
			for(int j =0;j< width;j++){
				double temp = counter[i][j];
				for(int t=0;t<cs_list.size();t++){
					temp = (double)temp*cs_list[t].counter[i][j];
				}
				result[i]+= temp;
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




}

#endif //!defined(JS_HPP)