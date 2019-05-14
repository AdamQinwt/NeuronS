#ifndef GLOBALS_H
#define GLOBALS_H
/*原则：
		减少指针使用
		减少不必要的分支
*/
/*规定：
		x,w为横向
		y,h为纵向
		k,l为频道方向
		[0]为输入，[1]为输出
		数据文件以1个整数开头，标识其数据集大小，名称在训练及测试时指定
*/
/*注意：
		神经元结构仍为拓扑结构
*/
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<string.h>
#define PS system("pause");
#define FREE(x) if(x){free(x);x=NULL;}
#define MLC(type) (type*)malloc(sizeof(type))
#define MLN(type,n) (type*)malloc(n*sizeof(type))
#define MLD(n) (double*)malloc(n*sizeof(double))
#define RST(p,type,n) memset(p,0,n*sizeof(type))
#define RSD(p,n) if(p) memset(p,0,n*sizeof(double))
#define FOR(iter,from,to,step) for(iter=from;iter<to;iter+=step)
#define FORFROM0STEP1(iter,to) for(iter=0;iter<to;iter++)
#define FORCHAIN(iter,list) for(iter=list.head;iter;iter=iter->next)
#define ZERO_THRESH 0.00001	//小于此值时按0处理
#define NEAR_ZERO(x) if(x>-ZERO_THRESH&&x>ZERO_THRESH)
#define CONTINUE_IF_NEAR_ZERO(x) if(x>-ZERO_THRESH&&x<ZERO_THRESH) continue;
#define COPY_ARRAY(from,to,type,n) memcpy(to,from,n*sizeof(type))
#define COPY_DOUBLE_ARRAY(from,to,n) memcpy(to,from,n*sizeof(double))
double**** new4dDoubleArray(int d,int l, int h, int w);
double*** new3dDoubleArray(int l, int h, int w);
double** new2dDoubleArray(int h, int w);
void destroy2dDoubleArray(double** p, int h);
void destroy3dDoubleArray(double*** p, int l, int h);
void destroy4dDoubleArray(double**** p, int k, int l, int h);
double*** new3dDoubleArrayFrom1d(int l, int h, int w, int offset, double* p);
int argmax(double* a, int l);
double sigmoid(double x);
double dsigmoid(double y);
double relu(double x);
double drelu(double y);
double randomDouble(double absRange);
void assignRandomDoubleArray(double* a, int len, double absrange);
void assignZeroDoubleArray(double* a, int len);
void print1dArray(double* a, int w);
void print2dArray(double** a, int h,int w);
void print3dArray(double*** a, int l,int h,int w);
void print4dArray(double**** a, int k,int l,int h,int w);
void write1dArray(FILE* fp, double* a, int w);
void write2dArray(FILE* fp, double** a, int h, int w);
void write3dArray(FILE* fp, double*** a, int l, int h, int w);
void write4dArray(FILE* fp, double**** a, int k, int l, int h, int w);
void read1dArray(FILE* fp, double* a, int w);
void read2dArray(FILE* fp, double** a, int h, int w);
void read3dArray(FILE* fp, double*** a, int l, int h, int w);
void read4dArray(FILE* fp, double**** a, int k, int l, int h, int w);
double clipByValue(double x, double a, double b);
void square1dArray(double* to,double* from,int w);
void square2dArray(double** to,double** from,int h,int w);
void square3dArray(double*** to,double*** from,int l,int h,int w);
void square4dArray(double**** to,double**** from,int k,int l,int h,int w);
#endif