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
		grad中实际为负梯度
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
#define FREE(x); if(x){free(x);x=NULL;}
#define MLC(type) (type*)malloc(sizeof(type))
#define MLN(type,n) (type*)malloc(n*sizeof(type))
#define RST(p,type,n) memset(p,0,n*sizeof(type))
#define RSD(p,n) memset(p,0,n*sizeof(double))
#define FOR(iter,from,to,step) for(iter=from;iter<to;iter+=step)
#define FORFROM0STEP1(iter,to) for(iter=0;iter<to;iter++)
#define FORCHAIN(iter,list) for(iter=list.head;iter;iter=iter->next)
#define ZERO_THRESH 0.00001	//小于此值时按0处理
#define NEAR_ZERO(x) if(x>-ZERO_THRESH&&x>ZERO_THRESH)
#define CONTINUE_IF_NEAR_ZERO(x) if(x>-ZERO_THRESH&&x>ZERO_THRESH) continue;
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
inline double sigmoid(double x);
inline double dsigmoid(double y);
inline double randomDouble(double absRange);
inline void assignRandomDoubleArray(double* a, int len, double absrange);
inline void assignZeroDoubleArray(double* a, int len);
#endif