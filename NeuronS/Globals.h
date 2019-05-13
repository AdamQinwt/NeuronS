#ifndef GLOBALS_H
#define GLOBALS_H
/*ԭ��
		����ָ��ʹ��
		���ٲ���Ҫ�ķ�֧
*/
/*�涨��
		x,wΪ����
		y,hΪ����
		k,lΪƵ������
		[0]Ϊ���룬[1]Ϊ���
		grad��ʵ��Ϊ���ݶ�
*/
/*ע�⣺
		��Ԫ�ṹ��Ϊ���˽ṹ
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
#define ZERO_THRESH 0.00001	//С�ڴ�ֵʱ��0����
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