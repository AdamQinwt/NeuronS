#ifndef ARG_H
#define ARG_H
#define OPTIONAL_COUNT 4
#define ORIGINAL 0
#define GRAD 1
#define DELTA 2
#define SHADOW 3
typedef struct _FC_Arg
{
	double* bias;
	double** weight;
}FC_Arg;
typedef struct _Conv_Arg
{
	double* bias;
	double**** weight;
}Conv_Arg;
typedef union _Arg
{
	struct {
		FC_Arg original;	//原参数
		FC_Arg grad;	//梯度
		FC_Arg delta;	//变化量
		FC_Arg shadow;	//影子，用于一些训练算法
	}fc;
	struct {
		Conv_Arg original;	//原参数
		Conv_Arg delta;	//变化量
		Conv_Arg grad;	//梯度
		Conv_Arg shadow;	//影子，用于一些训练算法
	}conv;
}Arg;
#endif