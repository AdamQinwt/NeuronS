#ifndef NETWORK_H
#define NETWORK_H
#include"Neuron.h"
typedef struct _Network
{
	char name[50];	//名称
	int il, ih, iw, ol, oh, ow;
	double** x, **dx;
	double** y, **dy;
	int number;	//神经元数量
	Neuron* neurons;	//神经元数组
	int batch;	//batch size
	int trainingStep;	//已训练次数（用于学习率调整）
	void(*Optimizer)(struct _Network*);		//优化算法
	double(*Loss)(struct _Network*,int indx);		//损失函数
	void(*Dloss)(struct _Network*,int indx);		//损失函数的导数
	void(*InitNetworkArgs)(struct _Network*);	//参数初始化函数
	void(*LearningRateAdjustment)(struct _Network*);	//学习率调整
	double loss;	//总输出损失
	char needAlloc[OPTIONAL_COUNT];	//对应的空间是否需要开辟
}Network;
Network* newNetwork(int number,int batch,int il,int ih,int iw, int ol, int oh, int ow);
int train(Network* n, int count, double thresh, FILE* log);
void run(Network* n);
void bp(Network* n);
void AdaGrad_Optimizer(Network* n);
void RMSProp_Optimizer(Network* n);
void RMSProp_Nesterov_Optimizer(Network* n);
void SGD_Optimizer(Network* n);
void Adam_Optimizer(Network* n);
void BFGS_Optimizer(Network* n);
void Newton_Optimizer(Network* n);
void Delta_Bar_Delta(Network* n);
void Normalized_Initialization(Network* n);
void Set(Network* n);
void Dtor(Network* n);
void none(Network* n);
void InitArgs(Network* n);
double SquareLoss(struct _Network*,int indx);
void DSquareloss(struct _Network*,int indx);
#endif