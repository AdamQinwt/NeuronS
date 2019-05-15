#ifndef NETWORK_H
#define NETWORK_H
#include"Neuron.h"
#include"Globals.h"
//用于优化器的额外参数表
#define EXTRACOUNT 6
#define ADAGRAD 0	//小常数delta，常为1e-7
#define RMSPROP_1 0	//小常数delta，常为1e-6
#define RMSPROP_2 1	//衰减速率\rou
#define ADAM_DELTA 0	//小常数delta，常为1e-8
#define ADAM_RO1 1	//Adam中的\rou1，常为0.9
#define ADAM_RO2 2	//Adam中的\rou2，常为0.999
#define ADAM_EPSILON 3	//Adam中的步长，常为0.001
#define ADAM_RO1T 4	//Adam中的\rou1^t，t为迭代次数
#define ADAM_RO2T 5	//Adam中的\rou2^t，t为迭代次数
typedef struct _Network
{
	char name[50];	//名称
	int il, ih, iw, ol, oh, ow;
	double**** x, ***dx;
	double extraArg[EXTRACOUNT];	//用于优化器的额外参数表
	double*** out;	//实际运行输出
	double**** y;
	int number;	//神经元数量
	Neuron* neurons;	//神经元数组
	int batch;	//batch size
	int batchCount;	//how many batches are there
	int batchRemainder;	//the size of the last batch
	FILE* dataSet;
	int trainingStep;	//已训练次数（用于学习率调整）
	void(*Optimizer)(struct _Network*);		//优化算法
	double(*Loss)(struct _Network*,int indx);		//损失函数
	void(*Dloss)(struct _Network*,int indx);		//损失函数的导数
	void(*InitNetworkArgs)(struct _Network*);	//参数初始化函数
	void(*LearningRateAdjustment)(struct _Network*);	//学习率调整
	double loss;	//总输出损失
	char needAlloc[OPTIONAL_COUNT];	//对应的空间是否需要开辟
}Network;
Network* newNetwork(char* name,int il, int ih, int iw, int ol, int oh, int ow);
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
void Uniform_Initialization(Network* n);
void Set(Network* n);
void Dtor(Network* n);
void none(Network* n);
void InitArgs(Network* n);
void ResetLosses(Network* n);
double SquareLoss(struct _Network*,int indx);
double CrossEntropyLoss(struct _Network* n, int indx);
void DSquareloss(struct _Network*,int indx);
void DCrossEntropyLoss(struct _Network* n, int indx);
void RecordArgs(Network* n);
void SaveArgs(Network* n,FILE* fp);
void ReadArgs(Network* n,FILE* fp);
void ReadHeader(Network* n);
void ReadData(Network* n, int num);
#endif