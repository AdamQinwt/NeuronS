#ifndef NEURON_H
#define NEURON_H
#include"Arg.h"
#include"Data.h"
#include"Info.h"
typedef enum _NeuronType {NONE=0,FC=1,CONV=2,MAX_POOL=3,AVERAGE_POOL=4,SOFTMAX=5}NeuronType;
typedef struct _Neuron
{
	NeuronType type;
	Arg arg;
	Data data;
	int dataOffset;	//data中已分配的数量（最高维）
	Info info;
	int extraArgCount;	//额外存储的与参数关联的数据量（如一阶矩等等）的个数
	SimpleArg* extraArg;	//额外存储的与参数关联的数据量（如一阶矩等等）
	int count;	//已训练数据数
	double(*activate)(double);
	double(*dactivate)(double);
	void(*run)(struct _Neuron*);
	void(*bp)(struct _Neuron*);
	double learningRate, momentum;
	char needFree[2];	//输入输出空间是否需要释放
	char needClear[2];	//输出和输入（反向）是否需要清零
	char dimension[2];	//输入输出维度
}Neuron;
void Connect(Neuron* from, Neuron* to);	//根据维度连接
void runFC(struct _Neuron* n);
void runConv(struct _Neuron* n);
void runMaxPool(struct _Neuron* n);
void runAveragePool(struct _Neuron* n);
void runSoftmax(struct _Neuron* n);
void bpFC(struct _Neuron* n);
void bpConv(struct _Neuron* n);
void bpMaxPool(struct _Neuron* n);
void bpAveragePool(struct _Neuron* n);
void bpSoftmax(struct _Neuron* n);
void SetFC(Neuron* n, double learningRate, char* act,char* needAlloc);
void SetConv(Neuron* n, double learningRate, char* act,char* needAlloc);
void SetMaxPool(Neuron* n);
void SetAveragePool(Neuron* n);
void SetSoftmax(Neuron* n);
void DestroyFC(Neuron* n);
void DestroyConv(Neuron* n);
void DestroyMaxPool(Neuron* n);
void DestroyAveragePool(Neuron* n);
void DestroySoftmax(Neuron* n);
#endif
