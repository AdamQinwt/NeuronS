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
	int dataOffset;	//data���ѷ�������������ά��
	Info info;
	int count;	//��ѵ��������
	double(*activate)(double);
	double(*dactivate)(double);
	void(*run)(struct _Neuron*);
	void(*bp)(struct _Neuron*);
	double learningRate, momentum;
	char needFree[2];	//��������ռ��Ƿ���Ҫ�ͷ�
	char needClear[2];	//��������루�����Ƿ���Ҫ����
	char dimension[2];	//�������ά��
}Neuron;
void Connect(Neuron* from, Neuron* to);	//����ά������
void runFC(struct _Neuron* n);
void runConv(struct _Neuron* n);
void runMaxPool(struct _Neuron* n);
void runAveragePool(struct _Neuron* n);
void runSoftmax(struct _Neuron* n);
void bpFC(struct _Neuron* n);
void SetFC(Neuron* n, double learningRate, char* act,char* needAlloc);
void DestroyFC(Neuron* n);
#endif
