#ifndef NETWORK_H
#define NETWORK_H
#include"Neuron.h"
#include"Globals.h"
//�����Ż����Ķ��������
#define EXTRACOUNT 6
#define ADAGRAD 0	//С����delta����Ϊ1e-7
#define RMSPROP_1 0	//С����delta����Ϊ1e-6
#define RMSPROP_2 1	//˥������\rou
#define ADAM_DELTA 0	//С����delta����Ϊ1e-8
#define ADAM_RO1 1	//Adam�е�\rou1����Ϊ0.9
#define ADAM_RO2 2	//Adam�е�\rou2����Ϊ0.999
#define ADAM_EPSILON 3	//Adam�еĲ�������Ϊ0.001
#define ADAM_RO1T 4	//Adam�е�\rou1^t��tΪ��������
#define ADAM_RO2T 5	//Adam�е�\rou2^t��tΪ��������
typedef struct _Network
{
	char name[50];	//����
	int il, ih, iw, ol, oh, ow;
	double**** x, ***dx;
	double extraArg[EXTRACOUNT];	//�����Ż����Ķ��������
	double*** out;	//ʵ���������
	double**** y;
	int number;	//��Ԫ����
	Neuron* neurons;	//��Ԫ����
	int batch;	//batch size
	int batchCount;	//how many batches are there
	int batchRemainder;	//the size of the last batch
	FILE* dataSet;
	int trainingStep;	//��ѵ������������ѧϰ�ʵ�����
	void(*Optimizer)(struct _Network*);		//�Ż��㷨
	double(*Loss)(struct _Network*,int indx);		//��ʧ����
	void(*Dloss)(struct _Network*,int indx);		//��ʧ�����ĵ���
	void(*InitNetworkArgs)(struct _Network*);	//������ʼ������
	void(*LearningRateAdjustment)(struct _Network*);	//ѧϰ�ʵ���
	double loss;	//�������ʧ
	char needAlloc[OPTIONAL_COUNT];	//��Ӧ�Ŀռ��Ƿ���Ҫ����
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