#ifndef NETWORK_H
#define NETWORK_H
#include"Neuron.h"
typedef struct _Network
{
	char name[50];	//����
	int il, ih, iw, ol, oh, ow;
	double** x, **dx;
	double** y, **dy;
	int number;	//��Ԫ����
	Neuron* neurons;	//��Ԫ����
	int batch;	//batch size
	int trainingStep;	//��ѵ������������ѧϰ�ʵ�����
	void(*Optimizer)(struct _Network*);		//�Ż��㷨
	double(*Loss)(struct _Network*,int indx);		//��ʧ����
	void(*Dloss)(struct _Network*,int indx);		//��ʧ�����ĵ���
	void(*InitNetworkArgs)(struct _Network*);	//������ʼ������
	void(*LearningRateAdjustment)(struct _Network*);	//ѧϰ�ʵ���
	double loss;	//�������ʧ
	char needAlloc[OPTIONAL_COUNT];	//��Ӧ�Ŀռ��Ƿ���Ҫ����
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