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
		FC_Arg original;	//ԭ����
		FC_Arg grad;	//�ݶ�
		FC_Arg delta;	//�仯��
		FC_Arg shadow;	//Ӱ�ӣ�����һЩѵ���㷨
	}fc;
	struct {
		Conv_Arg original;	//ԭ����
		Conv_Arg delta;	//�仯��
		Conv_Arg grad;	//�ݶ�
		Conv_Arg shadow;	//Ӱ�ӣ�����һЩѵ���㷨
	}conv;
}Arg;
#endif