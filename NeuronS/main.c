#include"Network.h"
main()
{
	srand(time(NULL));
	FILE* dataset = fopen("xor/data.txt", "r");
	Network* p = newNetwork("xor", 1, 1, 2, 1, 1, 2);
	p->dataSet = dataset;
	p->batch = 4;
	p->number = 2;
	p->neurons = MLN(Neuron, p->number);
	p->neurons[0].type = FC;
	p->neurons[1].type = FC;
	p->neurons[0].info.fc.in = 2;
	p->neurons[0].info.fc.out = 3;
	p->neurons[1].info.fc.in = 3;
	p->neurons[1].info.fc.out = 2;
	p->needAlloc[ORIGINAL] = 1;
	p->needAlloc[GRAD] = 1;
	p->needAlloc[DELTA] = 1;
	p->needAlloc[SHADOW] = 1;
	p->neurons[0].extraArgCount = 0;
	p->neurons[1].extraArgCount = 0;
	//使用AdaGrad
	//p->Optimizer = AdaGrad_Optimizer;
	//p->extraArg[ADAGRAD] = 0.00000001;
	//使用RMSProp
	p->Optimizer = RMSProp_Optimizer;
	p->extraArg[RMSPROP_1] = 0.0000001;
	p->extraArg[RMSPROP_2] = 0.3;
	//使用Adam
	/*
	p->Optimizer = Adam_Optimizer;
	p->extraArg[ADAM_DELTA] = 1e-8;
	p->extraArg[ADAM_EPSILON] = 0.001;
	p->extraArg[ADAM_RO1] = 0.9;
	p->extraArg[ADAM_RO2] = 0.999;
	p->extraArg[ADAM_RO1T] = p->extraArg[ADAM_RO1];
	p->extraArg[ADAM_RO2T] = p->extraArg[ADAM_RO2];
	p->neurons[0].extraArgCount = 1;
	p->neurons[1].extraArgCount = 1;
	*/
	Set(p);
	InitArgs(p);
	Connect(p->neurons, p->neurons+1);
	train(p, 10000, 0.01, stdout);
	RecordArgs(p);
	Dtor(p);
	fclose(dataset);
	PS;
}