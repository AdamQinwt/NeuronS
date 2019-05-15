#include"Network.h"
//全连接异或运算例程
/*
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
	p->Optimizer = Adam_Optimizer;
	p->extraArg[ADAM_DELTA] = 1e-8;
	p->extraArg[ADAM_EPSILON] = 0.001;
	p->extraArg[ADAM_RO1] = 0.9;
	p->extraArg[ADAM_RO2] = 0.999;
	p->extraArg[ADAM_RO1T] = p->extraArg[ADAM_RO1];
	p->extraArg[ADAM_RO2T] = p->extraArg[ADAM_RO2];
	p->neurons[0].extraArgCount = 1;
	p->neurons[1].extraArgCount = 1;
	
	Set(p);
	InitArgs(p);
	Connect(p->neurons, p->neurons+1);
	train(p, 10000, 0.01, stdout);
	RecordArgs(p);
	Dtor(p);
	fclose(dataset);
	PS;
}*/
//卷积mnist识别例程
main()
{
	srand(time(NULL));
	FILE* dataset = fopen("mnist/data.txt", "r");
	Network* p = newNetwork("mnist", 1, 28, 28, 1, 1, 10);
	p->dataSet = dataset;
	p->batch = 50;
	p->number = 5;
	p->neurons = MLN(Neuron, p->number);
	p->neurons[0].type = CONV;
	p->neurons[1].type = MAX_POOL;
	p->neurons[2].type = FC;
	p->neurons[3].type = FC;
	p->neurons[4].type = FC;

	p->neurons[0].info.conv.il = 1;
	p->neurons[0].info.conv.ih = 28;
	p->neurons[0].info.conv.iw = 28;
	p->neurons[0].info.conv.kh = 3;
	p->neurons[0].info.conv.kw = 3;
	p->neurons[0].info.conv.ol = 8;
	p->neurons[0].info.conv.oh = 28;
	p->neurons[0].info.conv.ow = 28;
	p->neurons[0].info.conv.ph = 1;
	p->neurons[0].info.conv.pw = 1;
	p->neurons[0].info.conv.sh = 1;
	p->neurons[0].info.conv.sw = 1;

	p->neurons[1].info.conv.il = 8;
	p->neurons[1].info.conv.ih = 28;
	p->neurons[1].info.conv.iw = 28;
	p->neurons[1].info.conv.kh = 2;
	p->neurons[1].info.conv.kw = 2;
	p->neurons[1].info.conv.ol = 8;
	p->neurons[1].info.conv.oh = 14;
	p->neurons[1].info.conv.ow = 14;
	p->neurons[1].info.conv.ph = 0;
	p->neurons[1].info.conv.pw = 0;
	p->neurons[1].info.conv.sh = 2;
	p->neurons[1].info.conv.sw = 2;

	p->neurons[2].info.fc.in = 1568;
	p->neurons[2].info.fc.out = 512;

	p->neurons[3].info.fc.in = 512;
	p->neurons[3].info.fc.out = 128;

	p->neurons[4].info.fc.in = 128;
	p->neurons[4].info.fc.out = 10;

	p->needAlloc[ORIGINAL] = 1;
	p->needAlloc[GRAD] = 1;
	p->needAlloc[DELTA] = 1;
	p->needAlloc[SHADOW] = 1;
	//使用AdaGrad
	//p->Optimizer = AdaGrad_Optimizer;
	//p->extraArg[ADAGRAD] = 0.00000001;
	//使用RMSProp
	//p->Optimizer = RMSProp_Optimizer;
	//p->extraArg[RMSPROP_1] = 0.0000001;
	//p->extraArg[RMSPROP_2] = 0.3;

	//使用Adam
	p->Optimizer = Adam_Optimizer;
	p->extraArg[ADAM_DELTA] = 1e-8;
	p->extraArg[ADAM_EPSILON] = 0.001;
	p->extraArg[ADAM_RO1] = 0.9;
	p->extraArg[ADAM_RO2] = 0.999;
	p->extraArg[ADAM_RO1T] = p->extraArg[ADAM_RO1];
	p->extraArg[ADAM_RO2T] = p->extraArg[ADAM_RO2];
	p->neurons[0].extraArgCount = 1;
	p->neurons[1].extraArgCount = 1;
	p->neurons[2].extraArgCount = 1;
	p->neurons[3].extraArgCount = 1;
	p->neurons[4].extraArgCount = 1;

	Set(p);
	InitArgs(p);
	Connect(p->neurons, p->neurons + 1);
	Connect(p->neurons + 1, p->neurons + 2);
	Connect(p->neurons + 2, p->neurons + 3);
	Connect(p->neurons + 3, p->neurons + 4);
	train(p, 100, 0.01, stdout);
	RecordArgs(p);
	Dtor(p);
	fclose(dataset);
	PS;
}