#include"Neuron.h"
#include"Globals.h"
void bpSoftmax(struct _Neuron* n)
{
	int i;
	FORFROM0STEP1(i, n->info.fc.in)
	{
		n->data.d11.din[i] = n->data.d11.out[i] * n->data.d11.out[i] * (n->learningRate - n->data.d11.out[i]);
	}
}
void runSoftmax(struct _Neuron* n)
{
	//if (n->needClear[1]) RSD(n->data.d11.out, n->info.fc.out); //不需要重置
	int i;
	n->learningRate = 0;
	FORFROM0STEP1(i, n->info.fc.in)
	{
		//softmax中，将e^in[i]存入arg的original bias
		n->arg.fc.original.bias[i] = exp(n->data.d11.in[i]);
		n->learningRate += n->arg.fc.original.bias[i];
	}
	FORFROM0STEP1(i, n->info.fc.out)
	{
		n->data.d11.out[i] = n->arg.fc.original.bias[i] / n->learningRate;
	}
}
void DestroySoftmax(Neuron* n)
{
	//destroy data
	if (n->needFree[0])
	{
		//destroy input
		FREE(n->data.d11.in);
		FREE(n->data.d11.din);
	}
	if (n->needFree[1])
	{
		//destroy output
		FREE(n->data.d11.out);
		FREE(n->data.d11.dout);
	}
	FREE(n->arg.fc.original.bias);
}
void SetSoftmax(Neuron* n)
{
	//将学习率用于exp(x)的和
	//n->type = FC;
	n->run = runSoftmax;
	n->bp = bpSoftmax;
	n->dataOffset = 0;
	n->needFree[0] = 0;
	n->needFree[1] = 0;
	n->needClear[0] = 1;
	n->needClear[1] = 1;
	n->dimension[0] = 1;
	n->dimension[1] = 1;
	n->learningRate = 0;
	n->momentum = 0;
	n->count = 0;
	n->arg.fc.original.bias = MLD(n->info.fc.out);
	RSD(n->arg.fc.original.bias, n->info.fc.out);
	n->arg.fc.grad.bias = NULL;
	n->arg.fc.delta.bias = NULL;
	n->arg.fc.shadow.bias = NULL;
	n->arg.fc.original.weight = NULL;
	n->arg.fc.grad.weight = NULL;
	n->arg.fc.delta.weight = NULL;
	n->arg.fc.shadow.weight = NULL;
	n->extraArg = NULL;
}