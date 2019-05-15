#include"Neuron.h"
#include"Globals.h"
void runFC(struct _Neuron* n)
{
	if (n->needClear[1]) RSD(n->data.d11.out, n->info.fc.out);
	int i, j;
	double t1, t2;
	for (j = 0; j < n->info.fc.out; j++)
	{
		for (i = 0; i < n->info.fc.in; i++)
		{
			t1 = n->data.d11.in[i];
			CONTINUE_IF_NEAR_ZERO(t1);
			t2 = n->arg.fc.original.weight[i][j];
			CONTINUE_IF_NEAR_ZERO(t2);
			n->data.d11.out[j] += n->data.d11.in[i] * n->arg.fc.original.weight[i][j];
		}
		n->data.d11.out[j] = n->activate(n->data.d11.out[j] + n->arg.fc.original.bias[j]);
	}
	//print1dArray(n->data.d11.out, j);
}
void bpFC(struct _Neuron* n)
{
	if (n->needClear[0]) RSD(n->data.d11.din, n->info.fc.in);
	double d;
	int i, j;
	n->count++;
	FORFROM0STEP1(j, n->info.fc.out)
	{
		d = n->dactivate(n->data.d11.out[j])*n->data.d11.dout[j];
		n->arg.fc.grad.bias[j] += d;
		CONTINUE_IF_NEAR_ZERO(d);
		FORFROM0STEP1(i, n->info.fc.in)
		{
			//CONTINUE_IF_NEAR_ZERO(n->arg.fc.original.weight[i][j]);
			n->data.d11.din[i] += d * n->arg.fc.original.weight[i][j];
			n->arg.fc.grad.weight[i][j] += d * n->data.d11.in[i];
		}
	}
}
void SetFC(Neuron* n, double learningRate, char* act, char* needAlloc)
{
	int i;
	//n->type = FC;
	n->run = runFC;
	n->bp = bpFC;
	if (!strcmp(act, "sigmoid"))
	{
		n->activate = sigmoid;
		n->dactivate = dsigmoid;
	}
	else if (!strcmp(act, "relu"))
	{
		n->activate = relu;
		n->dactivate = drelu;
	}
	n->dataOffset = 0;
	n->needFree[0] = 0;
	n->needFree[1] = 0;
	n->needClear[0] = 1;
	n->needClear[1] = 1;
	n->dimension[0] = 1;
	n->dimension[1] = 1;
	n->learningRate = learningRate;
	n->momentum = 0;
	n->count = 0;
	n->arg.fc.original.bias = needAlloc[ORIGINAL] ? MLN(double, n->info.fc.out) : NULL;
	n->arg.fc.grad.bias = needAlloc[GRAD] ? MLN(double, n->info.fc.out) : NULL;
	n->arg.fc.delta.bias = needAlloc[DELTA] ? MLN(double, n->info.fc.out) : NULL;
	n->arg.fc.shadow.bias = needAlloc[SHADOW] ? MLN(double, n->info.fc.out) : NULL;
	n->arg.fc.original.weight = needAlloc[ORIGINAL] ? new2dDoubleArray(n->info.fc.in, n->info.fc.out) : NULL;
	n->arg.fc.grad.weight = needAlloc[GRAD] ? new2dDoubleArray(n->info.fc.in, n->info.fc.out) : NULL;
	n->arg.fc.delta.weight = needAlloc[DELTA] ? new2dDoubleArray(n->info.fc.in, n->info.fc.out) : NULL;
	n->arg.fc.shadow.weight = needAlloc[SHADOW] ? new2dDoubleArray(n->info.fc.in, n->info.fc.out) : NULL;
	RSD(n->arg.fc.grad.bias, n->info.fc.out);
	RSD(n->arg.fc.shadow.bias, n->info.fc.out);
	RSD(n->arg.fc.delta.bias, n->info.fc.out);
	if (n->extraArgCount)
	{
		//开辟额外参数空间
		n->extraArg = MLN(SimpleArg, n->extraArgCount);
		FORFROM0STEP1(i, n->extraArgCount)
		{
			n->extraArg[i].fc.bias = MLN(double, n->info.fc.out);
			RSD(n->extraArg[i].fc.bias, n->info.fc.out);
			n->extraArg[i].fc.weight = new2dDoubleArray(n->info.fc.in, n->info.fc.out);
		}
	}
	else
	{
		n->extraArg = NULL;
	}
}
void DestroyFC(Neuron* n)
{
	int i;
	//destroy arg
	destroy2dDoubleArray(n->arg.fc.original.weight, n->info.fc.in);
	FREE(n->arg.fc.original.bias);
	destroy2dDoubleArray(n->arg.fc.delta.weight, n->info.fc.in);
	FREE(n->arg.fc.delta.bias);
	destroy2dDoubleArray(n->arg.fc.grad.weight, n->info.fc.in);
	FREE(n->arg.fc.grad.bias);
	destroy2dDoubleArray(n->arg.fc.shadow.weight, n->info.fc.in);
	FREE(n->arg.fc.shadow.bias);
	FORFROM0STEP1(i, n->extraArgCount)
	{
		destroy2dDoubleArray(n->extraArg[i].fc.weight, n->info.fc.in);
		FREE(n->extraArg[i].fc.bias);
	}
	FREE(n->extraArg);
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
}