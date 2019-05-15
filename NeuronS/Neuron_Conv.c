#include"Neuron.h"
#include"Globals.h"
#include"Arg.h"
void runConv(struct _Neuron* n)
{
	int i, j, k, l;
	double t1, t2;
	ChainNode4Int* tmp;
	if (n->needClear[1])
	{
		//clear outputs
		FORFROM0STEP1(l, n->info.conv.ol)
		{
			FORFROM0STEP1(i, n->info.conv.oh)
			{
				FORFROM0STEP1(j, n->info.conv.ow)
				{
					n->data.d33.out[l][i][j] = n->arg.conv.original.bias[l];
				}
			}
		}
	}
	FORFROM0STEP1(k, n->info.conv.ol)
	{
		FORFROM0STEP1(i, n->info.conv.oh)
		{
			FORFROM0STEP1(j, n->info.conv.ow)
			{
				FORCHAIN(tmp, n->info.conv.pairs[i][j])
				{
					FORFROM0STEP1(l, n->info.conv.il)
					{
						t1 = n->data.d33.in[l][tmp->ay][tmp->ax];
						t2 = n->arg.conv.original.weight[k][l][tmp->by][tmp->bx];
						CONTINUE_IF_NEAR_ZERO(t1);
						CONTINUE_IF_NEAR_ZERO(t2);
						n->data.d33.out[k][i][j] += n->data.d33.in[l][tmp->ay][tmp->ax] * n->arg.conv.original.weight[k][l][tmp->by][tmp->bx];
					}
				}
				n->data.d33.out[k][i][j] = n->activate(n->data.d33.out[k][i][j]);
			}
		}
	}
}
void bpConv(struct _Neuron* n)
{
	int i, j, k, l;
	double d;
	ChainNode4Int* tmp;
	n->count++;
	if (n->needClear[0])
	{
		//clear dinputs
		FORFROM0STEP1(l, n->info.conv.il)
		{
			FORFROM0STEP1(i, n->info.conv.ih)
			{
				FORFROM0STEP1(j, n->info.conv.iw)
				{
					n->data.d33.din[l][i][j] = 0;
				}
			}
		}
	}

	FORFROM0STEP1(i, n->info.conv.oh)
	{
		FORFROM0STEP1(j, n->info.conv.ow)
		{
			FORFROM0STEP1(k, n->info.conv.ol)
			{
				d = n->dactivate(n->data.d33.out[k][i][l])*n->data.d33.dout[k][i][l];
				n->arg.conv.grad.bias[i] += d;
				CONTINUE_IF_NEAR_ZERO(d);
				FORCHAIN(tmp, n->info.conv.pairs[i][j])
				{
					FORFROM0STEP1(l, n->info.conv.il)
					{
						n->data.d33.din[l][tmp->ay][tmp->ax] += d * n->arg.conv.original.weight[k][l][tmp->by][tmp->bx];
						n->arg.conv.grad.weight[k][l][tmp->by][tmp->bx] += d * n->data.d33.in[l][tmp->ay][tmp->ax];
					}
				}
			}
		}
	}
}
void DestroyConv(Neuron* n)
{
	//输入输出
	//参数
	//info
	int i;
	//destroy arg
	destroy4dDoubleArray(n->arg.conv.original.weight, n->info.conv.ol, n->info.conv.il, n->info.conv.kh);
	FREE(n->arg.conv.original.bias);
	destroy4dDoubleArray(n->arg.conv.delta.weight, n->info.conv.ol, n->info.conv.il, n->info.conv.kh);
	FREE(n->arg.conv.delta.bias);
	destroy4dDoubleArray(n->arg.conv.grad.weight, n->info.conv.ol, n->info.conv.il, n->info.conv.kh);
	FREE(n->arg.conv.grad.bias);
	destroy4dDoubleArray(n->arg.conv.shadow.weight, n->info.conv.ol, n->info.conv.il, n->info.conv.kh);
	FREE(n->arg.conv.shadow.bias);
	FORFROM0STEP1(i, n->extraArgCount)
	{
		destroy4dDoubleArray(n->extraArg[i].conv.weight, n->info.conv.ol, n->info.conv.il, n->info.conv.kh);
		FREE(n->extraArg[i].conv.bias);
	}
	FREE(n->extraArg);
	//destroy data
	if (n->needFree[0])
	{
		//destroy input
		destroy3dDoubleArray(n->data.d33.in, n->info.conv.il, n->info.conv.ih);
		destroy3dDoubleArray(n->data.d33.din, n->info.conv.il, n->info.conv.ih);
	}
	if (n->needFree[1])
	{
		//destroy output
		destroy3dDoubleArray(n->data.d33.out, n->info.conv.ol, n->info.conv.oh);
		destroy3dDoubleArray(n->data.d33.dout, n->info.conv.ol, n->info.conv.oh);
	}
	DestroyPairsInInfo(&(n->info));
}
void SetConv(Neuron* n, double learningRate, char* act, char* needAlloc)
{
	int i;
	//n->type = FC;
	n->run = runConv;
	n->bp = bpConv;
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
	n->dimension[0] = 3;
	n->dimension[1] = 3;
	n->learningRate = learningRate;
	n->momentum = 0;
	n->count = 0;
	n->arg.conv.original.bias = needAlloc[ORIGINAL] ? MLN(double, n->info.conv.ol) : NULL;
	n->arg.conv.grad.bias = needAlloc[GRAD] ? MLN(double, n->info.conv.ol) : NULL;
	n->arg.conv.delta.bias = needAlloc[DELTA] ? MLN(double, n->info.conv.ol) : NULL;
	n->arg.conv.shadow.bias = needAlloc[SHADOW] ? MLN(double, n->info.conv.ol) : NULL;
	n->arg.conv.original.weight = needAlloc[ORIGINAL] ? new4dDoubleArray(n->info.conv.ol, n->info.conv.il, n->info.conv.kh, n->info.conv.kw) : NULL;
	n->arg.conv.grad.weight = needAlloc[GRAD] ? new4dDoubleArray(n->info.conv.ol, n->info.conv.il, n->info.conv.kh, n->info.conv.kw) : NULL;
	n->arg.conv.delta.weight = needAlloc[DELTA] ? new4dDoubleArray(n->info.conv.ol, n->info.conv.il, n->info.conv.kh, n->info.conv.kw) : NULL;
	n->arg.conv.shadow.weight = needAlloc[SHADOW] ? new4dDoubleArray(n->info.conv.ol, n->info.conv.il, n->info.conv.kh, n->info.conv.kw) : NULL;
	RSD(n->arg.conv.grad.bias, n->info.fc.out);
	RSD(n->arg.conv.shadow.bias, n->info.fc.out);
	RSD(n->arg.conv.delta.bias, n->info.fc.out);
	if (n->extraArgCount)
	{
		//开辟额外参数空间
		n->extraArg = MLN(SimpleArg, n->extraArgCount);
		FORFROM0STEP1(i, n->extraArgCount)
		{
			n->extraArg[i].conv.bias = MLN(double, n->info.conv.ol);
			RSD(n->extraArg[i].conv.bias, n->info.conv.ol);
			n->extraArg[i].conv.weight = new4dDoubleArray(n->info.conv.ol, n->info.conv.il, n->info.conv.kh, n->info.conv.kw);
		}
	}
	else
	{
		n->extraArg = NULL;
	}
	SetPairsInInfo(&(n->info));
}