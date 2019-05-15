#include"Neuron.h"
#include"Globals.h"
#include"Info.h"
void runAveragePool(struct _Neuron* n)
{
	int i, j, k, l;
	double t1, t2 = n->info.conv.kh*n->info.conv.kw;
	t2 = 1 / t2;
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
					n->data.d33.out[l][i][j] = 0;
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
					t1 = n->data.d33.in[l][tmp->ay][tmp->ax];
					CONTINUE_IF_NEAR_ZERO(t1);
					n->data.d33.out[k][i][j] = n->data.d33.in[l][tmp->ay][tmp->ax];
				}
				n->data.d33.out[k][i][j] *= t2;
			}
		}
	}
}
void bpAveragePool(struct _Neuron* n)
{
	int i, j, l;
	double d = n->info.conv.kh*n->info.conv.kw;
	ChainNode4Int* tmp;
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
			FORCHAIN(tmp, n->info.conv.pairs[i][j])
			{
				FORFROM0STEP1(l, n->info.conv.ol)
				{
					n->data.d33.din[l][tmp->ay][tmp->ax] += n->data.d33.dout[l][i][j] * d;
				}
			}
		}
	}
}
void DestroyAveragePool(Neuron* n)
{
	//ÊäÈëÊä³ö
	//info
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
void SetAveragePool(Neuron* n)
{
	//n->type = FC;
	n->run = runAveragePool;
	n->bp = bpAveragePool;
	n->dataOffset = 0;
	n->needFree[0] = 0;
	n->needFree[1] = 0;
	n->needClear[0] = 1;
	n->needClear[1] = 1;
	n->dimension[0] = 3;
	n->dimension[1] = 3;
	n->learningRate = 0;
	n->momentum = 0;
	n->count = 0;
	n->arg.conv.original.bias = NULL;
	n->arg.conv.grad.bias = NULL;
	n->arg.conv.delta.bias = NULL;
	n->arg.conv.shadow.bias = NULL;
	n->arg.conv.original.weight = NULL;
	n->arg.conv.grad.weight = NULL;
	n->arg.conv.delta.weight = NULL;
	n->arg.conv.shadow.weight = NULL;
	n->extraArg = NULL;
	SetPairsInInfo(&(n->info));
}