#include"Neuron.h"
#include"Globals.h"
#include"Info.h"
void runMaxPool(struct _Neuron* n)
{
	int i, j, l;
	int mm, nn;
	int m0, n0;
	int x, y;	//池化计数变量
	double max;
	ChainNode4Int* tmp;
	m0 = -(n->info.conv.ph);
	FORFROM0STEP1(i, n->info.conv.oh)
	{
		n0 = -(n->info.conv.pw);
		FORFROM0STEP1(j, n->info.conv.ow)
		{
			tmp = n->info.conv.pairs[i][j].head;
			FORFROM0STEP1(l, n->info.conv.ol)	//每一层
			{
				mm = m0;
				max = INT_MIN;
				FORFROM0STEP1(x, n->info.conv.kh)
				{
					if (mm < 0) continue;
					if (mm >= n->info.conv.ih) break;
					nn = n0;
					FORFROM0STEP1(y, n->info.conv.kw)
					{
						if (nn < 0) continue;
						if (nn >= n->info.conv.iw) break;
						if (n->data.d33.in[l][mm][nn] > max)
						{
							max = n->data.d33.in[l][mm][nn];
							tmp->ax = nn;
							tmp->ay = mm;
						}
						nn++;
					}
					mm++;
				}
				n->data.d33.out[l][i][j] = max;
				tmp = tmp->next;
			}
			n0 += n->info.conv.sw;
		}
		m0 += n->info.conv.sh;
	}
}
void bpMaxPool(struct _Neuron* n)
{
	int i, j, l;
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
			tmp = n->info.conv.pairs[i][j].head;
			FORFROM0STEP1(l, n->info.conv.ol)
			{
				n->data.d33.din[l][tmp->ay][tmp->ax] += n->data.d33.dout[l][i][j];
				tmp = tmp->next;
			}
		}
	}
}
void SetMaxPool(Neuron* n)
{
	//n->type = FC;
	n->run = runMaxPool;
	n->bp = bpMaxPool;
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
	AllocPairsInInfo(&(n->info));
}
void DestroyMaxPool(Neuron* n)
{
	//输入输出
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