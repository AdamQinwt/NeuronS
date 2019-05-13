#include"Neuron.h"
#include"Globals.h"
void Connect(Neuron* from, Neuron* to)	//根据维度连接
{
	//判断连接方式
	//连接
	/*
		优先在一维侧分配
		优先在输出端分配
	*/
	if (from->dimension[1] == 3)
	{
		if (to->dimension[0] == 3)
		{
			//3->3
			//输出端分配
			from->needFree[1] = 1;
			to->needFree[0] = 0;
			from->data.d33.out = new3dDoubleArray(from->info.conv.ol, from->info.conv.oh, from->info.conv.ow);
			from->data.d33.dout = new3dDoubleArray(from->info.conv.ol, from->info.conv.oh, from->info.conv.ow);
			to->data.d33.in = from->data.d33.out+from->dataOffset;
			to->data.d33.din = from->data.d33.dout + from->dataOffset;
			from->dataOffset += from->info.conv.ol;
		}
		else
		{
			//3->1
			//输入端分配
			from->needFree[1] = 0;
			to->needFree[0] = 1;
			to->data.d11.in = MLN(double, to->info.fc.in);
			to->data.d11.din = MLN(double, to->info.fc.in);
			from->data.d33.out = new3dDoubleArrayFrom1d(from->info.conv.ol, from->info.conv.oh, from->info.conv.ow,0,to->data.d11.in);
			from->data.d33.dout = new3dDoubleArrayFrom1d(from->info.conv.ol, from->info.conv.oh, from->info.conv.ow,0,to->data.d11.din);
		}
	}
	else
	{
		//1->1
		//输入端分配
		from->needFree[1] = 1;
		to->needFree[0] = 0;
		from->data.d11.out = MLN(double, from->info.fc.out);
		from->data.d11.dout = MLN(double, from->info.fc.out);
		to->data.d11.in = from->data.d11.out;
		to->data.d11.din = from->data.d11.dout;
	}
}
void runFC(struct _Neuron* n)
{
	if (n->needClear[1]) RSD(n->data.d11.out,n->info.fc.out);
	int i, j;
	double t1, t2;
	for (j = 0; j < n->info.fc.out; j++)
	{
		for (i = 0; i < n->info.fc.in; i++)
		{
			t1 = n->data.d11.in[i];
			t2 = n->arg.fc.original.weight[i][j];
			CONTINUE_IF_NEAR_ZERO(t1);
			CONTINUE_IF_NEAR_ZERO(t2);
			n->data.d11.out[j] += n->data.d11.in[i]* n->arg.fc.original.weight[i][j];
		}
		n->data.d11.out[j] = n->activate(n->data.d11.out[j] + n->arg.fc.original.bias[j]);
	}
}
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
						n->data.d33.out[k][i][j] = n->data.d33.in[l][tmp->ay][tmp->ax]*n->arg.conv.original.weight[k][l][tmp->by][tmp->bx];
					}
				}
				n->data.d33.out[k][i][j] = n->activate(n->data.d33.out[k][i][j]);
			}
		}
	}
}
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
			n0+= n->info.conv.sw;
		}
		m0 += n->info.conv.sh;
	}
}
void runAveragePool(struct _Neuron* n)
{
	int i, j, l;
	int mm, nn;
	int m0, n0;
	int x, y;	//池化计数变量
	double w=1/n->info.conv.pairs[0][0].head->ax;
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
				n->data.d33.out[l][i][j] = 0;
				mm = m0;
				FORFROM0STEP1(x, n->info.conv.kh)
				{
					if (mm < 0) continue;
					if (mm >= n->info.conv.ih) break;
					nn = n0;
					FORFROM0STEP1(y, n->info.conv.kw)
					{
						if (nn < 0) continue;
						if (nn >= n->info.conv.iw) break;
						n->data.d33.out[l][i][j] += n->data.d33.in[l][mm][nn];
						nn++;
					}
					mm++;
				}
				n->data.d33.out[l][i][j] *= w;
				tmp = tmp->next;
			}
			n0 += n->info.conv.sw;
		}
		m0 += n->info.conv.sh;
	}
}
void runSoftmax(struct _Neuron* n)
{
	if (n->needClear[1]) RSD(n->data.d11.out, n->info.fc.out);
	int i;
	double sum=0;
	FORFROM0STEP1(i,n->info.fc.in)
	{
		//softmax中，将e^in[i]存入arg的original bias
		//sum存入original bias的额外单元
		n->arg.fc.original.bias[i] = exp(n->data.d11.in[i]);
		sum += n->arg.fc.original.bias[i];
	}
	n->arg.fc.original.bias[i] = sum;
	FORFROM0STEP1(i, n->info.fc.out)
	{
		n->data.d11.out[i] = n->arg.fc.original.bias[i] / sum;
	}
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
		n->arg.fc.grad.bias[j] = d;
		CONTINUE_IF_NEAR_ZERO(d);
		FORFROM0STEP1(i, n->info.fc.in)
		{
			//CONTINUE_IF_NEAR_ZERO(n->arg.fc.original.weight[i][j]);
			n->data.d11.din[i] += d * n->arg.fc.original.weight[i][j];
			n->arg.fc.grad.weight[i][j] = d * n->data.d11.in[i];
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
void bpMaxPool(struct _Neuron* n);
void bpAveragePool(struct _Neuron* n);
void bpSoftmax(struct _Neuron* n);
void SetFC(Neuron* n,double learningRate,char* act)
{
	//n->type = FC;
	n->run = runFC;
	n->bp = bpFC;
	n->activate = sigmoid;
	n->dactivate = dsigmoid;
	n->dataOffset = 0;
	n->needFree[0] = 0;
	n->needFree[1] = 0;
	n->needClear[0] = 1;
	n->needClear[1] = 1;
	n->dimension[0] = 1;
	n->dimension[1] = 1;
	n->learningRate = learningRate;
	n->momentum = 0;
	n->arg.fc.original.bias = MLN(double, n->info.fc.out);
	n->arg.fc.grad.bias = MLN(double, n->info.fc.out);
	n->arg.fc.delta.bias = MLN(double, n->info.fc.out);
	n->arg.fc.original.weight = new2dDoubleArray(n->info.fc.in,n->info.fc.out);
	n->arg.fc.grad.weight = new2dDoubleArray(n->info.fc.in,n->info.fc.out);
	n->arg.fc.delta.weight = new2dDoubleArray(n->info.fc.in,n->info.fc.out);
}
void DestroyFC(Neuron* n)
{
	//destroy arg
	destroy2dDoubleArray(n->arg.fc.original.weight,n->info.fc.in);
	FREE(n->arg.fc.original.bias);
	destroy2dDoubleArray(n->arg.fc.delta.weight, n->info.fc.in);
	FREE(n->arg.fc.delta.bias);
	destroy2dDoubleArray(n->arg.fc.grad.weight, n->info.fc.in);
	FREE(n->arg.fc.grad.bias);
	destroy2dDoubleArray(n->arg.fc.shadow.weight, n->info.fc.in);
	FREE(n->arg.fc.shadow.bias);
	//destroy data
	if (n->needFree[0])
	{
		//destroy input
		FREE(n->data.d11.in);
	}
	if (n->needFree[1])
	{
		//destroy output
		FREE(n->data.d11.out);
	}
}