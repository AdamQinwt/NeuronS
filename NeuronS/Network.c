#include"Network.h"
#include"Globals.h"
void _SGD_Optimizer_FC(Neuron* n)
{
	int i, j;
	if (n->count == 1)
	{
		FORFROM0STEP1(j, n->info.fc.out)
		{
			n->arg.fc.delta.bias[j] = n->arg.fc.grad.bias[j];
			n->arg.fc.original.bias[j] += n->arg.fc.original.bias[j];
			CONTINUE_IF_NEAR_ZERO(n->arg.fc.delta.bias[j]);
			FORFROM0STEP1(i, n->info.fc.in)
			{
				n->arg.fc.delta.weight[i][j] = n->arg.fc.grad.weight[i][j];
				n->arg.fc.original.weight[i][j] += n->arg.fc.delta.weight[i][j];
			}
		}
	}
	else //(n->count >= 1)
	{
		FORFROM0STEP1(j, n->info.fc.out)
		{
			n->arg.fc.delta.bias[j] = n->arg.fc.grad.bias[j]/n->count;
			n->arg.fc.original.bias[j] += n->arg.fc.original.bias[j];
			CONTINUE_IF_NEAR_ZERO(n->arg.fc.delta.bias[j]);
			FORFROM0STEP1(i, n->info.fc.in)
			{
				n->arg.fc.delta.weight[i][j] = n->arg.fc.grad.weight[i][j] / n->count;
				n->arg.fc.original.weight[i][j] += n->arg.fc.delta.weight[i][j];
			}
		}
	}
}
void _SGD_Optimizer_Conv(Neuron* n)
{
	int i, j, k, l;
	FORFROM0STEP1(k, n->info.conv.ol)
	{
		n->arg.conv.delta.bias[k] = n->arg.conv.grad.bias[k];
		FORFROM0STEP1(l, n->info.conv.il)
		{
			FORFROM0STEP1(i, n->info.conv.ph)
			{
				FORFROM0STEP1(j, n->info.conv.pw)
				{
					n->arg.conv.delta.weight[k][l][i][j] = n->arg.conv.grad.weight[k][l][i][j];
				}
			}
		}
	}
}
void SGD_Optimizer(Network* n)
{
	int num;
	FORFROM0STEP1(num, n->number)
	{
		switch (n->neurons[num].type)
		{
		case FC: _SGD_Optimizer_FC(n->neurons + num); break;
		case CONV: _SGD_Optimizer_Conv(n->neurons + num); break;
		default:
			break;
		}
		n->neurons[num].count = 0;
	}
}
void _Normalized_Initialization_FC(Neuron* n,double absRange)
{
	int i;
	FORFROM0STEP1(i, n->info.fc.in)
	{
		assignRandomDoubleArray(n->arg.fc.original.weight[i], n->info.fc.out, absRange);
	}
	assignZeroDoubleArray(n->arg.fc.original.bias, n->info.fc.out);
}
void _Normalized_Initialization_Conv(Neuron* n, double absRange)
{
	int k,i,j;
	FORFROM0STEP1(k, n->info.conv.ol)
	{
		FORFROM0STEP1(i, n->info.conv.il)
		{
			FORFROM0STEP1(j, n->info.conv.oh)
			{
				assignRandomDoubleArray(n->arg.conv.original.weight[k][i][j], n->info.conv.oh, absRange);
			}
		}
	}
	assignZeroDoubleArray(n->arg.conv.original.bias, n->info.conv.ol);
}
void Normalized_Initialization(Network* n)
{
	double absRange=sqrt((double)6 / ((double)(n->oh*n->ol*n->ow + n->ih*n->il*n->iw)));
	int num;
	FORFROM0STEP1(num, n->number)
	{
		switch (n->neurons[num].type)
		{
		case FC: _Normalized_Initialization_FC(n->neurons + num,absRange); break;
		case CONV: _Normalized_Initialization_Conv(n->neurons + num,absRange); break;
		default:
			break;
		}
		n->neurons[num].count = 0;
	}
}
void run(Network* n)
{
	int i;
	FORFROM0STEP1(i, n->number)
	{
		switch (n->neurons[i].type)
		{
		case FC:runFC(n->neurons + i); break;
		default:
			break;
		}
	}
}
void bp(Network* n)
{
	int i;
	for(i=n->number-1;i>-1;i--)
	{
		switch (n->neurons[i].type)
		{
		case FC:bpFC(n->neurons + i); break;
		default:
			break;
		}
	}
}
void Set(Network* n)
{
	int i;
	FORFROM0STEP1(i, n->number)
	{
		switch (n->neurons[i].type)
		{
		case FC:SetFC(n->neurons + i, 0.3, ""); break;
		default:
			break;
		}
	}
	//反向输入空间连接，输出连接，输出偏差连接
	
}
void Dtor(Network* n)
{
	//反向输入空间释放，输出释放，输出偏差释放
	int i;
	FORFROM0STEP1(i, n->number)
	{
		switch (n->neurons[i].type)
		{
		case FC:DestroyFC(n->neurons + i); break;
		default:
			break;
		}
	}
	FREE(n->neurons);
	FREE(n);
}
Network* newNetwork(char* name,int number, int batch, int il, int ih, int iw,int ol, int oh, int ow)
{
	Network* n = MLC(Network);
	n->number = number;
	n->batch = batch;
	n->il = il;
	n->ih = ih;
	n->iw = iw;
	n->ol = ol;
	n->oh = oh;
	n->ow = ow;
	n->neurons = MLN(Neuron, n->number);
	//n->LearningRateAdjustment = none;
	//n->needAlloc[ORIGINAL] = 1;
	n->Loss = SquareLoss;
	n->Dloss = DSquareloss;
	n->trainingStep = 0;
	n->InitNetworkArgs = Normalized_Initialization;
	n->LearningRateAdjustment = none;
	n->Optimizer = SGD_Optimizer;
	//开辟输入输出空间、答案空间、反向输入空间
	return n;
}
int train(Network* n, int count, double thresh, FILE* log)
{
	int b;
	//初始化过程
	FORFROM0STEP1(n->trainingStep, count)
	{
		ResetLosses(n);
		FORFROM0STEP1(b, n->batch)
		{
			//设置输入
			n->neurons[0].data.d11.in = n->x[b];
			//前向
			run(n);
			//后向
			bp(n);
			//计算误差
			n->Loss(n, b);
		}
		//优化
		n->Optimizer(n);
		//输出结果
		if(!(n->trainingStep&7)) fprintf(log, "iter:\t%d\tloss=%lf\n", n->trainingStep, n->loss);
		//判断误差
		if (n->loss < thresh)
		{
			//完成
			fprintf(log, "Complete!", n->trainingStep, n->loss);
			break;
		}
	}
}
void none(Network* n) 
{
	//do nothing
}
double SquareLoss(struct _Network* n,int indx)
{
	int k;
	double tmp,sum=0;
	FORFROM0STEP1(k, n->ow)
	{
		tmp = n->neurons[n->number - 1].data.d11.out[k] - n->y[indx][k];
		sum += tmp * tmp;
	}
	n->loss += sum/n->ow;
	return n->loss;
}
void DSquareloss(struct _Network* n, int indx)
{
	int i;
	double d = ((double)2) / n->ow;
	FORFROM0STEP1(i, n->ow)
	{
		n->neurons[n->number - 1].data.d11.dout[i] += d*(n->neurons[n->number - 1].data.d11.out[i] - n->y[indx][i]);
	}
}
void ResetLosses(struct _Network* n)
{
	int i;
	n->loss = 0;
	FORFROM0STEP1(i, n->ow)
	{
		n->neurons[n->number - 1].data.d11.dout[i] = 0;
	}
}
void InitArgs(Network* n)
{
	//尝试读文件
	//若文件为空，调用初始化函数
	n->InitNetworkArgs(n);
}