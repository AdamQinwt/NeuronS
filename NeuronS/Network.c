#include"Network.h"
void _SGD_Optimizer_FC(Neuron* n)
{
	int i, j;
	if (n->count == 1)
	{
		FORFROM0STEP1(j, n->info.fc.out)
		{
			n->arg.fc.delta.bias[j] = -n->arg.fc.grad.bias[j]*n->learningRate;
			n->arg.fc.original.bias[j] += n->arg.fc.delta.bias[j];
			n->arg.fc.grad.bias[j] = 0;
			CONTINUE_IF_NEAR_ZERO(n->arg.fc.delta.bias[j]);
			FORFROM0STEP1(i, n->info.fc.in)
			{
				n->arg.fc.delta.weight[i][j] = -n->arg.fc.grad.weight[i][j] * n->learningRate;
				n->arg.fc.original.weight[i][j] += n->arg.fc.delta.weight[i][j];
				n->arg.fc.grad.weight[i][j] = 0;
			}
		}
	}
	else //(n->count >= 1)
	{
		FORFROM0STEP1(j, n->info.fc.out)
		{
			n->arg.fc.delta.bias[j] = -n->arg.fc.grad.bias[j]/n->count*n->learningRate;
			n->arg.fc.original.bias[j] += n->arg.fc.delta.bias[j];
			n->arg.fc.grad.bias[j] = 0;
			//CONTINUE_IF_NEAR_ZERO(n->arg.fc.delta.bias[j]);
			FORFROM0STEP1(i, n->info.fc.in)
			{
				n->arg.fc.delta.weight[i][j] = -n->arg.fc.grad.weight[i][j] / n->count*n->learningRate;
				n->arg.fc.original.weight[i][j] += n->arg.fc.delta.weight[i][j];
				n->arg.fc.grad.weight[i][j] = 0;
			}
		}
	}
}
void _SGD_Optimizer_Conv(Neuron* n)
{
	int i, j, k, l;
	if (n->count == 1)
	{
		FORFROM0STEP1(k, n->info.conv.ol)
		{
			n->arg.conv.delta.bias[k] = -n->arg.conv.grad.bias[k] * n->learningRate;
			n->arg.conv.original.bias[k] += n->arg.conv.delta.bias[k];
			n->arg.fc.grad.bias[k] = 0;
			CONTINUE_IF_NEAR_ZERO(n->arg.fc.delta.bias[j]);
			FORFROM0STEP1(l, n->info.conv.il)
			{
				FORFROM0STEP1(i, n->info.conv.kh)
				{
					FORFROM0STEP1(j, n->info.conv.kw)
					{
						n->arg.conv.delta.weight[k][l][i][j] = -n->arg.conv.grad.weight[k][l][i][j] * n->learningRate;
						n->arg.conv.original.weight[k][l][i][j] += n->arg.conv.delta.weight[k][l][i][j];
						n->arg.conv.grad.weight[k][l][i][j] = 0;
					}
				}
			}
		}
	}
	else //(n->count >= 1)
	{
		FORFROM0STEP1(k, n->info.conv.ol)
		{
			n->arg.conv.delta.bias[k] = -n->arg.conv.grad.bias[k] * n->learningRate/n->count;
			n->arg.conv.original.bias[k] += n->arg.conv.delta.bias[k];
			n->arg.fc.grad.bias[k] = 0;
			CONTINUE_IF_NEAR_ZERO(n->arg.fc.delta.bias[j]);
			FORFROM0STEP1(l, n->info.conv.il)
			{
				FORFROM0STEP1(i, n->info.conv.kh)
				{
					FORFROM0STEP1(j, n->info.conv.kw)
					{
						n->arg.conv.delta.weight[k][l][i][j] = -n->arg.conv.grad.weight[k][l][i][j] * n->learningRate / n->count;
						n->arg.conv.original.weight[k][l][i][j] += n->arg.conv.delta.weight[k][l][i][j];
						n->arg.conv.grad.weight[k][l][i][j] = 0;
					}
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
void _AdaGrad_Optimizer_FC(Neuron* n,double delta)
{
	int i, j;
	if (n->count == 1)
	{
		FORFROM0STEP1(j, n->info.fc.out)
		{
			n->arg.fc.shadow.bias[j] += n->arg.fc.grad.bias[j] * n->arg.fc.grad.bias[j];
			n->arg.fc.delta.bias[j] = -n->arg.fc.grad.bias[j] * (n->learningRate/(delta+sqrt(n->arg.fc.shadow.bias[j])));
			n->arg.fc.original.bias[j] += n->arg.fc.delta.bias[j];
			n->arg.fc.grad.bias[j] = 0;
			CONTINUE_IF_NEAR_ZERO(n->arg.fc.delta.bias[j]);
			FORFROM0STEP1(i, n->info.fc.in)
			{
				n->arg.fc.shadow.weight[i][j] += n->arg.fc.grad.weight[i][j] * n->arg.fc.grad.weight[i][j];
				n->arg.fc.delta.weight[i][j] = -n->arg.fc.grad.weight[i][j] * (n->learningRate / (delta + sqrt(n->arg.fc.shadow.weight[i][j])));
				n->arg.fc.original.weight[i][j] += n->arg.fc.delta.weight[i][j];
				n->arg.fc.grad.weight[i][j] = 0;
			}
		}
	}
	else //(n->count >= 1)
	{
		FORFROM0STEP1(j, n->info.fc.out)
		{
			n->arg.fc.grad.bias[j] /= n->count;
			n->arg.fc.shadow.bias[j] += n->arg.fc.grad.bias[j] * n->arg.fc.grad.bias[j];
			n->arg.fc.delta.bias[j] = -n->arg.fc.grad.bias[j] * (n->learningRate / (delta + sqrt(n->arg.fc.shadow.bias[j])));
			n->arg.fc.original.bias[j] += n->arg.fc.delta.bias[j];
			n->arg.fc.grad.bias[j] = 0;
			CONTINUE_IF_NEAR_ZERO(n->arg.fc.delta.bias[j]);
			FORFROM0STEP1(i, n->info.fc.in)
			{
				n->arg.fc.grad.bias[j] /= n->count;
				n->arg.fc.shadow.weight[i][j] += n->arg.fc.grad.weight[i][j] * n->arg.fc.grad.weight[i][j];
				n->arg.fc.delta.weight[i][j] = -n->arg.fc.grad.weight[i][j] * (n->learningRate / (delta + sqrt(n->arg.fc.shadow.weight[i][j])));
				n->arg.fc.original.weight[i][j] += n->arg.fc.delta.weight[i][j];
				n->arg.fc.grad.weight[i][j] = 0;
			}
		}
	}
}
void _AdaGrad_Optimizer_CONV(Neuron* n, double delta) 
{
}
void AdaGrad_Optimizer(Network* n)
{
	//С����delta��Ϊn�е�extraArg[ADAGRAD]
	//ȫ��ѧϰ��epsilon��Ϊn�е�learningRate
	int num;
	FORFROM0STEP1(num, n->number)
	{
		switch (n->neurons[num].type)
		{
		case FC: _AdaGrad_Optimizer_FC(n->neurons + num,n->extraArg[ADAGRAD]); break;
		case CONV: _AdaGrad_Optimizer_CONV(n->neurons + num, n->extraArg[ADAGRAD]); break;
		default:
			break;
		}
		n->neurons[num].count = 0;
	}
}
void _RMSProp_Optimizer_FC(Neuron* n, double delta,double ro)
{
	int i, j;
	if (n->count == 1)
	{
		FORFROM0STEP1(j, n->info.fc.out)
		{
			n->arg.fc.shadow.bias[j] += n->arg.fc.grad.bias[j] * n->arg.fc.grad.bias[j];
			n->arg.fc.delta.bias[j] = -n->arg.fc.grad.bias[j] * (n->learningRate / (delta + sqrt(n->arg.fc.shadow.bias[j])));
			n->arg.fc.original.bias[j] += n->arg.fc.delta.bias[j];
			n->arg.fc.grad.bias[j] = 0;
			CONTINUE_IF_NEAR_ZERO(n->arg.fc.delta.bias[j]);
			FORFROM0STEP1(i, n->info.fc.in)
			{
				n->arg.fc.shadow.weight[i][j] += n->arg.fc.grad.weight[i][j] * n->arg.fc.grad.weight[i][j];
				n->arg.fc.delta.weight[i][j] = -n->arg.fc.grad.weight[i][j] * (n->learningRate / (delta + sqrt(n->arg.fc.shadow.weight[i][j])));
				n->arg.fc.original.weight[i][j] += n->arg.fc.delta.weight[i][j];
				n->arg.fc.grad.weight[i][j] = 0;
			}
		}
	}
	else //(n->count >= 1)
	{
		FORFROM0STEP1(j, n->info.fc.out)
		{
			n->arg.fc.grad.bias[j] /= n->count;
			n->arg.fc.shadow.bias[j] = ro* n->arg.fc.shadow.bias[j]+(1-ro)*n->arg.fc.grad.bias[j] * n->arg.fc.grad.bias[j];
			n->arg.fc.delta.bias[j] = -n->arg.fc.grad.bias[j] * (n->learningRate / sqrt(n->arg.fc.shadow.bias[j]+delta));
			n->arg.fc.original.bias[j] += n->arg.fc.delta.bias[j];
			n->arg.fc.grad.bias[j] = 0;
			CONTINUE_IF_NEAR_ZERO(n->arg.fc.delta.bias[j]);
			FORFROM0STEP1(i, n->info.fc.in)
			{
				n->arg.fc.grad.weight[i][j] /= n->count;
				n->arg.fc.shadow.weight[i][j] = ro*n->arg.fc.shadow.weight[i][j]+(1-ro)*n->arg.fc.grad.weight[i][j] * n->arg.fc.grad.weight[i][j];
				n->arg.fc.delta.weight[i][j] = -n->arg.fc.grad.weight[i][j] * (n->learningRate / sqrt(n->arg.fc.shadow.weight[i][j])+delta);
				n->arg.fc.original.weight[i][j] += n->arg.fc.delta.weight[i][j];
				n->arg.fc.grad.weight[i][j] = 0;
			}
		}
	}
}
void _RMSProp_Optimizer_CONV(Neuron* n, double delta,double ro)
{
}
void RMSProp_Optimizer(Network* n)
{
	//С����delta��Ϊn�е�extraArg[ADAGRAD]
	//ȫ��ѧϰ��epsilon��Ϊn�е�learningRate
	int num;
	FORFROM0STEP1(num, n->number)
	{
		switch (n->neurons[num].type)
		{
		case FC: _RMSProp_Optimizer_FC(n->neurons + num, n->extraArg[RMSPROP_1],n->extraArg[RMSPROP_2]); break;
		case CONV: _RMSProp_Optimizer_CONV(n->neurons + num, n->extraArg[RMSPROP_1], n->extraArg[RMSPROP_2]); break;
		default:
			break;
		}
		n->neurons[num].count = 0;
	}
}
void _Adam_Optimizer_FC(Neuron* n, double delta, double ro1, double ro2,double ro1t,double ro2t, double epsilon)
{
	//�ı�ṹ������һ�������ı���Ӱ�ӣ�����һ�׾�ƫ��s���˴�Ϊn->extraArg[0]
	//����r����shadow��
	int i, j;
	if (n->count == 1)
	{
		FORFROM0STEP1(j, n->info.fc.out)
		{
			n->arg.fc.shadow.bias[j] = ro2* n->arg.fc.shadow.bias[j]+(1-ro2)*n->arg.fc.grad.bias[j] * n->arg.fc.grad.bias[j];
			n->extraArg[0].fc.bias[j] = ro1 * n->extraArg[0].fc.bias[j] + (1 - ro1)*n->arg.fc.grad.bias[j];
			n->arg.fc.delta.bias[j] = -epsilon * n->extraArg[0].fc.bias[j] / (1 - ro1t) / (sqrt(n->arg.fc.shadow.bias[j] / (1 - ro2t)) + delta);
			n->arg.fc.original.bias[j] += n->arg.fc.delta.bias[j];
			n->arg.fc.grad.bias[j] = 0;
			CONTINUE_IF_NEAR_ZERO(n->arg.fc.delta.bias[j]);
			FORFROM0STEP1(i, n->info.fc.in)
			{
				n->arg.fc.shadow.weight[i][j] = ro2 * n->arg.fc.shadow.weight[i][j] + (1 - ro2)*n->arg.fc.grad.weight[i][j] * n->arg.fc.grad.weight[i][j];
				n->extraArg[0].fc.weight[i][j] = ro1 * n->extraArg[0].fc.weight[i][j] + (1 - ro1)*n->arg.fc.grad.weight[i][j];
				n->arg.fc.delta.weight[i][j] = -epsilon * n->extraArg[0].fc.weight[i][j] / (1 - ro1t) / (sqrt(n->arg.fc.shadow.weight[i][j] / (1 - ro2t)) + delta);
				n->arg.fc.original.weight[i][j] += n->arg.fc.delta.weight[i][j];
				n->arg.fc.grad.weight[i][j] = 0;
			}
		}
	}
	else //(n->count >= 1)
	{
		FORFROM0STEP1(j, n->info.fc.out)
		{
			n->arg.fc.grad.bias[j] /= n->count;
			n->arg.fc.shadow.bias[j] = ro2 * n->arg.fc.shadow.bias[j] + (1 - ro2)*n->arg.fc.grad.bias[j] * n->arg.fc.grad.bias[j];
			n->extraArg[0].fc.bias[j] = ro1 * n->extraArg[0].fc.bias[j] + (1 - ro1)*n->arg.fc.grad.bias[j];
			n->arg.fc.delta.bias[j] = -epsilon * n->extraArg[0].fc.bias[j] / (1 - ro1t) / (sqrt(n->arg.fc.shadow.bias[j] / (1 - ro2t)) + delta);
			n->arg.fc.original.bias[j] += n->arg.fc.delta.bias[j];
			n->arg.fc.grad.bias[j] = 0;
			CONTINUE_IF_NEAR_ZERO(n->arg.fc.delta.bias[j]);
			FORFROM0STEP1(i, n->info.fc.in)
			{
				n->arg.fc.grad.weight[i][j] /= n->count;
				n->arg.fc.shadow.weight[i][j] = ro2 * n->arg.fc.shadow.weight[i][j] + (1 - ro2)*n->arg.fc.grad.weight[i][j] * n->arg.fc.grad.weight[i][j];
				n->extraArg[0].fc.weight[i][j] = ro1 * n->extraArg[0].fc.weight[i][j] + (1 - ro1)*n->arg.fc.grad.weight[i][j];
				n->arg.fc.delta.weight[i][j] = -epsilon * n->extraArg[0].fc.weight[i][j] / (1 - ro1t) / (sqrt(n->arg.fc.shadow.weight[i][j] / (1 - ro2t)) + delta);
				n->arg.fc.original.weight[i][j] += n->arg.fc.delta.weight[i][j];
				n->arg.fc.grad.weight[i][j] = 0;
			}
		}
	}
}
void _Adam_Optimizer_CONV(Neuron* n, double delta, double ro1, double ro2, double ro1t, double ro2t, double epsilon)
{
	//
}
void Adam_Optimizer(Network* n)
{
	//С����delta��Ϊn�е�extraArg[ADAM_DELTA]
	//ָ��˥���ʷֱ�Ϊn�е�extraArg[ADAM_RO1],extraArg[ADAM_RO2]
	//ȫ��ѧϰ��epsilon��Ϊn�е�extraArg[ADAM_EPSILON]
	int num;
	FORFROM0STEP1(num, n->number)
	{
		switch (n->neurons[num].type)
		{
		case FC: _Adam_Optimizer_FC(n->neurons + num, n->extraArg[ADAM_DELTA], n->extraArg[ADAM_RO1], n->extraArg[ADAM_RO2], n->extraArg[ADAM_RO1T], n->extraArg[ADAM_RO2T], n->extraArg[ADAM_EPSILON]); break;
		case CONV: _Adam_Optimizer_CONV(n->neurons + num, n->extraArg[ADAM_DELTA], n->extraArg[ADAM_RO1], n->extraArg[ADAM_RO2], n->extraArg[ADAM_RO1T], n->extraArg[ADAM_RO2T], n->extraArg[ADAM_EPSILON]); break;
		default:
			break;
		}
		n->neurons[num].count = 0;
	}
	n->extraArg[ADAM_RO1T] *= n->extraArg[ADAM_RO1];
	n->extraArg[ADAM_RO2T] *= n->extraArg[ADAM_RO2];
}
void _Normalized_Initialization_FC(Neuron* n,double absRange)
{
	int i;
	FORFROM0STEP1(i, n->info.fc.in)
	{
		assignRandomDoubleArray(n->arg.fc.original.weight[i], n->info.fc.out, absRange);
		print1dArray(n->arg.fc.original.weight[i], n->info.fc.out);
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
void Uniform_Initialization(Network* n)
{
	int num;
	FORFROM0STEP1(num, n->number)
	{
		switch (n->neurons[num].type)
		{
		case FC: _Normalized_Initialization_FC(n->neurons + num,0.5); break;
		case CONV: _Normalized_Initialization_Conv(n->neurons + num,0.5); break;
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
		n->neurons[i].run(n->neurons + i);
	}
}
void bp(Network* n)
{
	int i;
	for(i=n->number-1;i>=0;i--)
	{
		n->neurons[i].bp(n->neurons+i);
	}
}
void Set(Network* n)
{
	int i;
	FORFROM0STEP1(i, n->number)
	{
		switch (n->neurons[i].type)
		{
		//case FC:SetFC(n->neurons + i, 0.3, (i==n->number-1)?"sigmoid":"relu"); break;
		case FC:SetFC(n->neurons + i, 0.3, "sigmoid",n->needAlloc); break;
		default:
			break;
		}
	}
	//��������ռ����ӣ�������ӣ����ƫ������
	n->dx = MLD(n->neurons[0].info.fc.in);
	n->neurons[0].data.d11.din = n->dx;
	Neuron* last = n->neurons+n->number - 1;
	n->out = MLD(last->info.fc.out);
	last->data.d11.dout = MLD(last->info.fc.out);
	last->needFree[1] = 1;
	last->data.d11.out = n->out;
}
void Dtor(Network* n)
{
	//��������ռ��ͷţ�����ͷţ����ƫ���ͷ�
	FREE(n->neurons[0].data.d11.din);
	Neuron* last = n->neurons + n->number - 1;
	//FREE(n->out);
	FREE(last->data.d11.dout);
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
Network* newNetwork(char* name,int il, int ih, int iw,int ol, int oh, int ow)
{
	Network* n = MLC(Network);
	strcpy(n->name, name);
	n->batch = 0;
	n->batchCount = 0;
	n->batchRemainder = 0;
	n->il = il;
	n->ih = ih;
	n->iw = iw;
	n->ol = ol;
	n->oh = oh;
	n->ow = ow;
	n->x = NULL;
	n->y = NULL;
	n->dx = NULL;
	n->out = NULL;
	//n->LearningRateAdjustment = none;
	//n->needAlloc[ORIGINAL] = 1;
	//n->Loss = SquareLoss;
	//n->Dloss = DSquareloss;
	n->Loss = CrossEntropyLoss;
	n->Dloss = DCrossEntropyLoss;
	n->trainingStep = 0;
	n->InitNetworkArgs = Normalized_Initialization;
	//n->InitNetworkArgs = Uniform_Initialization;
	n->LearningRateAdjustment = none;
	n->Optimizer = SGD_Optimizer;
	//������������ռ䡢�𰸿ռ䣨���������ɣ�
	return n;
}
int train(Network* n, int count, double thresh, FILE* log)
{
	int b,num,rem;
	//��ʼ������
	//���������ļ�ͷ
	ReadHeader(n);
	if (n->batchCount == 1)
	{
		ReadData(n,n->batch);
		FORFROM0STEP1(n->trainingStep, count)
		{
			ResetLosses(n);
			FORFROM0STEP1(b, n->batch)
			{
				//��������
				n->neurons[0].data.d11.in = n->x[b];
				//ǰ��
				run(n);
				//print1dArray(n->out, n->ow);
				//�������
				n->Loss(n, b);
				n->Dloss(n, b);
				//����
				bp(n);
			}
			//�Ż�
			n->Optimizer(n);
			//������
			//if(!(n->trainingStep&0xff)) 
			fprintf(log, "iter:\t%d\tloss=%lf\n", n->trainingStep, n->loss);
			//�ж����
			if (n->loss < thresh)
			{
				//���
				fprintf(log, "Complete!\n");
				break;
			}
		}
		FORFROM0STEP1(b, n->batch)
		{
			//��������
			n->neurons[0].data.d11.in = n->x[b];
			//ǰ��
			run(n);
			print1dArray(n->out, n->ow);
			print1dArray(n->y[b], n->ow);
		}
	}
	else
	{
		FORFROM0STEP1(n->trainingStep, count)
		{
			FORFROM0STEP1(num, n->batchCount)
			{
				rem = num==n->batchCount - 1 ? n->batchRemainder : n->batch;
				ReadData(n,rem);
				ResetLosses(n);
				FORFROM0STEP1(b, rem)
				{
					//��������
					n->neurons[0].data.d11.in = n->x[b];
					//ǰ��
					run(n);
					//print1dArray(n->out, n->ow);
					//�������
					n->Loss(n, b);
					n->Dloss(n, b);
					//����
					bp(n);
				}
				//�Ż�
				n->Optimizer(n);
				//������
				//if(!(n->trainingStep&0xff)) 
				fprintf(log, "iter:\t%d\tloss=%lf\n", n->trainingStep, n->loss);
				//�ж����
				if (n->loss < thresh)
				{
					//���
					fprintf(log, "Complete!\n");
					break;
				}
			}
		}
	}
	return n->trainingStep;
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
		tmp = n->out[k] - n->y[indx][k];
		sum += tmp * tmp;
	}
	n->loss += sum/n->batch;
	return n->loss;
}
double CrossEntropyLoss(struct _Network* n, int indx)
{
	int k;
	double sum = 0;
	FORFROM0STEP1(k, n->ow)
	{
		sum -= n->y[indx][k] * log(n->out[k]);
		sum-= (1-n->y[indx][k]) * log(1-n->out[k]);
	}
	n->loss += sum / n->batch;
	return n->loss;
}
void DSquareloss(struct _Network* n, int indx)
{
	int i;
	double d = ((double)2) / n->batch;
	FORFROM0STEP1(i, n->ow)
	{
		n->neurons[n->number - 1].data.d11.dout[i] = d*(n->out[i] - n->y[indx][i]);
	}
}
void DCrossEntropyLoss(struct _Network* n, int indx)
{
	int i;
	//double d = ((double)2) / n->batch;
	FORFROM0STEP1(i, n->ow)
	{
	//	CONTINUE_IF_NEAR_ZERO(n->y[indx][i]);
		n->neurons[n->number - 1].data.d11.dout[i]= -(n->y[indx][i]/n->out[i]+ (n->y[indx][i]-1) / (1-n->out[i]))/n->batch;
	}
}
void ResetLosses(Network* n)
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
	//���Զ��ļ�
	char fname[50];
	sprintf(fname,"%s/arg", n->name);
	FILE* fp = fopen(fname,"rb");
	if (fp) ReadArgs(n, fp);
	//���ļ�Ϊ�գ����ó�ʼ������
	else n->InitNetworkArgs(n);
}
void SaveArgs(Network* n, FILE* fp)
{
	int num;
	FORFROM0STEP1(num, n->number)
	{
		if (n->neurons[num].type == FC)
		{
			//write bias
			write1dArray(fp, n->neurons[num].arg.fc.original.bias, n->neurons[num].info.fc.out);
			//write arg
			write2dArray(fp, n->neurons[num].arg.fc.original.weight, n->neurons[num].info.fc.in, n->neurons[num].info.fc.out);
		}
		else if (n->neurons[num].type == CONV)
		{
			//write bias
			write1dArray(fp, n->neurons[num].arg.conv.original.bias, n->neurons[num].info.conv.ol);
			//write arg
			write4dArray(fp, n->neurons[num].arg.conv.original.weight, n->neurons[num].info.conv.ol, n->neurons[num].info.conv.il, n->neurons[num].info.conv.kh, n->neurons[num].info.conv.kw);
		}
	}
}
void ReadArgs(Network* n, FILE* fp)
{
	int num;
	FORFROM0STEP1(num, n->number)
	{
		if (n->neurons[num].type == FC)
		{
			//write bias
			read1dArray(fp, n->neurons[num].arg.fc.original.bias, n->neurons[num].info.fc.out);
			//write arg
			read2dArray(fp, n->neurons[num].arg.fc.original.weight, n->neurons[num].info.fc.in, n->neurons[num].info.fc.out);
		}
		else if (n->neurons[num].type == CONV)
		{
			//write bias
			read1dArray(fp, n->neurons[num].arg.conv.original.bias, n->neurons[num].info.conv.ol);
			//write arg
			read4dArray(fp, n->neurons[num].arg.conv.original.weight, n->neurons[num].info.conv.ol, n->neurons[num].info.conv.il, n->neurons[num].info.conv.kh, n->neurons[num].info.conv.kw);
		}
	}
}
void RecordArgs(Network* n)
{
	char fname[50];
	sprintf(fname, "%s/arg", n->name);
	FILE* fp = fopen(fname, "wb");
	if (fp) SaveArgs(n, fp);
}
void ReadHeader(Network* n)
{
	//�����ļ�Ϊ�ı��ļ�
	//�ɵ�����Ԥ��ָ��batch��batchCount��ReadData����ֻ���������������ռ䲢��������
	//�����ļ���һ��Ϊһ��������ͷ����ʶ���ݼ���С
	//�����ļ��ָ�����Ϊ ' ',',','\t','\r','\n'
	int cnt;	//���ݼ���С
	fscanf(n->dataSet, "%d", &cnt);
	if (n->batch)
	{
		n->batchCount = cnt / n->batch;
		n->batchRemainder = cnt % n->batch;
		if (n->batchRemainder) n->batchCount++;
	}
	else if (n->batchCount)
	{
		n->batch = cnt / n->batchCount;
		n->batchRemainder = cnt - n->batch*n->batchCount;
		if (n->batchRemainder)
		{
			n->batch++;
			n->batchCount = cnt / n->batch;
			n->batchRemainder = cnt % n->batch;
			if (n->batchRemainder) n->batchCount++;
		}
	}
	//����ռ�
	//һά���
	if (!(n->x)) n->x = new2dDoubleArray(n->batch, n->iw);
	if (!(n->y)) n->y = new2dDoubleArray(n->batch, n->ow);
}
void ReadData(Network* n,int num)
{
	int i, j;
	FORFROM0STEP1(i, num)
	{
		FORFROM0STEP1(j, n->iw)
		{
			fscanf(n->dataSet, "%lf", n->x[i] + j);
		}
		FORFROM0STEP1(j, n->ow)
		{
			fscanf(n->dataSet, "%lf", n->y[i] + j);
		}
	}
}