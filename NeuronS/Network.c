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
			CONTINUE_IF_NEAR_ZERO(n->arg.fc.delta.bias[k]);
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
			CONTINUE_IF_NEAR_ZERO(n->arg.fc.delta.bias[k]);
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
	int i, j, k, l;
	if (n->count == 1)
	{
		FORFROM0STEP1(k, n->info.conv.ol)
		{
			n->arg.conv.shadow.bias[k] += n->arg.conv.grad.bias[k] * n->arg.conv.grad.bias[k];
			n->arg.conv.delta.bias[k] = -n->arg.conv.grad.bias[k] * (n->learningRate / (delta + sqrt(n->arg.conv.shadow.bias[k])));
			n->arg.conv.original.bias[k] += n->arg.conv.delta.bias[k];
			n->arg.conv.grad.bias[k] = 0;
			CONTINUE_IF_NEAR_ZERO(n->arg.fc.delta.bias[k]);
			FORFROM0STEP1(l, n->info.conv.il)
			{
				FORFROM0STEP1(i, n->info.conv.kh)
				{
					FORFROM0STEP1(j, n->info.conv.kw)
					{
						n->arg.conv.shadow.weight[k][l][i][j] += n->arg.conv.grad.weight[k][l][i][j] * n->arg.conv.grad.weight[k][l][i][j];
						n->arg.conv.delta.weight[k][l][i][j] = -n->arg.conv.grad.weight[k][l][i][j] * (n->learningRate / (delta + sqrt(n->arg.conv.shadow.weight[k][l][i][j])));
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
			n->arg.conv.grad.bias[k] /= n->count;
			n->arg.conv.shadow.bias[k] += n->arg.conv.grad.bias[k] * n->arg.conv.grad.bias[k];
			n->arg.conv.delta.bias[k] = -n->arg.conv.grad.bias[k] * (n->learningRate / (delta + sqrt(n->arg.conv.shadow.bias[k])));
			n->arg.conv.original.bias[k] += n->arg.conv.delta.bias[k];
			n->arg.conv.grad.bias[k] = 0;
			CONTINUE_IF_NEAR_ZERO(n->arg.fc.delta.bias[k]);
			FORFROM0STEP1(l, n->info.conv.il)
			{
				FORFROM0STEP1(i, n->info.conv.kh)
				{
					FORFROM0STEP1(j, n->info.conv.kw)
					{
						n->arg.conv.grad.weight[k][l][i][j] /= n->count;
						n->arg.conv.shadow.weight[k][l][i][j] += n->arg.conv.grad.weight[k][l][i][j] * n->arg.conv.grad.weight[k][l][i][j];
						n->arg.conv.delta.weight[k][l][i][j] = -n->arg.conv.grad.weight[k][l][i][j] * (n->learningRate / (delta + sqrt(n->arg.conv.shadow.weight[k][l][i][j])));
						n->arg.conv.original.weight[k][l][i][j] += n->arg.conv.delta.weight[k][l][i][j];
						n->arg.conv.grad.weight[k][l][i][j] = 0;
					}
				}
			}
		}
	}
}
void AdaGrad_Optimizer(Network* n)
{
	//小常数delta定为n中的extraArg[ADAGRAD]
	//全局学习率epsilon定为n中的learningRate
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
			n->arg.fc.shadow.bias[j] = ro* n->arg.fc.shadow.bias[j]+(1-ro)*n->arg.fc.grad.bias[j] * n->arg.fc.grad.bias[j];
			n->arg.fc.delta.bias[j] = -n->arg.fc.grad.bias[j] * n->learningRate / sqrt(delta + n->arg.fc.shadow.bias[j]);
			n->arg.fc.original.bias[j] += n->arg.fc.delta.bias[j];
			n->arg.fc.grad.bias[j] = 0;
			CONTINUE_IF_NEAR_ZERO(n->arg.fc.delta.bias[j]);
			FORFROM0STEP1(i, n->info.fc.in)
			{
				n->arg.fc.shadow.weight[i][j] = ro * n->arg.fc.shadow.weight[i][j] + (1 - ro)*n->arg.fc.grad.weight[i][j] * n->arg.fc.grad.weight[i][j];
				n->arg.fc.delta.weight[i][j] = -n->arg.fc.grad.weight[i][j] * n->learningRate / sqrt(delta + n->arg.fc.shadow.weight[i][j]);
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
	int i, j, k, l;
	if (n->count == 1)
	{
		FORFROM0STEP1(k, n->info.conv.ol)
		{
			n->arg.conv.shadow.bias[k] = ro* n->arg.conv.shadow.bias[k]+(1-ro)*n->arg.conv.grad.bias[k] * n->arg.conv.grad.bias[k];
			n->arg.conv.delta.bias[k] = -n->arg.conv.grad.bias[k] * n->learningRate / sqrt(delta + n->arg.conv.shadow.bias[k]);
			n->arg.conv.original.bias[k] += n->arg.conv.delta.bias[k];
			n->arg.conv.grad.bias[k] = 0;
			//CONTINUE_IF_NEAR_ZERO(n->arg.fc.delta.bias[k]);
			FORFROM0STEP1(l, n->info.conv.il)
			{
				FORFROM0STEP1(i, n->info.conv.kh)
				{
					FORFROM0STEP1(j, n->info.conv.kw)
					{
						n->arg.conv.shadow.weight[k][l][i][j] = ro * n->arg.conv.shadow.weight[k][l][i][j] + (1 - ro)*n->arg.conv.grad.weight[k][l][i][j] * n->arg.conv.grad.weight[k][l][i][j];
						n->arg.conv.delta.weight[k][l][i][j] = -n->arg.conv.grad.weight[k][l][i][j] * n->learningRate / sqrt(delta + n->arg.conv.shadow.weight[k][l][i][j]);
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
			n->arg.conv.shadow.bias[k] /= n->count;
			n->arg.conv.shadow.bias[k] = ro * n->arg.conv.shadow.bias[k] + (1 - ro)*n->arg.conv.grad.bias[k] * n->arg.conv.grad.bias[k];
			n->arg.conv.delta.bias[k] = -n->arg.conv.grad.bias[k] * n->learningRate / sqrt(delta + n->arg.conv.shadow.bias[k]);
			n->arg.conv.original.bias[k] += n->arg.conv.delta.bias[k];
			n->arg.conv.grad.bias[k] = 0;
			//CONTINUE_IF_NEAR_ZERO(n->arg.fc.delta.bias[k]);
			FORFROM0STEP1(l, n->info.conv.il)
			{
				FORFROM0STEP1(i, n->info.conv.kh)
				{
					FORFROM0STEP1(j, n->info.conv.kw)
					{
						n->arg.conv.grad.weight[k][l][i][j] /= n->count;
						n->arg.conv.shadow.weight[k][l][i][j] = ro * n->arg.conv.shadow.weight[k][l][i][j] + (1 - ro)*n->arg.conv.grad.weight[k][l][i][j] * n->arg.conv.grad.weight[k][l][i][j];
						n->arg.conv.delta.weight[k][l][i][j] = -n->arg.conv.grad.weight[k][l][i][j] * n->learningRate / sqrt(delta + n->arg.conv.shadow.weight[k][l][i][j]);
						n->arg.conv.original.weight[k][l][i][j] += n->arg.conv.delta.weight[k][l][i][j];
						n->arg.conv.grad.weight[k][l][i][j] = 0;
					}
				}
			}
		}
	}
}
void RMSProp_Optimizer(Network* n)
{
	//小常数delta定为n中的extraArg[ADAGRAD]
	//全局学习率epsilon定为n中的learningRate
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
	//改变结构，加入一个参数的保留影子（用于一阶矩偏差s）此处为n->extraArg[0]
	//二阶r存入shadow中
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
	int i, j, k, l;
	if (n->count == 1)
	{
		FORFROM0STEP1(k, n->info.conv.ol)
		{
			n->arg.conv.shadow.bias[k] = ro2 * n->arg.conv.shadow.bias[k] + (1 - ro2)*n->arg.conv.grad.bias[k] * n->arg.conv.grad.bias[k];
			n->extraArg[0].conv.bias[k] = ro1 * n->extraArg[0].conv.bias[k] + (1 - ro1)*n->arg.conv.grad.bias[k];
			n->arg.conv.delta.bias[k] = -epsilon * n->extraArg[0].conv.bias[k] / (1 - ro1t) / (sqrt(n->arg.conv.shadow.bias[k] / (1 - ro2t)) + delta);
			n->arg.conv.original.bias[k] += n->arg.conv.delta.bias[k];
			n->arg.conv.grad.bias[k] = 0;
			//CONTINUE_IF_NEAR_ZERO(n->arg.fc.delta.bias[k]);
			FORFROM0STEP1(l, n->info.conv.il)
			{
				FORFROM0STEP1(i, n->info.conv.kh)
				{
					FORFROM0STEP1(j, n->info.conv.kw)
					{
						n->arg.conv.shadow.weight[k][l][i][j] = ro2 * n->arg.conv.shadow.weight[k][l][i][j] + (1 - ro2)*n->arg.conv.grad.bias[k] * n->arg.conv.grad.weight[k][l][i][j];
						n->extraArg[0].conv.weight[k][l][i][j] = ro1 * n->extraArg[0].conv.weight[k][l][i][j] + (1 - ro1)*n->arg.conv.grad.weight[k][l][i][j];
						n->arg.conv.delta.weight[k][l][i][j] = -epsilon * n->extraArg[0].conv.weight[k][l][i][j] / (1 - ro1t) / (sqrt(n->arg.conv.shadow.weight[k][l][i][j] / (1 - ro2t)) + delta);
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
			n->arg.conv.shadow.bias[k] /= n->count;
			n->arg.conv.shadow.bias[k] = ro2 * n->arg.conv.shadow.bias[k] + (1 - ro2)*n->arg.conv.grad.bias[k] * n->arg.conv.grad.bias[k];
			n->extraArg[0].conv.bias[k] = ro1 * n->extraArg[0].conv.bias[k] + (1 - ro1)*n->arg.conv.grad.bias[k];
			n->arg.conv.delta.bias[k] = -epsilon * n->extraArg[0].conv.bias[k] / (1 - ro1t) / (sqrt(n->arg.conv.shadow.bias[k] / (1 - ro2t)) + delta);
			n->arg.conv.original.bias[k] += n->arg.conv.delta.bias[k];
			n->arg.conv.grad.bias[k] = 0;
			//CONTINUE_IF_NEAR_ZERO(n->arg.fc.delta.bias[k]);
			FORFROM0STEP1(l, n->info.conv.il)
			{
				FORFROM0STEP1(i, n->info.conv.kh)
				{
					FORFROM0STEP1(j, n->info.conv.kw)
					{
						n->arg.conv.shadow.weight[k][l][i][j] /= n->count;
						n->arg.conv.shadow.weight[k][l][i][j] = ro2 * n->arg.conv.shadow.weight[k][l][i][j] + (1 - ro2)*n->arg.conv.grad.bias[k] * n->arg.conv.grad.weight[k][l][i][j];
						n->extraArg[0].conv.weight[k][l][i][j] = ro1 * n->extraArg[0].conv.weight[k][l][i][j] + (1 - ro1)*n->arg.conv.grad.weight[k][l][i][j];
						n->arg.conv.delta.weight[k][l][i][j] = -epsilon * n->extraArg[0].conv.weight[k][l][i][j] / (1 - ro1t) / (sqrt(n->arg.conv.shadow.weight[k][l][i][j] / (1 - ro2t)) + delta);
						n->arg.conv.original.weight[k][l][i][j] += n->arg.conv.delta.weight[k][l][i][j];
						n->arg.conv.grad.weight[k][l][i][j] = 0;
					}
				}
			}
		}
	}
}
void Adam_Optimizer(Network* n)
{
	//小常数delta定为n中的extraArg[ADAM_DELTA]
	//指数衰减率分别定为n中的extraArg[ADAM_RO1],extraArg[ADAM_RO2]
	//全局学习率epsilon定为n中的extraArg[ADAM_EPSILON]
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
		//print1dArray(n->arg.fc.original.weight[i], n->info.fc.out);
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
			FORFROM0STEP1(j, n->info.conv.kh)
			{
				assignRandomDoubleArray(n->arg.conv.original.weight[k][i][j], n->info.conv.kw, absRange);
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
		case CONV:SetConv(n->neurons + i, 0.3, "relu",n->needAlloc); break;
		case MAX_POOL:SetMaxPool(n->neurons + i); break;
		case AVERAGE_POOL:SetAveragePool(n->neurons + i); break;
		case SOFTMAX:SetSoftmax(n->neurons + i); break;
		default:
			break;
		}
	}
	//反向输入空间连接，输出连接，输出偏差连接

	//n->dx = MLD(n->neurons[0].info.fc.in);
	n->dx = new3dDoubleArray(n->il, n->ih, n->iw);
	//n->out = MLD(last->info.fc.out);
	n->out = new3dDoubleArray(n->ol, n->oh, n->ow);

	//n->neurons[0].data.d11.din = n->dx;
	if(n->neurons[0].dimension[0]==1) n->neurons[0].data.d11.din = n->dx[0][0];
	else n->neurons[0].data.d33.din = n->dx;

	Neuron* last = n->neurons+n->number - 1;
	if (last->dimension[1] == 1)
	{
		last->data.d11.out = n->out[0][0];
		last->data.d11.dout = MLD(last->info.fc.out);
	}
	else
	{
		last->data.d33.out = n->out;
		last->data.d33.dout = new3dDoubleArray(n->ol, n->oh, n->ow);
	}
	last->needFree[1] = 0;
}
void Dtor(Network* n)
{
	//输入释放，反向输入空间释放，输出释放，输出偏差释放
	destroy4dDoubleArray(n->x, n->batch,n->il, n->ih);
	destroy4dDoubleArray(n->y, n->batch,n->ol, n->oh);
	destroy3dDoubleArray(n->dx, n->il, n->ih);
	destroy3dDoubleArray(n->out, n->ol, n->oh);
	//FREE(n->neurons[0].data.d11.din);
	Neuron* last = n->neurons + n->number - 1;
	//FREE(n->out);
	if (last->dimension[1] == 1)
	{
		FREE(last->data.d11.dout);
	}
	else destroy3dDoubleArray(last->data.d33.dout, n->ol, n->oh);
	int i;
	FORFROM0STEP1(i, n->number)
	{
		switch (n->neurons[i].type)
		{
		case FC:DestroyFC(n->neurons + i); break;
		case CONV:DestroyConv(n->neurons + i); break;
		case MAX_POOL:DestroyMaxPool(n->neurons + i); break;
		case AVERAGE_POOL:DestroyAveragePool(n->neurons + i); break;
		case SOFTMAX:DestroySoftmax(n->neurons + i); break;
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
	//开辟输入输出空间、答案空间（先在外界完成）
	return n;
}
int train(Network* n, int count, double thresh, FILE* flog)
{
	int b,num,rem;
	//初始化过程
	//读入数据文件头
	ReadHeader(n);
	if (n->batchCount == 1)
	{
		ReadData(n,n->batch);
		FORFROM0STEP1(n->trainingStep, count)
		{
			ResetLosses(n);
			FORFROM0STEP1(b, n->batch)
			{
				//设置输入
				if(n->neurons[0].dimension[0]==1) n->neurons[0].data.d11.in = n->x[b][0][0];
				else n->neurons[0].data.d33.in = n->x[b];
				//前向
				run(n);
				//print1dArray(n->out, n->ow);
				//计算误差
				n->Loss(n, b);
				n->Dloss(n, b);
				//后向
				bp(n);
			}
			//优化
			n->Optimizer(n);
			//输出结果
			//if(!(n->trainingStep&0xff)) 
			fprintf(flog, "iter:\t%d\tloss=%lf\n", n->trainingStep, n->loss);
			//判断误差
			if (n->loss < thresh)
			{
				//完成
				fprintf(flog, "Complete!\n");
				break;
			}
		}
		FORFROM0STEP1(b, n->batch)
		{
			//设置输入
			if (n->neurons[0].dimension[0] == 1) n->neurons[0].data.d11.in = n->x[b][0][0];
			else n->neurons[0].data.d33.in = n->x[b];
			//前向
			run(n);
			if (n->neurons[n->number - 1].dimension[1] == 1)
			{
				print1dArray(flog,n->out[0][0], n->ow);
				print1dArray(flog, n->y[b][0][0], n->ow);
			}
			else
			{
				print3dArray(flog, n->out, n->ol,n->oh,n->ow);
				print3dArray(flog, n->y[b], n->ol, n->oh, n->ow);
			}
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
					//设置输入
					if (n->neurons[0].dimension[0] == 1) n->neurons[0].data.d11.in = n->x[b][0][0];
					else n->neurons[0].data.d33.in = n->x[b];
					//前向
					run(n);
					//print1dArray(n->out, n->ow);
					//计算误差
					n->Loss(n, b);
					n->Dloss(n, b);
					//后向
					bp(n);
				}
				//优化
				n->Optimizer(n);
				//输出结果
				//if(!(n->trainingStep&0xff)) 
				fprintf(flog, "iter:\t%d\tloss=%lf\n", n->trainingStep, n->loss);
				//判断误差
				if (n->loss < thresh)
				{
					//完成
					fprintf(flog, "Complete!\n");
					break;
				}
			}
		}
	}
	return n->trainingStep;
}
int test(Network* n, FILE* flog)
{
	int b, num, rem;
	int correct=0;
	int c;
	int ans, actual;
	int* cntAns=MLN(int,n->ow);
	memset(cntAns, 0, n->ow * sizeof(int));
	int* cntCorrect = MLN(int, n->ow);
	memset(cntCorrect, 0, n->ow * sizeof(int));
	//初始化过程
	//读入数据文件头
	c=ReadHeader(n);
	printf("c=%d\n", c);
	if (n->batchCount == 1)
	{
		ReadData(n, n->batch);
		ResetLosses(n);
		FORFROM0STEP1(b, n->batch)
		{
			//设置输入
			if (n->neurons[0].dimension[0] == 1) n->neurons[0].data.d11.in = n->x[b][0][0];
			else n->neurons[0].data.d33.in = n->x[b];
			//前向
			run(n);
			if (n->neurons[n->number - 1].dimension[1] == 1)
			{
				print1dArray(flog, n->out[0][0], n->ow);
				actual = argmax(n->out[0][0], n->ow);
				ans = argmax(n->y[b][0][0], n->ow);
				cntAns[ans]++;
				fprintf(flog, "Output is %d.\n Answer is %d.\n", actual, ans);
				if (ans == actual)
				{
					fputs("correct\n", flog);
					cntCorrect[actual]++;
					correct++;
				}
				else
				{
					fputs("wrong\n", flog);
				}
				//print1dArray(n->y[b][0][0], n->ow);
			}
			else
			{
				print3dArray(flog, n->out, n->ol, n->oh, n->ow);
				print3dArray(flog, n->y[b], n->ol, n->oh, n->ow);
			}
		}
	}
	else
	{
		FORFROM0STEP1(num, n->batchCount)
		{
			rem = num == n->batchCount - 1 ? n->batchRemainder : n->batch;
			ReadData(n, rem);
			ResetLosses(n);
			FORFROM0STEP1(b, rem)
			{
				//设置输入
				if (n->neurons[0].dimension[0] == 1) n->neurons[0].data.d11.in = n->x[b][0][0];
				else n->neurons[0].data.d33.in = n->x[b];
				//前向
				run(n);
				if (n->neurons[n->number - 1].dimension[1] == 1)
				{
					print1dArray(flog, n->out[0][0], n->ow);
					actual = argmax(n->out[0][0], n->ow);
					ans = argmax(n->y[b][0][0], n->ow);
					cntAns[ans]++;
					fprintf(flog, "Output is %d.\n Answer is %d.\n", actual, ans);
					if (ans == actual)
					{
						fputs("correct\n", flog);
						cntCorrect[actual]++;
						correct++;
					}
					else
					{
						fputs("wrong\n", flog);
					}
					//print1dArray(n->y[b][0][0], n->ow);
				}
				else
				{
					print3dArray(flog, n->out, n->ol, n->oh, n->ow);
					print3dArray(flog, n->y[b], n->ol, n->oh, n->ow);
				}
			}
		}
	}
	fprintf(flog, "%d inputs tested.\n%d correct.\nAccuracy is %.2lf%%\n", c,correct, 100*((double)correct) / c);
	FORFROM0STEP1(b, n->ow)
	{
		fprintf(flog, "%d inputs for answer=%d tested.\n%d correct.\nAccuracy is %.2lf%%\n", cntAns[b], b,cntCorrect[b], 100 * ((double)cntCorrect[b]) / cntAns[b]);
	}
	free(cntAns);
	free(cntCorrect);
	return correct;
}
void none(Network* n) 
{
	//do nothing
}
double SquareLoss(struct _Network* n,int indx)
{
	double tmp,sum=0;
	int i, j, l;
	if (n->neurons[n->number - 1].dimension[1] == 1)
	{
		FORFROM0STEP1(i, n->ow)
		{
			tmp = n->out[0][0][i] - n->y[indx][0][0][i];
			CONTINUE_IF_NEAR_ZERO(tmp);
			sum += tmp * tmp;
		}
	}
	else
	{
		FORFROM0STEP1(l, n->ol)
		{
			FORFROM0STEP1(i, n->oh)
			{
				FORFROM0STEP1(j, n->ow)
				{
					tmp = n->out[l][i][j] - n->y[indx][l][i][j];
					CONTINUE_IF_NEAR_ZERO(tmp);
					sum += tmp * tmp;
				}
			}
		}
	}
	n->loss += sum / n->batch;
	return n->loss;
}
double CrossEntropyLoss(struct _Network* n, int indx)
{
	double sum = 0;
	int i, j, l;
	if (n->neurons[n->number - 1].dimension[1] == 1)
	{
		FORFROM0STEP1(i, n->ow)
		{
			//	CONTINUE_IF_NEAR_ZERO(n->y[indx][i]);
			sum -= n->y[indx][0][0][i] * log(n->out[0][0][i]);
			sum -= (1 - n->y[indx][0][0][i]) * log(1 - n->out[0][0][i]);
		}
	}
	else
	{
		FORFROM0STEP1(l, n->ol)
		{
			FORFROM0STEP1(i, n->oh)
			{
				FORFROM0STEP1(j, n->ow)
				{
					sum -= n->y[indx][l][i][j] * log(n->out[l][i][j]);
					sum -= (1 - n->y[indx][l][i][j]) * log(1 - n->out[l][i][j]);
				}
			}
		}
	}
	n->loss += sum / n->batch;
	return n->loss;
}
void DSquareloss(struct _Network* n, int indx)
{
	double d = ((double)2) / n->batch;
	int i, j, l;
	if (n->neurons[n->number - 1].dimension[1] == 1)
	{
		FORFROM0STEP1(i, n->ow)
		{
			//	CONTINUE_IF_NEAR_ZERO(n->y[indx][i]);
			n->neurons[n->number - 1].data.d11.dout[i] = d * (n->out[0][0][i] - n->y[indx][0][0][i]);
		}
	}
	else
	{
		FORFROM0STEP1(l, n->ol)
		{
			FORFROM0STEP1(i, n->oh)
			{
				FORFROM0STEP1(j, n->ow)
				{
					//	CONTINUE_IF_NEAR_ZERO(n->y[indx][i]);
					n->neurons[n->number - 1].data.d33.dout[l][i][j] = d * (n->out[l][i][j] - n->y[indx][l][i][j]);
				}
			}
		}
	}
}
void DCrossEntropyLoss(struct _Network* n, int indx)
{
	int i,j,l;
	if (n->neurons[n->number - 1].dimension[1] == 1)
	{
		FORFROM0STEP1(i, n->ow)
		{
			//	CONTINUE_IF_NEAR_ZERO(n->y[indx][i]);
			n->neurons[n->number - 1].data.d11.dout[i] = -(n->y[indx][0][0][i] / n->out[0][0][i] + (n->y[indx][0][0][i] - 1) / (1 - n->out[0][0][i])) / n->batch;
		}
	}
	else
	{
		FORFROM0STEP1(l, n->ol)
		{
			FORFROM0STEP1(i, n->oh)
			{
				FORFROM0STEP1(j, n->ow)
				{
					//	CONTINUE_IF_NEAR_ZERO(n->y[indx][i]);
					n->neurons[n->number - 1].data.d33.dout[l][i][j] = -(n->y[indx][l][i][j] / n->out[l][i][j] + (n->y[indx][l][i][j] - 1) / (1 - n->out[l][i][j])) / n->batch;
				}
			}
		}
	}
}
void ResetLosses(Network* n)
{
	n->loss = 0;
	int i, j, l;
	//double d = ((double)2) / n->batch;
	if (n->neurons[n->number - 1].dimension[1] == 1)
	{
		FORFROM0STEP1(i, n->ow)
		{
			//	CONTINUE_IF_NEAR_ZERO(n->y[indx][i]);
			n->neurons[n->number - 1].data.d11.dout[i] = 0;
		}
	}
	else
	{
		FORFROM0STEP1(l, n->ol)
		{
			FORFROM0STEP1(i, n->oh)
			{
				FORFROM0STEP1(j, n->ow)
				{
					//	CONTINUE_IF_NEAR_ZERO(n->y[indx][i]);
					n->neurons[n->number - 1].data.d33.dout[l][i][j] = 0;
				}
			}
		}
	}
}
void InitArgs(Network* n)
{
	//尝试读文件
	char fname[50];
	sprintf(fname,"%s/arg", n->name);
	FILE* fp = fopen(fname,"rb");
	if (fp) ReadArgs(n, fp);
	//若文件为空，调用初始化函数
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
	fclose(fp);
}
int ReadHeader(Network* n)
{
	//数据文件为文本文件
	//由调用者预先指定batch或batchCount，ReadData函数只负责分配输入输出空间并读入数据
	//数据文件第一行为一个整数开头，标识数据集大小
	//数据文件分隔符可为 ' ',',','\t','\r','\n'
	int cnt;	//数据集大小
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
	//分配空间
	//一维情况
	if (!(n->x)) n->x = new4dDoubleArray(n->batch, n->il, n->ih, n->iw);
	if (!(n->y)) n->y = new4dDoubleArray(n->batch, n->ol, n->oh, n->ow);
	return cnt;
}
void ReadData(Network* n,int num)
{
	int i, j,l,d;
	FORFROM0STEP1(d, num)
	{
		FORFROM0STEP1(l, n->il)
		{
			FORFROM0STEP1(i, n->ih)
			{
				FORFROM0STEP1(j, n->iw)
				{
					fscanf(n->dataSet, "%lf", n->x[d][l][i] + j);
					//printf("%.3lf\t", n->x[d][l][i][j]);
				}
				//putchar('\n');
			}
		}
		//PS;
		FORFROM0STEP1(l, n->ol)
		{
			FORFROM0STEP1(i, n->oh)
			{
				FORFROM0STEP1(j, n->ow)
				{
					fscanf(n->dataSet, "%lf", n->y[d][l][i] + j);
					//printf("%.3lf\t", n->y[d][l][i][j]);
				}
				//putchar('\n');
				//PS;
			}
		}
	}
}