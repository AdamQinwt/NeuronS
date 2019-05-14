#include"Network.h"
main()
{
	srand(time(NULL));
	double** x = new2dDoubleArray(4, 2);
	double** y = new2dDoubleArray(4, 1);
	x[0][0] = 0;
	x[0][1] = 0;
	x[1][0] = 0;
	x[1][1] = 1;
	x[2][0] = 1;
	x[2][1] = 0;
	x[3][0] = 1;
	x[3][1] = 1;
	y[0][0] = 0;
	y[1][0] = 1;
	y[2][0] = 1;
	y[3][0] = 0;
	Network* p = newNetwork("xor",2, 4, 1, 1, 2, 1, 1, 1);
	p->neurons[0].type = FC;
	p->neurons[1].type = FC;
	p->x = x;
	p->y = y;
	p->neurons[0].info.fc.in = 2;
	p->neurons[0].info.fc.out = 3;
	p->neurons[1].info.fc.in = 3;
	p->neurons[1].info.fc.out = 1;
	Set(p);
	InitArgs(p);
	Connect(p->neurons, p->neurons+1);
	train(p, 100, 0.01, stdout);
	Dtor(p);
	PS;
}