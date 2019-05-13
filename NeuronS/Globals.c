#include"Globals.h"
double**** new4dDoubleArray(int d, int l, int h, int w)
{
	int i, j,k;
	int t1 = w * sizeof(double);
	int t2 = h * sizeof(double*);
	int t3 = l * sizeof(double***);
	double**** r = MLN(double***, d);
	for (k = 0; k < d; k++)
	{
		r[k] = (double***)malloc(t3);
		for (i = 0; i < l; i++)
		{
			r[k][i] = (double**)malloc(t2);
			for (j = 0; j < h; j++)
			{
				r[k][i][j] = (double*)malloc(t1);
			}
		}
	}
	return r;
}
double*** new3dDoubleArray(int l, int h, int w)
{
	int i, j;
	int t1 = w * sizeof(double);
	int t2 = h * sizeof(double*);
	double*** r = MLN(double**, l);
	for (i = 0; i < l; i++)
	{
		r[i] = (double**)malloc(t2);
		for (j = 0; j < h; j++)
		{
			r[i][j] = (double*)malloc(t1);
		}
	}
	return r;
}
double** new2dDoubleArray(int h, int w)
{
	int i, j;
	int t = w * sizeof(double);
	double** r = MLN(double*, h);
	for (i = 0; i < h; i++)
	{
		r[i] = (double*)malloc(t);
	}
	return r;
}
void destroy2dDoubleArray(double** p, int h)
{
	if (!p) return;
	for (h--; h>=0; h--) FREE(p[h]);
	FREE(p);
}
double*** new3dDoubleArrayFrom1d(int l, int h, int w, int offset, double* p)
{
	int i, j;
	int t2 = h * sizeof(double*);
	double*** r = MLN(double**, l);
	for (i = 0; i < l; i++)
	{
		r[i] = (double**)malloc(t2);
		for (j = 0; j < h; j++)
		{
			r[i][j] = p + offset;
			offset += w;
		}
	}
	return r;
}
void destroy3dDoubleArray(double*** p, int l, int h)
{
	if (!p) return;
	int i, j;
	for (i = 0; i < l; i++)
	{
		for (j = 0; j < h; j++)
		{
			FREE(p[i][j]);
		}
		FREE(p[i]);
	}
	FREE(p);
}
void destroy4dDoubleArray(double**** p, int k, int l, int h)
{
	if (!p) return;
	int i, j,m;
	for (i = 0; i < k; i++)
	{
		for (j = 0; j < l; j++)
		{
			for (m = 0; m < h; m++)
			{
				FREE(p[i][j][m]);
			}
			FREE(p[i][j]);
		}
		FREE(p[i]);
	}
	FREE(p);
}
int argmax(double* a, int l)
{
	int i, mi=0;
	double max = a[0];
	for (i = 1; i < l; i++)
	{
		if (a[i] > max)
		{
			max = a[i];
			mi = i;
		}
	}
	return mi;
}
inline double randomDouble(double absRange)
{
	double r = (rand() & 0xffff)-0x7fff;
	r /= (double)0x8000;
	return r*absRange;
}
inline void assignRandomDoubleArray(double* a, int len, double absrange)
{
	int i;
	FORFROM0STEP1(i, len) a[i] = randomDouble(absrange);
}
inline void assignZeroDoubleArray(double* a, int len)
{
	int i;
	FORFROM0STEP1(i, len)
	{
		a[i] = 0;
	}
}
inline double sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}
inline double dsigmoid(double y)
{
	return y * (1 - y);
}