#include"Globals.h"
double**** new4dDoubleArray(int d, int l, int h, int w)
{
	int i, j, k;
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
	int i;
	int t = w * sizeof(double);
	double** r = MLN(double*, h);
	for (i = 0; i < h; i++)
	{
		r[i] = (double*)malloc(t);
	}
	return r;
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
void destroy2dDoubleArray(double** p, int h)
{
	if (!p) return;
	int i;
	for(i=0;i<h;i++)
	{
		FREE(p[i]);
		//FREE(p[h]);
	}
	FREE(p);
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
double randomDouble(double absRange)
{
	double r = (rand() & 0xfff)-0x7ff;
	r /= (double)0x800;
	return r*absRange;
}
void assignRandomDoubleArray(double* a, int len, double absrange)
{
	int i;
	FORFROM0STEP1(i, len) a[i] = randomDouble(absrange);
}
void assignZeroDoubleArray(double* a, int len)
{
	int i;
	for(i=0;i<len;i++)
	{
		a[i] = 0;
	}
}
double sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}
double dsigmoid(double y)
{
	return y * (1 - y);
}
void print1dArray(double* a, int w)
{
	int i;
	for (i = 0; i < w; i++) printf("%.3lf\t", a[i]);
	putchar('\n');
}
void print2dArray(double** a, int h, int w)
{
	int i,j;
	for (i = 0; i < h; i++)
	{
		for (j = 0; j < w; j++) printf("%.3lf\t", a[i][j]);
	}
}
void print3dArray(double*** a, int l, int h, int w);
void print4dArray(double**** a, int k, int l, int h, int w);
void write1dArray(FILE* fp,double* a, int w)
{
	fwrite(a, sizeof(double), w, fp);
}
void write2dArray(FILE* fp, double** a, int h, int w)
{
	int i;
	for (i = 0; i < h; i++)
	{
		fwrite(a[i], sizeof(double), w, fp);
	}
}
void write3dArray(FILE* fp, double*** a, int l, int h, int w)
{
	int i,j;
	for (i = 0; i < l; i++)
	{
		for (j = 0; j < h; j++)
		{
			fwrite(a[i][j], sizeof(double), w, fp);
		}
	}
}
void write4dArray(FILE* fp, double**** a, int k, int l, int h, int w)
{
	int i, j,m;
	for (m = 0; m < k; m++)
	{
		for (i = 0; i < l; i++)
		{
			for (j = 0; j < h; j++)
			{
				fwrite(a[m][i][j], sizeof(double), w, fp);
			}
		}
	}
}
void read1dArray(FILE* fp, double* a, int w)
{
	fread(a, sizeof(double), w, fp);
}
void read2dArray(FILE* fp, double** a, int h, int w)
{
	int i;
	for (i = 0; i < h; i++)
	{
		fread(a[i], sizeof(double), w, fp);
	}
}
void read3dArray(FILE* fp, double*** a, int l, int h, int w)
{
	int i, j;
	for (i = 0; i < l; i++)
	{
		for (j = 0; j < h; j++)
		{
			fread(a[i][j], sizeof(double), w, fp);
		}
	}
}
void read4dArray(FILE* fp, double**** a, int k, int l, int h, int w)
{
	int i, j, m;
	for (m = 0; m < k; m++)
	{
		for (i = 0; i < l; i++)
		{
			for (j = 0; j < h; j++)
			{
				fread(a[m][i][j], sizeof(double), w, fp);
			}
		}
	}
}