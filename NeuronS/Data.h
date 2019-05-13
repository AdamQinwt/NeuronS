#ifndef DATA_H
#define DATA_H
typedef struct _Data11D
{
	double* in;
	double* din;
	double* out;
	double* dout;
}Data11D;
typedef struct _Data33D
{
	double*** in;
	double*** din;
	double*** out;
	double*** dout;
}Data33D;
typedef union _Data
{
	Data11D d11;
	Data33D d33;
}Data;
#endif