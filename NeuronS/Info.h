#ifndef INFO_H
#define INFO_H
typedef struct _ChainNode4Int
{
	int ax, ay;
	int bx, by;
	struct _ChainNode4Int* next;
}ChainNode4Int;
typedef struct _ChainList4Int
{
	ChainNode4Int* head;
	ChainNode4Int* tail;
}ChainList4Int;
void AppendToChain4Int(ChainList4Int* list,int _ax, int _ay, int _bx, int _by);
typedef struct _FC_Info
{
	int in, out;
}FC_Info;
typedef struct _Conv_Info
{
	int il, iw, ih;
	int ol, ow, oh;
	int sw, sh;	//stride
	int kw, kh;	//kernel
	int pw, ph;	//padding(each side)
	//关于点对的链表数组
	ChainList4Int** pairs;
}Conv_Info;
typedef union _Info
{
	FC_Info fc;
	Conv_Info conv;
}Info;
#endif