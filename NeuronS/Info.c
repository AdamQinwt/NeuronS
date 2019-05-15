#include"Info.h"
#include<stdlib.h>
void AppendToChain4Int(ChainList4Int* list, int _ax, int _ay, int _bx, int _by)
{
	if (list->tail)
	{
		list->tail->next = (ChainNode4Int*)malloc(sizeof(ChainNode4Int));
		list->tail = list->tail->next;
	}
	else
	{
		list->head = (ChainNode4Int*)malloc(sizeof(ChainNode4Int));
		list->tail = list->head;
	}
	list->tail->ax = _ax;
	list->tail->ay = _ay;
	list->tail->bx = _bx;
	list->tail->by = _by;
	list->tail->next = NULL;
}
void SetPairsInInfo(Info* p)
{
	//在卷积和平均池化的初始化中调用，从层info分配pairs并赋值
	int i0, j0, i, j, m, n;
	int t = p->conv.ow * sizeof(ChainNode4Int);
	p->conv.pairs = (ChainList4Int**)malloc(p->conv.oh * sizeof(ChainList4Int*));
	for (i = 0, i0 = -(p->conv.ph); i < p->conv.oh; i++)
	{
		p->conv.pairs[i] = (ChainList4Int*)malloc(t);
		for (j = 0, j0 = -(p->conv.pw); j < p->conv.ow; j++)
		{
			p->conv.pairs[i][j].tail = NULL;
			for (m = 0; m < p->conv.kh; m++)
			{
				if (i0 + m < 0) continue;
				if (i0 + m >= p->conv.ih) break;
				for (n = 0; n < p->conv.kw; n++)
				{
					if (j0 + n < 0) continue;
					if (j0 + n >= p->conv.iw) break;
					AppendToChain4Int(p->conv.pairs[i] + j, j0 + n, i0 + m,n,m );
				}
			}
			j0 += p->conv.sw;
		}
		i0 += p->conv.sh;
	}
}
void AllocPairsInInfo(Info* p)
{
	//在最大池化初始化中调用，用于分配pairs空间
	int i, j, l;
	int t = p->conv.ow * sizeof(ChainList4Int);
	p->conv.pairs = (ChainList4Int**)malloc(p->conv.oh * sizeof(ChainList4Int*));
	for (i = 0; i < p->conv.oh; i++)
	{
		p->conv.pairs[i] = (ChainList4Int*)malloc(t);
		for (j = 0; j < p->conv.ow; j++)
		{
			p->conv.pairs[i][j].head = NULL;
			p->conv.pairs[i][j].tail = NULL;
			for (l = 0; l < p->conv.ol; l++)
			{
				AppendToChain4Int(p->conv.pairs[i] + j, 0, 0, 0, 0);
			}
		}
	}
}
void DestroyPairsInInfo(Info* p)
{
	//所有带有pairs的层都应调用
	ChainNode4Int* t;
	int i, j;
	for (i = 0; i < p->conv.oh; i++)
	{
		for (j = 0; j < p->conv.ow; j++)
		{
			if (p->conv.pairs[i][j].head)
			{
				t = p->conv.pairs[i][j].head->next;
			free_start:
				free(p->conv.pairs[i][j].head);
				p->conv.pairs[i][j].head = t;
				if (t)
				{
					t = t->next;
					goto free_start;
				}
			}
		}
		free(p->conv.pairs[i]);
	}
	free(p->conv.pairs);
}