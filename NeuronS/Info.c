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
	//�ھ����ƽ���ػ��ĳ�ʼ���е��ã��Ӳ�info����pairs����ֵ
	int i0, j0, i, j, m, n;
	for (i = 0, i0 = -(p->conv.ph); i < p->conv.oh; i++)
	{
		for (j = 0, j0 = -(p->conv.pw); j < p->conv.ow; j++)
		{
			for (m = 0; m < p->conv.kh; m++)
			{
				if (i0 + m < 0) continue;
				if (i0 + m > p->conv.ih) break;
				for (n = 0; n < p->conv.kw; n++)
				{
					if (j0 + n < 0) continue;
					if (j0 + n > p->conv.iw) break;
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
	//�����ػ���ʼ���е��ã����ڷ���pairs�ռ�
	int i, j, l;
	for (i = 0; i < p->conv.oh; i++)
	{
		for (j = 0; j < p->conv.ow; j++)
		{
			for (l = 0; l < p->conv.ol; i++)
			{
				AppendToChain4Int(p->conv.pairs[i] + j, 0, 0, 0, 0);
			}
		}
	}
}
void DestroyPairsInInfo(Info* p)
{
	//���д���pairs�Ĳ㶼Ӧ����
	ChainNode4Int* t;
	int i, j;
	for (i = 0; i < p->conv.oh; i++)
	{
		for (j = 0; j < p->conv.ow; j++)
		{
			if (p->conv.pairs[i][j].head)
			{
				t = p->conv.pairs[i][j].head;
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
	}
}