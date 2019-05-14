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