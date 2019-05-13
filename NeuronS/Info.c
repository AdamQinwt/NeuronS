#include"Info.h"
#include"Globals.h"
void AppendToChain4Int(ChainList4Int* list, int _ax, int _ay, int _bx, int _by)
{
	if (list->tail)
	{
		list->tail->next = MLC(ChainNode4Int);
		list->tail = list->tail->next;
	}
	else
	{
		list->head = MLC(ChainNode4Int);
		list->tail = list->head;
	}
	list->tail->ax = _ax;
	list->tail->ay = _ay;
	list->tail->bx = _bx;
	list->tail->by = _by;
	list->tail->next = NULL;
}