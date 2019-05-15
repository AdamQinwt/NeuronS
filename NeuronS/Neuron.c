#include"Neuron.h"
#include"Globals.h"
void Connect(Neuron* from, Neuron* to)	//根据维度连接
{
	//判断连接方式
	//连接
	/*
		优先在一维侧分配
		优先在输出端分配
	*/
	if (from->dimension[1] == 3)
	{
		if (to->dimension[0] == 3)
		{
			//3->3
			//输出端分配
			from->needFree[1] = 1;
			to->needFree[0] = 0;
			from->data.d33.out = new3dDoubleArray(from->info.conv.ol, from->info.conv.oh, from->info.conv.ow);
			from->data.d33.dout = new3dDoubleArray(from->info.conv.ol, from->info.conv.oh, from->info.conv.ow);
			to->data.d33.in = from->data.d33.out+from->dataOffset;
			to->data.d33.din = from->data.d33.dout + from->dataOffset;
			from->dataOffset += from->info.conv.ol;
		}
		else
		{
			//3->1
			//输入端分配
			from->needFree[1] = 0;
			to->needFree[0] = 1;
			to->data.d11.in = MLN(double, to->info.fc.in);
			to->data.d11.din = MLN(double, to->info.fc.in);
			from->data.d33.out = new3dDoubleArrayFrom1d(from->info.conv.ol, from->info.conv.oh, from->info.conv.ow,0,to->data.d11.in);
			from->data.d33.dout = new3dDoubleArrayFrom1d(from->info.conv.ol, from->info.conv.oh, from->info.conv.ow,0,to->data.d11.din);
		}
	}
	else
	{
		//1->1
		//输入端分配
		from->needFree[1] = 1;
		to->needFree[0] = 0;
		from->data.d11.out = MLN(double, from->info.fc.out);
		from->data.d11.dout = MLN(double, from->info.fc.out);
		to->data.d11.in = from->data.d11.out;
		to->data.d11.din = from->data.d11.dout;
	}
}