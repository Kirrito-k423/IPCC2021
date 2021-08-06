/*
 * @Descripttion: 
 * @version: 
 * @Author: Shaojie Tan
 * @Date: 2021-08-05 13:46:15
 * @LastEditors: Shaojie Tan
 * @LastEditTime: 2021-08-05 14:26:11
 */

#include <cstdio>
#include <cstdlib>
 
class _ppm
{
private:
	int pX;								//类型
	int sizeX,sizeY;					//大小
	unsigned char maxColor;				//颜色范围
	unsigned char *image;				//保存像素
 
public:
	~_ppm ()
	{
		if (image != NULL)
			free (image);
		image = NULL;
	}
	void write_image (char *ch);
	void creat_image ();
};
 
 
void _ppm::write_image (char *ch)
{//图片输出路径
	FILE *fp = fopen (ch, "wb");
	fprintf (fp, "P%d\n%d %d\n%d\n", pX, sizeX, sizeY, maxColor);
	if (pX == 5)
	{
		fwrite (image, sizeof (unsigned char), sizeX * sizeY, fp);
	}
	else if (pX == 6)
	{
		fwrite (image, sizeof (unsigned char), sizeX * sizeY * 3, fp);
	}
	fclose (fp);
	fp = NULL;
}
 
void _ppm::creat_image ()
{//构造一张图片
	pX = 6;	//5 是黑白图片
	sizeX = 2560;//长度
	sizeY = 34000;//宽度
	maxColor = 255;//色彩范围
	image = (unsigned char *) calloc (sizeX * sizeY * 3, sizeof (unsigned char));
	for (int i = 0; i < sizeY; i++)
	{
		for (int j = 0; j < sizeX; j++)
		{
			image[3 * i * sizeX + j] = j%maxColor;
			image[3 * i * sizeX + j+1] = (j+1)%maxColor;
			image[3 * i * sizeX + j+2] = (j+2)%maxColor;
		}
	}
}
 
int main ()
{
	_ppm img;
	img.creat_image();
	img.write_image("./homework1.ppm");
	return 0;
}

// ————————————————
// 版权声明：本文为CSDN博主「WellerZhao」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
// 原文链接：https://blog.csdn.net/wellerzhao/article/details/11912739