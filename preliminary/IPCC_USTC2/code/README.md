## 优化原代码

### 队伍

鸿雁超算队2 队长：付佳伟

### 代码结构

1. SLIC.h
2. SLIC.cpp

### 运行流程
```
sbatch slurm.sh
cat out.log
```

### 注意事项
1. 其中check.ppm input_image.ppm都是case2的。
2. 如果要测试case1和case3,请自行替换同位置check.ppm input_image.ppm。并将SLIC.cpp里m_spcount的值改成200或者150

