## Utilility Functions for 3D Point Cloud Deep Learning

## visualization tool
```shell
sh compile_render_balls_so.sh
python show3d_balls.py
```
### Tips

  **keyboard shortcut:**

  |**Shortcut**| **Function**|
  |:--:|:--|
  |`g`|Show Ground Truth|
  |`p`|Show Prediction Labels|
  |`r`|Show Residual|
  |`n`|Zoom in|
  |`m`|Zoom out|
  |`r`|Reset|
  |`s`|Save image|
  |`q`|Exit|
   
## Samples extend tool
```shell
sh compile_render_balls_so.sh
python sample_extend_mouse.py
```
### Tips

  **keyboard shortcut:**

  |**Shortcut**| **Function**|
  |:--:|:--|
  |`e`|Exit rect part|
  |`g`|Show Ground Truth|
  |`p`|Show Prediction Labels|
  |`r`|Show Residual|
  |`n`|Zoom in|
  |`m`|Zoom out|
  |`r`|Reset|
  |`s`|Save image|
  |`q`|Quit|

### Update
1. **配置**：
    主函数`line 317,318`：
    `root_dir` 点云文件和类别文件目录
    `save_dir` 扩充样本保存目录
    `pred_root` 预测文件，如果有 predict=True, 否则predict=False, pred_root可以为空


2. **标注流程**
    程序运行后，可使用`g` GT,`p` Pred, `r` Res查看真实类别，预测值和差别
    按`d`进入画图程序,每次框选后，`s`保存;
    按`c`清除上一次框选结果，继续框选
    按`e`保存上一次框选结果并退出;
    按`d`不保存上一次结果并退出
    按`n`标注下一张图像;
    按`q`退出程序，停止标注


3. **注意事项**
    标注时不需要每次标注所有样本，每次标注时会显示当前标注的图像的序号，
    标注后记住序号，下次可以继续标
    可在主函数`line 323` `for`循环 选择标注序号
    扩充样本点云文件命名方式为 样本名+`_PC3_extend.txt`
    扩充样本类别文件命名方式为 样本名+`_CLS_extend.txt`
    可在 `render`函数中`line 203 204` `file_path ``label_path`修改