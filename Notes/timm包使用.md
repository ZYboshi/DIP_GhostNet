# timm使用
> 导入方式： timm / torchvision  ->timm  方便比较ghostnet

## 查看可用接口
torchvision直接查看源码：ctrl+MB1，点击models.mobilenet_v3_small
```python
import torchvision.models as models

model = models.mobilenet_v3_small(pretrained=False)
```

## 模型结构
```python
model = models.mobilenet_v3_small(pretrained=False)
print("完整的 classifier 结构:")
print(model.classifier)
```
Sequential(
  (0): Linear(in_features=576, out_features=1024, bias=True)
  (1): Hardswish()
  (1): Hardswish()
  (2): Dropout(p=0.2, inplace=True)
  (3): Linear(in_features=1024, out_features=1000, bias=True)
)


## 通过timm获取模型方式
`model = timm.create_model(model_name,pretrained,numclasses)`




