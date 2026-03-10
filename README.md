# ImageClassifierWeb

一个基于 PyTorch 和 Flask 的图像分类 Web 应用，使用 CIFAR-10 数据集训练的简单 CNN 模型，支持用户上传图片实时分类（飞机、汽车、鸟、猫等 10 类）。

## 技术栈
- **深度学习**：PyTorch + Torchvision
- **Web 框架**：Flask
- **前端**：Bootstrap + HTML
- **部署准备**：gunicorn + Render.com（免费）

## 快速开始（本地运行）

1. 克隆仓库
```bash
git clone https://github.com/xinyulian053-arch/ImageClassifierWeb.git
cd ImageClassifierWeb

2.创建并激活虚拟环境
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

3.安装依赖
pip install -r requirements.txt

4.训练模型（只需运行一次，生成 trained_model.pth)
python model.py

5.启动web应用
python app.py

浏览器访问浏览器访问：http://127.0.0.1:5000
```
## 项目特点
- **端到端实现**：数据加载 → 模型训练 → Web 部署
- **简单但完整的 CNN 结构（Conv → Pool → FC）**
- **支持置信度显示**
- **已配置 .gitignore，避免上传大模型文件和虚拟环境**

## 未来改进方向
- **使用预训练模型（如 ResNet18）提升准确率**
- **添加更多类别或支持自定义数据集**
- **部署到 Render.com / Railway.app（免费层）**
- **添加 Docker 支持**
