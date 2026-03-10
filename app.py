from flask import Flask, render_template, request
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import SimpleCNN, classes

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("trained_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='没有选择文件')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='文件名为空')

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img = Image.open(filepath).convert('RGB')
            img_t = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_t)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                pred_idx = torch.argmax(probabilities).item()
                confidence = probabilities[pred_idx].item() * 100

            # 可选：删除上传的文件，节省空间
            # os.remove(filepath)

            return render_template('index.html',
                                 prediction=classes[pred_idx],
                                 confidence=f"{confidence:.2f}%",
                                 image_filename=file.filename)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)