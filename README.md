# Chest X-ray Classifier -- Static Site (runs 100% in the browser)

No backend. No server. No memory limits. No paid tiers. The AI model runs
directly on the visitor's device using ONNX Runtime Web.

## How to deploy on GitHub Pages (free, no card, ever)

### Step 1 -- Get the ONNX model file
In your Colab notebook, after training, add and run this cell:

```python
model.eval()
dummy_input = torch.randn(1, 3, 224, 224).to(device)

torch.onnx.export(
    model,
    dummy_input,
    'chest_xray_model.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    opset_version=13
)

from google.colab import files
files.download('chest_xray_model.onnx')
```

This downloads `chest_xray_model.onnx` to your computer.

### Step 2 -- Put the model file in this folder
Place `chest_xray_model.onnx` directly inside this folder, next to
`index.html` and `script.js`.

### Step 3 -- Push to GitHub
1. Create a new GitHub repository (public)
2. Upload all 3 files: `index.html`, `script.js`, `chest_xray_model.onnx`

### Step 4 -- Enable GitHub Pages
1. In your repo, go to **Settings -> Pages**
2. Under "Source", select **Deploy from a branch**
3. Branch: `main`, folder: `/ (root)`
4. Click **Save**

Wait 1-2 minutes. Your site goes live at:
```
https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/
```

That's it -- a real, public, free-forever website. No server to keep alive,
no memory limits, no sleeping/cold starts. The model runs on each visitor's
own device.

## Important: class order must match training

`CLASS_NAMES` in `script.js` must be in the *exact* same order your model
was trained with. Check the order printed during training
(`get_dataloaders` prints `Classes (in index order): [...]`) and update
this line in `script.js` if it differs:

```js
const CLASS_NAMES = ["COVID", "Lung_Opacity", "Normal", "Viral_Pneumonia"];
```

## Notes on model file size
ONNX exports of ResNet50 are typically 90-100MB. GitHub allows files up to
100MB via normal upload; if yours is right at that edge, use Git LFS
(`git lfs track "*.onnx"`) to be safe.
