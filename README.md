# BME FYP

<a href="#0.">0. Quick start</a>

<a href="#1.">1. Packages management</a>

<a href="#2.">2. Select python.exe in VSCode</a>

<a href="#3.">3. Update submodules</a>

<a href="#4.">4. Networks</a>

---

<h1 id="0.">0. Quick start</h1>

<h2>0.0. Set up</h2>

(NOTE: Tested with python 3.9.10)

```cmd
git submodule update --init --recursive 

python -m venv dixon_fyp_venv
dixon_fyp_venv\Scripts\activate

pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -r requirements.txt
```

<h2>0.1. Training</h2>

0. Provide data in `data` as follow:
    ```
    data\
    ├── test\
    │   ├── gts\
    │   │   ├── 1001.png
    │   │   ├── 1002.png
    │   │   └── . . .
    │   └── images\
    │       ├── 1001.png
    │       ├── 1002.png
    │       └── . . .
    ├── train\
    │   ├── gts\
    │   │   ├── 1003.png
    │   │   ├── 1004.png
    │   │   └── . . .
    │   └── images\
    │       ├── 1003.png
    │       ├── 1004.png
    │       └── . . .
    └── valid\
        ├── gts\
        │   ├── 1005.png
        │   ├── 1006.png
        │   └── . . .
        └── images\
            ├── 1005.png
            ├── 1006.png
            └── . . .
    ```
1. Run `pre_train.ipynb` to choose configurations for training

2. Copy `helper_pytorch/dataset.py` to direcotry's root, than modify it for the project

3. Modify `train.py` for the project

4.  ```cmd
    python .\train.py
    ```

<h2>0.2. Testing</h2>

0. Modify `post_processing.py` for the project

1. Modify and run `test.ipynb`

<h1 id="1.">1. Packages management</h1>

<h2>1.0. Create requirements.txt</h2>

```cmd
pip freeze > requirements.txt
```

<h1 id="2.">2. Select python.exe in VSCode</h1>

<h2>General</h2>

`Ctrl+Shift+P` => `Python: Select Interpreter` => `.\dixon_fyp_venv\Scripts\python.exe`

<h2>In Jupyter notebook</h2>

`Ctrl+Shift+P` => `Notebook: Select Notebook Kernel` => `.\dixon_fyp_venv\Scripts\python.exe`

<h1 id="3.">3. Git submodules</h1>

<h2>3.0. Add git submodule</h2>

```cmd
git submodule add https://url path/to/directory
```

<h2>3.1. Update submodule on GitHub</h2>

1. Login to GitHub

2. Click icon at upper-right corner => `Settings` => `Developer settings` => `Personal access tokens` => `Generate new token`

3. Checkout write:packages

4. cd to repository of submodule

4. ```cmd
    git add .
    git commit -m "message"
    git push origin main
    ```
5. Delete personal access token in GitHub

<h1 id="4.">4. Networks</h1>

This project used different networks and have some modifications

<h2>CENet</h2>

Forked and modified from <a href="https://github.com/David-zaiwang/Image_segmentation_framework.git">github</a>

<h2>UNet</h2>

Forked and modified from <a href="https://github.com/milesial/Pytorch-UNet">github</a>

<h2>UACANet</h2>

Copy and modified from <a href="https://github.com/plemeri/UACANet">github</a>

<h2>ResNet</h2>

Modified from <a href="https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py">github</a>

<h2>Segmenter</h2>

Forked and modified from <a href="https://github.com/rstrudel/segmenter">github</a>

<h2>HRNet + OCR + SegFix</h2>

Forked and modified from <a href="https://github.com/openseg-group/openseg.pytorch"github</a>

