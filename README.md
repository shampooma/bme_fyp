# BME FYP

<a href="#0.">0. Quick start</a>

<a href="#1.">1. Install packages</a>

<a href="#2.">2. Select python.exe in VSCode</a>

<a href="#3.">3. Update helper_pytorch</a>

<a href="#4.">4. Networks</a>

---

<h1 id="0.">0. Quick start</h1>

<h2>0.0. Training</h2>

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

2. Check `dataset.py` is correct

3.  ```cmd
    .\dixon_fyp_venv_windows\Scripts\activate
    pip install -r requirements.txt
    python .\train.py
    ```

<h2>0.1. Testing</h2>

0. Check `post_processing.py` is correct

1. Run `test.ipynb`

<h1 id="1.">1. Packages management</h1>

<h2>1.0. Install packages</h2>

```cmd
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -r requirements.txt
```

<h2>1.1. Create requirements.txt</h2>

```cmd
pip freeze > requirements.txt
```

<h1 id="2.">2. Select python.exe in VSCode</h1>

<h2>General</h2>

`Ctrl+Shift+P` => `Python: Select Interpreter` => `.\dixon_fyp_venv\Scripts\python.exe`

<h2>In Jupyter notebook</h2>

`Ctrl+Shift+P` => `Notebook: Select Notebook Kernel` => `.\dixon_fyp_venv\Scripts\python.exe`

<h1 id="3.">3. Update helper_pytorch</h1>

Following the procedures in [https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)

1. Login to GitHub

2. Click `upper-right corner` => `Settings` => `Developer settings` => `Personal access tokens` => `Generate new token`

3. Checkout write:packages

4. ```cmd
    cd .\helper_pytorch
    git add .
    git commit -m "message"
    git push origin main
    ```
5. Delete personal access token

<h1 id="4.">4. Networks</h1>

This project used different networks and have some modifications

<h2>CENet</h2>

<a href="https://github.com/David-zaiwang/Image_segmentation_framework.git">github</a>

Changed input channels to 1

<h2>UNet</h2>

<a href="https://github.com/milesial/Pytorch-UNet">github</a>

Changed batch normalization to group normalization

<h1 id="5.">5. Git submodules</h1>

<h2>5.0. Add git submodule</h2>

```sh
# Add git submodule
git submodule add 
```

<h1 id="6.">6. PaddlePaddle</h1>

python predict.py --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml    --model_path https://bj.bcebos.com/paddleseg/dygraph/optic_disc/bisenet_optic_disc_512x512_1k/model.pdparams --image_path docs/images/optic_test_image.jpg --save_dir output/result