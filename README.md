# mask2face

マスクの下の顔を予測するAIモデル


## Usage
1. install python=3.8

2. `pip install -r requirements.txt`

3. `mkdir data`

4. [AddMask](https://github.com/mandelamane/AddMask.git)より画像データを作成し，dataにディレクトリを移動させる．このとき，ディレクトリ名はfaceおよびmaskにするとよい．

5. `python datasets.py -id [images directory] -o [output file basename] -r [size to resize image]`

6. `python train.py -fd [face datasets basename] -md [mask datasets basename] -m [model architecher] -b [batch size] -e [number of epoch] -lr [learning rate]`

    ただし，モデルの構造は，cycleganおよびUnetのモデルに対応している．

7. resultにテストデータに対するモデルの適用結果, modelにが苦渋済みモデルが保存される．


### 参考
[str-ml-mask2face](https://github.com/strvcom/strv-ml-mask2face)