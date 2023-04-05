# mask2face

マスクの下の顔を予測するAIモデル


## Usage
1. install python=3.9

2. `pip install -r requirements.txt`　(ただし，M1 Pro mac対応)

3. `mkdir data`

4. [AddMask](https://github.com/mandelamane/AddMask.git)より画像データを作成し，trainおよびtest内部のディレクトリをdataにディレクトリを移動させる．このとき，ディレクトリ名はfacexxおよびmaskxxにするとよい．

5. `python train.py -tid [train mask dir] -ttd [train face dir] -vid [validation mask dir] -vtd [validation face dir] -is [image size (width and height)] -m [model architecher] -b [batch size] -e [number of epoch] -lr [learning rate]`

    ただし，モデルの構造は，cycleganおよびUnetのみ対応している．

6. resultにテストデータに対するモデルの適用結果, modelにが学習済みモデルが保存される．


### 参考
[str-ml-mask2face](https://github.com/strvcom/strv-ml-mask2face)