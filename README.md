# jpeg-dcgan-pytorch
JPEG圧縮を応用した画像生成プロジェクトです。

## 参考論文
### [Faster Neural Networks Straight from JPEG](https://papers.nips.cc/paper/2018/file/7af6266cc52234b5aa339b16695f7fc4-Paper.pdf)
> Gueguen, L., Sergeev, A., Kadlec, B., Liu, R., & Yosinski, J. (2018). Faster neural networks straight from jpeg. In Advances in Neural Information Processing Systems (pp. 3933-3944).

JPEGのDCT係数を入力として画像分類をする。<br>
分類精度を維持しながら高速化を達成した。

公式実装: [uber-research / jpeg2dct](https://github.com/uber-research/jpeg2dct)

### [Toward Joint Image Generation and Compression using Generative Adversarial Networks](https://arxiv.org/abs/1901.07838)
> Kang, B., Tripathi, S., & Nguyen, T. Q. (2019). Toward Joint Image Generation and Compression using Generative Adversarial Networks. arXiv preprint arXiv:1901.07838.

高解像度画像は圧縮されているはずなのでGeneratorにJPEGデコード機構を追加する事で画像にJPEG的な特徴を付与する。<br>
通常のモデルに追加する形なので高速化ではなくFIDスコアの向上が目的である。

- Locally Connected Layer<br>
入出力特徴平面上の小領域間でのみ結合があるような畳み込みを行う。

### [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)
> Goyal, P., Dollár, P., Girshick, R., Noordhuis, P., Wesolowski, L., Kyrola, A., Tulloch, A., Jia, Y., & He, K. (2017). Accurate, large minibatch sgd: Training imagenet in 1 hour. arXiv preprint arXiv:1706.02677.

- Linear Scaling<br>
学習率をバッチサイズに比例させる。

### [cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637)
> Miyato, T., & Koyama, M. (2018). cGANs with projection discriminator. arXiv preprint arXiv:1802.05637.

- Projection Discriminator<br>
Discriminatorの最終層の特徴とクラスを埋め込んだベクトルの内積を取る。

### [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500)
> Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017). Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Advances in neural information processing systems (pp. 6626-6637).

- Two Time-Scale Update Rule (TTUR)<br>
GeneratorとDiscriminatorでそれぞれ異なる学習率を適用する。

### [Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318)
> Zhang, H., Goodfellow, I., Metaxas, D., & Odena, A. (2019, May). Self-attention generative adversarial networks. In International Conference on Machine Learning (pp. 7354-7363). PMLR.

GとDの両方にSelf-Attentionを導入して大域的な画素の関係を考慮できるようにする。

### [Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis](https://openreview.net/forum?id=1Fqg133qRaI)
> Anonymous authors (Paper under double-blind review). ICLR2021.

- Skip-Layer channel-wise Excitation Module (SLE Module)<br>
ResidualConnectionやSEモジュールのような勾配補強のレイヤー間接続を作成する。
- Self-Supervised Discriminator<br>
Dの中間特徴から元画像を復元できるようにDを訓練し正則化する。

非公式実装: [lucidrains / lightweight-gan](https://github.com/lucidrains/lightweight-gan)

## 参考記事など
1. [soumith/ganhacks - GitHub](https://github.com/soumith/ganhacks)
2. [Tips for Training Stable Generative Adversarial Networks](https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/)
3. [Keep Calm and train a GAN. Pitfalls and Tips on training Generative Adversarial Networks](https://medium.com/@utk.is.here/keep-calm-and-train-a-gan-pitfalls-and-tips-on-training-generative-adversarial-networks-edd529764aa9)
4. [GAN — Ways to improve GAN performance](https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b)
5. [GAN(Generative Adversarial Networks)を学習させる際の14のテクニック](https://qiita.com/underfitting/items/a0cbb035568dea33b2d7)
6. [DCGAN(Deep Convolutional GAN)｜DeepLearning論文の原文を読む #12](https://lib-arts.hatenablog.com/entry/paper12_DCGAN)
7. [DCGAN](https://medium.com/@liyin2015/dcgan-79af14a1c247)
8. [GANで学習がうまくいかないときに見るべき資料](https://gangango.com/2018/11/16/post-322/)
9. [個人的GANのTipsまとめ](https://qiita.com/pacifinapacific/items/6811b711eee1a5ebbb03)
10. [PyTorchでDCGANやってみた](https://blog.shikoan.com/pytorch-dcgan/)
11. [CIFAR10を混ぜたままChainerのDCGANに突っ込んだら名状しがたい何かが生成された話](https://ensekitt.hatenablog.com/entry/2017/11/07/123000)
12. [TonyMooori/dct_2dim.py - GitHub Gist](https://gist.github.com/TonyMooori/661a2da7cbb389f0a99c)
13. [GANの発展の歴史を振り返る！GANの包括的なサーベイ論文の紹介(アルゴリズム編)](https://ai-scholar.tech/articles/treatise/gansurvey-ai-371)
14. [GANの発展の歴史を振り返る！GANの包括的なサーベイ論文の紹介(応用編)](https://ai-scholar.tech/articles/treatise/gan-survey-ai-375)

## 参考ライブラリ
- [kornia](https://kornia.github.io/) ([Documentation](https://kornia.readthedocs.io/en/latest/index.html))<br>
GPU上で動作するData Augmentationライブラリ