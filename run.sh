python train.py mnist -e 500 --save --sample-interval 50 --save-interval 10000 --ms-lambda 0.1 --first-sample --dir-name R$1 --seed $1
python train.py fashion_mnist -e 500 --save --sample-interval 50 --save-interval 10000 --ms-lambda 0.1 --first-sample --dir-name R$1 --seed $1
python train.py cifar10 -e 500 --save --sample-interval 50 --save-interval 10000 --ms-lambda 0.1 --first-sample --dir-name R$1 --seed $1

python train.py mnist -e 500 --g-jpeg --d-jpeg --save --sample-interval 50 --save-interval 10000 --ms-lambda 0.1 --first-sample --dir-name R$1 --seed $1
python train.py fashion_mnist -e 500 --g-jpeg --d-jpeg --save --sample-interval 50 --save-interval 10000 --ms-lambda 0.1 --first-sample --dir-name R$1 --seed $1
python train.py cifar10 -e 500 --g-jpeg --d-jpeg --save --sample-interval 50 --save-interval 10000 --ms-lambda 0.1 --first-sample --dir-name R$1 --seed $1

python train.py mnist -e 100 --g-jpeg --save --sample-interval 50 --save-interval 10000 --ms-lambda 0.1 --first-sample --dir-name ND$1 --seed $1
python train.py fashion_mnist -e 100 --g-jpeg --save --sample-interval 50 --save-interval 10000 --ms-lambda 0.1 --first-sample --dir-name ND$1 --seed $1
python train.py cifar10 -e 100 --g-jpeg --save --sample-interval 50 --save-interval 10000 --ms-lambda 0.1 --first-sample --dir-name ND$1 --seed $1

python generate.py outputs/mnist/R$1/epoch500/generator.pt --seed $1
python generate.py outputs/fashion_mnist/R$1/epoch500/generator.pt --seed $1
python generate.py outputs/cifar10/R$1/epoch500/generator.pt --seed $1

python generate.py outputs/mnist_jpeg/R$1/epoch500/generator.pt --seed $1
python generate.py outputs/fashion_mnist_jpeg/R$1/epoch500/generator.pt --seed $1
python generate.py outputs/cifar10_jpeg/R$1/epoch500/generator.pt --seed $1

python generate.py outputs/mnist_jpeg/ND$1/epoch100/generator.pt --seed $1
python generate.py outputs/fashion_mnist_jpeg/ND$1/epoch100/generator.pt --seed $1
python generate.py outputs/cifar10_jpeg/ND$1/epoch100/generator.pt --seed $1

python evaluate.py outputs/mnist/R$1/epoch500/generator.pt --seed $1
python evaluate.py outputs/fashion_mnist/R$1/epoch500/generator.pt --seed $1
python evaluate.py outputs/cifar10/R$1/epoch500/generator.pt --seed $1

python evaluate.py outputs/mnist_jpeg/R$1/epoch500/generator.pt --seed $1
python evaluate.py outputs/fashion_mnist_jpeg/R$1/epoch500/generator.pt --seed $1
python evaluate.py outputs/cifar10_jpeg/R$1/epoch500/generator.pt --seed $1

python evaluate.py outputs/mnist_jpeg/ND$1/epoch100/generator.pt --seed $1
python evaluate.py outputs/fashion_mnist_jpeg/ND$1/epoch100/generator.pt --seed $1
python evaluate.py outputs/cifar10_jpeg/ND$1/epoch100/generator.pt --seed $1


python train.py stl10 -b 25 -e 500 --save --sample-interval 50 --save-interval 10000 --ms-lambda 0.1 --first-sample --dir-name R$1 --seed $1
python train.py cifar10 --grayscale -e 500 --save --sample-interval 50 --save-interval 10000 --ms-lambda 0.1 --first-sample --dir-name R$1 --seed $1
python train.py stl10 --grayscale -b 25 -e 500 --save --sample-interval 50 --save-interval 10000 --ms-lambda 0.1 --first-sample --dir-name R$1 --seed $1

python train.py stl10 -b 25 -e 500 --g-jpeg --d-jpeg --save --sample-interval 50 --save-interval 10000 --ms-lambda 0.1 --first-sample --dir-name R$1 --seed $1
python train.py cifar10 --grayscale -e 500 --g-jpeg --d-jpeg --save --sample-interval 50 --save-interval 10000 --ms-lambda 0.1 --first-sample --dir-name R$1 --seed $1
python train.py stl10 --grayscale -b 25 -e 500 --g-jpeg --d-jpeg --save --sample-interval 50 --save-interval 10000 --ms-lambda 0.1 --first-sample --dir-name R$1 --seed $1

python train.py stl10 -b 25 -e 100 --g-jpeg --save --sample-interval 50 --save-interval 10000 --ms-lambda 0.1 --first-sample --dir-name ND$1 --seed $1
python train.py cifar10 --grayscale -e 100 --g-jpeg --save --sample-interval 50 --save-interval 10000 --ms-lambda 0.1 --first-sample --dir-name ND$1 --seed $1
python train.py stl10 --grayscale -b 25 -e 100 --g-jpeg --save --sample-interval 50 --save-interval 10000 --ms-lambda 0.1 --first-sample --dir-name ND$1 --seed $1

python generate.py outputs/stl10/R$1/epoch500/generator.pt --seed $1
python generate.py outputs/cifar10_grayscale/R$1/epoch500/generator.pt --seed $1
python generate.py outputs/stl10_grayscale/R$1/epoch500/generator.pt --seed $1

python generate.py outputs/stl10_jpeg/R$1/epoch500/generator.pt --seed $1
python generate.py outputs/cifar10_grayscale_jpeg/R$1/epoch500/generator.pt --seed $1
python generate.py outputs/stl10_grayscale_jpeg/R$1/epoch500/generator.pt --seed $1

python generate.py outputs/stl10_jpeg/ND$1/epoch100/generator.pt --seed $1
python generate.py outputs/cifar10_grayscale_jpeg/ND$1/epoch100/generator.pt --seed $1
python generate.py outputs/stl10_grayscale_jpeg/ND$1/epoch100/generator.pt --seed $1

python evaluate.py outputs/stl10/R$1/epoch500/generator.pt --seed $1
python evaluate.py outputs/cifar10_grayscale/R$1/epoch500/generator.pt --seed $1
python evaluate.py outputs/stl10_grayscale/R$1/epoch500/generator.pt --seed $1

python evaluate.py outputs/stl10_jpeg/R$1/epoch500/generator.pt --seed $1
python evaluate.py outputs/cifar10_grayscale_jpeg/R$1/epoch500/generator.pt --seed $1
python evaluate.py outputs/stl10_grayscale_jpeg/R$1/epoch500/generator.pt --seed $1

python evaluate.py outputs/stl10_jpeg/ND$1/epoch100/generator.pt --seed $1
python evaluate.py outputs/cifar10_grayscale_jpeg/ND$1/epoch100/generator.pt --seed $1
python evaluate.py outputs/stl10_grayscale_jpeg/ND$1/epoch100/generator.pt --seed $1