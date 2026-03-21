# Artifact Restoration using Neural Networks

> NOTES:
> The purpose of this project is to use existing literature to learn
> machine learning image algorithms and techniques.
> Currently, only the denoised model is complete; the inpainting model
> is being worked on.
> 

# Setup
1. Open CMD within the project root and run:
```bash
pip install -r requirements.txt
```
  This will install all required python components for the project.
2. Download the trained model from [here](https://drive.google.com/file/d/1feZ_9RPZIrjEvDFaOQiKefmfGDQrI7Np/view?usp=sharing). If you would like to train your own model, run the command below (warning: this can take up to a day based on how powerful your CPU is unless trained via Google Collab).
```bash
python train.py --train_dir data/images/train --val_dir data/images/test --noise_sigma 25 --epochs 5 --batch_size 2 --patch_size 128
```
3. To test/evaluate the model, run this command:
```bash
python infer.py --model checkpoints/best_model.pth --input data/images/test --add_noise 25 --visualize
```
   This will denoise the images (in black and white) and show a comparison for one of the images.
