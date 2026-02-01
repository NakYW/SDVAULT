# STYLEVAULT: Distribution-Calibrated Competitive Attention for Training-Free Reference-Based Diffusion Style Transfer

![Overview](asset/01.png)

## Setup

### Environment

```
conda env create -r requirement.txt
conda activate STYLEVAULT
```

### Download Stable Diffusion Weights

Download Stable Diffusion 1.5 weights and place them under:[SD1.5](https://www.wikiart.org/)  <br>  

```
models/ldm/stable-diffusion-1.5/
```

### Dataset

Our curated dataset can be obtained here:[data]([(https://drive.google.com/drive/folders/1kCAR8hPgSFtplsudqIm5GZt0Rh7nA386))]  <br>  


## Run

First, extract style features:

```
python extract_style_features.py --sty <style_img_dir>
```

Then run style transfer:

```
python style_transfer.py --cnt <content_img_dir> --sty <style_img_dir> --output_path <output_dir>
```

## Evaluation

Before running the evaluation, duplicate the content and style images to match the number of stylized images. (24 styles, 24 contents -> 576 style images, 576 content images)

```
python util/copy_inputs.py --cnt data/cnt --sty data/sty
```

We use matthias-wright/art-fid and mahmoudnafifi/HistoGAN for evaluation.

### Art-fid

```
cd evaluation;
python eval_artfid.py --sty ../data/sty_eval --cnt ../data/cnt_eval --tar ../output
```

### Histogram loss

```
cd evaluation;
python eval_histogan.py --sty ../data/sty_eval --tar ../output
```


