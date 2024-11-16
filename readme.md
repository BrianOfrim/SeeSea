# SeeSea
A project that aims to **see** how much environmental information can be recovered from images of bodys of water (**seas**). For example: wind speed, wind direction, wave height, wave direction etc.

## Dataset
https://huggingface.co/datasets/brianofrim/NOAA_buoycams 

## Model
A multiheaded regression model was chosen with an ouput head for each parameter we want to perdict. 

For the model backbone a pretrained image classification model is used with the final classification layer removed to expose it's input features. These features are then pooled and connected the the regression heads.

The output of each regression head is a continuous floating point value. Eg for wind speed the regression head ouput is the prediction in m/s.

Different pretrained image classification models were tested such a various sizes of ResNets and MobileVits.

Current best results on the test data split are:  

wind_speed_mps:
- Mean Squared Error: 1.095
- Mean Absolute Error: 0.777

wave_height_m:
- Mean Squared Error: 0.386
- Mean Absolute Error: 0.439

Overall (All regression heads combined):
- Mean Squared Error: 0.741
- Mean Absolute Error: 0.607

These results were achieved with the following training script input:
```
python -m seesea.model.train_multihead --log DEBUG --input /Volumes/external/NOAA_buoycams --output /Volumes/external/seesea/train --batch-size 64 --model "microsoft/resnet-50" --epochs 50 --learning-rate 0.0005 --output-names wind_speed_mps wave_height_m --rotation 12
```
Current best known hyperperameters:
- Backbone pretrained image classification model: ResNet-50
- Optimizer: AdamW
- Learning rate: scheduler type: linear, peak=0.0005, warmup ratio:0.1
- Epochs: 50 (~50000 total training steps)
- Batch size: 64

## Sample output

![Z10A_2024_10_11_1110_0](readme_assets/Z10A_2024_10_11_1110_0_vis.png)
![Z67A_2024_10_06_0110_3](readme_assets/Z67A_2024_10_06_0110_3_vis.png)
![W11A_2024_10_04_2210_3](readme_assets/W11A_2024_10_04_2210_3_vis.png)
![Z25A_2024_09_26_1810_3](readme_assets/Z25A_2024_09_26_1810_3_vis.png)
![W31A_2024_10_02_0410_5](readme_assets/W31A_2024_10_02_0410_5_vis.png)
![W38A_2024_09_26_1818_1](readme_assets/W38A_2024_09_26_1818_1_vis.png)
![W64A_2024_09_19_1710_0](readme_assets/W64A_2024_09_19_1710_0_vis.png)
![W81A_2024_09_21_1910_3](readme_assets/W81A_2024_09_21_1910_3_vis.png)
![Z27A_2024_09_18_2110_5](readme_assets/Z27A_2024_09_18_2110_5_vis.png)
![W92A_2024_10_11_1410_0](readme_assets/W92A_2024_10_11_1410_0_vis.png)

#### Other potential data sources
Great Lakes Environmental Research Labs webcam images: 
- https://www.glerl.noaa.gov/metdata/

Other Great Lakes BuoyCams (images/video):
- https://limno.io/stations/

Regional Science Consortium of Presque Isle (large galary of buoy videos):
- https://wqdatalive.com/public/55

USGS Costal cameras:
- eg: https://cmgds.marine.usgs.gov/data/islaverde/

Live surf cameras:
- https://thesurfersview.com
- https://www.surfline.com 
