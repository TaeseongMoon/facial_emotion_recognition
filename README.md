# YoloV3 Facial emotion recognition

### installation

```bash
pip install -r requirements.txt 
```

### run inference 

```bash
python detect.py --classes ./data/emotion.names --weights {weights_file} --tiny --image {test_img_folder} --num_classes 5
```

