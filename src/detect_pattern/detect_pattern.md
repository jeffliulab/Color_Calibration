(terminal) detect & crop  >> 

`yolo task=detect mode=predict model=best.pt source=input/ save_crop=True`

(terminal) detect/predict >> 

`yolo predict model=first_detect_object.pt source={picture_path} save=True`