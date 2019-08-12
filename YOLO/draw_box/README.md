# Box generation for YOLO

![Language](https://img.shields.io/badge/language-Python-blue.svg)

## Bounding box generation for object in images

**Credits**: This module is an extension/adaptation from [source](https://github.com/genomexyz/detect_object)

* From the project directory **execute** $```python ./detect_object/drawbox.py -h``` seek help on type of argument that can be passed. 
```
Pass arguments to set width and height, image-directory, csv-filename, pickle-
filename

optional arguments:
  -h, --help            show this help message and exit
  -cw CANVAS_WIDTH, --canvas_width CANVAS_WIDTH
                        Give the width of canvas on which image will be
                        displayed and should be at least 100px more than max
                        width from all of the images
  -ch CANVAS_HEIGHT, --canvas_height CANVAS_HEIGHT
                        Give the height of canvas on which image will be
                        displayed and should be at least 100px more than max
                        height from all of the images
  -d DIR_IMG, --dir_img DIR_IMG
                        Give fullpath to directory where images are located or
                        relative from current dir
  -s SAVE_CSV_FILE, --save_csv_file SAVE_CSV_FILE
                        GIve filename where object box data will be saved
  -p PK_FILE, --pk_file PK_FILE
                        Give pickle filename where object-box info is stored
                        in dictionary with key image name
```

*  To get more details about the module **execute** ```pydoc ./detect_object/drawbox.py``` which provides details about how: 
	* this module helps in generating bounding box around object in given image; it is currently adapted for two classes only but can be extended for any number of classes.
	* After running the module use mouse to bound an object on image and repeat this process until all the object are bound then
		* press key **s** to save all the boxes
		* press key **d** to delete if there was any mistake. Please note that for now user has to redraw on all of the object in  currnt image if key **d** was used.
		* Use left and right key to navigate and skip if no is object present in the image. 	   







