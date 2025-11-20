from PIL import Image
import os
files=os.path.join(os.getcwd(),'assets')
for file in os.listdir(files):
    with Image.open(files+'/'+file) as img:
        print(img.info)