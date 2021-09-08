import os
from PIL import Image
from wand import image as xdds_image#for .dds conversion
import cv2
import shutil
import json

class Crawler():

    def __init__(self):
        self.i = 0
        self.dir_len = 0
        self.error_count = 0
        self.error_list = []
        self.json_dir = r"X:\Projects\Super_Rez\Fallout NV\Output"
        self.forbidden_regex = "_n.png" #let's not process negatives.
        self.delete_regex = "_n.dds"


    def deleter(self, source_dir):
        r"""
        Crawl over folder and subfolders, populate with new format and optional delete of old format.
        """
        os.chdir(source_dir)
        source_dir_list = os.listdir()

        for file in source_dir_list:
            os.chdir(source_dir)
            file_name = str(file)
            try:
                
                if os.path.isdir(file_name):#if its a folder
                    new_dir = source_dir+f"\{file_name}"
                    self.deleter(source_dir=new_dir)

                elif not os.path.isdir(file_name) and file_name[-6:]==self.delete_regex: #extensions match
                    print(f"{source_dir[self.dir_len:]}...{file_name}")
                    os.chdir(source_dir)#In case we just finished a subfolder
                    os.remove(file) #delete this file.

                print(file_name)

            except Exception as e: 
                print(file_name, e)


    def crawler(self, source_dir, load_extension="", save_extension="", delete_empty=False, replace=False):
        r"""
        Crawl over folder and subfolders, populate with new format and optional delete of old format.
        """
        os.chdir(source_dir)
        source_dir_list = os.listdir()
        no_img_bool = True
        for file in source_dir_list:
            os.chdir(source_dir)
            file_name = str(file)
            try:
                
                if os.path.isdir(file_name):#if its a folder
                    new_dir = source_dir+f"\{file_name}"
                    self.crawler(source_dir=new_dir, load_extension=load_extension, save_extension=save_extension, delete_empty=delete_empty, replace=replace,)

                elif not os.path.isdir(file_name) and file_name[-4:] == load_extension: #extensions match

                    print(f"{source_dir[self.dir_len:]}...{file_name}")
                    os.chdir(source_dir)#In case we just finished a subfolder

                    if file_name[-6:]==self.forbidden_regex:
                        os.remove(file) #delete this file.

                    elif file_name[-6:]!=self.forbidden_regex:
                        no_img_bool = False #we have images in this folder.

                        if save_extension == ".dds":
                            with xdds_image.Image(filename=file_name) as img:
                                save_name = file_name[0:-4]
                                img.compression = 'dxt5'
                                save_name = save_name+'.dds'
                                img.save(filename=save_name)

                        elif save_extension == ".png":
                            img = Image.open(file_name)
                            save_name = file_name[0:-4]
                            img.save(save_name+save_extension)

                        if replace:
                            os.remove(file) #delete the old file
                        self.i+=1
                        #if i%100 == 0:
                        print(f"Image #:{self.i} Errors:{self.error_count} ",end="")
                    """
                    elif not os.path.isdir(file_name) and file_name[-4:]!=load_extension and file_name[-4:]!=save_extension: #wrong type, delete
                        print(f"{source_dir}...{file_name}")
                        os.chdir(source_dir)#In case we just finished a subfolder
                        os.remove(file) #delete the old file
                    """

            except Exception as e: 
                print(file_name, e)
                self.error_count+=1
                os.chdir(self.json_dir)#In case we just finished a subfolder
                error_dir = source_dir+f"\{file_name}"
                self.error_list.append(error_dir)
                with open("error_list.txt", "w") as fp:
                    json.dump(self.error_list, fp)
                os.chdir(source_dir)
                """
                with open("test.txt", "r") as fp:
                    b = json.load(fp)
                os.chdir(source_dir)#In case we just finished a subfolder
                os.remove(file) #delete the error file
                """

            #this is an empty folder, delete it.
        if delete_empty and (no_img_bool or source_dir_list==[]):
            print("Deleting folder")
            #os.chdir(source_dir) #cant delete a folder if you're in it
            shutil.rmtree(source_dir+f"\{file_name}")

def reconciler(working_dir, donor_dir, load_extension=""):
    r"""
    Crawl over folder and subfolders, populate with new format and optional delete of old format.
    """
    os.chdir(donor_dir)
    donor_dir_list = os.listdir()

    os.chdir(working_dir)
    working_dir_list = os.listdir()
    i = 0
    for file in working_dir_list:
        file_name = str(file)
        print(f"{working_dir}  {file_name}")
        
        if "." not in file_name:#if its a folder
            new_dir = working_dir+f"\{file_name}"
            reconciler(working_dir=new_dir, donor_dir=donor_dir, load_extension=load_extension)

        elif file_name[-4:] == load_extension: #extensions match
            os.chdir(working_dir)#In case we just finished a subfolder

            if file_name in donor_dir_list:
                os.remove(file) #delete the old png
                shutil.copy(donor_dir+"//"+file_name, working_dir) #copy in new files
                i+=1
                #if i%100 == 0:
                print(f"Image Count:{i}",end="")
        

def main(mode):

    if mode == "single":

        output_dir = r"X:\Projects\Super_Rez\Fallout NV\Output\textures\architecture\strip"
        source_dir = r"X:\Projects\Super_Rez\Fallout NV\Output\textures\architecture\strip"
        file_name = 'l38base02.dds'

        os.chdir(source_dir)

        img = Image.open(file_name)
        save_name = file_name[0:-4]
        img.save(save_name+'.png')

        os.remove(file_name) #delete the old png
        print("Done")

    elif mode == "crawler":
        r"""
        Crawl over folder and subfolders, populate with new format and optional delete of old format.
        """
        source_dir = r"X:\Projects\Super_Rez\Fallout NV\Output\DLCs"
        crawler_class = Crawler()
        crawler_class.dir_len = len(source_dir)
        crawler_class.crawler(source_dir=source_dir, load_extension=".png", save_extension=".dds", delete_empty=False, replace=True)
        #crawler_class.crawler(source_dir=source_dir, load_extension=".dds", save_extension=".png", delete_empty=False, replace=True)
        #crawler_class.deleter(source_dir=source_dir)

    elif mode == "reconciler":
        r"""
        Crawl over folder and subfolders, populate with new format and optional delete of old format.
        """
        donor_dir = r"X:\Projects\Super_Rez\Fallout NV\Output\upgrade - Copy"
        working_dir = r"X:\GOG\Fallout New Vegas\Data\textures\dungeons"
        #crawler(source_dir=source_dir, load_extension=".dds", save_extension=".png", delete_empty=False, replace=True)
        reconciler(working_dir=working_dir, donor_dir=donor_dir, load_extension=".dds")

if __name__ == "__main__":
    main(mode="crawler")