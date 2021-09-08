import shutil
import os
import time
import pyautogui
import pydirectinput #this module requires pyautogui to run
import subprocess


#What this script will do?
#for every file in "C:\Users\Ezeab\Documents\Python\Super_Rez\Source_files\models"
#EXCEPTION "player" folder.
#
#1) Create a subfolder with the name of the file sans extension.
#2) Movet the file into the subfolder.
#3) in MilkShape:
#    - Tools > HalfLife > Decompile normal MDL
#    - Enter subfolder path > hit enter.
#    - Enter MDL file name
#    - uncheck all boxes except textrues
#    - click okay

class Extractor():

    def __init__(self):
        self.source_dir = r"C:\Users\Ezeab\Documents\Python\Super_Rez\Source_files\models"
        self.dest_dir = r"C:\Users\Ezeab\Documents\Python\Super_Rez\Output\models"
        self.source_dir_list = []
        self.file_name = ""
        self.sub_dir = ""


    def folder_works(self):
        os.chdir(self.source_dir)
        self.source_dir_list = os.listdir()
        for file in self.source_dir_list:
            self.file_name = str(file)
            self.sub_dir = self.dest_dir+"//"+str(self.file_name[0:-4]) #cut off that extension.
            os.mkdir(self.sub_dir)
            shutil.copy(self.source_dir+"//"+self.file_name, self.sub_dir) #copy in new files
            #os.remove(self.file_name)  #delete the old

            self.milk_shape_wrapper()
            os.chdir(self.sub_dir)
            os.remove(file) #delete the old

    def milk_shape_wrapper(self):
        program_dir = r"C:\Program Files (x86)\MilkShape 3D 1.8.4\ms3d.exe"
        milk = subprocess.Popen(program_dir)
        time.sleep(1.5)
        text_dir = self.dest_dir + f"\{self.file_name[0:-4]}"

        #open menu
        pyautogui.moveTo(513, 78, duration=0.2) #Move to "tools"
        pyautogui.click()
        pyautogui.moveTo(520, 120, duration=0.2) #Move to "Halflife"
        pyautogui.moveTo(1445, 120, duration=0.2) 
        pyautogui.moveTo(1445, 222, duration=0.2) #Decompile normal MDL"
        pyautogui.click()

        #address bar
        pyautogui.moveTo(1145, 212, duration=0.2)
        pyautogui.click()
        pyautogui.write(text_dir)
        pydirectinput.press('enter')

        #file name
        pyautogui.moveTo(620, 1040, duration=0.2)
        pyautogui.click()
        pyautogui.write(self.file_name)
        pydirectinput.press('enter')

        #checkboxes
        pyautogui.moveTo(1725, 1050, duration=0.05) #references
        pyautogui.click()
        pyautogui.moveTo(1725, 1095, duration=0.05) #sequences
        pyautogui.click()
        pyautogui.moveTo(1725, 1180, duration=0.05) #qc file
        pyautogui.click()
        pyautogui.moveTo(1915, 1250, duration=0.05) #click okay
        pyautogui.click()

        time.sleep(3)
        milk.terminate()
        return

def main(mode):
    if mode == "folder":
        time.sleep(3)
        extractor = Extractor()
        extractor.folder_works()
    elif mode == "milk":
        time.sleep(3)
        extractor = Extractor()
        extractor.milk_shape_wrapper()

if __name__ == "__main__":
    main(mode="folder")

