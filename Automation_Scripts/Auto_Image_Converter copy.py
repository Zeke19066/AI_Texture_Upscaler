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
        self.source_dir = r"X:\Projects\Super_Rez\Fallout NV\Output\upgrade_ddn"
        self.dest_dir = r"X:\Projects\Super_Rez\Fallout NV\Output\upgrade_ddn"
        self.source_dir_list = []
        self.file_name = ""
        program_dir = r"C:\Program Files\GIMP 2\bin\gimp-2.10"
        self.gimp = subprocess.Popen(program_dir)

    def folder_works(self):
        os.chdir(self.source_dir)
        self.source_dir_list = os.listdir()
        for file in self.source_dir_list:
            self.file_name = str(file)[0:-4]

            self.gimp_wrapper()
            os.remove(file) #delete the old

    def gimp_wrapper(self):
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

    def terminate(self):
            self.gimp.terminate()

def main(mode):
    if mode == "folder":
        time.sleep(3)
        extractor = Extractor()
        extractor.folder_works()
        extractor.terminate()

if __name__ == "__main__":
    main(mode="folder")

