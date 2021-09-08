
import os

out_dir = r"X:\Projects\Super_Rez\Fallout NV\Output\Lonesome\butts\double_butts"

check = os.path.isdir(out_dir)
print(check)

if not os.path.isdir(out_dir): #Create output folder if it doesnt exist.
    os.mkdir(out_dir)

check = os.path.isdir(out_dir)
print(check)