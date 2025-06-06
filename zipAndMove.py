import os
import py7zr

path = r"D:\WhyNotFreeNames\Work\DAS(ПИШ)\RshSDK"
remove_dirs = True
move_zips = True
path_to_move_zips = r'D:\WhyNotFreeNames\Work\DAS(ПИШ)\RSHMOVE'

if os.path.exists(path) and os.path.isdir(path):
    os.chdir(path)
    print(os.getcwd())
    list_all_dirs = os.listdir(path)
    list_all_dirs = list(filter(lambda x: os.path.isdir(path + "\\" + x), list_all_dirs))
    print("Full list:")
    for ind, file in enumerate(list_all_dirs):
        print(str(ind).rjust(6), file)
    print()
    for ind, dir in enumerate(list_all_dirs):
        with py7zr.SevenZipFile(dir + '.7z', mode='w') as zip_file:
            print(str(ind).rjust(6), " ", "Convert '", dir, "'... ", sep='', end='')
            zip_file.writeall(dir)
        print("complete")

        if remove_dirs:
            flag = True
            os.chdir(dir)
            print(str(ind).rjust(6), "Removing files in", os.getcwd(), end='... ')
            for filename in os.listdir(os.getcwd()):
                try:
                    os.remove(filename)
                except PermissionError as err:
                    flag = False
                    print("\n" + str(ind).rjust(6), "Error while removing file:", err, end='')
            if flag:
                print('complete.')
            else:
                print()
            os.chdir('..')
            print(str(ind).rjust(6), "Back to main folder:", os.getcwd())
            print(str(ind).rjust(6), "Trying to remove dir", end='... ')
            try:
                os.rmdir(dir)
            except OSError as err:
                print("\n" + str(ind).rjust(6), "Error while removing folder:", err, end='\n')
            else:
                print('complete.')

        if move_zips:
            print(str(ind).rjust(6), "Moving zip", dir + '.7z', "to", path_to_move_zips, end='... ')
            os.replace(path + "\\" + dir + '.7z', path_to_move_zips + "\\" + dir + ".7z")
            print("complete.")
        print()

