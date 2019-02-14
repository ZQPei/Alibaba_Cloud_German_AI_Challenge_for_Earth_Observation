import os
import sys

def GetAbsoluteFIlePath(folderpath):
    absolute_file_path_list = []
    for folderpath, subfolderlist, filelist in os.walk(folderpath):
        for filename in filelist:
            filepath = os.path.join(folderpath, filename)
            absolute_file_path_list.append(filepath)
    return absolute_file_path_list

def progress_bar(current, total):
    sys.stdout.write("\rProgress[%.1f%%]:%d/%d "%(100*current/total, current, total))
    sys.stdout.flush()

if __name__ == "__main__":
    files = GetAbsoluteFIlePath('model')
    print(files)