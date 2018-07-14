# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 10:20:07 2018

@author: For_Gondor
"""

import os


def search(content: str, suffix: str = '', path: str = os.getcwd()):
    """
    Find content in multiple files.
    The results will be printed in the console.
    
    Args:
        content: string that you want to find.
        suffix: files' suffix, could be empty.
        path: all files in the path will be searched (including files in subdirectories).
    """
    list_root = []
    list_root1 = []
    list_dirs = []
    list_files = []
    for root, dirs, files in os.walk(path):
        list_root.append(root)
        list_dirs.append(dirs)
        list_files.append(files)
        
    list_root1.append('')
    for i in range(1, len(list_root)):
        list_root1.append(list_root[i].replace(list_root[0] + '\\', ''))
    
    for x in range(len(list_files)):
        if not list_files[x]:
            continue
        for file in list_files[x]:
            if suffix and not file.endswith(suffix):
                continue
            f = open(os.path.join(list_root[x], file), 'rb')
            lines = f.readlines()
            f.close()
            for i in range(len(lines)):
                line = lines[i].decode('UTF-8', 'ignore')
                if content in line:
                    print('File "' + os.path.join(list_root1[x], file) +
                          '" contents "{}" in line {}.'.format(content, i + 1))
                    print('\t' + line.strip())
                    print()
