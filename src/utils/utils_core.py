################ imports ################
import logging
import os
import numpy as np

# ----------------------------------------
#             PATH processing
# ----------------------------------------
def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

# def get_jpgs(path):
#     # read a folder, return the image name
#     ret = []
#     for root, dirs, files in os.walk(path):
#         for filespath in files:
#             ret.append(filespath)
#     return ret

def get_jpgs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if filespath.lower().endswith(('.png', '.jpg', '.jpeg')):
                if filespath.lower() != 'SYNOFILE_THUMB_M.png'.lower():
                    ret.append(filespath)      
    ret.sort()             # change                 
    return ret

def text_save(content, filename, mode = 'a'):
    # save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()


# Checks if path exist, creates if it doesn't
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


# ----------------------------------------
#           Logging help
# ----------------------------------------
def logg(text, logs=True, verbose=True):
    if logs:
        logging.info(text)
    if verbose:
        print(text)

