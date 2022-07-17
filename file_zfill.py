path = '/home/nas4_user/jaeseonglee/sandbox/WWZRPTh-irU#004992#005154'

import os, glob

orig_list = glob.glob(path+'/*.jpg')

print(len(orig_list))

for i in orig_list:
    parse_list = i.split('/')
    new_name = "/".join(parse_list[:-1])+'/'+parse_list[-1][:-4].zfill(4)+'.jpg'
    old_name = i

    os.rename(old_name, new_name)

