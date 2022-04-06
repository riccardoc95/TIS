import configparser
import os


# DEFAULT PARAMETERS
parameters = {"data_folder": "/",
              }



# READ PARAMETERS FROM .CONF FILE
parser = configparser.ConfigParser()

config_file_path = os.path.join("","tis.conf")

parser.read(config_file_path)

for section_name in parser.sections():
    if section_name == "USER PARAMETERS":
        for key, value in parser.items(section_name):
            print(key,value)
