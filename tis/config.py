import configparser

parser = configparser.ConfigParser()

print(parser.read("../tis.conf"))