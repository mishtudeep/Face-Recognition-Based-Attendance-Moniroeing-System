import cx_Freeze
import sys
import matplotlib
import os

base = None

if sys.platform == 'win32':
    base = "Win32GUI"

executables = [cx_Freeze.Executable("FaceRecognition.py", base=base)]

os.environ['TCL_LIBRARY'] = r'C:\Program Files\Python36-32\tcl\tkinter'
os.environ['TK_LIBRARY'] = r'C:\Program Files\Python36-32\tcl\tcl8'

additional_mods = ['numpy.core._methods', 'numpy.lib.format']

cx_Freeze.setup(
    name = "FaceRecognition",
    options = {"build_exe": {'includes': additional_mods,"packages":["tkinter","os","cv2","keras","tensorflow","glob","utils"], "include_files":["facial-recognition-ibm-1440x920.gif","model.h5","haarcascade_frontalface_default.xml"]}},
    version = "0.01",
    description = "Face recognition application",
	author="Sourav Maiti",
    executables = executables
    )