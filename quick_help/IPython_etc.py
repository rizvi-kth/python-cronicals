
# *** Allow printing for each command in Jupyter cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"



# *** Change IPyhton working directory
import os
os.getcwd()
os.chdir("Infer/")
os.getcwd()



# *** Fixing an import in the IPython
import os
os.getcwd()
import sys
sys.path.extend([os.getcwd() + "/my-package-folder"])
# sys.path.insert(0, os.path.abspath('/my-package-folder'))
# sys.path.append(".")
# Now /my-package-folder/my-package should be imported in IPyhton
import my-package




# *** How to remove the RED UNDER LINE from the import commmand?

# 1. This is working in Python Console because
#     a. The default working directory of Python Console is the Project directory itself.
#     b. Extending the sys.path with "/python-scripts/DAL" will allow discovering modules in this directory.

# 2. This will work with Run/Debug also because
#     a. In the Run Configuration the working directory is set to "/Users/test/elasticsearch-phase-1/python-scripts/DAL"

# !!! Only the problem is the RED UNDER LINE with import which does not allow code navigation.

# To solve it
# 1. Make the "/DAL" as "Sources Root" (Right click DAL > Make directory as > Sources Root )

