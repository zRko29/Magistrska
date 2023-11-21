# from google.colab import drive

# drive.mount('/content/drive')
# os.chdir("/content/drive/My Drive/Colab Notebooks")

from utils.mapping_helper import StandardMap

map = StandardMap()

map.do_mapping()
map.save_data()
map.plot_data()
