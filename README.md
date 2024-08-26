# Randi
An image database maker coupled with a randomizer for classification

> [!CAUTION]
> Work in progress!
> 
> The main functionalities are implemented.
> 
> No securities in many cases, which can lead the program to crash. -> Work in progress!
> 
> Lot of assumptions as well, see below


# Requirement
The folder arborescence is critical, the database maker assumes you have multiple conditions by folder, and that these conditions are ordered in subfolders.

Image format: tif ONLY

Image Dimension: ZXY, XY, ZCXY and CXY

NOT recursive (TODO, auto process the Split folder if such a folder is present, which can be a result of a ImageJ macro or Vesa's plugin)

# Create a database
Under the tab **database** > **new/load**

Select the folder where the images are located.

![Folder organization example](manual%20image/Folder.png)

In this case, images are ZXY images split by their channel (‘C1-’, ‘C2-’...).

![File organization](manual%20image/files.png)

If a database has **NOT** been create, it will open a popup:

![Database maker popup](manual%20image/database_maker_popup.png)

This is a pop-up for the ZXY dimension, the program will read the metadata of the first tif image to get it and assume that ALL other images have the same dimension (*TODO, security*).

The `best plane selector` and `Z projection` are exclusive, you can only select one. The best plane here is defined as the **largest continuous** area detected after thresholding in the Z-plane. The crop will be just in this area. These options are compute-intensive. The `channel for the selection` indicates with channel to use to perform such selection. The main purpose of these options is to select the center plane of the nucleus, but it can also be used for other purposes.

If such an option is not good, you can apply the Z-projection. 3 types are available: Max, Mean, and Sum. It can be combined with `crop to selection` to focus only on the nucleus, for example, while ignoring the best plane. The Z-projection alone is extremely quick, limited by the read time of the drive where the data are located.

The database will be generated on the channel designed in `Channel for database`.

The final option is the final size of each image in the database. 100x100 pixels is the default, and it should be just enough to be able to identify the type of data while removing unnecessary details (that the AI will try to focus on).

> [!CAUTION]
> If you want to change the database itself (other settings, mistakes…), you need to manually delete the database folder and the log.

# Load a database
If you just created the database, it will be loaded immediately.

In case the folder already contains a database (a log), Randi will automatically open it.

It will give you information about the database, and then display the data.

![Database info display](manual%20image/Database_log.png)

You can also load the database itself (folder name ends with `database`).

In this case, you lose the database information on its settings creation.

![Database display](manual%20image/Database.png)

# Classify
Whether you load the database directly or not, you can classify the data on **Classfier** > **New**

It will open a popup that requires two information, user name and the number of categories that you want to classify the data.

![Classifiy popup](manual%20image/Classify_popup.png)

> [!NOTE]
> If you want to do multiple classifications on the same data, do **NOT** put the same name! Use others such as Simon1, Simon_1…

Then the classification itself starts. The images can be rotated and/or flip and their order of appearance is **random** so that you have no idea where the images are coming from. In this example, I put 2 as the number of categories, so 1 and 2 are accepted as input, as well as 0 (data to discard).

![Classifiy example](manual%20image/Classify_ex.png)

> [!CAUTION]
> You cannot undo!

> [!NOTE]
> If you put no number, it assumes 0

A popup will open once you finish the classification and the database with your entry will be automatically saved.

You should then see the results of your classification.

![Result example](manual%20image/Results.png)

> [!CAUTION]
> The database will **NOT** be saved if you do not finish the classification.

The database.csv can then be used with the database folder to train an AI to identify the categories!



TODO. Security popup window `Are you sure to quit? Yes/No`

TODO. Display a graph on the number of categories/folders, with the option to save it
