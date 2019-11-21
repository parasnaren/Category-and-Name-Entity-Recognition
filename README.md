# Category-and-Name-Entity-Recognition
Program that performs named entity and category recognition on English sentences. We provide a User Interface that takes sentences as input. We developed 2 models that independently perform both tasks simultaneously on the input from the UI.

* **Named Entities** are colored by the corresponding tags that they belong
* **Categories** are displayed at the bottom in Black


## Results

1. For *Category classification*, we achieved a accuracy of **88%** over *17 categories* using our custom fastText embeddings trained on a large Indian news corpora. We employed a Bi-Lstm for classification, as it achieved the best results.

2. For *NER task*, we achieved micro average f1-measure of **93.05** on *9 tags*, trained on the OntoNotes5.0 dataset, using a combination of Flair and Custom fastText embeddings. We employed a Bi-Lstm with a CRF layer on top for the sequence labelling task.


## User Interface

![ui](https://user-images.githubusercontent.com/29833297/69327371-7318d100-0c73-11ea-86b6-c052c991e76a.PNG)


## Steps to run the program

1.  Clone the repository

    `git clone https://github.com/parasnaren/Category-and-Name-Entity-Recognition.git`
    
2.  Install the dependencies in a virtual environment, with **python3.6+**

    `pip install -r requirements.txt`
     
3.  Download **checkpoint.pt** model from [here](https://drive.google.com/file/d/1XZWm5nGf8s_FLrJlxYamEPbPIRCYrHBH/view?usp=sharing) and store it in the directory.

4.  Run `python3 app.py`

5.  Enter sentences into the textbox and view the Entity Tags as well as Categories of the text entered.

