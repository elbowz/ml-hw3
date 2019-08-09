# Machine Learning: Missing Value forecast
### more detailed information in ITALIAN: [REPORT.MD](REPORT.MD)  

Compute probability and precision of hitting missing value through different models (*MLP, SVM, RandomForest, KNN*).

## Utilized libs 
    
* **pandas**:  load and manipulation of **DataSet** 
* **numpy, collections**: advanced data structures
* **sklearn**:  algo, stats and splitting of **TrainingSet**
* **argparse**: command line script parsing
* **matplotlib**: graph plotting
  
## Script usage  
  
```bash
main.py [-h] [--test-size TEST_SIZE] [--question-marks-ts QM_REPETED_TS] [--no-split]
        [--img-tag IMG_TAG] [--verbose]
        [DATASET]
```

* **DATASET**: [breast-cancer|letters|poker] - select *DataSet*. Default is *breast-cancer*
* **--test-size TEST_SIZE, -t TEST_SIZE**: [0-1] - *TestSet* dimension. Default 0.2
* **--question-marks-ts QM_REPETED_TS, -q QM_REPETED_TS**: {0,1,2...} - Number of samples added to the DataSet for all *'?'* in the query. Default 1
* **--no-split, -s**: use the whole *DataSet* for training. Disabled by default
* **--img-tag IMG_TAG, -i IMG_TAG**: string - append a string to image graph filename
* **--verbose, -v**: verbose mode (increase it by repeating)
* **-h, --help**: show help

> **nota**: Others options could be set by source code.

## DataSets

The **DataSet** taken from [archive.ics.uci.edu](https://archive.ics.uci.edu/ml/index.php) are: 
* [Breast Cancer Wisconsin](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
* [Letter Recognition](https://archive.ics.uci.edu/ml/datasets/Letter+Recognition)
* [Poker Hand](https://archive.ics.uci.edu/ml/datasets/Poker+Hand)