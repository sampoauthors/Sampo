![](docs/sampo.jpg "In Finnish mythology, the Sampo was a magical quern or mill of some sort that made flour, salt, and gold out of thin air. ")
# Sampo: Unsupervised Knowledge-Base Construction from Reviews
This repository implements Sampo, a framework for automatically building Knowledge-bases (KB) from review data. More specifically, Sampo is designed to build KBs that capture how modifier-aspect pairs extracted from reviews relate to each other. For instance in the hotel domain, _"cold coffee"_ implies _"bad breakfast"_ which further implies _"poor service"_. 

### 1. Installation
You can install Sampo as a python package via the following command:
```
$ pip install -U -e .
```

You also need to download the spacy model for English via the following command:
```
$ python -m spacy download en_core_web_md
```

### 2. Tools and Scripts 
All the scripts are located under the "sampo" folder. What follows is a description of each tool.

#### 2.1 - Micropine
Micropine is a simple rule-based opinion extractor which extracts (modifier, aspect) pairs from reviews. You can find the set of rules used to find the modifiers and aspects [here](https://github.com/vladsandulescu/phrases). We recommend using a stronger opinion extractor if such extractor is available for your domain of interest. 

Micropine accepts the review corpora in a _.csv_ format. You need to specify the column that contains the reviews (using the ```-r``` option) as well as the column that identifies the item being reviewed (using the ```-i``` option). For instance, if the review is stored in column ```review``` and the item is represented in column ```item_id``` then the following command creates the desired dataset of modifer-aspect pairs and stores the results in file specified via the ```-o``` option.
```
$ python sampo/micropine.py -f ../data/raw_reviews/toys.csv -o ../data/toys.csv -i item_id -r review
```
The output _.csv_ file contains 4 columns: *item_id*, *review_id*, *modifier*, and *aspect*. Micropine assigns a *review_id* to each review by simply hashing the review text. The *item_id* is the same as the ids in the input _.csv_ file and the *modifier* and *aspect* are spans extracted by micropine.

#### 2.2 - Normalizing extractions
While the _.cvs_ file produced by micropine (or any other opinion-extractor tool) can be used for creating tensors directly, reducing the sparsitiy of the data can improve the factorization results. To increase the density of the final tensors, we can merge modifiers and aspects that are (for the most part) equivalent. For instance, _"nice view"_ and _"very nice views"_ can be grouped together to improve the results.

Given a _.csv_ file with the format described in the previous section (2.1), the ```normalize_text.py``` script creates an updated csv files where the compatible modifiers and aspects are merged respectively. The script also creates two news files which store all the surface forms and what they have been map to.
```
$ python sampo/normalize_text.py -f ../data/toys.csv -o ../data/merged_toys/ 
```
_Note: The output should point to a folder (not a file). The folder both contains the mappings as well as the merged .csv file._

#### 2.3 - Building tensors
The script is ```sampo/make_tensor.py``` which builds tensors and matrices from the _csv_ file described in Section 2.1. You can use the following command to see the options that the tool provides:
```
$ python sampo/make_tensor.py -h
```
But the following examples should be sufficient to understand how the script works.

__1.__ Build a tensor and a matrix using 100 most reviewed hotels and 500 mostly mentioned modifier-aspect pairs and store the results in a folder called ```small_data```:
```
$ python sampo/make_tensor.py -f data/hotel.csv -p small_data -i 100 -e 500
```
__2.__ Build a tensor and a matrix using 200 most reviewed hotels and 1000 mostly mentioned modifier-aspect pairs and store the results in a folder called ```large_data```:
```
$ python sampo/make_tensor.py -f data/hotel.csv -p large_data -i 200 -e 1000
```

#### 2.4 - Factorizing tensors and matrices
##### 2.4.1 - PARAFAC
PARAFAC is the most fundamental method to factorize tensors. You can run PARAFAC on any tensor/matrix created in Section 2.3. To do so, you need to specify a few arguments:

__1.__ You need to specify the path to the created matrix and tensor using the ```-p``` option.

__2.__ You need to provide the rank (i.e., the embedding dimension) that you desire for your factorization using a ```-d``` option. 

__3.__ You need to assign a name to the factorization via the ```--name``` option. This is because we often factorize a tensor via different tehcnique and using different parameters, and assinging names enables us to all obtained results more easily. 
For instance, we can factorize the ```small_data``` with a rank of 20 using the following command:
```
$ python sampo/parafac.py -p small_data -d 10  --name simple_dim10
```
or 
```
$ python sampo/parafac.py -p large_data -d 20 --name simple_dim20
```
> Note: The factorization results will be stored in the same directory as the tensors under the specified name. Consequently, the results will be over-written if you run the factorization with different parameters using the same name. 

The following are additional parameters that provides more control over the factorization. 

__4.__ The ```-i``` specifies the number of times, we repeat the factorization. This option is provided since the factorization is a randomized process and you can obtain more stable results by repeating the factorization and store all obtained results.

__5.__ To test the significance of obtained results, you can add noise to the created tensor before factorization. The options are ```--gause``` and ```gen``` and each add noise to the tensor as follows: ```gause``` adds a gaussian noise with mean=0 and std=1, and ```gen``` first factorizes the matrix to estimate the mean and std of the noise and then adds a gaussian with those parameters.

__6.__ The ```--fixed``` option is only valid when either a gaussian or poisson noise is added to the matrix and simply states that the noisy matrix should be fixed accross different repetitions of factorizations. This option is mainly for analysis of the tool and it's not likely you need it for running statistical tests.

__7.__ By default the tool, both factorizes a 2D matrix as well as the 3D tensor that is created from the input data. However, if you are only interested in factorizing one of these data structure, you can specify that using the ```--matrix``` or ```--tensor``` options.

__8.__ By using ```--cuda``` singals that GPUs are available and the factorization should be done over GPU.

##### 2.4.2 - Non-Negative PARAFAC
This is a non-negative factorization technique which enforces the final learned embeddings and the reconstructed tensor to be all positive. The common observation is that non-negative factorization often yields more interpretable results (e.g., when you want to compare the learned embeddings which is what we are doing here!). You can use this factorization by simply adding a ```-n``` or ```--nonnegative``` option to the ```parafac.py``` script introduced above:
```
$ python sampo/parafac.py -p small_data -d 10  --name nonneg_dim10 --nonnegative
```
We generally recommend using nonnegative factorization to achieve better results.

##### 2.4.3 - Coupled Matrix-Tensor Factorization (CMTF)
This factorizatoin technqiue allows for additional matrices to be coupled and jointly factorized with the tensor. More precisely, we can specify one additional matrix per each dimension of the tensor. Normally, the additional matrix is used to provided some new information about the relationship between items in a dimension. In our case, we use this additional matrix to model the linguistic similarity between modifiers (similarly aspects) of the tensor. That is, we create a matrix where each row stores the avergate word embedding of the modifier (or aspect) associated with the row. You can run this factorization using the same interface as the ```parafac.py```:
```
$ python sampo/cmtf.py -p small_data -d 10 --name simple_dim10
``` 

#### 2.5 - Nearest-neighbor search on modifier-aspect pair embeddings
You can find the most similar (measured via cosine similarity) modifer-aspect pairs to a provided list of modifier-aspect queries using the script ```sampo/nn_report.py``` as follows. You need to specify a _csv_ file of queries with ```modifier``` and ```aspect``` among the columns, and specify how many neighbors you would like the script to fetch. See below for some examples.

__1.__ Find the 3 most similar modifier-aspect pairs to _good-staff_:
```
$ echo "modifer,aspect\nnice,view" > query.csv
$ python sampo/nn_report.py -p small_data -f query.csv -n 3 --name simple_dim10
```
and here is what the output would look like:
```
Query: nice;view, Count: 1664
| nn              |   count |      sim |   rank |
|-----------------+---------+----------+--------|
| good;view       |     726 | 0.964709 |    1.6 |
| great;view      |    3556 | 0.963318 |    5.6 |
| amazing;view    |    1118 | 0.930998 |    5.6 |
====================
```
> Note: The numbers that follows each neighboring modifier-aspect are (1) the count of the neighboring modifier and aspect pair, (2) the cosine similarity between their embeddings and embedding of the input query, and (3) its rank among the nearest neighbors. You might see the rank is not 1, 2, and 3 as expected. This is because the factorization has been done 5 times and the results you see is the average rank across these 5 iterations. The same is true for similarity values.

Also, we have to note that given that the results are average over multiple factorizations. Ranking the neighbors based on their average order vs. their average similarity can produce different results. Thus, the script provides a ```sort-by``` option which should be set either to _rank_ or _sim_. The default ordering is using the average similarity values.

__2.__ Find the 5 most similar modifier-aspect pairs to _good-staff_ but with a different aspect:
```
$ echo "modifer,aspect\ngood,staff" > query.csv
$ python sampo/nn_report.py -p small_data -f query.csv -n 5 --unique_aspect
```

__3.__ Find the most similar modifier-aspect pairs to _fresh-coffee_ but with a different modifier and a different aspect:
```
$ echo "modifier,aspect\nfresh,coffee" > query.csv
$ python sampo/nn_report.py -p small_data -f query.csv -n 1 --unique_aspect --unique_modifier
```

If you don't have a query file to process, you can simply ignore the ```-f``` option, and the script will find
the nearest neighbors of each opinion-aspect pair in the original dataset. For example:
```
$ python sampo/nn_report.py -p small_data -n 1 --unique_aspect --unique_modifier
```

Finally, you can get the nearest neighbor resutls in a _.csv_ or _.json_ format using the ```--csv``` and ```--json``` options followed by an output file. For example, you can save the output of the previous command in a csv file as follows:
```
$ python sampo/nn_report.py -p small_data -n 1 --unique_aspect --unique_modifier --csv stored_results.csv
```
