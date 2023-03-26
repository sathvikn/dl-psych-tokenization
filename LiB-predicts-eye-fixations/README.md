The **online demo** of LiB is here:
https://hub-binder.mybinder.ovh/user/ray306-lib_demo-qsr3qu0q/doc/tree/Quick_Demo.ipynb
You can run the Jupyter notebook to see the segmentation result.

---

# Unsupervised text segmentation predicts eye ﬁxations during reading
PDF here: https://psyarxiv.com/eyvu7/
## Abstract
Words typically form the basis of psycholinguistic and computational linguistic studies about sentence processing. However, recent evidence shows the basic units during reading, i.e., the items in the mental lexicon, are not always words, but could also be sub-word and supra-word units. To recognize these units, human readers require a cognitive mechanism to learn and detect them. In this paper, we assume eye fixations during reading reveal the locations of the cognitive units, and that the cognitive units are analogous with the text units discovered by unsupervised segmentation models. We predict eye fixations by model-segmented units on both English and Dutch text. The results show the model-segmented units predict eye fixations better than word units. This finding suggests that the predictive performance of model-segmented units indicates their plausibility as cognitive units. The Less-is-Better (LiB) model, which finds the units that minimize both long-term and working memory load, offers advantages both in terms of prediction score and efficiency among alternative models. Our results also suggest that modeling the least-effort principle on the management of long-term and working memory can lead to inferring cognitive units. Overall, the study supports the theory that the mental lexicon stores not only words but also smaller and larger units, suggests that fixation locations during reading depend on these units, and shows that unsupervised segmentation models can discover these units.

#### Anaylsis code
See `[Open It] LiB_evaluation_on_GECO.ipynb`. 

#### The LiB model code
`LiB.py` is the main script of the LiB model. It depends on `structures.py`, which defines the basic data structure of LiB. 

#### The AG model & the CBL model
See `Other models`.

#### Data files
The files without name extension are the pre-processed corpora and eye-fixation data of GECO. 
Since the file size limitation of Github, the pre-processed large corpora (COCA and SoNaR) are uploaded to https://osf.io/ydr7w/. 

## Quick tutorial
![image](https://user-images.githubusercontent.com/1559890/134485015-54208a26-852c-439c-9152-8150ae44d4d6.png)
--
![image](https://user-images.githubusercontent.com/1559890/134485184-acf9dd51-c1bf-439c-bc4e-40ef2380d913.png)
--
![image](https://user-images.githubusercontent.com/1559890/134485407-8022f636-c1c0-4ebe-9f75-d19ba9d0a657.png)
--
![image](https://user-images.githubusercontent.com/1559890/134485253-dc77ae10-7be7-4d7a-a447-00e209f3bbd1.png)
--
![image](https://user-images.githubusercontent.com/1559890/134485282-42c9812e-f71a-4c86-9dac-5ba73964a706.png)




