Follow the instructions to run this code:
During main function loop:
1. modify FLAG variable ('HK' or 'MY' or 'WV'or 'multi') to choose classification data.
2. DM_REDUCTION is a boot variable. It shows whether or not dimension reduction is enabled.
3. call classifiers at the end of main loop. All six classifiers are available, the functions are: 
 	knnClf(), -> k nearest neighbor,
 	treeClf(), -> decision tree, 
 	RFClf(), -> random forest
 	SVMClf(), -> supporting vector machine
	ANNClf(), -> Artificial neural network
	NaiveBayesianClf() -> Naive Bayes
4. Each classification function will output hyper-parameter tuning results, prediction time 
    and prediction accuracy. The hyper parameters are number of nearest neighbors k, tree depth, 
    decision tree numbers, soft margin parameter C,  activation function and smoothing variable, 
    respectively.

