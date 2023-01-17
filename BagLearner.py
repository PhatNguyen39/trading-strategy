import numpy as np
from scipy.stats import mode

class BagLearner(object):
    """
  		  	   		  	  		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  	  		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		  	  		  		  		    	 		 		   		 		  
    """

    def __init__(self, learner, kwargs={}, bags=20, boost=False, verbose=False):
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Constructor method  		  	   		  	  		  		  		    	 		 		   		 		  
        """
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

    def author(self):
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        :return: The GT username of the student  		  	   		  	  		  		  		    	 		 		   		 		  
        :rtype: str  		  	   		  	  		  		  		    	 		 		   		 		  
        """
        return "pnguyen340"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        self.learner_list = []
        for i in range(self.bags):
            bag = np.random.choice(data_x.shape[0], data_x.shape[0], replace=True)
            learner = self.learner(**self.kwargs)
            learner.add_evidence(data_x[bag], data_y[bag])
            self.learner_list.append(learner)
        return self

    def query(self, points):
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Estimate a set of test points given the model we built.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		  	  		  		  		    	 		 		   		 		  
        :type points: numpy.ndarray  		  	   		  	  		  		  		    	 		 		   		 		  
        :return: The predicted result of the input data according to the trained model  		  	   		  	  		  		  		    	 		 		   		 		  
        :rtype: numpy.ndarray  		  	   		  	  		  		  		    	 		 		   		 		  
        """
        result = np.zeros((points.shape[0], self.bags))

        for j in range(self.bags):
            result[:, j] = self.learner_list[j].query(points)
        y_predict =  np.squeeze(mode(result, axis=1)[0]) #np.mean(result, axis=1)
        return y_predict
def author():
    return "pnguyen340"  # replace tb34 with your Georgia Tech username