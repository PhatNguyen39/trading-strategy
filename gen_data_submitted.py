""""""
"""  		  	   		  	  		  		  		    	 		 		   		 		  
template for generating data to fool learners (c) 2016 Tucker Balch  		  	   		  	  		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  	  		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		  	  		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  	  		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		  	  		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		  	  		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		  	  		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		  	  		  		  		    	 		 		   		 		  
or edited.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		  	  		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		  	  		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  	  		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: pnguyen340 (replace with your User ID)  		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903749038 (replace with your GT ID)  		  	   		  	  		  		  		    	 		 		   		 		  
"""

import math

import numpy as np


# this function should return a dataset (X and Y) that will work  		  	   		  	  		  		  		    	 		 		   		 		  
# better for linear regression than decision trees  		  	   		  	  		  		  		    	 		 		   		 		  
def best_4_lin_reg(seed=1489683273):
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		  	  		  		  		    	 		 		   		 		  
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		  	  		  		  		    	 		 		   		 		  
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
    :param seed: The random seed for your data generation.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type seed: int  		  	   		  	  		  		  		    	 		 		   		 		  
    :return: Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		  	  		  		  		    	 		 		   		 		  
    :rtype: numpy.ndarray  		  	   		  	  		  		  		    	 		 		   		 		  
    """
    np.random.seed(seed)
    intercept = np.random.random()
    x = np.random.choice(200, (500, 3))
    y = x[:, 0] * 2 + x[:, 1] * 10 + x[:, 2] + intercept
    noise = np.random.random(y.shape) * 20
    y_noise = y + noise

    # Here's is an example of creating a Y from randomly generated  		  	   		  	  		  		  		    	 		 		   		 		  
    # X with multiple columns  		  	   		  	  		  		  		    	 		 		   		 		  
    # y = x[:,0] + np.sin(x[:,1]) + x[:,2]**2 + x[:,3]**3  		  	   		  	  		  		  		    	 		 		   		 		  
    return x, y_noise


def best_4_dt(seed=1489683273):
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		  	  		  		  		    	 		 		   		 		  
    The data set should include from 2 to 10 columns in x, and one column in Y.  		  	   		  	  		  		  		    	 		 		   		 		  
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
    :param seed: The random seed for your data generation.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type seed: int  		  	   		  	  		  		  		    	 		 		   		 		  
    :return: Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		  	  		  		  		    	 		 		   		 		  
    :rtype: numpy.ndarray  		  	   		  	  		  		  		    	 		 		   		 		  
    """

    np.random.seed(seed)
    rows = 500
    x = np.random.choice(200, (rows, 3))
    coefficient = np.random.choice(10, 3)
    param = np.random.choice(20, 3)
    first_feature = (x[:, 0] > np.mean(x[:, 0])) * coefficient[0] + param[0]
    second_feature = (x[:, 1] < np.mean(x[:, 1])) * coefficient[1] + param[1]
    third_feature = (x[:, 2] < np.mean(x[:, 2])) * coefficient[2] + param[2]
    y = first_feature ** 2 + second_feature + third_feature

    return x, y


def author():
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		  	  		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		  	  		  		  		    	 		 		   		 		  
    """
    return "pnguyen340"  # Change this to your user ID
